# xai_neighbors_gradcam_with_text.py
"""
Like xai_neighbors_gradcam.py but uses text embeddings to infer a short description
for each analysis. The description is used to name the folder and added as the title
on the 2xK composite image.

Usage example:
  python xai_tests.py --embed_file data/coco_resnet50_img.pt --text_emb_file data/coco_allroberta_text.pt --coco_root ./coco_dataset --split train2017 --n_queries 5 --topk 5 --top_text_k 5

If --text_emb_file is omitted, folder names use the query image id only.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import torch.nn.functional as F
from pycocotools.coco import COCO
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import Ridge

# ---------- Helpers (loads & extraction) ----------
def load_embedding_file(path):
    """
    Load .pt file, return (image_ids or None, torch.Tensor embeddings on CPU).
    """
    obj = torch.load(path, map_location='cpu')
    if torch.is_tensor(obj):
        return None, obj
    if isinstance(obj, dict):
        # try typical keys
        for k in ('embeddings', 'image_embeddings', 'image_representations', 'image_reps', 'emb', 'text_embeddings'):
            if k in obj and torch.is_tensor(obj[k]):
                return obj.get('image_ids', None), obj[k]
        # fallback: recursive search
        def _find(o):
            if torch.is_tensor(o): return o
            if isinstance(o, dict):
                for v in o.values():
                    t = _find(v)
                    if t is not None: return t
            if isinstance(o, (list, tuple)):
                for v in o:
                    t = _find(v)
                    if t is not None: return t
            return None
        t = _find(obj)
        return obj.get('image_ids', None), t
    raise RuntimeError(f"Unsupported embedding file format: {path}")

def ensure_path_exists(p):
    os.makedirs(p, exist_ok=True)

def sanitize_folder_name(s, maxlen=60):
    # remove newlines, trim, replace non-alnum with underscores
    s = s.replace('\n', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^A-Za-z0-9 _\-]', '_', s)
    if len(s) > maxlen:
        s = s[:maxlen].rsplit(' ', 0)[0]
    s = s.strip()
    if len(s) == 0:
        s = 'desc'
    return s.replace(' ', '_')

# ---------- Grad-CAM core for ResNet50 ----------
class GradCAMResNet50:
    def __init__(self, device):
        self.device = device
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.target_layer = self.model.layer4[-1].conv3
        self.saved_activations = {}
        self.saved_gradients = {}

    def _forward_hook(self, module, inp, outp):
        self.saved_activations['value'] = outp.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.saved_gradients['value'] = grad_out[0].detach()

    def run_on_pil(self, pil_img):
        fh = self.target_layer.register_forward_hook(self._forward_hook)
        bh = self.target_layer.register_backward_hook(self._backward_hook)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        input_t = transform(pil_img).unsqueeze(0).to(self.device)
        input_t.requires_grad = True

        out = self.model(input_t)
        pred_class = int(out.argmax(dim=1).item())

        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, pred_class] = 1.0
        out.backward(gradient=one_hot)

        if 'value' not in self.saved_activations or 'value' not in self.saved_gradients:
            fh.remove(); bh.remove()
            raise RuntimeError("Grad-CAM: failed to capture activations or gradients.")

        activations = self.saved_activations['value']
        gradients = self.saved_gradients['value']

        weights = torch.mean(gradients, dim=(2,3), keepdim=True)
        cam_map = (weights * activations).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)

        cam = cam_map.squeeze(0).squeeze(0).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        orig_w, orig_h = pil_img.size
        cam_img = Image.fromarray(np.uint8(cam * 255)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        cam_np = np.array(cam_img).astype(np.float32) / 255.0

        img_np = np.array(pil_img).astype(np.float32) / 255.0
        heatmap = plt.get_cmap('jet')(cam_np)[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)

        cam_gray_u8 = np.uint8(cam_np * 255)
        overlay_u8 = np.uint8(overlay * 255)
        orig_u8 = np.uint8(img_np * 255)

        fh.remove(); bh.remove()
        self.saved_activations.clear(); self.saved_gradients.clear()

        return orig_u8, cam_gray_u8, overlay_u8

# ---------- composite builder with title ----------
def save_comparison_grid_with_title(orig_images, overlay_images, title_text, out_path, thumb_size=(256,256), title_height=40, bg_color=(255,255,255)):
    """
    Save 2xK grid with a title at top. Title is centered in a top banner of height title_height.
    orig_images / overlay_images: lists of PIL.Image or numpy arrays (HxWx3 uint8).
    """
    from PIL import ImageDraw, ImageFont

    K = len(orig_images)
    if K == 0:
        return
    tw, th = thumb_size
    grid_w = tw * K
    grid_h = title_height + (th * 2)
    grid = Image.new('RGB', (grid_w, grid_h), color=bg_color)

    # draw title area
    draw = ImageDraw.Draw(grid)
    # choose default font
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    # helper to measure text (robust across Pillow versions)
    def measure_text(txt, fnt):
        # Try textbbox (preferred, gives width+height)
        try:
            bbox = draw.textbbox((0, 0), txt, font=fnt)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            return int(w), int(h)
        except Exception:
            pass
        # Try textsize
        try:
            w, h = draw.textsize(txt, font=fnt)
            return int(w), int(h)
        except Exception:
            pass
        # Try font.getsize
        try:
            w, h = fnt.getsize(txt)
            return int(w), int(h)
        except Exception:
            pass
        # Fallback estimate
        return max(10, int(len(txt) * 6)), 11

    # wrap text into at most two lines to fit width
    max_width = grid_w - 10
    words = title_text.split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        tw_text, th_text = measure_text(trial, font)
        if tw_text <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
        # stop if we already have two lines
        if len(lines) >= 2:
            break
    if cur and len(lines) < 2:
        lines.append(cur)
    # ensure at least one line
    if len(lines) == 0:
        lines = [title_text[:max(20, min(len(title_text), 60))]]

    # Draw lines centered
    y0 = max(4, (title_height - sum(measure_text(ln, font)[1] for ln in lines) - (2 * (len(lines)-1)))//2)
    for ln in lines[:2]:
        tw_text, th_text = measure_text(ln, font)
        draw.text(((grid_w - tw_text) // 2, y0), ln, fill=(0,0,0), font=font)
        y0 += th_text + 2

    # paste images below title
    for i in range(K):
        o = orig_images[i]
        ov = overlay_images[i]
        if isinstance(o, np.ndarray):
            o_pil = Image.fromarray(o)
        else:
            o_pil = o
        if isinstance(ov, np.ndarray):
            ov_pil = Image.fromarray(ov)
        else:
            ov_pil = ov
        o_th = o_pil.resize((tw, th), resample=Image.BILINEAR)
        ov_th = ov_pil.resize((tw, th), resample=Image.BILINEAR)
        grid.paste(o_th, (i*tw, title_height))
        grid.paste(ov_th, (i*tw, title_height + th))

    grid.save(out_path)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_file', default='data/coco_resnet50_img.pt', help='image embedding .pt file (image ids + embeddings)')
    parser.add_argument('--text_emb_file', default=None, help='text embedding .pt file (optional) used to infer descriptions')
    parser.add_argument('--coco_root', default='./coco_dataset', help='COCO root folder containing train2017/val2017 and annotations/')
    parser.add_argument('--split', default='val2017', help='coco split folder name (train2017 or val2017)')
    parser.add_argument('--n_queries', type=int, default=3, help='number of random queries to process')
    parser.add_argument('--topk', type=int, default=5, help='top-k neighbors per query (including query itself)')
    parser.add_argument('--query_ids', default=None, help='comma-separated image ids to use as queries (overrides random sampling)')
    parser.add_argument('--out_dir', default='XAI_results', help='directory to save outputs')
    parser.add_argument('--device', default=None, help='cuda device string (e.g. cuda:0) or cpu (default: auto)')
    parser.add_argument('--thumb_size', default='256,256', help='thumbnail size for comparison grid as W,H (default 256,256)')
    parser.add_argument('--seed', type=int, default=None, help='optional random seed for reproducible sampling')
    parser.add_argument('--top_text_k', type=int, default=5, help='how many nearest text embeddings to consult per image')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(args.embed_file):
        alt = args.embed_file.replace('_val2017_img.pt', '_img.pt')
        if os.path.isfile(alt):
            args.embed_file = alt
        else:
            raise FileNotFoundError(f"Embedding file not found: {args.embed_file}")

    image_ids, emb_tensor = load_embedding_file(args.embed_file)
    embs = emb_tensor.cpu().numpy()
    N, D = embs.shape
    print(f"Loaded image embeddings: shape={embs.shape}")

    # load text embeddings if provided
    text_image_ids = None
    text_embs = None
    if args.text_emb_file:
        ...
        text_image_ids, text_tensor = load_embedding_file(args.text_emb_file)
        text_embs = text_tensor.cpu().numpy()
        print(f"Loaded text embeddings: shape={text_embs.shape}")

    map_image_to_text = None
    text_embs_norm = None

    if text_embs is not None:
        img_dim = embs.shape[1]
        txt_dim = text_embs.shape[1]
        if img_dim != txt_dim:
            print(f"[INFO] image dim ({img_dim}) != text dim ({txt_dim}) — training linear mapper image->text (Ridge).")
            # Train ridge on all pairs (X image -> Y text)
            # You can reduce data / subsample if training is slow
            reg = Ridge(alpha=1.0, fit_intercept=True)
            reg.fit(embs, text_embs)  # emb shape (N,img_dim) -> text_embs shape (N,txt_dim)
            def map_image_to_text(x):
                # x: numpy array shape (n, img_dim) -> returns (n, txt_dim)
                return reg.predict(x)
            # Pre-normalize text embeddings for fast cosine sims
            txt_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
            txt_norms[txt_norms == 0] = 1.0
            text_embs_norm = text_embs / txt_norms
        else:
            # same dimensionality — no mapping needed
            def map_image_to_text(x):
                return x
            # normalize text embeddings too
            txt_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
            txt_norms[txt_norms == 0] = 1.0
            text_embs_norm = text_embs / txt_norms

    ann_file = os.path.join(args.coco_root, 'annotations', f'captions_{"val" if args.split=="val2017" else "train"}2017.json')
    coco = COCO(ann_file)

    # choose queries
    if args.query_ids:
        raw = [int(x.strip()) for x in args.query_ids.split(',') if x.strip()]
        if image_ids is None:
            raise RuntimeError("Embedding file did not include image_ids; cannot map COCO ids to indices.")
        id_to_idx = {int(i): idx for idx, i in enumerate(image_ids)}
        query_indices = []
        for iid in raw:
            if iid not in id_to_idx:
                print(f"Warning: image id {iid} not found in embeddings (skipping).")
                continue
            query_indices.append(id_to_idx[iid])
        if len(query_indices) == 0:
            raise RuntimeError("No valid query ids found.")
    else:
        if args.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(args.seed)
        query_indices = rng.choice(np.arange(N), size=min(args.n_queries, N), replace=False).tolist()

    print("Query indices (embedding-array positions):", query_indices)

    gradcam = GradCAMResNet50(device=device)
    ensure_path_exists(args.out_dir)
    tw, th = [int(x) for x in args.thumb_size.split(',')]

    for qi, qidx in enumerate(query_indices):
        q_img_id = int(image_ids[qidx]) if image_ids is not None else int(qidx)

        # If text embeddings exist, infer a short description for the QUERY image itself:
        description = f"img_{q_img_id}"
        if text_embs is not None:
            q_emb = embs[qidx:qidx+1]
            mapped_q = map_image_to_text(q_emb) if map_image_to_text is not None else q_emb
            # normalize mapped_q
            mq_norms = np.linalg.norm(mapped_q, axis=1, keepdims=True)
            mq_norms[mq_norms == 0] = 1.0
            mapped_q_norm = mapped_q / mq_norms

            # compute cosine sims against pre-normalized text embeddings (fast dot product)
            sims_text = (mapped_q_norm @ text_embs_norm.T)[0]
            top_text_idxs = np.argsort(-sims_text)[:args.top_text_k]
            # collect captions from COCO for those text_image_ids
            found_caps = []
            for tidx in top_text_idxs:
                if text_image_ids is not None:
                    tid = int(text_image_ids[tidx])
                else:
                    tid = None
                if tid is not None:
                    anns = coco.imgToAnns.get(tid, [])
                    for a in anns:
                        txt = a.get('caption','').strip()
                        if txt and txt not in found_caps:
                            found_caps.append(txt)
                # stop early if we have enough unique lines
                if len(found_caps) >= 3:
                    break
            # build a short summary
            if len(found_caps) > 0:
                summary = " / ".join(found_caps[:2])
                description = summary
            else:
                description = f"img_{q_img_id}"

        # sanitize folder name and create folder
        safe = sanitize_folder_name(description, maxlen=60)
        q_out_dir = os.path.join(args.out_dir, f"{q_img_id}__{safe}")
        ensure_path_exists(q_out_dir)

        # compute nearest neighbors
        q_emb = embs[qidx:qidx+1]  # (1,D)
        sims = cosine_similarity(q_emb, embs)[0]
        order = np.argsort(-sims)[:args.topk]
        if order[0] != qidx:
            order = np.concatenate(([qidx], order[order != qidx]))[:args.topk]

        print(f"[{qi+1}/{len(query_indices)}] Query image id {q_img_id} -> description: {description}. topk indices: {order}")

        orig_list = []
        overlay_list = []

        for rank, idx in enumerate(order):
            img_id = int(image_ids[idx]) if image_ids is not None else int(idx)
            meta = coco.loadImgs(img_id)[0]
            fname = meta['file_name']
            img_path = os.path.join(args.coco_root, args.split, fname)
            pil = Image.open(img_path).convert('RGB')

            # infer text description for this neighbor too (optional, not used for folder name)
            # (we won't change folder name, it's based on query only)

            orig_u8, cam_gray_u8, overlay_u8 = gradcam.run_on_pil(pil)

            base_prefix = f"rank_{rank:02d}_{img_id}"
            Image.fromarray(orig_u8).save(os.path.join(q_out_dir, base_prefix + "_orig.jpg"))
            Image.fromarray(cam_gray_u8).convert('L').save(os.path.join(q_out_dir, base_prefix + "_heatmap.jpg"))
            Image.fromarray(overlay_u8).save(os.path.join(q_out_dir, base_prefix + "_overlay.jpg"))

            orig_list.append(Image.fromarray(orig_u8))
            overlay_list.append(Image.fromarray(overlay_u8))

        # build title (use the short summary we computed)
        title_text = description
        comp_path = os.path.join(q_out_dir, f"comparison_{q_img_id}.jpg")
        save_comparison_grid_with_title(orig_list, overlay_list, title_text, comp_path, thumb_size=(tw, th))
        print(f"Saved to: {q_out_dir}")

if __name__ == "__main__":
    main()
