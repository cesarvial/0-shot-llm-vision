#!/usr/bin/env python3
"""
xai_full_explain.py (updated: save ResNet predictions + annotated overlays)

Usage (example):
 python xai_full_explain.py --img_emb data/coco_resnet50_img.pt \
   --text_emb data/coco_allroberta_text.pt --coco_root ./coco_dataset \
   --split train2017 --n_queries 3 --topk 5 --top_text_k 5 --nsamples 200

Notes:
 - If you want human-readable ImageNet labels, put a file named
   `imagenet_classes.txt` next to this script (1000 lines, one label per index 0..999).
 - Otherwise the script will use fallback labels like "class_123".
"""
import os
import json
import argparse
import re
import random
import hashlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer
import shap
import matplotlib.pyplot as plt

# ----------------- Helpers -----------------
def load_embedding_file(path):
    obj = torch.load(path, map_location='cpu')
    if torch.is_tensor(obj):
        return None, obj
    if isinstance(obj, dict):
        for k in ('embeddings','image_embeddings','image_representations','image_reps','emb','text_embeddings'):
            if k in obj and torch.is_tensor(obj[k]):
                return obj.get('image_ids', None), obj[k]
        def _find(o):
            if torch.is_tensor(o): return o
            if isinstance(o, dict):
                for v in o.values():
                    t = _find(v)
                    if t is not None: return t
            if isinstance(o, (list,tuple)):
                for v in o:
                    t = _find(v)
                    if t is not None: return t
            return None
        t = _find(obj)
        return obj.get('image_ids', None), t
    raise RuntimeError(f"Unsupported embedding file format: {path}")

def ensure_dir(p):
    if p is None or p == '':
        return
    os.makedirs(p, exist_ok=True)

def sanitize_folder_name(s, maxlen=80):
    s = s.replace('\n',' ').strip()
    s = re.sub(r'\s+',' ', s)
    s = re.sub(r'[^A-Za-z0-9 _\-\.,]', '_', s)
    if len(s) > maxlen:
        s = s[:maxlen].rsplit(' ',0)[0]
    s = s.strip()
    return s.replace(' ','_') if s else 'desc'

def short_hash(s, n=8):
    import hashlib
    return hashlib.sha1(s.encode('utf8')).hexdigest()[:n]

def find_image_file(coco_root, file_name, splits=('val2017','train2017')):
    for sp in splits:
        p = os.path.join(coco_root, sp, file_name)
        if os.path.isfile(p):
            return p
    if os.path.isfile(file_name):
        return file_name
    return None

def load_merged_coco_captions(coco_root, splits=('val2017','train2017')):
    imgs = {}
    imgToAnns = {}
    for sp in splits:
        ann_path = os.path.join(coco_root, 'annotations', f'captions_{sp}.json')
        if not os.path.isfile(ann_path): continue
        with open(ann_path, 'r', encoding='utf8') as f:
            j = json.load(f)
        for im in j.get('images', []):
            imgs[int(im['id'])] = im
        for ann in j.get('annotations', []):
            iid = int(ann['image_id'])
            imgToAnns.setdefault(iid, []).append(ann)
    class MiniCOCO:
        def __init__(self, imgs_map, imgToAnns_map):
            self.imgs = imgs_map
            self.imgToAnns = imgToAnns_map
        def loadImgs(self, ids):
            return [self.imgs[int(i)] for i in ids]
    return MiniCOCO(imgs, imgToAnns)

def tokenize_text(s):
    return s.strip().split()

def detokenize(tokens):
    return " ".join(tokens).strip()

def build_texts_from_masks(tokens, mask_matrix):
    texts = []
    for row in mask_matrix:
        kept = [t for t,m in zip(tokens,row) if float(m) > 0.5]
        texts.append(detokenize(kept) if kept else "")
    return texts

# robust text size measurement
def measure_text(draw, txt, font):
    try:
        bbox = draw.textbbox((0,0), txt, font=font)
        return int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
    except Exception:
        pass
    try:
        return draw.textsize(txt, font=font)
    except Exception:
        pass
    try:
        return font.getsize(txt)
    except Exception:
        pass
    return max(8, int(len(txt)*6)), 11

def save_token_highlight(tokens, shap_vals, out_path, cmap_name='RdBu_r', font_size=14, pad=4):
    ensure_dir(os.path.dirname(out_path) or ".")
    max_abs = max(1e-8, float(np.max(np.abs(shap_vals))))
    norm = [float(v)/max_abs for v in shap_vals]
    cmap = plt.get_cmap(cmap_name)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new('RGB', (10,10))
    draw = ImageDraw.Draw(dummy)
    widths = [measure_text(draw, t, font)[0] for t in tokens]
    heights = [measure_text(draw, t, font)[1] for t in tokens]
    h = max(heights) + 2*pad
    total_w = sum(widths) + (len(tokens)+1)*pad
    img = Image.new('RGB', (int(total_w), int(h)), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    x = pad
    for i, t in enumerate(tokens):
        val = norm[i]
        rgba = cmap((val+1)/2.0)
        color = tuple(int(255*c) for c in rgba[:3])
        w = widths[i]
        draw.rectangle([x-pad//2, pad//2, x + w + pad//2, h-pad//2], fill=color)
        draw.text((x, pad//2), t, fill=(0,0,0), font=font)
        x += w + pad
    img.save(out_path)

def save_waterfall_or_bar(tokens, shap_vals, explainer_expected_value, out_path):
    ensure_dir(os.path.dirname(out_path) or ".")
    try:
        exp = shap.Explanation(values=np.atleast_2d(shap_vals), base_values=np.atleast_1d(explainer_expected_value), data=np.atleast_2d(tokens))
        try:
            shap.plots.waterfall(exp[0], show=False)
            plt.gcf().savefig(out_path, bbox_inches='tight', dpi=150)
            plt.close()
            return
        except Exception:
            try:
                shap.plots._waterfall.waterfall_legacy(exp[0])
                plt.gcf().savefig(out_path, bbox_inches='tight', dpi=150)
                plt.close()
                return
            except Exception:
                pass
    except Exception:
        pass
    plt.figure(figsize=(max(4, len(tokens)*0.5), 3))
    colors = ['tab:red' if v>0 else 'tab:blue' for v in shap_vals]
    plt.bar(range(len(tokens)), shap_vals, color=colors)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right', fontsize=8)
    plt.ylabel('SHAP value')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- GradCAM for ResNet50 ----------
class GradCAMResNet50:
    def __init__(self, device, imagenet_labels=None):
        self.device = device
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.target_layer = self.model.layer4[-1].conv3
        self.saved_acts = {}
        self.saved_grads = {}
        self.imagenet_labels = imagenet_labels

    def _fh(self, module, inp, outp): self.saved_acts['val'] = outp.detach()
    def _bh(self, module, grad_in, grad_out): self.saved_grads['val'] = grad_out[0].detach()

    def run_on_pil(self, pil_img, annotate_label=True):
        """
        Returns: orig_u8, cam_gray_u8, overlay_u8, pred_class_index, top5_list
        where top5_list = [(idx, prob), ...]
        """
        fh = self.target_layer.register_forward_hook(self._fh)
        bh = self.target_layer.register_backward_hook(self._bh)
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        inp = transform(pil_img).unsqueeze(0).to(self.device)
        inp.requires_grad = True

        out = self.model(inp)  # logits
        probs = F.softmax(out, dim=1)
        topk = torch.topk(probs, k=5, dim=1)
        topk_vals = topk.values[0].detach().cpu().numpy().tolist()
        topk_idx = topk.indices[0].detach().cpu().numpy().tolist()
        pred = int(out.argmax(dim=1).item())

        # backward on predicted class (standard Grad-CAM)
        self.model.zero_grad()
        one_hot = torch.zeros_like(out); one_hot[0,pred] = 1.0
        out.backward(gradient=one_hot)

        if 'val' not in self.saved_acts or 'val' not in self.saved_grads:
            fh.remove(); bh.remove()
            raise RuntimeError("Grad-CAM hooks failed")
        acts = self.saved_acts['val']; grads = self.saved_grads['val']
        weights = torch.mean(grads, dim=(2,3), keepdim=True)
        cam_map = (weights * acts).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)
        cam = cam_map.squeeze(0).squeeze(0).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0: cam = cam / cam.max()
        ow, oh = pil_img.size
        cam_img = Image.fromarray((cam*255).astype('uint8')).resize((ow,oh), resample=Image.BILINEAR)
        cam_np = np.array(cam_img).astype(np.float32)/255.0
        img_np = np.array(pil_img).astype(np.float32)/255.0
        heatmap = plt.get_cmap('jet')(cam_np)[:,:,:3]
        overlay = 0.5*img_np + 0.5*heatmap; overlay = np.clip(overlay,0,1)
        fh.remove(); bh.remove()
        self.saved_acts.clear(); self.saved_grads.clear()

        orig_u8 = np.uint8(img_np*255)
        cam_gray_u8 = np.uint8(cam_np*255)
        overlay_u8 = np.uint8(overlay*255)

        # annotate overlay image with predicted label (if labels available)
        top5 = list(zip(topk_idx, topk_vals))
        if annotate_label:
            label = None
            if self.imagenet_labels is not None and len(self.imagenet_labels) > pred:
                label = self.imagenet_labels[pred]
            else:
                label = f"class_{pred}"
            labtext = f"{label} ({topk_vals[0]:.2f})"
            try:
                pil_ov = Image.fromarray(overlay_u8).convert('RGB')
                draw = ImageDraw.Draw(pil_ov)
                # make font 2.5x larger than previous default (previously 16 -> now 40)
                base_font_size = 16
                label_font_size = max(12, int(base_font_size * 2.5))
                try:
                    font = ImageFont.truetype("arial.ttf", label_font_size)
                except Exception:
                    font = ImageFont.load_default()
                text_w, text_h = measure_text(draw, labtext, font)
                # draw semi-transparent rectangle and text
                margin = max(4, label_font_size // 6)
                rect_xy = (6, 6, 6 + text_w + 2*margin, 6 + text_h + 2*margin)
                # Use a slightly transparent rectangle: create RGBA then paste if available
                try:
                    overlay_layer = Image.new('RGBA', pil_ov.size, (255,255,255,0))
                    ov_draw = ImageDraw.Draw(overlay_layer)
                    ov_draw.rectangle(rect_xy, fill=(255,255,255,200))
                    pil_ov = Image.alpha_composite(pil_ov.convert('RGBA'), overlay_layer).convert('RGB')
                    draw = ImageDraw.Draw(pil_ov)
                except Exception:
                    draw.rectangle(rect_xy, fill=(255,255,255))
                draw.text((6+margin, 6+margin), labtext, fill=(0,0,0), font=font)
                overlay_u8 = np.array(pil_ov)
            except Exception:
                pass

        return orig_u8, cam_gray_u8, overlay_u8, pred, top5

# composite 2xK with title (originals top, overlays bottom)
def save_2xk_with_title(orig_images, overlay_images, title_text, out_path, thumb_size=(256,256), title_height=48, bg=(255,255,255)):
    from PIL import ImageDraw, ImageFont
    K = len(orig_images)
    if K == 0: return
    tw, th = thumb_size
    W = tw * K
    H = title_height + th * 2
    canvas = Image.new('RGB', (W, H), color=bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    max_w = W - 10
    words = title_text.split()
    lines = []; cur=""
    for w in words:
        trial = (cur + " " + w).strip()
        tw_text, th_text = measure_text(draw, trial, font)
        if tw_text <= max_w:
            cur = trial
        else:
            if cur: lines.append(cur)
            cur = w
        if len(lines) >= 2: break
    if cur and len(lines) < 2: lines.append(cur)
    if len(lines)==0: lines=[title_text[:min(len(title_text),60)]]
    y = max(4, (title_height - sum(measure_text(draw,ln,font)[1] for ln in lines) - (len(lines)-1)*2)//2)
    for ln in lines[:2]:
        tw_text, th_text = measure_text(draw, ln, font)
        draw.text(((W - tw_text)//2, y), ln, fill=(0,0,0), font=font)
        y += th_text + 2
    for i in range(K):
        o = orig_images[i]; ov = overlay_images[i]
        o_p = Image.fromarray(o) if isinstance(o, np.ndarray) else o
        ov_p = Image.fromarray(ov) if isinstance(ov, np.ndarray) else ov
        o_r = o_p.resize((tw, th), resample=Image.BILINEAR)
        ov_r = ov_p.resize((tw, th), resample=Image.BILINEAR)
        canvas.paste(o_r, (i*tw, title_height))
        canvas.paste(ov_r, (i*tw, title_height + th))
    ensure_dir(os.path.dirname(out_path) or ".")
    canvas.save(out_path)

    # Add near top of script (imports needed: os, urllib.request, warnings)
import os, warnings
try:
    import urllib.request
except Exception:
    urllib = None

def ensure_imagenet_labels(save_path="imagenet_classes.txt", prefer_torchvision=True, verbose=True):
    """
    Ensure a local file 'imagenet_classes.txt' exists and return labels list.
    Strategy:
      1) Try torchvision's weights.meta['categories'] (if torchvision >= ~0.13).
      2) Otherwise, download the canonical file from PyTorch Hub raw on GitHub.
    Returns: list of 1000 labels (strings) or None on failure.
    """
    # 1) If already present, read and return:
    if os.path.isfile(save_path):
        if verbose: print(f"Found existing labels file: {save_path}")
        with open(save_path, 'r', encoding='utf8') as f:
            labels = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(labels) >= 1000:
            return labels
        else:
            if verbose: print("Found labels file but it has unexpected length; will try to regenerate.")

    # 2) Try torchvision weights metadata
    if prefer_torchvision:
        try:
            import torchvision
            # modern torchvision provides weights enum with meta
            # safe attempt for ResNet50 weights
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT
                meta = weights.meta
                cats = meta.get("categories", None)
                if cats and len(cats) >= 1000:
                    # save to file
                    with open(save_path, 'w', encoding='utf8') as f:
                        for c in cats:
                            f.write(c + "\n")
                    if verbose: print("Saved ImageNet labels from torchvision weights ->", save_path)
                    return cats
            except Exception:
                # older torchvision API fallback (some versions expose weights differently)
                try:
                    # try model_zoo style
                    from torchvision import models
                    m = models.resnet50(pretrained=False)
                    # no meta available; fallthrough to download
                except Exception:
                    pass
        except Exception:
            if verbose: print("torchvision not available or no meta labels; will try to download.")

    # 3) Fallback: download canonical file from PyTorch Hub raw (GitHub)
    # URL: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    if urllib is not None:
        try:
            if verbose: print("Downloading ImageNet labels from", url)
            resp = urllib.request.urlopen(url, timeout=20)
            data = resp.read().decode('utf8').splitlines()
            labels = [ln.strip() for ln in data if ln.strip()]
            if len(labels) >= 1000:
                with open(save_path, 'w', encoding='utf8') as f:
                    for c in labels:
                        f.write(c + "\n")
                if verbose: print("Saved downloaded ImageNet labels ->", save_path)
                return labels
            else:
                if verbose: print("Downloaded labels but length unexpected:", len(labels))
        except Exception as e:
            if verbose: print("Failed to download ImageNet labels:", e)
    else:
        if verbose: print("urllib not available; cannot download labels")

    # If all fails return None
    if verbose: print("Could not obtain ImageNet labels automatically.")
    return None


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_emb', default='data/coco_resnet50_img.pt')
    parser.add_argument('--text_emb', default='data/coco_allroberta_text.pt')
    parser.add_argument('--coco_root', default='./coco_dataset')
    parser.add_argument('--split', default='train2017')
    parser.add_argument('--n_queries', type=int, default=3)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--top_text_k', type=int, default=5)
    parser.add_argument('--nsamples', type=int, default=200)
    parser.add_argument('--map_subsample', type=int, default=20000, help='0 for no subsample')
    parser.add_argument('--device', default=None)
    parser.add_argument('--out_dir', default='XAI_results')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # optionally load ImageNet human labels if present
    imagenet_labels = ensure_imagenet_labels(save_path='imagenet_classes.txt', prefer_torchvision=True)

    # check input files
    if not os.path.isfile(args.img_emb):
        alt = args.img_emb.replace('_img.pt','_val2017_img.pt')
        if os.path.isfile(alt): args.img_emb = alt
        else: raise FileNotFoundError(args.img_emb)
    if not os.path.isfile(args.text_emb):
        alt = args.text_emb.replace('_text.pt','_val2017_text.pt')
        if os.path.isfile(alt): args.text_emb = alt
        else: raise FileNotFoundError(args.text_emb)

    img_ids, img_tensor = load_embedding_file(args.img_emb)
    text_ids, text_tensor = load_embedding_file(args.text_emb)
    if img_tensor is None or text_tensor is None:
        raise RuntimeError("Could not load embeddings.")
    img_embs = img_tensor.cpu().numpy()
    text_embs = text_tensor.cpu().numpy()
    print("Image emb shape:", img_embs.shape, "Text emb shape:", text_embs.shape)

    # train mapper if dims differ
    if img_embs.shape[1] != text_embs.shape[1]:
        print("[INFO] dims differ: training linear mapper image->text (Ridge)")
        N = img_embs.shape[0]
        if args.map_subsample and args.map_subsample > 0 and args.map_subsample < N:
            idxs = np.random.default_rng().choice(N, size=args.map_subsample, replace=False)
            X = img_embs[idxs]; Y = text_embs[idxs]
            print("  training on subsample", len(idxs))
        else:
            X = img_embs; Y = text_embs
            print("  training on all pairs (may be slow)")
        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(X, Y)
        map_image_to_text = lambda x: reg.predict(x)
        text_embs_norm = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-12)
    else:
        map_image_to_text = lambda x: x
        text_embs_norm = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-12)

    # text encoder for SHAP
    print("Loading SentenceTransformer (for SHAP encodes)...")
    text_encoder = SentenceTransformer('all-roberta-large-v1', device='cpu')
    print("loaded text encoder.")

    # load captions metadata
    coco = load_merged_coco_captions(args.coco_root, splits=('val2017','train2017'))
    splits_search = ('val2017','train2017')

    N = img_embs.shape[0]
    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random.default_rng()
    query_idxs = rng.choice(np.arange(N), size=min(args.n_queries, N), replace=False).tolist()
    print("Query positions:", query_idxs)

    gradcam = GradCAMResNet50(device=device, imagenet_labels=imagenet_labels)
    ensure_dir(args.out_dir)

    for qpos in query_idxs:
        q_img_id = int(img_ids[qpos]) if img_ids is not None else int(qpos)
        q_out_dir = os.path.join(args.out_dir, str(q_img_id))
        ensure_dir(q_out_dir)

        # derive short description via text embeddings (for title)
        mapped_q = map_image_to_text(img_embs[qpos:qpos+1])
        mq_norm = mapped_q / (np.linalg.norm(mapped_q, axis=1, keepdims=True) + 1e-12)
        sims_text = (mq_norm @ text_embs_norm.T)[0]
        top_text_idxs = np.argsort(-sims_text)[:args.top_text_k]
        found_caps = []
        for tidx in top_text_idxs:
            timgid = int(text_ids[tidx]) if text_ids is not None else None
            if timgid is not None:
                caps = [a.get('caption','').strip() for a in coco.imgToAnns.get(timgid, []) if a.get('caption','').strip()]
                for c in caps:
                    if c not in found_caps: found_caps.append(c)
            if len(found_caps) >= 3: break
        desc = " / ".join(found_caps[:2]) if found_caps else f"img_{q_img_id}"

        # nearest neighbor images (visual)
        q_emb = img_embs[qpos:qpos+1]
        sims_imgs = (q_emb @ img_embs.T)[0]
        order = np.argsort(-sims_imgs)[:args.topk]
        if order[0] != qpos:
            order = np.concatenate(([qpos], order[order != qpos]))[:args.topk]

        orig_list = []; overlay_list = []

        # storage for predictions summary (top-5)
        preds_summary = []

        for rank, idx in enumerate(order):
            imgid = int(img_ids[idx]) if img_ids is not None else int(idx)
            meta = coco.imgs.get(imgid, None)
            if meta is None:
                print(f"Warning: metadata for {imgid} not found (skipping neighbor).")
                continue
            fname = meta.get('file_name')
            img_path = find_image_file(args.coco_root, fname, splits_search)
            if img_path is None:
                print(f"Warning: image file for {imgid} not found (skipping).")
                continue
            pil = Image.open(img_path).convert('RGB')

            orig_u8, cam_gray_u8, overlay_u8, pred, top5 = gradcam.run_on_pil(pil, annotate_label=True)
            # save files per neighbor
            base = f"rank{rank:02d}_{imgid}"
            ensure_dir(q_out_dir)
            Image.fromarray(orig_u8).save(os.path.join(q_out_dir, base + "_orig.jpg"))
            Image.fromarray(cam_gray_u8).convert('L').save(os.path.join(q_out_dir, base + "_heatmap.jpg"))
            Image.fromarray(overlay_u8).save(os.path.join(q_out_dir, base + "_overlay.jpg"))
            # also save overlay label included version (same file already has label)
            Image.fromarray(overlay_u8).save(os.path.join(q_out_dir, base + "_overlay_label.jpg"))

            # append for composite
            orig_list.append(orig_u8); overlay_list.append(overlay_u8)

            # map top5 to labels if possible
            mapped_top5 = []
            for cidx, prob in top5:
                lbl = None
                if imagenet_labels is not None and len(imagenet_labels) > cidx:
                    lbl = imagenet_labels[cidx]
                else:
                    lbl = f"class_{cidx}"
                mapped_top5.append((cidx, lbl, float(prob)))
            preds_summary.append((imgid, mapped_top5))

        # save a predictions.txt listing top-5 for each neighbor
        txt_path = os.path.join(q_out_dir, "predictions.txt")
        with open(txt_path, 'w', encoding='utf8') as f:
            f.write(f"Query image id: {q_img_id}\nDescription (inferred): {desc}\n\n")
            for imgid, top5 in preds_summary:
                f.write(f"Image {imgid} top-5 predictions:\n")
                for idx, lbl, prob in top5:
                    f.write(f"  {idx}\t{lbl}\t{prob:.4f}\n")
                f.write("\n")

        # save 2xK comparison with overlays that already contain the label text
        comp_path = os.path.join(q_out_dir, f"comparison_{q_img_id}.jpg")
        save_2xk_with_title(orig_list, overlay_list, desc, comp_path, thumb_size=(256,256))
        print(f"Saved visual comparison to {comp_path} and predictions saved to {txt_path}")

        # For each caption found (from earlier top_text_idxs), run SHAP and save results
        all_captions = []
        for tidx in top_text_idxs:
            timgid = int(text_ids[tidx]) if text_ids is not None else None
            if timgid is None: continue
            caps = [a.get('caption','').strip() for a in coco.imgToAnns.get(timgid, []) if a.get('caption','').strip()]
            for c in caps:
                if c not in all_captions: all_captions.append(c)
        if len(all_captions) == 0:
            caps_q = [a.get('caption','').strip() for a in coco.imgToAnns.get(q_img_id, []) if a.get('caption','').strip()]
            all_captions = caps_q[:args.top_text_k]

        mapped_q = map_image_to_text(img_embs[qpos:qpos+1])
        mapped_q = mapped_q / (np.linalg.norm(mapped_q, axis=1, keepdims=True) + 1e-12)

        for ci, caption in enumerate(all_captions[:args.top_text_k]):
            if not caption or caption.strip()=="":
                continue
            tokens = tokenize_text(caption)
            if len(tokens)==0: continue
            if len(tokens) > 30:
                tokens = tokens[:30]
            num_tokens = len(tokens)
            def model_from_mask(mask_matrix):
                texts = build_texts_from_masks(tokens, mask_matrix)
                emb = text_encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
                sims = (emb_n @ mapped_q.T).reshape(-1)
                return sims
            background = np.zeros((1, num_tokens))
            explainer = shap.KernelExplainer(model_from_mask, background)
            x_to_explain = np.ones((1, num_tokens))
            print(f"Running SHAP for image {q_img_id} caption #{ci+1}/{len(all_captions)} (nsamples={args.nsamples})")
            shap_vals = explainer.shap_values(x_to_explain, nsamples=args.nsamples)
            if isinstance(shap_vals, list):
                shap_arr = np.array(shap_vals[0]).reshape(-1)
            else:
                shap_arr = np.array(shap_vals).reshape(-1)
            h = short_hash(caption)
            token_path = os.path.join(q_out_dir, f"shap_tokens_{ci:02d}_{h}.png")
            waterfall_path = os.path.join(q_out_dir, f"shap_waterfall_{ci:02d}_{h}.png")
            ensure_dir(q_out_dir)
            save_token_highlight(tokens, shap_arr, token_path)
            ev = getattr(explainer, "expected_value", 0.0)
            save_waterfall_or_bar(tokens, shap_arr, ev, waterfall_path)
            with open(os.path.join(q_out_dir, f"caption_{ci:02d}_{h}.txt"), 'w', encoding='utf8') as f:
                f.write(caption + "\n")
            print("  saved SHAP outputs:", token_path, waterfall_path)

    print("Done.")

if __name__ == "__main__":
    main()
