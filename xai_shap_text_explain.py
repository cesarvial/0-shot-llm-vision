#!/usr/bin/env python3
"""
xai_shap_text_explain.py (patched, robust)

Explain which words in the caption drive similarity to an image embedding using SHAP
(KernelExplainer). Also saves Grad-CAM overlays.

Usage example:
 python xai_shap_text_explain.py --img_emb data/coco_resnet50_img.pt \
   --text_emb data/coco_allroberta_text.pt --coco_root ./coco_dataset \
   --n_queries 2 --nsamples 100 --map_subsample 20000

Key fixes compared to earlier version:
 - merges captions_train2017.json + captions_val2017.json to avoid KeyError
 - searches for images in both train2017/ and val2017/ folders
 - skips empty caption entries and falls back to the query's COCO captions
 - optional subsampling for Ridge training (--map_subsample)

 python xai_shap_text_explain.py --img_emb data/coco_resnet50_img.pt --text_emb data/coco_allroberta_text.pt --n_queries 2 --nsamples 100 --map_subsample 20000
"""
import os
import json
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import re
import random

# --------- Helpers ----------
def load_embedding_file(path):
    obj = torch.load(path, map_location='cpu')
    if torch.is_tensor(obj):
        return None, obj
    if isinstance(obj, dict):
        for k in ('embeddings','image_embeddings','image_representations','image_reps','emb','text_embeddings'):
            if k in obj and torch.is_tensor(obj[k]):
                return obj.get('image_ids', None), obj[k]
        # fallback: recursive search for first tensor
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

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def sanitize_folder_name(s, maxlen=80):
    s = s.replace('\n',' ').strip()
    s = re.sub(r'\s+',' ', s)
    s = re.sub(r'[^A-Za-z0-9 _\-\.,]', '_', s)
    if len(s) > maxlen:
        s = s[:maxlen].rsplit(' ',0)[0]
    return s.replace(' ', '_')

def find_image_file(coco_root, file_name, splits=('val2017','train2017')):
    for sp in splits:
        p = os.path.join(coco_root, sp, file_name)
        if os.path.isfile(p):
            return p
    # last resort: try file_name as-is
    if os.path.isfile(file_name):
        return file_name
    return None

# Build a merged minimal COCO object from train+val caption files
def load_merged_coco_captions(coco_root, splits=('val2017','train2017')):
    imgs = {}
    imgToAnns = {}
    for sp in splits:
        ann_path = os.path.join(coco_root, 'annotations', f'captions_{sp}.json')
        if not os.path.isfile(ann_path):
            continue
        with open(ann_path, 'r', encoding='utf8') as f:
            j = json.load(f)
        for im in j.get('images', []):
            imgs[int(im['id'])] = im
        for ann in j.get('annotations', []):
            iid = int(ann['image_id'])
            imgToAnns.setdefault(iid, []).append(ann)
    # minimal wrapper like pycocotools COCO for loadImgs and imgToAnns lookup
    class MiniCOCO:
        def __init__(self, imgs_map, imgToAnns_map):
            self.imgs = imgs_map
            self.imgToAnns = imgToAnns_map
        def loadImgs(self, ids):
            return [self.imgs[int(i)] for i in ids]
    return MiniCOCO(imgs, imgToAnns)

# simple whitespace tokenizer and detokenizer
def tokenize_text(s):
    return s.strip().split()

def detokenize(tokens):
    return " ".join(tokens).strip()

def build_texts_from_masks(tokens, mask_matrix):
    texts = []
    for row in mask_matrix:
        kept = [t for t,m in zip(tokens, row) if float(m) > 0.5]
        if len(kept) == 0:
            texts.append("")   # SentenceTransformer can encode empty string
        else:
            texts.append(detokenize(kept))
    return texts

# Save token highlight image
def save_token_highlight(tokens, shap_vals, out_path, cmap_name='RdBu_r', font_size=14, pad=4):
    """
    Draw tokens on a horizontal strip where each token's background color
    encodes its SHAP value (red = positive, blue = negative).
    Robust across Pillow versions (uses textbbox / font.getsize fallbacks).
    """
    max_abs = max(1e-8, np.max(np.abs(shap_vals)))
    norm = shap_vals / max_abs
    cmap = plt.get_cmap(cmap_name)

    # choose font with fallback
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # helper to measure text size robustly
    # uses an ImageDraw instance's textbbox if available, otherwise font.getsize/getbbox, then a rough estimate
    dummy = Image.new('RGB', (10, 10))
    draw = ImageDraw.Draw(dummy)

    def measure_text(txt):
        # prefer draw.textbbox (gives precise bbox with font)
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        except Exception:
            pass
        # fallback to font.getsize
        try:
            size = font.getsize(txt)
            return int(size[0]), int(size[1])
        except Exception:
            pass
        # fallback to font.getbbox (newer Pillow)
        try:
            bbox = font.getbbox(txt)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        except Exception:
            pass
        # final cheap estimate (per-character)
        return max(8, int(len(txt) * (font_size * 0.6))), font_size + 2

    widths = [measure_text(t)[0] for t in tokens]
    heights = [measure_text(t)[1] for t in tokens]
    h = max(heights) + 2 * pad
    total_w = sum(widths) + (len(tokens) + 1) * pad

    img = Image.new('RGB', (int(total_w), int(h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = pad
    for i, t in enumerate(tokens):
        val = float(norm[i])
        rgba = cmap((val + 1) / 2.0)  # map -1..1 to 0..1
        color = tuple(int(255 * c) for c in rgba[:3])
        w = widths[i]
        draw.rectangle([x - pad // 2, pad // 2, x + w + pad // 2, h - pad // 2], fill=color)
        draw.text((x, pad // 2), t, fill=(0, 0, 0), font=font)
        x += w + pad
    img.save(out_path)


def save_shap_bar(tokens, shap_vals, out_path):
    plt.figure(figsize=(max(4, len(tokens)*0.5), 3))
    colors = ['tab:red' if v>0 else 'tab:blue' for v in shap_vals]
    plt.bar(range(len(tokens)), shap_vals, color=colors)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right', fontsize=8)
    plt.ylabel('SHAP value (impact on similarity)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Grad-CAM ResNet50 (same approach)
class GradCAMResNet50:
    def __init__(self, device):
        self.device = device
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.target_layer = self.model.layer4[-1].conv3
        self.saved_acts = {}
        self.saved_grads = {}

    def _fh(self, module, inp, outp): self.saved_acts['val'] = outp.detach()
    def _bh(self, module, grad_in, grad_out): self.saved_grads['val'] = grad_out[0].detach()

    def run_on_pil(self, pil_img):
        fh = self.target_layer.register_forward_hook(self._fh)
        bh = self.target_layer.register_backward_hook(self._bh)
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        inp = transform(pil_img).unsqueeze(0).to(self.device)
        inp.requires_grad = True
        out = self.model(inp)
        pred = int(out.argmax(dim=1).item())
        self.model.zero_grad()
        one_hot = torch.zeros_like(out); one_hot[0,pred] = 1.0
        out.backward(gradient=one_hot)
        if 'val' not in self.saved_acts or 'val' not in self.saved_grads:
            fh.remove(); bh.remove(); raise RuntimeError("Grad-CAM hooks failed")
        acts = self.saved_acts['val']; grads = self.saved_grads['val']
        weights = torch.mean(grads, dim=(2,3), keepdim=True)
        cam_map = (weights * acts).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)
        cam = cam_map.squeeze(0).squeeze(0).cpu().numpy()
        cam -= cam.min(); 
        if cam.max() > 0: cam = cam / cam.max()
        ow, oh = pil_img.size
        cam_img = Image.fromarray((cam*255).astype('uint8')).resize((ow,oh), resample=Image.BILINEAR)
        cam_np = np.array(cam_img).astype(np.float32)/255.0
        img_np = np.array(pil_img).astype(np.float32)/255.0
        heatmap = plt.get_cmap('jet')(cam_np)[:,:,:3]
        overlay = 0.5*img_np + 0.5*heatmap; overlay = np.clip(overlay, 0, 1)
        fh.remove(); bh.remove()
        self.saved_acts.clear(); self.saved_grads.clear()
        return np.uint8(img_np*255), np.uint8(cam_np*255), np.uint8(overlay*255)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_emb', default='data/coco_resnet50_img.pt')
    parser.add_argument('--text_emb', default='data/coco_allroberta_text.pt')
    parser.add_argument('--coco_root', default='./coco_dataset')
    parser.add_argument('--split', default='train2017', help='primary split for image file lookup (train2017 or val2017)')
    parser.add_argument('--n_queries', type=int, default=3)
    parser.add_argument('--query_ids', default=None, help='comma-separated COCO image ids to analyze (overrides random)')
    parser.add_argument('--top_text_k', type=int, default=5, help='how many nearest text embeddings to consider')
    parser.add_argument('--max_tokens', type=int, default=20)
    parser.add_argument('--nsamples', type=int, default=200, help='SHAP KernelExplainer nsamples')
    parser.add_argument('--device', default=None, help='cuda:0 or cpu (auto)')
    parser.add_argument('--out_dir', default='XAI_results')
    parser.add_argument('--map_subsample', type=int, default=20000, help='subsample size to train Ridge mapper (0=use all)')
    args = parser.parse_args()

    # device for gradcam (image model)
    device = torch.device(args.device) if args.device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # load embeddings
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
        raise RuntimeError("Could not load embeddings correctly.")
    img_embs = img_tensor.cpu().numpy()
    text_embs = text_tensor.cpu().numpy()
    print("Loaded image embeddings:", img_embs.shape)
    print("Loaded text embeddings:", text_embs.shape)

    # train mapper if dims differ (optionally subsample)
    map_image_to_text = None
    text_embs_norm = None
    if img_embs.shape[1] != text_embs.shape[1]:
        print("[INFO] dims differ: training Ridge image->text mapper...")
        N_total = img_embs.shape[0]
        if args.map_subsample and args.map_subsample > 0 and args.map_subsample < N_total:
            idxs = np.random.default_rng().choice(N_total, size=args.map_subsample, replace=False)
            X = img_embs[idxs]
            Y = text_embs[idxs]
            print(f"  training on subsample {len(idxs)} / {N_total}")
        else:
            X = img_embs
            Y = text_embs
            print(f"  training on all {N_total} pairs (this may take a while)")
        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(X, Y)
        def map_fn(x): return reg.predict(x)
        map_image_to_text = map_fn
        norms = np.linalg.norm(text_embs, axis=1, keepdims=True); norms[norms==0]=1.0
        text_embs_norm = text_embs / norms
    else:
        map_image_to_text = lambda x: x
        norms = np.linalg.norm(text_embs, axis=1, keepdims=True); norms[norms==0]=1.0
        text_embs_norm = text_embs / norms

    # load sentence-transformer (CPU is fine for many small encodes)
    print("Loading sentence-transformer model (this may take a while)...")
    text_encoder = SentenceTransformer('all-roberta-large-v1', device='cpu')
    print("Text encoder loaded.")

    # load merged COCO captions (train+val)
    coco = load_merged_coco_captions(args.coco_root, splits=('val2017','train2017'))
    # For image file lookup we will search in both splits
    splits_search = ('val2017','train2017')

    # choose query indices
    N = img_embs.shape[0]
    if args.query_ids:
        req = [int(x.strip()) for x in args.query_ids.split(',') if x.strip()]
        if img_ids is None:
            raise RuntimeError("Image embedding file lacks image_ids; cannot use --query_ids.")
        id2idx = {int(i): idx for idx, i in enumerate(img_ids)}
        query_indices = [id2idx[i] for i in req if i in id2idx]
        if len(query_indices) == 0:
            raise RuntimeError("No valid query ids found in embeddings.")
    else:
        rng = np.random.default_rng()
        query_indices = rng.choice(np.arange(N), size=min(args.n_queries, N), replace=False).tolist()

    # ensure query indices are plain ints
    query_indices = [int(x) for x in query_indices]
    print("Query indices (positions):", query_indices)

    gradcam = GradCAMResNet50(device=device)
    ensure_dir(args.out_dir)

    for qpos in query_indices:
        q_img_id = int(img_ids[qpos]) if img_ids is not None else int(qpos)
        q_out_dir = os.path.join(args.out_dir, str(q_img_id))
        ensure_dir(q_out_dir)

        # map image embedding -> text space and normalize
        q_emb = img_embs[qpos:qpos+1]
        mapped = map_image_to_text(q_emb)
        mn = np.linalg.norm(mapped, axis=1, keepdims=True); mn[mn==0]=1.0
        mapped_n = mapped / mn

        # retrieve top text indices
        sims = (mapped_n @ text_embs_norm.T)[0]
        top_text_idxs = np.argsort(-sims)[:args.top_text_k]
        print(f"[Image {q_img_id}] top text idxs: {top_text_idxs} (sims: {sims[top_text_idxs]})")

        # pick first top_text idx that has associated caption(s) via text_ids -> coco
        chosen_caption = None
        chosen_text_idx = None
        for tidx in top_text_idxs:
            # map text idx -> possible image id (if text_ids present)
            text_img_id = int(text_ids[tidx]) if text_ids is not None else None
            captions = []
            if text_img_id is not None:
                captions = [a.get('caption','').strip() for a in coco.imgToAnns.get(text_img_id, []) if a.get('caption','').strip()]
            # if captions found, pick first
            if captions:
                chosen_caption = captions[0]
                chosen_text_idx = int(tidx)
                break
        # fallback: use captions of the query image itself (if any)
        if chosen_caption is None:
            anns_q = coco.imgToAnns.get(q_img_id, [])
            caps_q = [a.get('caption','').strip() for a in anns_q if a.get('caption','').strip()]
            if caps_q:
                chosen_caption = caps_q[0]
                chosen_text_idx = None
                print(f"  fallback: using query image's own caption(s)")
            else:
                # last resort: empty string
                chosen_caption = ""
                chosen_text_idx = None
                print(f"  warning: no captions available for top texts or query image; skipping SHAP for this query")

        # if chosen_caption empty -> skip SHAP (but still compute Grad-CAM)
        if chosen_caption.strip() == "":
            print(f"Caption empty for image {q_img_id}; skipping SHAP and computing only Grad-CAM.")
        else:
            # tokenize and maybe truncate
            tokens = tokenize_text(chosen_caption)
            if len(tokens) == 0:
                print("Tokenization produced empty list; skipping SHAP.")
            else:
                if len(tokens) > args.max_tokens:
                    tokens = tokens[:args.max_tokens]
                    print(f"Truncated caption to first {args.max_tokens} tokens for SHAP.")
                num_tokens = len(tokens)

                # model wrapper maps binary masks -> similarity score
                def model_from_mask(mask_matrix):
                    texts = build_texts_from_masks(tokens, mask_matrix)
                    emb = text_encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                    norms_e = np.linalg.norm(emb, axis=1, keepdims=True); norms_e[norms_e==0]=1.0
                    emb_n = emb / norms_e
                    sims_local = (emb_n @ mapped_n.T).reshape(-1)
                    return sims_local

                # background: zero-mask (all tokens removed)
                background = np.zeros((1, num_tokens))
                explainer = shap.KernelExplainer(model_from_mask, background)
                x_to_explain = np.ones((1, num_tokens))
                print("Running SHAP KernelExplainer (nsamples=", args.nsamples, ") ...")
                shap_vals = explainer.shap_values(x_to_explain, nsamples=args.nsamples)
                # convert shap_vals into flat 1D array
                if isinstance(shap_vals, list):
                    shap_arr = np.array(shap_vals[0]).reshape(-1)
                else:
                    shap_arr = np.array(shap_vals).reshape(-1)
                # save visualizations
                out_prefix = os.path.join(q_out_dir, f"shap_text_top1")
                save_token_highlight(tokens, shap_arr, out_prefix + ".png")
                save_shap_bar(tokens, shap_arr, out_prefix + "_bar.png")
                print("Saved SHAP visualizations to", q_out_dir)

        # compute and save Grad-CAM overlay for the query image
        # find path for the image file
        meta = None
        try:
            meta = coco.loadImgs([q_img_id])[0]
        except Exception as e:
            print("Warning: coco.loadImgs failed for id", q_img_id, "->", e)
            meta = None
        if meta is None or 'file_name' not in meta:
            print(f"Could not find metadata for image id {q_img_id} in merged captions. Skipping Grad-CAM.")
        else:
            fname = meta['file_name']
            img_path = find_image_file(args.coco_root, fname, splits_search)
            if img_path is None:
                print(f"Image file for id {q_img_id} ({fname}) not found in expected folders; skipping Grad-CAM.")
            else:
                pil = Image.open(img_path).convert('RGB')
                orig_u8, cam_gray_u8, overlay_u8 = gradcam.run_on_pil(pil)
                Image.fromarray(orig_u8).save(os.path.join(q_out_dir, f"{q_img_id}_orig.jpg"))
                Image.fromarray(cam_gray_u8).convert('L').save(os.path.join(q_out_dir, f"{q_img_id}_heatmap.jpg"))
                Image.fromarray(overlay_u8).save(os.path.join(q_out_dir, f"{q_img_id}_overlay.jpg"))
                print("Saved Grad-CAM overlays to", q_out_dir)

    print("All done.")

if __name__ == "__main__":
    main()
