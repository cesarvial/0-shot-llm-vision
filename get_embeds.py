# get_embeds.py
"""
Full get_embeds.py adapted for your project layout (./coco_dataset).

This script extracts image embeddings from several supported pretrained
backbones and saves them as .pt files in a data/ directory.

Supported models (attempted in this order):
  - dinov2   (via timm / huggingface if available)  -- name: 'dinov2'
  - clip     (OpenAI CLIP, via the 'clip' package or HuggingFace)
  - resnet50 (torchvision.models.resnet50) -- name: 'resnet50'
  - any timm model name as fallback

Usage examples:
  python get_embeds.py --m dinov2 --d coco --split val2017 --gpu 0
  python get_embeds.py --m clip --d coco --split train2017 --coco_root ./coco_dataset --batch_size 64

  Nosso caso:
  python get_embeds.py --m resnet50 --d coco --split val2017 --gpu 0 --batch_size 64
  python get_embeds.py --m allroberta --d coco --split val2017 --gpu 0

Notes:
 - You need `pycocotools`, `torch`, `torchvision`. For CLIP use `git+https://github.com/openai/CLIP.git` or `transformers`/`huggingface` equivalents.
 - If a model package is missing, the script will raise an ImportError describing what to install.
 - This script extracts global image embeddings. If you need patch/region embeddings, adapt the forward pass accordingly.
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoCaptions

# Helper: default COCO dataset layout in repo root
PROJECT_ROOT = os.path.abspath(os.getcwd())
COCO_DATA_ROOT = os.path.join(PROJECT_ROOT, "coco_dataset")
COCO_ROOT = os.path.join(COCO_DATA_ROOT, "val2017")
COCO_ANN = os.path.join(COCO_DATA_ROOT, "annotations", "instances_val2017.json")


def get_coco_paths(coco_root=None, split='val2017'):
    """
    Return (images_dir, ann_file) for a COCO layout located in the project root under ./coco_dataset.
    This prefers captions_{split}.json if present, otherwise falls back to instances_{split}.json.
    """
    if coco_root is None:
        coco_root = COCO_DATA_ROOT
    coco_root = os.path.abspath(coco_root)
    if split not in ('train2017', 'val2017'):
        raise ValueError("--split must be 'train2017' or 'val2017'")

    images_dir = os.path.join(coco_root, split)

    # prefer captions annotation file if available (required by CocoCaptions)
    cap_fname = f'captions_{"train" if split=="train2017" else "val"}2017.json'
    inst_fname = f'instances_{"train" if split=="train2017" else "val"}2017.json'

    ann_dir = os.path.join(coco_root, 'annotations')
    cap_path = os.path.join(ann_dir, cap_fname)
    inst_path = os.path.join(ann_dir, inst_fname)

    if os.path.isfile(cap_path):
        ann_file = cap_path
    elif os.path.isfile(inst_path):
        ann_file = inst_path
    else:
        raise FileNotFoundError(f"Neither {cap_fname} nor {inst_fname} found in {ann_dir}")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    return images_dir, ann_file


def build_transform(image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform


class CocoWrapper(CocoCaptions):
    """
    Wraps CocoCaptions to return (PIL image, caption_text_list, image_id)
    so mapping back to image ids is straightforward.
    """
    def __init__(self, root, annFile, transform=None):
        super().__init__(root=root, annFile=annFile)
        self.transform = transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img_id = self.ids[index]
        if self.transform is not None:
            img = self.transform(img)
        # target is a list of caption strings
        return img, target, img_id


def load_clip_openai(device):
    """
    Try loading OpenAI CLIP package model+preprocess.
    """
    try:
        import clip
        model, preprocess = clip.load('ViT-B/32', device=device)
        model.eval()
        return (model, preprocess), model.visual.output_dim
    except Exception:
        # Try HF CLIP
        try:
            from transformers import CLIPVisionModel, CLIPProcessor
            model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
            proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            model.eval()
            return (model, proc), model.config.hidden_size
        except Exception as e:
            raise ImportError("Could not load CLIP (openai clip or hf). Install one.")


def load_model(model_name, device):
    """
    Load supported models and return (model_or_tuple, feat_dim_or_None)

    model_or_tuple can be:
     - (model, preprocess) for CLIP style
     - model only for simple torch models
    """
    model_name = model_name.lower()
    # DINOv2 via timm / huggingface - try timm first
    if model_name == 'dinov2':
        try:
            import timm
            # try a common dinov2 variant, fallback to ViT if not present
            try:
                m = timm.create_model('dinov2_vit_base', pretrained=True, num_classes=0, global_pool='avg')
            except Exception:
                m = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='avg')
            m.eval()
            m.to(device)
            # many timm ViT models output 768-d features; but this may vary
            return m, None
        except Exception as e:
            raise ImportError("timm is required for dinov2/vit support. pip install timm")

    if model_name == 'clip':
        return load_clip_openai(device)

    if model_name in ('resnet50', 'resnet'):
        try:
            import torchvision.models as models
        except Exception:
            raise ImportError('torchvision is required to load ResNet models')
        m = models.resnet50(pretrained=True)
        m.fc = torch.nn.Identity()
        m.eval()
        m.to(device)
        return m, 2048

    # fallback: try timm for arbitrary model_name
    try:
        import timm
        m = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
        m.eval()
        m.to(device)
        return m, None
    except Exception:
        raise ValueError(f'Unsupported model: {model_name}. Install timm or use resnet50/clip/dinov2')


def extract_embeddings(model_tuple, dataloader, device, out_file, model_name):
    """
    Runs images through the model and writes embeddings + image ids to out_file (.pt)
    model_tuple: if CLIP returns (model, preprocess), else model alone.
    """
    model = model_tuple
    preprocess = None
    is_clip = False
    if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
        model, preprocess = model_tuple
        if hasattr(model, 'encode_image') or hasattr(model, 'visual'):
            is_clip = True

    emb_list = []
    id_list = []
    captions_list = []

    model_device = device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeds'):
            # For our CocoWrapper, batch is (list_of_imgs, list_of_targets, list_of_img_ids)
            imgs, targets, img_ids = batch
            # imgs may be a list of PIL or preprocessed tensors
            if is_clip and preprocess is not None and not torch.is_tensor(imgs[0]):
                # openai clip preprocess expects PIL per item
                proc_tensors = [preprocess(img).unsqueeze(0) for img in imgs]
                x = torch.cat(proc_tensors, dim=0).to(model_device)
                # openai clip has encode_image
                if hasattr(model, 'encode_image'):
                    emb = model.encode_image(x).float().cpu()
                else:
                    # HF CLIP vision model returns last_hidden_state
                    outputs = model(pixel_values=x)
                    emb = outputs.last_hidden_state.mean(dim=1).cpu()
            else:
                # imgs are already tensors (assuming dataloader collate stacked)
                if isinstance(imgs, (list, tuple)):
                    imgs = torch.stack(imgs, dim=0)
                x = imgs.to(model_device)
                # Some timm models accept x and return (B, C) features
                try:
                    out = model(x)
                except Exception:
                    # try forward_features for timm/vit
                    if hasattr(model, 'forward_features'):
                        out = model.forward_features(x)
                    else:
                        out = model(x)
                # If out has spatial dim, pool
                if isinstance(out, torch.Tensor):
                    if out.dim() == 4:
                        emb = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1).cpu()
                    else:
                        emb = out.cpu()
                else:
                    # fallback: if model returns dict or object with last_hidden_state
                    if hasattr(out, 'last_hidden_state'):
                        emb = out.last_hidden_state.mean(dim=1).cpu()
                    else:
                        raise RuntimeError("Unknown model output format; please adapt extract_embeddings for your model.")
            # embeddings and ids
            # emb shape (B, D) ; img_ids length B
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            emb_list.append(emb)
            # collect captions (targets) as simple list-of-lists of strings
            captions_list.extend(targets)
            id_list.extend(list(img_ids))

    embeddings = torch.cat(emb_list, dim=0)
    out = {
        'image_ids': id_list,
        'captions': captions_list,
        'embeddings': embeddings,
        'model_name': model_name,
    }
    torch.save(out, out_file)
    print(f"Saved embeddings to: {out_file} (num={len(id_list)}, dim={embeddings.shape[1]})")

# collate: return list of images, list of captions, list of ids
def collate_fn(batch):
    imgs = [b[0] for b in batch]
    caps = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    return imgs, caps, ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', '--model', dest='model', default='dinov2', help='model name: dinov2, clip, resnet50, or timm model name')
    parser.add_argument('--d', '--dataset', dest='dataset', default='coco', help='dataset: coco (only supported in this script)')
    parser.add_argument('--split', dest='split', default='val2017', help="coco split: train2017 or val2017")
    parser.add_argument('--coco_root', dest='coco_root', default=None, help='path to coco_dataset folder (defaults to ./coco_dataset)')
    parser.add_argument('--gpu', dest='gpu', default='0', help='GPU id or \"cpu\"')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--out_dir', dest='out_dir', default='data')
    parser.add_argument('--image_size', dest='image_size', type=int, default=224)
    args = parser.parse_args()

    if args.gpu == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset.lower() != 'coco':
        raise NotImplementedError('This helper currently only supports COCO via --d coco')

    images_dir, ann_file = get_coco_paths(coco_root=args.coco_root, split=args.split)
    print('Images dir:', images_dir)
    print('Ann file  :', ann_file)

        # ---------- REPLACEMENT START ----------
    model_name = args.model.lower()

    # If user asked for a text encoder (allroberta), run a text-embedding pipeline
    if model_name == 'allroberta':
        # we will compute per-image text embeddings by averaging the caption embeddings
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            print("sentence-transformers is required for --m allroberta. Install with: pip install sentence-transformers")
            sys.exit(1)

        print("Loading SentenceTransformer all-roberta-large-v1 (this may download ~1.4GB)...")
        st_model = SentenceTransformer('all-roberta-large-v1')  # blocks while downloading

        # Build COCO dataset with no image transform (we only use captions)
        dataset = CocoWrapper(root=images_dir, annFile=ann_file, transform=None)

        # Windows-safe default: use num_workers=0 to avoid pickling issues
        num_workers = 0 if os.name == 'nt' else 4

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        print("Encoding captions to text embeddings (this may take a while)...")
        image_ids_out = []
        text_emb_list = []

        # For each image, dataset returns (img, captions_list, img_id)
        # We encode each image's caption list and average (convert_to_tensor -> torch)
        for imgs, captions_batch, img_ids in tqdm(dataloader, desc='Encoding captions'):
            # captions_batch is list of lists
            for captions_for_img, img_id in zip(captions_batch, img_ids):
                # captions_for_img is a list of strings (may be empty)
                if len(captions_for_img) == 0:
                    emb = st_model.encode("", convert_to_tensor=True)
                else:
                    # encode all captions for this image, then average
                    encs = st_model.encode(captions_for_img, convert_to_tensor=True)
                    # encs shape (num_caps, dim)
                    emb = encs.mean(dim=0)
                text_emb_list.append(emb.cpu())
                image_ids_out.append(img_id)

        text_embeddings = torch.stack(text_emb_list, dim=0)  # (N, D)
        out_file = os.path.join(args.out_dir, f'{args.dataset}_{model_name}_{args.split}_text.pt')
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            'image_ids': image_ids_out,
            'text_embeddings': text_embeddings,
            'model_name': model_name
        }, out_file)
        print(f"Saved text embeddings to: {out_file}  shape: {text_embeddings.shape}")
        sys.exit(0)

    # Otherwise proceed to load a vision model and extract image embeddings
    try:
        model_tuple, feat_dim = load_model(model_name, device)
    except Exception as e:
        print('Error loading model:', e)
        sys.exit(1)

    # Determine transform: CLIP preprocess or generic transform
    preprocess = None
    if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
        _, preprocess = model_tuple
        transform = None
    else:
        transform = build_transform(image_size=args.image_size)

    dataset = CocoWrapper(root=images_dir, annFile=ann_file, transform=transform)

    # Windows-safe default: num_workers = 0 to avoid pickling issues if on Windows
    num_workers = 0 if os.name == 'nt' else 4

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    out_file = os.path.join(args.out_dir, f'{args.dataset}_{model_name}_{args.split}_img.pt')
    extract_embeddings(model_tuple, dataloader, device, out_file, model_name)
    # ---------- REPLACEMENT END ----------



if __name__ == '__main__':
    main()
