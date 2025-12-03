# inspect_embeds.py
import torch
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Config: adapt if needed
EMBED_FILE = "data/coco_resnet50_img.pt"   # produced by get_embeds.py
COCO_ROOT = "./coco_dataset"                      # your project-local coco_dataset
SPLIT = "val2017"

# Load saved embeddings
d = torch.load(EMBED_FILE)
image_ids = d['image_ids']        # list of ints
embs = d['embeddings'].numpy()    # shape (N, D)
print("Loaded embeddings:", embs.shape, "num images:", len(image_ids))

# Load COCO API to map image_id -> filename
ann_file = os.path.join(COCO_ROOT, "annotations", f"instances_{ 'val' if SPLIT=='val2017' else 'train'}2017.json")
coco = COCO(ann_file)

def show_image_by_id(img_id, title=None):
    meta = coco.loadImgs(img_id)[0]
    fname = meta['file_name']
    path = os.path.join(COCO_ROOT, SPLIT, fname)
    img = Image.open(path).convert('RGB')
    plt.imshow(img); plt.axis('off')
    if title: plt.title(title)

# Example: show 3 random images and their IDs
import random
sample_idx = random.sample(range(len(image_ids)), 3)
plt.figure(figsize=(12,4))
for i, idx in enumerate(sample_idx):
    plt.subplot(1,3,i+1)
    show_image_by_id(image_ids[idx], title=f"id={image_ids[idx]}")
plt.show()

# Example: nearest neighbors of the first sampled image
q_idx = sample_idx[0]
q_emb = embs[q_idx:q_idx+1]  # shape (1, D)
sims = cosine_similarity(q_emb, embs)[0]  # length N
topk = 5
order = np.argsort(-sims)[:topk]
print("Top-k indices:", order)
plt.figure(figsize=(15,3))
for i, idx in enumerate(order):
    plt.subplot(1,topk,i+1)
    show_image_by_id(image_ids[idx], title=f"rank={i} sim={sims[idx]:.3f}")
plt.show()
