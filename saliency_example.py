# simple_saliency.py
import torch, torchvision.transforms as T
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import os

COCO_ROOT = "./coco_dataset"; SPLIT="val2017"
ANN = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
coco = COCO(ANN)
d = torch.load("data/coco_resnet50_val2017_img.pt")
image_ids = d['image_ids']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device); model.eval()

transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def load_tensor(img_id):
    meta = coco.loadImgs(img_id)[0]
    path = os.path.join(COCO_ROOT, SPLIT, meta['file_name'])
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    return t, img

img_id = image_ids[0]
inp, pil = load_tensor(img_id)
inp.requires_grad_()
out = model(inp)
cls = out.argmax(dim=1)
score = out[0, cls]
score.backward()
saliency = inp.grad.abs().squeeze().cpu().numpy()  # C,H,W
sal = saliency.max(axis=0)  # H,W
sal = (sal - sal.min()) / (sal.max()-sal.min() + 1e-8)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(pil); plt.axis('off'); plt.title(f"id {img_id}")
plt.subplot(1,2,2); plt.imshow(pil.resize((224,224))); plt.imshow(sal, cmap='hot', alpha=0.6); plt.axis('off'); plt.title("Input-grad saliency")
plt.show()
