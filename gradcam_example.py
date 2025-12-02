# gradcam_manual.py
# Self-contained Grad-CAM for torchvision ResNet50 (no external grad-cam package)

import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# ---------- CONFIG ----------
COCO_ROOT = "./coco_dataset"
SPLIT = "val2017"
ANN = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
EMBED_FILE = "data/coco_resnet50_val2017_img.pt"   # optional, used to pick image ids
# choose index 0 by default if no embed file
PICK_INDEX = 20
# ----------------------------

# Preprocessing must match get_embeds.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Helper: load COCO and pick an image id
coco = COCO(ANN)
image_ids = None
if os.path.isfile(EMBED_FILE):
    d = torch.load(EMBED_FILE)
    image_ids = d.get('image_ids', None)

if image_ids is None or len(image_ids) == 0:
    image_id = coco.getImgIds()[0]
else:
    image_id = image_ids[PICK_INDEX]

meta = coco.loadImgs(image_id)[0]
img_file = os.path.join(COCO_ROOT, SPLIT, meta['file_name'])
if not os.path.isfile(img_file):
    raise FileNotFoundError(f"Image file not found: {img_file}")

# Load image
orig_pil = Image.open(img_file).convert('RGB')
inp_tensor = transform(orig_pil).unsqueeze(0)  # 1,C,H,W

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Target layer: last conv in ResNet bottleneck
target_layer = model.layer4[-1].conv3

# Storage for hooks
saved_activations = {}
saved_gradients = {}

def forward_hook(module, input, output):
    # output shape: (B, C, H, W)
    saved_activations['value'] = output.detach()

def backward_hook(module, grad_in, grad_out):
    # grad_out is a tuple; grad_out[0] is gradient w.r.t. output
    saved_gradients['value'] = grad_out[0].detach()

# register hooks
fh = target_layer.register_forward_hook(forward_hook)
bh = target_layer.register_backward_hook(backward_hook)

# Move input to device
inp = inp_tensor.to(device)
inp.requires_grad = True

# Forward pass
out = model(inp)  # shape (1, 1000)
pred_class = int(out.argmax(dim=1).item())
print(f"Predicted class: {pred_class}")

# Backprop for the predicted class
model.zero_grad()
one_hot = torch.zeros_like(out)
one_hot[0, pred_class] = 1.0
out.backward(gradient=one_hot)

# Get saved activations and gradients
if 'value' not in saved_activations or 'value' not in saved_gradients:
    # remove hooks
    fh.remove(); bh.remove()
    raise RuntimeError("Failed to capture activations or gradients from hooks.")

activations = saved_activations['value']           # shape: (1, C, H, W)
gradients = saved_gradients['value']               # shape: (1, C, H, W)

# Compute weights: global average pooling of gradients over spatial dims
weights = torch.mean(gradients, dim=(2,3), keepdim=True)   # shape (1, C, 1, 1)
# Weighted combination of activations
weighted_maps = (weights * activations).sum(dim=1, keepdim=True)  # shape (1,1,H,W)
cam = F.relu(weighted_maps)  # relu

# Squeeze and convert to numpy, normalize to [0,1]
cam = cam.squeeze(0).squeeze(0).cpu().numpy()  # shape (H,W)
cam -= cam.min()
if cam.max() > 0:
    cam = cam / cam.max()

# Upsample to original image size (use PIL size)
orig_w, orig_h = orig_pil.size
cam_resized = Image.fromarray(np.uint8(cam*255)).resize((orig_w, orig_h), resample=Image.BILINEAR)
cam_resized = np.array(cam_resized).astype(np.float32)/255.0  # H,W in [0,1]

# Create heatmap overlay
img_np = np.array(orig_pil).astype(np.float32)/255.0
heatmap = plt.get_cmap('jet')(cam_resized)[:,:,:3]  # H,W,3
overlay = 0.5 * img_np + 0.5 * heatmap
overlay = np.clip(overlay, 0, 1)

# Display
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(orig_pil); plt.title(f"Original: id={image_id}"); plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(cam_resized, cmap='jet'); plt.title("Grad-CAM (grayscale)"); plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(overlay); plt.title("Overlay"); plt.axis('off')
plt.show()

# Clean up hooks
fh.remove(); bh.remove()
