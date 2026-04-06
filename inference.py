"""Inference and evaluation
"""
import os, sys
import torch
import wandb
from models.multitask import MultiTaskPerceptionModel
import matplotlib.image as mpimg
import numpy as np
import albumentations as A


os.environ["WANDB_API_KEY"] = "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"

np.random.seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

wandb.init(project="da6401_Assigment_02_Weight_&_Biase")

model = MultiTaskPerceptionModel()
model.eval()

def colorize_mask(mask):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    colored[mask == 0] = [0, 0, 0]
    colored[mask == 1] = [0, 255, 0]
    colored[mask == 2] = [255, 0, 0]

    return colored

# Load image
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

img = mpimg.imread("data/images/Abyssinian_1.jpg").astype(np.float32)

if img.max() > 1:
    img = img / 255.0

aug = transform(image=img)
img = aug["image"]

img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0)

# Forward pass
with torch.no_grad():
    out = model(img_tensor)

# Outputs
pred_mask = torch.argmax(out["segmentation"][0], dim=0).cpu().numpy()
pred_box = out["localization"][0].cpu().numpy()

# Convert image for visualization
img_np = (img * 255).astype(np.uint8)

# Draw bounding box (no cv2)
def draw_box(img, box, color):
    img = img.copy()
    xc, yc, w, h = box

    x1 = int(xc - w/2)
    y1 = int(yc - h/2)
    x2 = int(xc + w/2)
    y2 = int(yc + h/2)

    img[y1:y1+2, x1:x2] = color
    img[y2-2:y2, x1:x2] = color
    img[y1:y2, x1:x1+2] = color
    img[y1:y2, x2-2:x2] = color

    return img

img_box = draw_box(img_np, pred_box, [255,0,0])

# Log to W&B
wandb.log({
    "final_pipeline_image": wandb.Image(img_np),
    "final_pipeline_mask": wandb.Image(colorize_mask(pred_mask)),
    "final_pipeline_bbox": wandb.Image(img_box)
})
