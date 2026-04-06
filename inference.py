"""Inference and evaluation
"""
import os, sys
import torch
import wandb
from models.multitask import MultiTaskPerceptionModel
import matplotlib.image as mpimg
import numpy as np

#os.environ["WANDB_API_KEY"] = "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"

np.random.seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

wandb.init(project="da6401_Assigment_02_Weight_&_Biase")

model = MultiTaskPerceptionModel()
model.eval()

# Load external image
img = mpimg.imread("data/images/Abyssinian_1.jpg").astype(np.float32)

if img.max() > 1:
    img = img / 255.0

img = torch.tensor(img).permute(2,0,1).unsqueeze(0)

with torch.no_grad():
    out = model(img)

pred_mask = torch.argmax(out["segmentation"][0], dim=0)

wandb.log({
    "final_input": wandb.Image(img[0].permute(1,2,0).numpy()),
    "final_pred_mask": wandb.Image(pred_mask.numpy() * 100)
})
