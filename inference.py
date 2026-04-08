"""Inference helpers for dataset and novel-image showcase."""

import os
import sys
from pathlib import Path

import albumentations as A
import matplotlib.image as mpimg
import numpy as np
import torch
import wandb

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.multitask import MultiTaskPerceptionModel


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask == 0] = [0, 0, 0]
    colored[mask == 1] = [0, 255, 0]
    colored[mask == 2] = [255, 0, 0]
    return colored


def draw_box(img: np.ndarray, box, color) -> np.ndarray:
    img = img.copy()
    xc, yc, w, h = box
    x1 = max(0, min(int(xc - w / 2), img.shape[1] - 1))
    y1 = max(0, min(int(yc - h / 2), img.shape[0] - 1))
    x2 = max(0, min(int(xc + w / 2), img.shape[1] - 1))
    y2 = max(0, min(int(yc + h / 2), img.shape[0] - 1))
    img[y1 : y1 + 2, x1:x2] = color
    img[max(y2 - 2, 0) : y2, x1:x2] = color
    img[y1:y2, x1 : x1 + 2] = color
    img[y1:y2, max(x2 - 2, 0) : x2] = color
    return img


def preprocess_image(path: str) -> tuple[np.ndarray, torch.Tensor]:
    image = mpimg.imread(path).astype(np.float32)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.max() > 1:
        image = image / 255.0

    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=1.0,
            ),
        ]
    )
    image_norm = transform(image=image)["image"]
    tensor = torch.tensor(image_norm).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


def predict_image(model: MultiTaskPerceptionModel, image_path: str) -> dict:
    image_raw, image_tensor = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    cls_probs = torch.softmax(output["classification"][0], dim=0).cpu().numpy()
    pred_label = int(np.argmax(cls_probs))
    pred_conf = float(np.max(cls_probs))
    pred_box = output["localization"][0].cpu().numpy()
    pred_mask = torch.argmax(output["segmentation"][0], dim=0).cpu().numpy()

    image_uint8 = (image_raw * 255).astype(np.uint8)
    image_resized = np.array(A.Resize(224, 224)(image=image_uint8)["image"])
    boxed = draw_box(image_resized, pred_box, [255, 0, 0])

    return {
        "image": image_resized,
        "boxed": boxed,
        "mask": pred_mask,
        "label": pred_label,
        "confidence": pred_conf,
        "path": image_path,
    }


def run_showcase(image_paths, project: str = "da6401_assignment_02_showcase", run_name: str = "novel_images") -> None:
    model = MultiTaskPerceptionModel()
    wandb.init(project=project, name=run_name, mode="online")

    table = wandb.Table(columns=["path", "label", "confidence", "image", "bbox", "mask"])
    for image_path in image_paths:
        pred = predict_image(model, image_path)
        table.add_data(
            pred["path"],
            pred["label"],
            pred["confidence"],
            wandb.Image(pred["image"]),
            wandb.Image(pred["boxed"]),
            wandb.Image(colorize_mask(pred["mask"])),
        )

    wandb.log({"novel_image_showcase": table})
    wandb.finish()


if __name__ == "__main__":
    default_image = Path("data/images/Abyssinian_1.jpg")
    if default_image.exists():
        run_showcase([str(default_image)], run_name="single_image_smoke_test")
