"""Dataset loader for Oxford-IIIT Pet"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import albumentations as A


class OxfordIIITPetDataset(Dataset):
    """
    Multi-task dataset:
    - classification
    - localization
    - segmentation
    """

    def __init__(self, root, split="train", crop_for_classification=False, crop_margin=0.15):
        self.root = root
        self.split = split
        self.crop_for_classification = crop_for_classification
        self.crop_margin = crop_margin

        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")

        if split == "train":
            self.split_file = os.path.join(root, "annotations", "trainval.txt")
        elif split == "val":
            self.split_file = os.path.join(root, "annotations", "test.txt")
        else:
            raise ValueError("split must be 'train' or 'val'")

        transforms = [A.Resize(224, 224)]
        if split == "train":
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])
        transforms.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
        self.transform = A.Compose(transforms)

        self.samples = []
        self.labels = {}

        with open(self.split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                name = parts[0]
                label = int(parts[1]) - 1
                self.samples.append(name)
                self.labels[name] = label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]

        img_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir, name + ".png")

        image = mpimg.imread(img_path).astype(np.float32)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.max() > 1:
            image = image / 255.0

        mask = mpimg.imread(mask_path)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        mask = np.round(mask).astype(np.uint8)

        mask_final = np.zeros_like(mask, dtype=np.int64)
        mask_final[mask == 1] = 1
        mask_final[mask == 2] = 0
        mask_final[mask == 3] = 2
        mask = np.clip(mask_final, 0, 2)

        ys_raw, xs_raw = np.where(mask == 1)
        if len(xs_raw) == 0 or len(ys_raw) == 0:
            x1_raw, y1_raw, x2_raw, y2_raw = 0, 0, image.shape[1] - 1, image.shape[0] - 1
        else:
            x1_raw, x2_raw = xs_raw.min(), xs_raw.max()
            y1_raw, y2_raw = ys_raw.min(), ys_raw.max()

        if self.crop_for_classification:
            pad_x = int((x2_raw - x1_raw + 1) * self.crop_margin)
            pad_y = int((y2_raw - y1_raw + 1) * self.crop_margin)
            x1_crop = max(0, x1_raw - pad_x)
            x2_crop = min(image.shape[1] - 1, x2_raw + pad_x)
            y1_crop = max(0, y1_raw - pad_y)
            y2_crop = min(image.shape[0] - 1, y2_raw + pad_y)
            image = image[y1_crop:y2_crop + 1, x1_crop:x2_crop + 1]
            mask = mask[y1_crop:y2_crop + 1, x1_crop:x2_crop + 1]

        augmented = self.transform(image=image, mask=mask)
        image = torch.tensor(augmented["image"]).permute(2, 0, 1)
        mask = torch.tensor(augmented["mask"]).long()
        label = torch.tensor(self.labels[name]).long()

        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            bbox = torch.tensor([112, 112, 50, 50], dtype=torch.float32)
        else:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)

        return image, label, bbox, mask
