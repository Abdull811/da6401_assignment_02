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
    - classification (breed)
    - localization (dummy bbox)
    - segmentation (trimap)
    """

    def __init__(self, root, split="train"):
        self.root = root
        self.split = split

        # PATHS
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")

        if split == "train":
            self.split_file = os.path.join(root, "annotations", "trainval.txt")
        elif split == "val":
            self.split_file = os.path.join(root, "annotations", "test.txt")

        # TRANSFORM
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ])

        self.samples = []
        self.labels = {}

        # LOAD SPLIT
        with open(self.split_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                name = parts[0]
                label = int(parts[1]) - 1  # 0-based

                self.samples.append(name)
                self.labels[name] = label

                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]

        img_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir, name + ".png")

        # LOAD IMAGE
        image = mpimg.imread(img_path).astype(np.float32)

        # Remove alpha channel (RGBA → RGB)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Normalize image to [0,1]
        if image.max() > 1:
            image = image / 255.0

        # LOAD MASK
        mask = mpimg.imread(mask_path)
        
        # Convert to single channel if needed
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Convert properly to uint8
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        mask = np.round(mask).astype(np.uint8)
        
        # FINAL LABEL MAPPING
        mask_final = np.zeros_like(mask, dtype=np.int64)
        
        mask_final[mask == 1] = 1   # pet
        mask_final[mask == 2] = 2   # boundary
        mask_final[mask == 3] = 0   # background
        
        mask_final = np.clip(mask_final, 0, 2)
        
        mask = mask_final

        # APPLY TRANSFORM
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        # TO TENSOR
        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).long()

        label = torch.tensor(self.labels[name]).long()
        
        # BBOX
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            bbox = torch.tensor([112,112,50,50], dtype=torch.float32)
        else:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
        
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
        
            bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        
        return image, label, bbox, mask
    
