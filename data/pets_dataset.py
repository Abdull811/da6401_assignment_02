"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.image as mpimg
import albumentations as A


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, images_dir, masks_dir, labels_dict, bboxes_dict, image_files=None):
        """
        Initialize dataset.
            images_dir: Path to directory containing input images.
            masks_dir: Path to directory containing segmentation masks.
            labels_dict: Dictionary mapping image filename > class label.
            bboxes_dict: Dictionary mapping image filename > bounding box [x1, y1, x2, y2].
        """ 
        # Store paths

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.labels_dict = labels_dict
        self.bboxes_dict = bboxes_dict
        
        # List of image filenames
        self.image_files = image_files if image_files is not None else sorted(labels_dict.keys())

        # Resize to fixed size (224 * 224) and normalize pixel values
        self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


    def __len__(self):
        # Return total number of samples in dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image filename for index
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + ".png")

        # Load image and mask
        img = mpimg.imread(img_path)
        mask = mpimg.imread(mask_path)

        if img.dtype != np.float32:
            img = img.astype(np.float32)

        if img.max() > 1.0:
            img = img / 255.0

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        if img.shape[2] > 3:
            img = img[:, :, :3]

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        orig_h, orig_w = img.shape[0], img.shape[1]


        # Apply image transform
        transformed = self.transform(image=img)
        img = transformed["image"]

        # Resize mask separately using nearest-neighbor
        mask = mask.astype(np.float32)
        mask = A.Resize(224, 224, interpolation=0)(image=mask)["image"]
        mask = mask.astype(np.int64)
        mask = np.clip(mask - 1, 0, 2)

        # Convert image to torch tensor
        img = torch.tensor(img).permute(2, 0, 1).float()

        # Load label
        label = torch.tensor(self.labels_dict[img_name]).long()

        # Load and resize bbox from [x1, y1, x2, y2] to [x_center, y_center, width, height]
        x1, y1, x2, y2 = self.bboxes_dict[img_name]

        x1 = x1 * 224.0 / orig_w
        x2 = x2 * 224.0 / orig_w
        y1 = y1 * 224.0 / orig_h
        y2 = y2 * 224.0 / orig_h

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        bbox = torch.tensor([x_center, y_center, width, height]).float()
        mask = torch.tensor(mask).long()

        # Return all components for multi-task
        return img, label, bbox, mask
