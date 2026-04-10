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

    def __init__(self, root, split="train", crop_for_classification: bool = False, crop_margin: float = 0.25):
        self.root = root
        self.split = split
        self.crop_for_classification = crop_for_classification
        self.crop_margin = crop_margin

        # PATHS
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")

        if split == "train":
            self.split_file = os.path.join(root, "annotations", "trainval.txt")
        elif split == "val":
            self.split_file = os.path.join(root, "annotations", "test.txt")
        else:
            raise ValueError("split must be 'train' or 'val'")

        # TRANSFORM
        transforms = [A.Resize(224, 224)]
        if split == "train":
            if crop_for_classification:
                transforms.extend(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.25),
                        # A.HorizontalFlip(p=0.5),
                        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        A.Affine(
                            translate_percent=(-0.03, 0.03),
                            scale=(0.92, 1.08),
                            rotate=(-12, 12),
                            border_mode=0,
                            p=0.4,
                        ),
                    ]
                )
            else:
                transforms.extend(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.3),
                    ]
                )
        transforms.append(
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=1.0,
            )
        )
        self.image_transform = A.Compose(transforms)
        self.task_transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
        )

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

        if image.max() > 1:
           image = image / 255.0
        
        # LOAD MASK
        mask = mpimg.imread(mask_path)

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.max() <= 1.0:
            # mpimg reads the trimap PNG as floats {1/255, 2/255, 3/255}
            mask = np.round(mask * 255).astype(np.uint8)
        else:
            mask = np.round(mask).astype(np.uint8)

        # Oxford-IIIT Pet trimap:
        # 1 = foreground, 2 = background, 3 = boundary
        # Remap to contiguous class ids for CE loss
        mask = np.clip(mask.astype(np.int64) - 1, 0, 2)

        # LOAD XML HEAD BOX
        xml_path = os.path.join(self.root, "annotations", "xmls", name + ".xml")
        if os.path.exists(xml_path):
            root = ET.parse(xml_path).getroot()
            box = root.find("object").find("bndbox")
            bbox_xyxy = [
                float(box.find("xmin").text) - 1.0,
                float(box.find("ymin").text) - 1.0,
                float(box.find("xmax").text) - 1.0,
                float(box.find("ymax").text) - 1.0,
            ]
        else:
            ys_raw, xs_raw = np.where(mask == 0)
            if len(xs_raw) == 0 or len(ys_raw) == 0:
                bbox_xyxy = [0.0, 0.0, float(image.shape[1] - 1), float(image.shape[0] - 1)]
            else:
                bbox_xyxy = [
                    float(xs_raw.min()),
                    float(ys_raw.min()),
                    float(xs_raw.max()),
                    float(ys_raw.max()),
                ]

        if self.crop_for_classification:
            augmented = self.image_transform(image=image, mask=mask)
            image = torch.tensor(augmented["image"]).permute(2, 0, 1)
            mask = torch.tensor(augmented["mask"]).long()
            label = torch.tensor(self.labels[name]).long()
            bbox = torch.tensor([112.0, 112.0, 50.0, 50.0], dtype=torch.float32)
            return image, label, bbox, mask

        augmented = self.task_transform(
            image=image,
            mask=mask,
            bboxes=[bbox_xyxy],
            bbox_labels=[0],
        )

        image = torch.tensor(augmented["image"]).permute(2, 0, 1)
        mask = torch.tensor(augmented["mask"]).long()
        label = torch.tensor(self.labels[name]).long()

        x1, y1, x2, y2 = augmented["bboxes"][0]
        bbox = torch.tensor(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                max(x2 - x1, 1.0),
                max(y2 - y1, 1.0),
            ],
            dtype=torch.float32,
        )

        return image, label, bbox, mask
    
