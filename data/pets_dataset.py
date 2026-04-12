"""Dataset loader for Oxford-IIIT Pet"""

import os

import albumentations as A
import matplotlib.image as mpimg
import numpy as np
import torch
from torch.utils.data import Dataset


class OxfordIIITPetDataset(Dataset):
    """
    Multi-task dataset:
    - classification (breed)
    - localization (head bbox)
    - segmentation (trimap)
    """

    def __init__(self, root, split="train", crop_for_classification: bool = False, crop_margin: float = 0.25):
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
            if crop_for_classification:
                transforms.extend(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.25),
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
                max_pixel_value=255.0,
            )
        )

        self.image_transform = A.Compose(transforms)
        self.task_transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
        )

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

    def _read_tag_value(self, text: str, tag: str) -> float:
        start_token = f"<{tag}>"
        end_token = f"</{tag}>"

        start = text.find(start_token)
        end = text.find(end_token)

        if start == -1 or end == -1:
            raise ValueError(f"Tag {tag} not found in XML")

        start += len(start_token)
        return float(text[start:end].strip())

    def _load_xml_bbox(self, xml_path):
        import xml.etree.ElementTree as ET
    
        tree = ET.parse(xml_path)
        root = tree.getroot()
    
        bndbox = root.find("object").find("bndbox")
    
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
    
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        name = self.samples[idx]

        img_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir, name + ".png")

        image = mpimg.imread(img_path).astype(np.float32)

        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        #if image.max() > 1:
        #    image = image / 255.0

        mask = mpimg.imread(mask_path)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if mask.max() <= 1.0:
            mask = np.round(mask * 255).astype(np.uint8)
        else:
            mask = np.round(mask).astype(np.uint8)

        # 1=foreground, 2=background, 3=boundary -> 0,1,2
        mask = np.clip(mask.astype(np.int64) - 1, 0, 2)

        xml_path = os.path.join(self.root, "annotations", "xmls", name + ".xml")
        if os.path.exists(xml_path):
            x1, y1, x2, y2 = self._load_xml_bbox(xml_path)
        else:
            # fallback (never used normally)
            h, w = image.shape[:2]
            x1, y1, x2, y2 = 0, 0, w - 1, h - 1
        bbox_xyxy = [x1, y1, x2, y2]
        
        bbox = torch.tensor([
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(x2 - x1, 1.0),
            max(y2 - y1, 1.0),
        ], dtype=torch.float32)

        if self.crop_for_classification:
            x1_box, y1_box, x2_box, y2_box = bbox_xyxy

            box_w = x2_box - x1_box + 1.0
            box_h = y2_box - y1_box + 1.0
            side = int(max(box_w, box_h) * (1.0 + self.crop_margin))
            side = max(side, 32)

            cx = int((x1_box + x2_box) / 2.0)
            cy = int((y1_box + y2_box) / 2.0)

            x1_crop = max(0, cx - side // 2)
            y1_crop = max(0, cy - side // 2)
            x2_crop = min(image.shape[1] - 1, x1_crop + side - 1)
            y2_crop = min(image.shape[0] - 1, y1_crop + side - 1)

            x1_crop = max(0, x2_crop - side + 1)
            y1_crop = max(0, y2_crop - side + 1)

            image = image[y1_crop:y2_crop + 1, x1_crop:x2_crop + 1]
            mask = mask[y1_crop:y2_crop + 1, x1_crop:x2_crop + 1]

            augmented = self.image_transform(image=image, mask=mask)
            image = torch.tensor(augmented["image"]).permute(2, 0, 1)
            mask = torch.tensor(augmented["mask"]).long()
            label = torch.tensor(self.labels[name]).long()
            bbox = torch.tensor([
                image.shape[2] / 2,
                image.shape[1] / 2,
                image.shape[2] * 0.5,
                image.shape[1] * 0.5,
            ], dtype=torch.float32)
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
