"""Training entrypoint for DA6401 Assignment-2."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR_CLASSIFIER = 3e-4
LR_LOCALIZER = 1e-4
LR_SEGMENTER = 1e-4
NUM_WORKERS = 0
SEED = 42

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

os.environ["WANDB_API_KEY"] = "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"

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu()
    img = img * STD + MEAN
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask == 0] = [0, 0, 0]
    colored[mask == 1] = [0, 255, 0]
    colored[mask == 2] = [255, 0, 0]
    return colored


def draw_box(img: np.ndarray, box, color) -> np.ndarray:
    img = img.copy()
    h, w, _ = img.shape
    xc, yc, bw, bh = box

    x1 = max(0, min(int(xc - bw / 2), w - 1))
    y1 = max(0, min(int(yc - bh / 2), h - 1))
    x2 = max(0, min(int(xc + bw / 2), w - 1))
    y2 = max(0, min(int(yc + bh / 2), h - 1))

    img[y1 : y1 + 2, x1:x2] = color
    img[max(y2 - 2, 0) : y2, x1:x2] = color
    img[y1:y2, x1 : x1 + 2] = color
    img[y1:y2, max(x2 - 2, 0) : x2] = color
    return img


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    score = 0.0
    for cls in range(1, 3):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        score += (2 * inter + 1e-6) / (union + 1e-6)
    return score / 2.0


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    inter = (probs * target_one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return 1.0 - dice.mean()


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean()


def compute_iou(box1, box2) -> float:
    def convert(box):
        xc, yc, bw, bh = box
        return xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2

    px1, py1, px2, py2 = convert(box1)
    tx1, ty1, tx2, ty2 = convert(box2)

    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)

    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = max(px2 - px1, 0.0) * max(py2 - py1, 0.0) + max(tx2 - tx1, 0.0) * max(ty2 - ty1, 0.0) - inter
    return inter / (union + 1e-6)


def save_checkpoint(model: nn.Module, path: str, epoch: int, best_metric: float) -> None:
    payload = {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_metric}
    torch.save(payload, path)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    torch.save(payload, checkpoints_dir / Path(path).name)


def build_loaders() -> tuple[DataLoader, DataLoader]:
    train_ds = OxfordIIITPetDataset("data", "train")
    val_ds = OxfordIIITPetDataset("data", "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def set_segmentation_freeze_mode(segmenter: VGG11UNet, freeze_mode: str) -> None:
    if freeze_mode == "freeze":
        for param in segmenter.encoder.parameters():
            param.requires_grad = False
    elif freeze_mode == "partial":
        params = list(segmenter.encoder.parameters())
        for param in params[: len(params) // 2]:
            param.requires_grad = False


def train(dropout_p: float = 0.5, freeze_mode: str = "full", wandb_mode: str = "online") -> None:
    set_seed()

    run = wandb.init(
        project="da6401_assignment_02",
        name=f"dropout_{dropout_p}_{freeze_mode}",
        config={
            "dropout": dropout_p,
            "freeze_mode": freeze_mode,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr_classifier": LR_CLASSIFIER,
            "lr_localizer": LR_LOCALIZER,
            "lr_segmenter": LR_SEGMENTER,
        },
        mode=wandb_mode,
    )

    train_loader, val_loader = build_loaders()

    classifier = VGG11Classifier(dropout_p=dropout_p).to(DEVICE)
    localizer = VGG11Localizer(dropout_p=dropout_p).to(DEVICE)
    segmenter = VGG11UNet(dropout_p=dropout_p).to(DEVICE)
    set_segmentation_freeze_mode(segmenter, freeze_mode)

    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loc_reg_loss = nn.SmoothL1Loss(beta=5.0)
    loc_iou_loss = IoULoss()
    seg_ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.0, 1.5], device=DEVICE))

    cls_opt = optim.Adam(classifier.parameters(), lr=LR_CLASSIFIER, weight_decay=1e-4)
    loc_opt = optim.Adam(localizer.parameters(), lr=LR_LOCALIZER, weight_decay=1e-5)
    seg_opt = optim.Adam(filter(lambda p: p.requires_grad, segmenter.parameters()), lr=LR_SEGMENTER, weight_decay=1e-5)

    cls_sched = optim.lr_scheduler.ReduceLROnPlateau(cls_opt, mode="max", factor=0.5, patience=2)
    loc_sched = optim.lr_scheduler.ReduceLROnPlateau(loc_opt, mode="max", factor=0.5, patience=2)
    seg_sched = optim.lr_scheduler.ReduceLROnPlateau(seg_opt, mode="max", factor=0.5, patience=2)

    best_cls_f1 = -1.0
    best_loc_iou = -1.0
    best_seg_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        localizer.train()
        segmenter.train()

        train_cls_loss = 0.0
        train_loc_loss = 0.0
        train_seg_loss = 0.0

        for step, (images, labels, bboxes, masks) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            bboxes = bboxes.to(DEVICE)
            masks = masks.to(DEVICE)

            cls_opt.zero_grad()
            cls_logits = classifier(images)
            cls_loss = cls_loss_fn(cls_logits, labels)
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
            cls_opt.step()

            loc_opt.zero_grad()
            pred_boxes = localizer(images)
            loc_loss = 0.5 * loc_reg_loss(pred_boxes, bboxes) + 1.0 * loc_iou_loss(pred_boxes, bboxes)
            loc_loss.backward()
            torch.nn.utils.clip_grad_norm_(localizer.parameters(), 5.0)
            loc_opt.step()

            seg_opt.zero_grad()
            seg_logits = segmenter(images)
            seg_loss = seg_ce_loss(seg_logits, masks) + 1.0 * dice_loss(seg_logits, masks)
            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(segmenter.parameters(), 5.0)
            seg_opt.step()

            train_cls_loss += cls_loss.item()
            train_loc_loss += loc_loss.item()
            train_seg_loss += seg_loss.item()

            if step == 1 or step % 20 == 0:
                print(
                    f"Epoch {epoch:02d} Step {step:03d} | "
                    f"CLS {cls_loss.item():.4f} | LOC {loc_loss.item():.4f} | SEG {seg_loss.item():.4f}"
                )

        classifier.eval()
        localizer.eval()
        segmenter.eval()

        val_cls_logits = []
        val_labels = []
        val_box_ious = []
        val_dice_total = 0.0
        val_pixel_total = 0.0
        vis_batch = None

        with torch.no_grad():
            for batch_idx, (images, labels, bboxes, masks) in enumerate(val_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                bboxes = bboxes.to(DEVICE)
                masks = masks.to(DEVICE)

                cls_logits = classifier(images)
                pred_boxes = localizer(images)
                seg_logits = segmenter(images)

                val_cls_logits.append(torch.argmax(cls_logits, dim=1).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                val_box_ious.extend(
                    [compute_iou(pred, target) for pred, target in zip(pred_boxes.cpu().numpy(), bboxes.cpu().numpy())]
                )
                val_dice_total += float(dice_score(seg_logits, masks).item())
                val_pixel_total += float(pixel_accuracy(seg_logits, masks).item())

                if batch_idx == 0:
                    vis_batch = (images, labels, bboxes, masks, cls_logits, pred_boxes, seg_logits)

                    act = images
                    act = classifier.encoder.conv1(act)
                    act = classifier.encoder.pool1(act)
                    act = classifier.encoder.conv2(act)
                    act = classifier.encoder.pool2(act)
                    act = classifier.encoder.conv3(act)
                    wandb.log({"conv3_activation": wandb.Histogram(act.detach().cpu().numpy())}, commit=False)

        val_preds = np.concatenate(val_cls_logits)
        val_targets = np.concatenate(val_labels)
        cls_macro_f1 = f1_score(val_targets, val_preds, average="macro")
        loc_iou_avg = float(np.mean(val_box_ious)) if val_box_ious else 0.0
        loc_acc_05 = float(np.mean(np.array(val_box_ious) >= 0.5)) if val_box_ious else 0.0
        loc_acc_075 = float(np.mean(np.array(val_box_ious) >= 0.75)) if val_box_ious else 0.0
        seg_dice_avg = val_dice_total / max(len(val_loader), 1)
        seg_pixel_avg = val_pixel_total / max(len(val_loader), 1)

        cls_sched.step(cls_macro_f1)
        loc_sched.step(loc_iou_avg)
        seg_sched.step(seg_dice_avg)

        log_payload = {
            "epoch": epoch,
            "train_cls_loss": train_cls_loss / max(len(train_loader), 1),
            "train_loc_loss": train_loc_loss / max(len(train_loader), 1),
            "train_seg_loss": train_seg_loss / max(len(train_loader), 1),
            "val_cls_macro_f1": cls_macro_f1,
            "val_loc_iou": loc_iou_avg,
            "val_loc_acc@0.5": loc_acc_05,
            "val_loc_acc@0.75": loc_acc_075,
            "val_dice": seg_dice_avg,
            "val_pixel_acc": seg_pixel_avg,
            "lr_classifier": cls_opt.param_groups[0]["lr"],
            "lr_localizer": loc_opt.param_groups[0]["lr"],
            "lr_segmenter": seg_opt.param_groups[0]["lr"],
        }

        if vis_batch is not None:
            images, labels, bboxes, masks, cls_logits, pred_boxes, seg_logits = vis_batch
            sample_img = (denormalize(images[0]) * 255).astype(np.uint8)
            gt_box = bboxes[0].detach().cpu().numpy()
            pred_box = pred_boxes[0].detach().cpu().numpy()
            sample_with_boxes = draw_box(draw_box(sample_img, gt_box, [0, 255, 0]), pred_box, [255, 0, 0])
            gt_mask = masks[0].detach().cpu().numpy()
            pred_mask = torch.argmax(seg_logits[0], dim=0).detach().cpu().numpy()

            first_map = segmenter.encoder.conv1(images)[0].mean(dim=0).detach().cpu().numpy()
            bottleneck = segmenter.encoder(images)[0].mean(dim=0).detach().cpu().numpy()
            first_map = ((first_map - first_map.min()) / (first_map.max() - first_map.min() + 1e-6) * 255).astype(np.uint8)
            bottleneck = ((bottleneck - bottleneck.min()) / (bottleneck.max() - bottleneck.min() + 1e-6) * 255).astype(np.uint8)

            log_payload.update(
                {
                    "bbox_visualization": wandb.Image(sample_with_boxes),
                    "gt_mask": wandb.Image(colorize_mask(gt_mask)),
                    "pred_mask": wandb.Image(colorize_mask(pred_mask)),
                    "first_layer_feature": wandb.Image(first_map),
                    "last_layer_feature": wandb.Image(bottleneck),
                }
            )

            bbox_table = wandb.Table(columns=["image", "iou"])
            limit = min(10, images.shape[0])
            for idx in range(limit):
                img_i = (denormalize(images[idx]) * 255).astype(np.uint8)
                pred_i = pred_boxes[idx].detach().cpu().numpy()
                gt_i = bboxes[idx].detach().cpu().numpy()
                overlay = draw_box(draw_box(img_i, gt_i, [0, 255, 0]), pred_i, [255, 0, 0])
                bbox_table.add_data(wandb.Image(overlay), compute_iou(pred_i, gt_i))
            log_payload["IoU_table"] = bbox_table

        wandb.log(log_payload)

        print(
            f"Epoch {epoch:02d} | "
            f"F1 {cls_macro_f1:.4f} | IoU {loc_iou_avg:.4f} | "
            f"Acc@0.5 {loc_acc_05:.4f} | Dice {seg_dice_avg:.4f}"
        )

        if cls_macro_f1 > best_cls_f1:
            best_cls_f1 = cls_macro_f1
            save_checkpoint(classifier, "classifier.pth", epoch, best_cls_f1)

        if loc_iou_avg > best_loc_iou:
            best_loc_iou = loc_iou_avg
            save_checkpoint(localizer, "localizer.pth", epoch, best_loc_iou)

        if seg_dice_avg > best_seg_dice:
            best_seg_dice = seg_dice_avg
            save_checkpoint(segmenter, "unet.pth", epoch, best_seg_dice)

    if best_cls_f1 < 0:
        save_checkpoint(classifier, "classifier.pth", EPOCHS, 0.0)
    if best_loc_iou < 0:
        save_checkpoint(localizer, "localizer.pth", EPOCHS, 0.0)
    if best_seg_dice < 0:
        save_checkpoint(segmenter, "unet.pth", EPOCHS, 0.0)

    if run is not None:
        wandb.finish()


def run_report_experiments(wandb_mode: str = "online") -> None:
    for dropout_p in [0.0, 0.2, 0.5]:
        train(dropout_p=dropout_p, freeze_mode="full", wandb_mode=wandb_mode)

    for freeze_mode in ["freeze", "partial", "full"]:
        train(dropout_p=0.2, freeze_mode=freeze_mode, wandb_mode=wandb_mode)


if __name__ == "__main__":
    train(dropout_p=0.2, freeze_mode="full", wandb_mode="online")
    
    # DROPOUT EXPERIMENTS
    #for d in [0.0, 0.2, 0.5]:
     #   train(dropout_p=d, freeze_mode="full")

    # TRANSFER LEARNING EXPERIMENTS
  #for mode in ["freeze", "partial", "full"]:
   #     train(dropout_p=0.5, freeze_mode=mode)    
