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

os.environ["WANDB_API_KEY"] = "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"

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 20
LOCALIZER_EPOCHS = 20
SEGMENTER_EPOCHS = 20
CLASSIFIER_LR = 1e-3
LOCALIZER_LR = 1e-4
SEGMENTER_LR = 1e-4
NUM_WORKERS = 0
SEED = 42

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


def build_loaders(crop_for_classification: bool = False) -> tuple[DataLoader, DataLoader]:
    train_ds = OxfordIIITPetDataset("data", "train", crop_for_classification=crop_for_classification)
    val_ds = OxfordIIITPetDataset("data", "val", crop_for_classification=crop_for_classification)
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


def save_checkpoint(model: nn.Module, filename: str, epoch: int, best_metric: float) -> None:
    payload = {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_metric}
    torch.save(payload, filename)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    torch.save(payload, checkpoints_dir / filename)


def load_checkpoint_into_model(model: nn.Module, filename: str) -> float:
    candidates = [Path(filename), Path("checkpoints") / filename]
    for candidate in candidates:
        if candidate.exists():
            payload = torch.load(candidate, map_location=DEVICE)
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            model.load_state_dict(state_dict, strict=False)
            if isinstance(payload, dict):
                return float(payload.get("best_metric", 0.0))
            return 0.0
    raise FileNotFoundError(f"Checkpoint not found for {filename}")


def log_feature_maps(classifier: VGG11Classifier, segmenter: VGG11UNet, images: torch.Tensor) -> dict:
    with torch.no_grad():
        x = images
        first_layer = classifier.encoder.conv1(x)[0].mean(dim=0).detach().cpu().numpy()
        bottleneck = segmenter.encoder(x)[0].mean(dim=0).detach().cpu().numpy()

    first_layer = ((first_layer - first_layer.min()) / (first_layer.max() - first_layer.min() + 1e-6) * 255).astype(np.uint8)
    bottleneck = ((bottleneck - bottleneck.min()) / (bottleneck.max() - bottleneck.min() + 1e-6) * 255).astype(np.uint8)
    return {
        "first_layer_feature": wandb.Image(first_layer),
        "last_layer_feature": wandb.Image(bottleneck),
    }


def train_classifier(model: VGG11Classifier, train_loader: DataLoader, val_loader: DataLoader) -> float:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.MultiStepLR(
    #   optimizer,
    #    milestones=[20, 35, 45],
    #    gamma=0.1,)
    
    best_f1 = -1.0

    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, (images, labels, _, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            train_total += labels.numel()
            if step == 1 or step % 20 == 0:
                print(f"[CLS] Epoch {epoch:02d} Step {step:03d} | Loss {loss.item():.4f}")

        model.eval()
        pred_labels = []
        true_labels = []
        conv3_hist = None
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (images, labels, _, _) in enumerate(val_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)
                pred_labels.append(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.append(labels.cpu().numpy())
                val_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                val_total += labels.numel()

                if batch_idx == 0:
                    act = images
                    act = model.encoder.conv1(act)
                    act = model.encoder.pool1(act)
                    act = model.encoder.conv2(act)
                    act = model.encoder.pool2(act)
                    act = model.encoder.conv3(act)
                    conv3_hist = wandb.Histogram(act.detach().cpu().numpy())

        y_true = np.concatenate(true_labels)
        y_pred = np.concatenate(pred_labels)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        # scheduler.step()

        wandb.log(
            {
                "cls_epoch": epoch,
                "cls_train_loss": train_loss / max(len(train_loader), 1),
                "cls_train_acc": train_acc,
                "val_cls_macro_f1": macro_f1,
                "val_cls_acc": val_acc,
                "cls_lr": optimizer.param_groups[0]["lr"],
                "conv3_activation": conv3_hist,
            }
        )
        print(
            f"[CLS] Epoch {epoch:02d} | "
            f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f} | Macro-F1 {macro_f1:.4f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            save_checkpoint(model, "classifier.pth", epoch, best_f1)

    return best_f1


def train_localizer(
    model: VGG11Localizer,
    classifier: VGG11Classifier,
    encoder_state,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> float:
    model.encoder.load_state_dict(encoder_state, strict=True)
    criterion_reg = nn.SmoothL1Loss(beta=5.0)
    criterion_iou = IoULoss()
    optimizer = optim.Adam(model.parameters(), lr=LOCALIZER_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_iou = -1.0

    for epoch in range(1, LOCALIZER_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for step, (images, _, boxes, _) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            boxes = boxes.to(DEVICE)

            optimizer.zero_grad()
            pred_boxes = model(images)
            loss = 0.5 * criterion_reg(pred_boxes, boxes) + criterion_iou(pred_boxes, boxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            if step == 1 or step % 20 == 0:
                print(f"[LOC] Epoch {epoch:02d} Step {step:03d} | Loss {loss.item():.4f}")

        model.eval()
        ious = []
        bbox_table = None

        with torch.no_grad():
            for batch_idx, (images, _, boxes, _) in enumerate(val_loader):
                images = images.to(DEVICE)
                boxes = boxes.to(DEVICE)
                pred_boxes = model(images)
                cls_probs = torch.softmax(classifier(images), dim=1)
                ious.extend([compute_iou(pred, target) for pred, target in zip(pred_boxes.cpu().numpy(), boxes.cpu().numpy())])

                if batch_idx == 0:
                    bbox_table = wandb.Table(columns=["image", "confidence", "iou"])
                    limit = min(10, images.shape[0])
                    for idx in range(limit):
                        img_i = (denormalize(images[idx]) * 255).astype(np.uint8)
                        pred_i = pred_boxes[idx].detach().cpu().numpy()
                        gt_i = boxes[idx].detach().cpu().numpy()
                        overlay = draw_box(draw_box(img_i, gt_i, [0, 255, 0]), pred_i, [255, 0, 0])
                        confidence = float(cls_probs[idx].max().item())
                        bbox_table.add_data(wandb.Image(overlay), confidence, compute_iou(pred_i, gt_i))

        mean_iou = float(np.mean(ious)) if ious else 0.0
        scheduler.step(mean_iou)
        wandb.log(
            {
                "loc_epoch": epoch,
                "loc_train_loss": train_loss / max(len(train_loader), 1),
                "val_loc_iou": mean_iou,
                "val_loc_acc@0.5": float(np.mean(np.array(ious) >= 0.5)) if ious else 0.0,
                "val_loc_acc@0.75": float(np.mean(np.array(ious) >= 0.75)) if ious else 0.0,
                "loc_lr": optimizer.param_groups[0]["lr"],
                "IoU_table": bbox_table,
            }
        )
        print(f"[LOC] Epoch {epoch:02d} | Mean-IoU {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            save_checkpoint(model, "localizer.pth", epoch, best_iou)

    return best_iou


def set_segmentation_freeze_mode(segmenter: VGG11UNet, freeze_mode: str) -> None:
    if freeze_mode == "freeze":
        for param in segmenter.encoder.parameters():
            param.requires_grad = False
    elif freeze_mode == "partial":
        params = list(segmenter.encoder.parameters())
        for param in params[: len(params) // 2]:
            param.requires_grad = False


def train_segmenter(
    classifier: VGG11Classifier,
    model: VGG11UNet,
    encoder_state,
    train_loader: DataLoader,
    val_loader: DataLoader,
    freeze_mode: str,
) -> float:
    model.encoder.load_state_dict(encoder_state, strict=True)
    set_segmentation_freeze_mode(model, freeze_mode)
    criterion_ce = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 2.0], device=DEVICE))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=SEGMENTER_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_dice = -1.0

    for epoch in range(1, SEGMENTER_EPOCHS + 1):
        model.train()
        # stabilize learning
        if epoch <= 5:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
                
        train_loss = 0.0

        for step, (images, _, _, masks) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion_ce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            if step == 1 or step % 20 == 0:
                print(f"[SEG] Epoch {epoch:02d} Step {step:03d} | Loss {loss.item():.4f}")

        model.eval()
        dice_values = []
        pixel_values = []
        visual_log = {}

        with torch.no_grad():
            for batch_idx, (images, _, boxes, masks) in enumerate(val_loader):
                images = images.to(DEVICE)
                boxes = boxes.to(DEVICE)
                masks = masks.to(DEVICE)
                logits = model(images)
                dice_values.append(float(dice_score(logits, masks).item()))
                pixel_values.append(float(pixel_accuracy(logits, masks).item()))

                if batch_idx == 0:
                    pred_masks = torch.argmax(logits, dim=1)
                    sample_img = (denormalize(images[0]) * 255).astype(np.uint8)
                    visual_log = {
                        "bbox_visualization": wandb.Image(draw_box(sample_img, boxes[0].detach().cpu().numpy(), [0, 255, 0])),
                        "gt_mask": wandb.Image(colorize_mask(masks[0].detach().cpu().numpy())),
                        "pred_mask": wandb.Image(colorize_mask(pred_masks[0].detach().cpu().numpy())),
                    }
                    visual_log.update(log_feature_maps(classifier, model, images))
                    for idx in range(min(5, images.shape[0])):
                        visual_log[f"seg_sample_{idx}_input"] = wandb.Image((denormalize(images[idx]) * 255).astype(np.uint8))
                        visual_log[f"seg_sample_{idx}_gt"] = wandb.Image(colorize_mask(masks[idx].detach().cpu().numpy()))
                        visual_log[f"seg_sample_{idx}_pred"] = wandb.Image(colorize_mask(pred_masks[idx].detach().cpu().numpy()))

        mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
        mean_pixel_acc = float(np.mean(pixel_values)) if pixel_values else 0.0
        scheduler.step(mean_dice)
        wandb.log(
            {
                "seg_epoch": epoch,
                "seg_train_loss": train_loss / max(len(train_loader), 1),
                "val_dice": mean_dice,
                "val_pixel_acc": mean_pixel_acc,
                "seg_lr": optimizer.param_groups[0]["lr"],
                **visual_log,
            }
        )
        print(f"[SEG] Epoch {epoch:02d} | Dice {mean_dice:.4f} | PixelAcc {mean_pixel_acc:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(model, "unet.pth", epoch, best_dice)

    return best_dice


def train(
    dropout_p: float = 0.2,
    freeze_mode: str = "full",
    wandb_mode: str = "online",
    use_batchnorm: bool = True,
) -> None:
    set_seed()
    wandb.init(
        project="da6401_assignment_02",
        name=f"dropout_{dropout_p}_{freeze_mode}",
        config={
            "dropout": dropout_p,
            "freeze_mode": freeze_mode,
            "use_batchnorm": use_batchnorm,
            "batch_size": BATCH_SIZE,
            "classifier_epochs": CLASSIFIER_EPOCHS,
            "localizer_epochs": LOCALIZER_EPOCHS,
            "segmenter_epochs": SEGMENTER_EPOCHS,
        },
        mode=wandb_mode,
    )

    # Classification was collapsing with crop-only training. Use full normalized images
    # so train/val/inference distributions stay aligned.
    cls_train_loader, cls_val_loader = build_loaders(crop_for_classification=False)
    task_train_loader, task_val_loader = build_loaders(crop_for_classification=False)

    # Keep the classifier simpler and more stable; BN was hurting validation generalization.
    classifier = VGG11Classifier(dropout_p=0.3, use_batchnorm=False).to(DEVICE)
    best_cls_f1 = train_classifier(classifier, cls_train_loader, cls_val_loader)
    load_checkpoint_into_model(classifier, "classifier.pth")
    encoder_state = classifier.encoder.state_dict()
    print(f"[CLS] Best saved Macro-F1: {best_cls_f1:.4f}")

    localizer = VGG11Localizer(dropout_p=dropout_p, use_batchnorm=use_batchnorm).to(DEVICE)
    best_loc_iou = train_localizer(localizer, classifier, encoder_state, task_train_loader, task_val_loader)

    segmenter = VGG11UNet(dropout_p=dropout_p, use_batchnorm=use_batchnorm).to(DEVICE)
    best_seg_dice = train_segmenter(classifier, segmenter, encoder_state, task_train_loader, task_val_loader, freeze_mode)

    wandb.summary["best_cls_macro_f1"] = best_cls_f1
    wandb.summary["best_loc_iou"] = best_loc_iou
    wandb.summary["best_seg_dice"] = best_seg_dice
    wandb.finish()


def run_report_experiments(wandb_mode: str = "online") -> None:
    for use_batchnorm in [False, True]:
        train(dropout_p=0.2, freeze_mode="full", wandb_mode=wandb_mode, use_batchnorm=use_batchnorm)

    for dropout_p in [0.0, 0.2, 0.5]:
        train(dropout_p=dropout_p, freeze_mode="full", wandb_mode=wandb_mode, use_batchnorm=True)

    for freeze_mode in ["freeze", "partial", "full"]:
        train(dropout_p=0.2, freeze_mode=freeze_mode, wandb_mode=wandb_mode, use_batchnorm=True)


if __name__ == "__main__":
    train(dropout_p=0.2, freeze_mode="full", wandb_mode="online")
