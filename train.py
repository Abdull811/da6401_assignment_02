"""Training entrypoint
"""

import os
import argparse

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "localization", "segmentation", "multitask"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--freeze_strategy",
        type=str,
        default="full",
        choices=["freeze", "partial", "full"],
    )
    return parser.parse_args()


def read_split_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Split file not found: " + path)

    files = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                files.append(parts[0] + ".jpg")
    return files


def load_labels(annotation_dir):
    labels = {}
    for split_name in ["trainval.txt", "test.txt"]:
        split_path = os.path.join(annotation_dir, split_name)
        if not os.path.exists(split_path):
            continue

        with open(split_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0] + ".jpg"] = int(parts[1]) - 1

    if len(labels) == 0:
        raise FileNotFoundError("Could not load labels from annotations folder: " + annotation_dir)

    return labels


def read_tag_value(text, tag):
    start_token = "<" + tag + ">"
    end_token = "</" + tag + ">"
    start = text.find(start_token)
    end = text.find(end_token)

    if start == -1 or end == -1:
        return None

    start += len(start_token)
    return text[start:end].strip()


def resolve_xml_dir(annotations_dir):
    candidates = [
        os.path.join(annotations_dir, "xmls"),
        os.path.join(annotations_dir, "xml"),
        os.path.join(annotations_dir, "annotations"),
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        "Could not find bbox xml directory. Expected one of: "
        + ", ".join(candidates)
    )


def load_bboxes(xml_dir):
    bboxes = {}
    for file_name in os.listdir(xml_dir):
        if not file_name.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, file_name)
        with open(xml_path, "r", encoding="utf-8") as file:
            text = file.read()

        xmin = read_tag_value(text, "xmin")
        ymin = read_tag_value(text, "ymin")
        xmax = read_tag_value(text, "xmax")
        ymax = read_tag_value(text, "ymax")

        if None in [xmin, ymin, xmax, ymax]:
            continue

        image_name = os.path.splitext(file_name)[0] + ".jpg"
        bboxes[image_name] = [float(xmin), float(ymin), float(xmax), float(ymax)]

    if len(bboxes) == 0:
        raise FileNotFoundError("No valid xml bbox annotations found in: " + xml_dir)

    return bboxes


def make_train_val_split(trainval_files, val_split=0.2, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(trainval_files), generator=generator).tolist()

    split_index = int(len(trainval_files) * (1.0 - val_split))
    if split_index <= 0:
        split_index = 1
    if split_index >= len(trainval_files):
        split_index = len(trainval_files) - 1

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_files = [trainval_files[i] for i in train_indices]
    val_files = [trainval_files[i] for i in val_indices]
    return train_files, val_files


def build_dataloaders(data_root, batch_size, num_workers, val_split):
    images_dir = os.path.join(data_root, "images")
    annotations_dir = os.path.join(data_root, "annotations")
    masks_dir = os.path.join(annotations_dir, "trimaps")
    xml_dir = resolve_xml_dir(annotations_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError("Images directory not found: " + images_dir)
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError("Annotations directory not found: " + annotations_dir)
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError("Trimaps directory not found: " + masks_dir)

    labels_dict = load_labels(annotations_dir)
    bboxes_dict = load_bboxes(xml_dir)

    trainval_path = os.path.join(annotations_dir, "trainval.txt")
    trainval_files = read_split_file(trainval_path)
    trainval_files = [name for name in trainval_files if name in labels_dict and name in bboxes_dict]

    if len(trainval_files) < 2:
        raise ValueError("Not enough training samples found after matching labels and bboxes.")

    train_files, val_files = make_train_val_split(trainval_files, val_split=val_split)

    train_dataset = OxfordIIITPetDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        labels_dict=labels_dict,
        bboxes_dict=bboxes_dict,
        image_files=train_files,
    )
    val_dataset = OxfordIIITPetDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        labels_dict=labels_dict,
        bboxes_dict=bboxes_dict,
        image_files=val_files,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def freeze_backbone(model, strategy):
    encoder = None
    if hasattr(model, "encoder"):
        encoder = model.encoder

    if encoder is None:
        return

    for parameter in encoder.parameters():
        parameter.requires_grad = True

    if strategy == "freeze":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    elif strategy == "partial":
        for block_name in ["block1", "block2", "block3"]:
            if hasattr(encoder, block_name):
                for parameter in getattr(encoder, block_name).parameters():
                    parameter.requires_grad = False


def get_model(task, dropout_p):
    if task == "classification":
        return VGG11Classifier(dropout_p=dropout_p)
    if task == "localization":
        return VGG11Localizer(dropout_p=dropout_p)
    if task == "segmentation":
        return VGG11UNet(dropout_p=dropout_p)
    return MultiTaskPerceptionModel()


def get_checkpoint_name(task):
    if task == "classification":
        return "classifier.pth"
    if task == "localization":
        return "localizer.pth"
    if task == "segmentation":
        return "unet.pth"
    return "multitask.pth"


def compute_iou_score(pred_boxes, target_boxes):
    loss = IoULoss(reduction="none")(pred_boxes, target_boxes)
    return (1.0 - loss).mean().item()


def dice_score(logits, targets, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    preds_fg = (preds > 0).float()
    targets_fg = (targets > 0).float()

    intersection = (preds_fg * targets_fg).sum(dim=(1, 2))
    union = preds_fg.sum(dim=(1, 2)) + targets_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def pixel_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def forward_and_loss(model, imgs, labels, bboxes, masks, task, cls_loss_fn, seg_loss_fn, iou_loss_fn):
    outputs = model(imgs)

    if task == "classification":
        loss = cls_loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        metrics = {
            "f1": f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
        }
        return loss, metrics

    if task == "localization":
        mse_loss = nn.functional.mse_loss(outputs, bboxes)
        iou_loss = iou_loss_fn(outputs, bboxes)
        loss = mse_loss + iou_loss
        metrics = {
            "iou": compute_iou_score(outputs.detach(), bboxes.detach())
        }
        return loss, metrics

    if task == "segmentation":
        loss = seg_loss_fn(outputs, masks)
        metrics = {
            "dice": dice_score(outputs.detach(), masks.detach()),
            "pixel_acc": pixel_accuracy(outputs.detach(), masks.detach()),
        }
        return loss, metrics

    cls_loss = cls_loss_fn(outputs["classification"], labels)
    loc_mse = nn.functional.mse_loss(outputs["localization"], bboxes)
    loc_iou = iou_loss_fn(outputs["localization"], bboxes)
    seg_loss = seg_loss_fn(outputs["segmentation"], masks)

    loss = cls_loss + loc_mse + loc_iou + seg_loss

    preds = torch.argmax(outputs["classification"], dim=1)
    metrics = {
        "f1": f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro"),
        "iou": compute_iou_score(outputs["localization"].detach(), bboxes.detach()),
        "dice": dice_score(outputs["segmentation"].detach(), masks.detach()),
        "pixel_acc": pixel_accuracy(outputs["segmentation"].detach(), masks.detach()),
    }
    return loss, metrics


def run_epoch(model, loader, optimizer, device, task, cls_loss_fn, seg_loss_fn, iou_loss_fn, training):
    if training:
        model.train()
    else:
        model.eval()

    totals = {"loss": 0.0}
    steps = 0

    for imgs, labels, bboxes, masks in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            loss, metrics = forward_and_loss(
                model,
                imgs,
                labels,
                bboxes,
                masks,
                task,
                cls_loss_fn,
                seg_loss_fn,
                iou_loss_fn,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        totals["loss"] += loss.item()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1

    for key in totals:
        totals[key] /= steps
    return totals


def wandb_start(args):
    if args.wandb_mode == "disabled":
        return None, args

    try:
        wandb.init(
            project="da6401_Assigment_02_Weight_&_Biase",
            config=vars(args),
            mode=args.wandb_mode,
        )
    except Exception:
        wandb.init(
            project="da6401_Assigment_02_Weight_&_Biase",
            config=vars(args),
            mode="offline",
        )

    return wandb.run, wandb.config


def wandb_log(data, run):
    if run is not None:
        wandb.log(data)


def wandb_finish(run):
    if run is not None:
        wandb.finish()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run, config = wandb_start(args)

    train_loader, val_loader = build_dataloaders(
        config.data_root,
        config.batch_size,
        config.num_workers,
        config.val_split,
    )

    device = torch.device(config.device)
    model = get_model(config.task, config.dropout_p).to(device)

    if config.task in ["localization", "segmentation"]:
        freeze_backbone(model, config.freeze_strategy)

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config.lr,
    )

    cls_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = nn.CrossEntropyLoss()
    iou_loss_fn = IoULoss(reduction="mean")

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        train_logs = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config.task,
            cls_loss_fn,
            seg_loss_fn,
            iou_loss_fn,
            training=True,
        )

        val_logs = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            config.task,
            cls_loss_fn,
            seg_loss_fn,
            iou_loss_fn,
            training=False,
        )

        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_logs["loss"],
            "val/loss": val_logs["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        for key, value in train_logs.items():
            if key != "loss":
                log_dict["train/" + key] = value

        for key, value in val_logs.items():
            if key != "loss":
                log_dict["val/" + key] = value

        wandb_log(log_dict, run)

        if val_logs["loss"] < best_val_loss:
            best_val_loss = val_logs["loss"]
            save_path = os.path.join(config.checkpoint_dir, get_checkpoint_name(config.task))
            torch.save({"state_dict": model.state_dict()}, save_path)

    wandb_finish(run)


if __name__ == "__main__":
    main()
