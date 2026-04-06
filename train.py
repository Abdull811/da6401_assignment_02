"""Training entrypoint for DA6401 Assignment-2"""

import os
import sys
import wandb

os.environ["WANDB_API_KEY"] = "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"

wandb.login()


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


np.random.seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4


# Metrics
def dice_score(pred, target):
    pred = torch.argmax(pred, dim=1)
    dice = 0
    for cls in range(1, 3):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice += (2 * inter + 1e-6) / (union + 1e-6)
    return dice / 2


def dice_loss(pred, target):
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=3)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    inter = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()


def pixel_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    return (pred == target).float().mean()


def compute_iou(box1, box2):
    def convert(box):
        xc, yc, w, h = box
        return xc-w/2, yc-h/2, xc+w/2, yc+h/2

    px1, py1, px2, py2 = convert(box1)
    tx1, ty1, tx2, ty2 = convert(box2)

    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)

    inter = max(ix2-ix1, 0) * max(iy2-iy1, 0)
    union = (px2-px1)*(py2-py1) + (tx2-tx1)*(ty2-ty1) - inter
    return inter / (union + 1e-6)


# Train 
def train(dropout_p=0.5, freeze_mode="full"):

    wandb.init(
        project="da6401_Assigment_02_Weight_&_Biase",
        name=f"dropout_{dropout_p}_{freeze_mode}",
        config={"dropout": dropout_p, "freeze": freeze_mode},
        mode="online"
    )
    print("W&B RUN:", wandb.run.url)


    # Data
    train_loader = DataLoader(
        OxfordIIITPetDataset("data", "train"),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        OxfordIIITPetDataset("data", "val"),
        batch_size=BATCH_SIZE
    )

    # Models
    classifier = VGG11Classifier(dropout_p=dropout_p).to(DEVICE)
    localizer = VGG11Localizer(dropout_p=dropout_p).to(DEVICE)
    segmenter = VGG11UNet(dropout_p=dropout_p).to(DEVICE)

    # Transfer Learning control
    # Freeze modes
    if freeze_mode == "freeze":
        for p in segmenter.encoder.parameters():
            p.requires_grad = False

    elif freeze_mode == "partial":
        params = list(segmenter.encoder.parameters())
        for p in params[:len(params)//2]:
            p.requires_grad = False

    # Losses
    cls_loss_fn = nn.CrossEntropyLoss()
    loc_loss_fn = nn.MSELoss()
    iou_loss_fn = IoULoss()
    seg_loss_fn = nn.CrossEntropyLoss()

    # Optimizers
    cls_opt = optim.Adam(classifier.parameters(), lr=LR)
    loc_opt = optim.Adam(localizer.parameters(), lr=LR)
    seg_opt = optim.Adam(segmenter.parameters(), lr=LR)

    
    # Train Loop
    for epoch in range(EPOCHS):

        print(f"\n===== EPOCH {epoch+1}/{EPOCHS} =====")

        classifier.train()
        localizer.train()
        segmenter.train()

        total_cls = total_loc = total_seg = 0

        for step, (images, labels, bboxes, masks) in enumerate(train_loader):

            if step == 0:  # Print shapes and stats for the first batch
                print("Image shape:", images.shape)
                print("Mask shape:", masks.shape)
                print("Mask unique values:", masks.unique())

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            bboxes = bboxes.to(DEVICE)
            masks = masks.to(DEVICE)

            # Classification
            cls_out = classifier(images)
            cls_loss = cls_loss_fn(cls_out, labels)

            cls_opt.zero_grad()
            cls_loss.backward()
            cls_opt.step()

            # Localization
            loc_out = localizer(images)
            loc_loss = loc_loss_fn(loc_out, bboxes) + iou_loss_fn(loc_out, bboxes)

            loc_opt.zero_grad()
            loc_loss.backward()
            loc_opt.step()

            # Segmentation
            seg_out = segmenter(images)
            seg_loss = seg_loss_fn(seg_out, masks) + dice_loss(seg_out, masks)

            seg_opt.zero_grad()
            seg_loss.backward()
            seg_opt.step()

            total_cls += cls_loss.item() / len(train_loader)
            total_loc += loc_loss.item() / len(train_loader)
            total_seg += seg_loss.item() / len(train_loader)

            # PRINT STEP PROGRESS
            if step % 20 == 0:
                print(f"[Step {step}] CLS: {cls_loss:.3f} | LOC: {loc_loss:.3f} | SEG: {seg_loss:.3f}")

        # VALIDATION
        classifier.eval()
        localizer.eval()
        segmenter.eval()

        dice_total = acc_total = 0

        with torch.no_grad():
            for images, labels, bboxes, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                seg_out = segmenter(images)

                dice_total += dice_score(seg_out, masks)
                acc_total += pixel_accuracy(seg_out, masks)

        dice_avg = dice_total / len(val_loader)
        acc_avg = acc_total / len(val_loader)

        with torch.no_grad():
            x = images
        
            # First layer
            x1 = segmenter.encoder.conv1(x)
            first_map = x1[0].mean(dim=0).detach().cpu()
        
            # Pass through encoder properly
            x = segmenter.encoder.pool(x1)
            x = segmenter.encoder.conv2(x)
            x = segmenter.encoder.pool(x)
            x = segmenter.encoder.conv3(x)
            x = segmenter.encoder.conv4(x)
            x = segmenter.encoder.pool(x)
            x = segmenter.encoder.conv5(x)
            x = segmenter.encoder.conv6(x)
            x = segmenter.encoder.pool(x)
        
            # Last conv block
            x_last = segmenter.encoder.conv7(x)
            x_last = segmenter.encoder.conv8(x_last)
        
            last_map = x_last[0].mean(dim=0).detach().cpu()

        # SEGMENTATION VISUALIZATION (W&B)
        img = images[0].cpu()
        img_vis = img.permute(1,2,0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-6)
        gt = masks[0].cpu()
        pred = torch.argmax(seg_out[0], dim=0).cpu()
        with torch.no_grad():
            feature = segmenter.encoder(images)

        feature = feature.detach().cpu()
        # convert [512,7,7] → [7,7]
        fm = feature[0].mean(dim=0)

        iou = compute_iou(
            loc_out[0].detach().cpu().numpy(),
            bboxes[0].detach().cpu().numpy()
        )

        wandb.log({
            "epoch": epoch,
            "train_cls_loss": total_cls,
            "train_loc_loss": total_loc,
            "train_seg_loss": total_seg,
            "val_dice": dice_avg,
            "val_pixel_acc": acc_avg,
            "iou": iou,
            
            # Image
            "input": wandb.Image(img_vis),
            "gt_mask": wandb.Image(gt.numpy()),
            "pred_mask": wandb.Image(pred.numpy()),

            # Feature map (FIXED)
            "feature_map": wandb.Image(fm.numpy() * 120)   
        })

        for i in range(min(3, images.shape[0])):
            wandb.log({
                f"sample_{i}_input": wandb.Image(images[i].cpu().permute(1,2,0).numpy()),
                f"sample_{i}_gt": wandb.Image(masks[i].cpu().numpy() * 100),
                f"sample_{i}_pred": wandb.Image(torch.argmax(seg_out[i], dim=0).cpu().numpy() * 100)
            })

        table = wandb.Table(columns=["image", "iou"])

        for i in range(min(5, images.shape[0])):
            img_i = images[i].cpu()
            img_i = img_i.permute(1,2,0).numpy()
            img_i = (img_i - img_i.min()) / (img_i.max() - img_i.min() + 1e-6)
            iou_i = compute_iou(
                loc_out[i].detach().cpu().numpy(),
                bboxes[i].detach().cpu().numpy()
            )
            table.add_data(wandb.Image(img_i), iou_i)
        
        wandb.log({"IoU_table": table})
        
        print(f"\nEpoch {epoch+1} DONE → CLS={total_cls:.3f}, LOC={total_loc:.3f}, SEG={total_seg:.3f}")
        print(f"Validation → Dice={dice_avg:.3f}, Acc={acc_avg:.3f}")

    # SAVE MODELS
    torch.save(classifier.state_dict(), "classifier.pth")
    torch.save(localizer.state_dict(), "localizer.pth")
    torch.save(segmenter.state_dict(), "unet.pth")

    wandb.finish()


# RUN
if __name__ == "__main__":
    train(dropout_p=0.5, freeze_mode="full")
    
    # DROPOUT EXPERIMENTS
    #for d in [0.0, 0.2, 0.5]:
     #   train(dropout_p=d, freeze_mode="full")

    # TRANSFER LEARNING EXPERIMENTS
    #for mode in ["freeze", "partial", "full"]:
     #   train(dropout_p=0.5, freeze_mode=mode)    
