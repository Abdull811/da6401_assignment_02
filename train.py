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
EPOCHS = 20
LR = 1e-4

# GLOBAL NORMALIZATION FIX
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def denormalize(img):
    img = img.cpu()
    img = img * std + mean
    img = img.clamp(0,1)
    return img.permute(1,2,0).numpy()

def colorize_mask(mask):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    colored[mask == 0] = [0, 0, 0]       # background
    colored[mask == 1] = [0, 255, 0]     # pet
    colored[mask == 2] = [255, 0, 0]     # boundary

    return colored
    
def draw_box(img, box, color):
    img = img.copy()

    H, W, _ = img.shape

    xc, yc, w, h = box

    # Convert normalized → pixel
    xc *= W
    yc *= H
    w *= W
    h *= H

    x1 = int(xc - w/2)
    y1 = int(yc - h/2)
    x2 = int(xc + w/2)
    y2 = int(yc + h/2)

    # Clamp to image boundaries
    x1 = max(0, min(x1, W-1))
    x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1))
    y2 = max(0, min(y2, H-1))

    img[y1:y1+2, x1:x2] = color
    img[y2-2:y2, x1:x2] = color
    img[y1:y2, x1:x1+2] = color
    img[y1:y2, x2-2:x2] = color

    return img

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
    # convert normalized → pixel
    box1 = box1.copy()
    box2 = box2.copy()

    box1 *= 224.0
    box2 *= 224.0

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
    cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loc_loss_fn = nn.MSELoss()
    iou_loss_fn = IoULoss()
    seg_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.3,1.0,1.0]).to(DEVICE))

    # Optimizers
    cls_opt = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
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
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
            cls_opt.step()

            # Localization
            loc_out = localizer(images)
            loc_loss = 0.2 * loc_loss_fn(loc_out, bboxes) + 1.0 * iou_loss_fn(loc_out, bboxes)

            loc_opt.zero_grad()
            loc_loss.backward()
            loc_opt.step()

            # Segmentation
            seg_out = segmenter(images)
            seg_loss = seg_loss_fn(seg_out, masks) + 1.5 * dice_loss(seg_out, masks)
            print("Pred classes:", torch.unique(torch.argmax(seg_out, dim=1)))

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
            for batch_idx, (images, labels, bboxes, masks) in enumerate(val_loader):

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                seg_out = segmenter(images)

                dice_total += dice_score(seg_out, masks)
                acc_total += pixel_accuracy(seg_out, masks)
                
                # Activation Histogram of 3rd conv layer
                x_act = images
                
                x_act = classifier.encoder.conv1(x_act)
                x_act = classifier.encoder.pool1(x_act)
                
                x_act = classifier.encoder.conv2(x_act)
                x_act = classifier.encoder.pool2(x_act)
                
                x_act = classifier.encoder.conv3(x_act)
                
                if batch_idx == 0:
                   wandb.log({
                       "conv3_activation": wandb.Histogram(x_act.detach().cpu().numpy())})
        

        dice_avg = dice_total / len(val_loader)
        acc_avg = acc_total / len(val_loader)
        vis_images = images
        vis_masks = masks
        
        with torch.no_grad():
            x = images
        
            # Block 1
            x1 = segmenter.encoder.conv1(x)
            first_map = x1[0].mean(dim=0).detach().cpu()
        
            x = segmenter.encoder.pool1(x1)
        
            # Block 2
            x = segmenter.encoder.conv2(x)
            x = segmenter.encoder.pool2(x)
        
            # Block 3
            x = segmenter.encoder.conv3(x)
            x = segmenter.encoder.conv4(x)
            x = segmenter.encoder.pool3(x)
        
            # Block 4
            x = segmenter.encoder.conv5(x)
            x = segmenter.encoder.conv6(x)
            x = segmenter.encoder.pool4(x)
        
            # Block 5
            x_last = segmenter.encoder.conv7(x)
            x_last = segmenter.encoder.conv8(x_last)
        
            last_map = x_last[0].mean(dim=0).detach().cpu()

        # SEGMENTATION VISUALIZATION (W&B)
        img_vis = denormalize(vis_images[0])

        img_np = (img_vis * 255).astype(np.uint8)

        gt_box = bboxes[0].cpu().numpy()
        pred_box = loc_out[0].detach().cpu().numpy()
        
        img_gt = draw_box(img_np, gt_box, [0,255,0])     # green GT
        img_pred = draw_box(img_gt, pred_box, [255,0,0]) # red pred
        
        gt = vis_masks[0].cpu()
        pred = torch.argmax(seg_out[0], dim=0).cpu()
        with torch.no_grad():
            feature = segmenter.encoder(images)

        feature = feature.detach().cpu()
        # convert [512,7,7] → [7,7]
        fm = feature[0].mean(dim=0)

        fm_vis = fm.numpy()
        fm_vis = (fm_vis - fm_vis.min()) / (fm_vis.max() - fm_vis.min() + 1e-6)
        
        iou = compute_iou(
            loc_out[0].detach().cpu().numpy(),
            bboxes[0].detach().cpu().numpy())
        

        wandb.log({
            "epoch": epoch,
            "train_cls_loss": total_cls,
            "train_loc_loss": total_loc,
            "train_seg_loss": total_seg,
            "val_dice": dice_avg,
            "val_pixel_acc": acc_avg,
            "iou": iou,
            
            # Image
            "input": wandb.Image(img_np),
            "bbox_visualization": wandb.Image(img_pred),
            "gt_mask": wandb.Image(colorize_mask(gt.numpy())),
            "pred_mask": wandb.Image(colorize_mask(pred.numpy())),
            "first_layer_feature": wandb.Image(
                ((first_map.numpy() - first_map.numpy().min()) /
                 (first_map.numpy().max() - first_map.numpy().min() + 1e-6) * 255).astype(np.uint8)
            ),
            "last_layer_feature": wandb.Image(
                ((last_map.numpy() - last_map.numpy().min()) /
                 (last_map.numpy().max() - last_map.numpy().min() + 1e-6) * 255).astype(np.uint8)
            ),

            # Feature map (FIXED)
            "feature_map": wandb.Image(
                ((fm_vis - fm_vis.min()) / (fm_vis.max() - fm_vis.min() + 1e-6) * 255).astype(np.uint8)
            )
        })

        for i in range(min(3, vis_images.shape[0])):
            wandb.log({
                f"sample_{i}_input": wandb.Image(denormalize(vis_images[i])),
                f"sample_{i}_gt": wandb.Image(colorize_mask(vis_masks[i].cpu().numpy())),
                f"sample_{i}_pred": wandb.Image(colorize_mask(torch.argmax(seg_out[i], dim=0).cpu().numpy()))
            })

        table = wandb.Table(columns=["image", "iou"])

        for i in range(min(5, vis_images.shape[0])):
            img_i = denormalize(vis_images[i])
            iou_i = compute_iou(
                loc_out[i].detach().cpu().numpy(),
                bboxes[i].detach().cpu().numpy()
            )
            table.add_data(wandb.Image(img_i), iou_i)
        
        wandb.log({"IoU_table": table})
        
        print(f"\nEpoch {epoch+1} DONE → CLS={total_cls:.3f}, LOC={total_loc:.3f}, SEG={total_seg:.3f}")
        print(f"Validation → Dice={dice_avg:.3f}, Acc={acc_avg:.3f}")

    # SAVE MODELS
    torch.save({"state_dict": classifier.state_dict()}, "classifier.pth")
    torch.save({"state_dict": localizer.state_dict()}, "localizer.pth")
    torch.save({"state_dict": segmenter.state_dict()}, "unet.pth")

    wandb.finish()


# RUN
if __name__ == "__main__":
    train(dropout_p=0.5, freeze_mode="full")
    
    # DROPOUT EXPERIMENTS
    #for d in [0.0, 0.2, 0.5]:
     #   train(dropout_p=d, freeze_mode="full")

    # TRANSFER LEARNING EXPERIMENTS
  #for mode in ["freeze", "partial", "full"]:
   #     train(dropout_p=0.5, freeze_mode=mode)    
