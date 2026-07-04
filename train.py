"""
train.py
---------------------------------------------------------
Trains a 3D U-Net to segment organ (1) and tumor (2) from
background (0) on MRI volumes, using MONAI's data pipeline,
Dice+CrossEntropy loss, and sliding-window validation.

Run:
    python3 train.py --data_dir ../data --epochs 30

Swap in real data: point --data_dir at a folder with
imagesTr/labelsTr/imagesVal/labelsVal in NIfTI format (same
layout as the Medical Segmentation Decathlon) -- no other
changes needed.
"""

import os
import argparse
import glob
import json
import time

import torch
import numpy as np

from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandRotate90d, RandGaussianNoised, ToTensord,
    AsDiscrete, Activations,
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model import build_unet


def make_loaders(data_dir, batch_size=2, num_workers=2):
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    val_images = sorted(glob.glob(os.path.join(data_dir, "imagesVal", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(data_dir, "labelsVal", "*.nii.gz")))

    train_files = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
    val_files = [{"image": i, "label": l} for i, l in zip(val_images, val_labels)]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.02),
        ToTensord(keys=["image", "label"]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/claude/mridian-ai/data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", type=str, default="/home/claude/mridian-ai/checkpoints")
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="resume from last_model.pth + history.json")
    parser.add_argument("--start_epoch", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders(args.data_dir, batch_size=args.batch_size)

    model = build_unet(in_channels=1, out_channels=3).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])
    post_label = Compose([AsDiscrete(to_onehot=3)])

    best_dice = -1.0
    history = {"epoch": [], "train_loss": [], "val_dice_organ": [], "val_dice_tumor": []}

    last_ckpt = os.path.join(args.ckpt_dir, "last_model.pth")
    hist_path = os.path.join(args.ckpt_dir, "history.json")
    if args.resume and os.path.exists(last_ckpt):
        model.load_state_dict(torch.load(last_ckpt, map_location=device))
        print(f"Resumed weights from {last_ckpt}")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            vals = [d for d in history["val_dice_organ"] if d is not None]
            tvals = [d for d in history["val_dice_tumor"] if d is not None]
            if vals and tvals:
                best_dice = max((v + t) / 2 for v, t in zip(vals, tvals))
            print(f"Resumed history with {len(history['epoch'])} prior epochs, best_dice so far={best_dice:.4f}")

    t0 = time.time()
    end_epoch = args.start_epoch + args.epochs - 1
    for epoch in range(args.start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(train_loader), 1)
        scheduler.step()

        log_line = f"Epoch {epoch}/{end_epoch} | train_loss={epoch_loss:.4f}"

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    roi_size = (64, 96, 96)
                    val_outputs = sliding_window_inference(val_inputs, roi_size, 1, model)
                    val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_list = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            dice_scores = dice_metric.aggregate()
            # dice_scores: per-class (organ, tumor) since include_background=False
            organ_dice = float(dice_scores[0])
            tumor_dice = float(dice_scores[1])
            mean_dice = (organ_dice + tumor_dice) / 2

            log_line += f" | val_dice_organ={organ_dice:.4f} val_dice_tumor={tumor_dice:.4f}"

            history["val_dice_organ"].append(organ_dice)
            history["val_dice_tumor"].append(tumor_dice)

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth"))
                log_line += " | ** new best, saved **"
        else:
            history["val_dice_organ"].append(None)
            history["val_dice_tumor"].append(None)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        print(log_line)

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        torch.save(model.state_dict(), last_ckpt)

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed/60:.1f} min. Best mean Dice: {best_dice:.4f}")

    with open(os.path.join(args.ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "last_model.pth"))


if __name__ == "__main__":
    main()
