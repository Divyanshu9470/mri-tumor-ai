"""
infer.py
---------------------------------------------------------
Loads a trained checkpoint and runs segmentation on a given
NIfTI volume, returning the predicted organ/tumor mask plus
per-class Dice score if ground truth is available.
"""

import torch
import numpy as np
import nibabel as nib

from monai.transforms import Compose, ScaleIntensity, ToTensor
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Activations

from model import build_unet


def load_model(ckpt_path, device="cpu"):
    model = build_unet(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def preprocess(volume_np):
    """volume_np: (D, H, W) float array -> tensor (1, 1, D, H, W)"""
    transform = Compose([ScaleIntensity(), ToTensor()])
    vol = transform(volume_np)
    vol = vol.unsqueeze(0).unsqueeze(0).float()
    return vol


def predict_volume(model, volume_np, device="cpu", roi_size=(64, 96, 96)):
    inputs = preprocess(volume_np).to(device)
    with torch.no_grad():
        outputs = sliding_window_inference(inputs, roi_size, 1, model)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1)
    pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    prob_np = probs.squeeze(0).cpu().numpy()  # (3, D, H, W) class probabilities
    return pred_np, prob_np


def dice_score(pred, label, cls):
    p = (pred == cls)
    g = (label == cls)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/claude/mridian-ai/checkpoints/best_model.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    img = nib.load(args.image).get_fdata().astype(np.float32)
    pred, prob = predict_volume(model, img, device)

    print("Prediction shape:", pred.shape)
    print("Organ voxels predicted:", (pred == 1).sum())
    print("Tumor voxels predicted:", (pred == 2).sum())

    if args.label:
        lbl = nib.load(args.label).get_fdata().astype(np.uint8)
        print("Dice (organ):", round(dice_score(pred, lbl, 1), 4))
        print("Dice (tumor):", round(dice_score(pred, lbl, 2), 4))
