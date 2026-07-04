"""
generate_synthetic_data.py
---------------------------------------------------------
Generates synthetic 3D "MRI" volumes + segmentation masks that
mimic abdominal MRI scans used in MR-guided radiotherapy
(e.g. pancreas / liver tumor segmentation, similar in spirit to
the Medical Segmentation Decathlon Task07_Pancreas / Task03_Liver).

WHY SYNTHETIC DATA:
Real datasets (BraTS, Medical Segmentation Decathlon) are hosted
on data portals that require registration/auth and are not
reachable from this sandboxed environment. This generator produces
data with the same *shape, format, and statistical structure*
(NIfTI volumes, organ + tumor masks, realistic noise/intensity
patterns) so the full training/inference pipeline below is provable
and runnable end-to-end.

>>> SWAPPING IN REAL DATA <<<
To use real data instead, just point `--data_dir` in train.py at a
folder containing:
    imagesTr/case_XXX.nii.gz   (3D MRI volume)
    labelsTr/case_XXX.nii.gz   (matching segmentation mask: 0=bg,1=organ,2=tumor)
This is the exact folder layout the Medical Segmentation Decathlon
uses, so no code changes are needed elsewhere in this project.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_dilation


def make_organ_blob(shape, center, radius, irregularity=0.25, seed=None):
    """Create a smooth, organ-like blob (e.g. pancreas/liver) via a
    noisy, blurred sphere — gives irregular but plausible organ shape."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    dist = np.sqrt(
        ((zz - center[0]) / radius[0]) ** 2
        + ((yy - center[1]) / radius[1]) ** 2
        + ((xx - center[2]) / radius[2]) ** 2
    )
    noise = gaussian_filter(rng.normal(size=shape), sigma=3)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    surface = dist + irregularity * (noise - 0.5)
    blob = (surface < 1.0).astype(np.uint8)
    return blob


def make_tumor_blob(shape, center, radius, seed=None):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    dist = np.sqrt(
        ((zz - center[0]) / radius[0]) ** 2
        + ((yy - center[1]) / radius[1]) ** 2
        + ((xx - center[2]) / radius[2]) ** 2
    )
    noise = gaussian_filter(rng.normal(size=shape), sigma=2)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    surface = dist + 0.35 * (noise - 0.5)
    blob = (surface < 1.0).astype(np.uint8)
    return blob


def generate_case(shape=(64, 96, 96), seed=0, tumor_present=True):
    rng = np.random.default_rng(seed)

    # Base "tissue" background: smooth low-frequency field + noise,
    # similar to T2-weighted MRI intensity distribution
    base = gaussian_filter(rng.normal(loc=0.4, scale=0.08, size=shape), sigma=2)
    base = np.clip(base, 0, 1)

    # Organ (e.g. pancreas) somewhere in the central region
    organ_center = (
        shape[0] // 2 + rng.integers(-4, 4),
        shape[1] // 2 + rng.integers(-10, 10),
        shape[2] // 2 + rng.integers(-10, 10),
    )
    organ_radius = (
        shape[0] * rng.uniform(0.12, 0.18),
        shape[1] * rng.uniform(0.16, 0.22),
        shape[2] * rng.uniform(0.14, 0.20),
    )
    organ_mask = make_organ_blob(shape, organ_center, organ_radius, seed=seed)

    label = np.zeros(shape, dtype=np.uint8)
    label[organ_mask == 1] = 1  # organ-at-risk class

    image = base.copy()
    # organ tissue is brighter / different intensity than background
    image[organ_mask == 1] += 0.18
    image = gaussian_filter(image, sigma=0.5)

    if tumor_present:
        # tumor nested inside / adjacent to organ, smaller, different intensity
        offset = rng.integers(-6, 6, size=3)
        tumor_center = tuple(np.array(organ_center) + offset)
        tumor_radius = tuple(r * rng.uniform(0.25, 0.45) for r in organ_radius)
        tumor_mask = make_tumor_blob(shape, tumor_center, tumor_radius, seed=seed + 1000)
        # ensure tumor mostly sits within/near organ for realism
        tumor_mask = np.logical_and(
            tumor_mask, binary_dilation(organ_mask, iterations=3)
        ).astype(np.uint8)

        label[tumor_mask == 1] = 2  # tumor class
        image[tumor_mask == 1] -= 0.12  # tumors often hypo/hyper-intense vs organ

    # Add scanner-like Rician/Gaussian noise
    image += rng.normal(0, 0.03, size=shape)
    image = np.clip(image, 0, 1).astype(np.float32)

    return image, label


def save_nifti(volume, path, dtype):
    affine = np.eye(4)
    img = nib.Nifti1Image(volume.astype(dtype), affine)
    nib.save(img, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/home/claude/mridian-ai/data")
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_val", type=int, default=8)
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 96, 96))
    args = parser.parse_args()

    for split, n, start_seed in [
        ("Tr", args.n_train, 0),
        ("Val", args.n_val, 10000),
    ]:
        img_dir = os.path.join(args.out_dir, f"images{split}")
        lbl_dir = os.path.join(args.out_dir, f"labels{split}")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(n):
            seed = start_seed + i
            tumor_present = True  # all cases have a tumor for this demo task
            image, label = generate_case(tuple(args.shape), seed=seed, tumor_present=tumor_present)
            case_name = f"case_{i:03d}.nii.gz"
            save_nifti(image, os.path.join(img_dir, case_name), np.float32)
            save_nifti(label, os.path.join(lbl_dir, case_name), np.uint8)

        print(f"[{split}] wrote {n} synthetic volumes to {img_dir} / {lbl_dir}")


if __name__ == "__main__":
    main()
