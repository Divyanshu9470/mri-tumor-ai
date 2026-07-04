"""
app.py
---------------------------------------------------------
Interactive demo: upload an MRI volume (NIfTI .nii/.nii.gz),
run the trained U-Net, and visualize predicted organ + tumor
contours slice-by-slice — a simplified version of the
auto-contouring workflow used in MR-guided radiotherapy
planning (e.g. ViewRay's MRIdian A3i).

Run:
    streamlit run app.py
"""

import os
import io
import tempfile

import numpy as np
import nibabel as nib
import streamlit as st
import matplotlib.pyplot as plt
import torch

from model import build_unet
from infer import load_model, predict_volume, dice_score

st.set_page_config(page_title="MRIdian-style AI Auto-Contouring", layout="wide")

CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth")
DEMO_IMG = os.path.join(os.path.dirname(__file__), "..", "data", "imagesVal", "case_000.nii.gz")
DEMO_LBL = os.path.join(os.path.dirname(__file__), "..", "data", "labelsVal", "case_000.nii.gz")


@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT_PATH, device)
    return model, device


def overlay_mask(image_slice, mask_slice):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_slice, cmap="gray")
    organ_overlay = np.ma.masked_where(mask_slice != 1, mask_slice)
    tumor_overlay = np.ma.masked_where(mask_slice != 2, mask_slice)
    ax.imshow(organ_overlay, cmap="Blues", alpha=0.45, vmin=0, vmax=2)
    ax.imshow(tumor_overlay, cmap="Reds", alpha=0.55, vmin=0, vmax=2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


st.title("🩻 AI Auto-Contouring for MR-Guided Radiotherapy")
st.caption(
    "A 3D U-Net (MONAI / PyTorch) trained to segment organ-at-risk (blue) and "
    "tumor (red) on MRI volumes — the same kind of task that powers real-time "
    "auto-contouring in MR-guided radiotherapy systems."
)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload MRI volume (.nii / .nii.gz)", type=["nii", "gz"])
    use_demo = st.button("Use demo MRI case instead")
    st.markdown("---")
    st.markdown(
        "**Note:** This model is trained on synthetic data generated to mimic "
        "abdominal MRI tumor segmentation tasks (pancreas / liver), since real "
        "clinical datasets (e.g. Medical Segmentation Decathlon) require "
        "registration and are not bundled here. Swap in real NIfTI data with "
        "zero code changes — see README."
    )

image_path = None
label_path = None

if uploaded is not None:
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        image_path = tmp.name
elif use_demo or "demo_loaded" in st.session_state:
    st.session_state["demo_loaded"] = True
    image_path = DEMO_IMG
    label_path = DEMO_LBL

if image_path is None:
    st.info("Upload a NIfTI MRI volume in the sidebar, or click 'Use demo MRI case' to try it instantly.")
    st.stop()

if not os.path.exists(CKPT_PATH):
    st.warning("No trained checkpoint found yet — training may still be in progress. Run train.py first.")
    st.stop()

model, device = get_model()

img_nii = nib.load(image_path)
img = img_nii.get_fdata().astype(np.float32)

with st.spinner("Running AI segmentation..."):
    pred, prob = predict_volume(model, img, device)

label = None
if label_path and os.path.exists(label_path):
    label = nib.load(label_path).get_fdata().astype(np.uint8)

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Volume info")
    st.write(f"Shape: {img.shape}")
    st.write(f"Predicted organ voxels: {(pred==1).sum():,}")
    st.write(f"Predicted tumor voxels: {(pred==2).sum():,}")
    if label is not None:
        st.subheader("Accuracy vs ground truth")
        od = dice_score(pred, label, 1)
        td = dice_score(pred, label, 2)
        st.metric("Organ Dice score", f"{od:.3f}")
        st.metric("Tumor Dice score", f"{td:.3f}")

    slice_idx = st.slider("Slice (axial)", 0, img.shape[0] - 1, img.shape[0] // 2)

with col1:
    img_slice = img[slice_idx]
    pred_slice = pred[slice_idx]
    fig = overlay_mask(img_slice, pred_slice)
    st.pyplot(fig, use_container_width=False)
    st.caption("Blue = AI-predicted organ-at-risk contour · Red = AI-predicted tumor contour")

    if label is not None:
        st.markdown("**Ground truth (ref.) overlay**")
        gt_slice = label[slice_idx]
        fig2 = overlay_mask(img_slice, gt_slice)
        st.pyplot(fig2, use_container_width=False)

st.markdown("---")
st.markdown(
    "Built as a deep-learning portfolio project demonstrating MR-guided "
    "radiotherapy-style auto-contouring: 3D U-Net, Dice+CrossEntropy loss, "
    "sliding-window inference, trained/served with MONAI + PyTorch + Streamlit."
)
