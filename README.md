# 🩻 MRI Tumor Segmentation AI

AI-powered auto-contouring for MR-guided radiotherapy — built with 3D U-Net, MONAI and PyTorch.

## Results
| Metric | Score |
|--------|-------|
| Organ Dice | 0.985 |
| Tumor Dice | 0.917 |

## What it does
Automatically segments organ-at-risk (blue) and tumor (red) regions on MRI scans — the same category of task that powers real-time auto-contouring in MR-guided radiotherapy systems like ViewRay MRIdian A3i.

## Tech Stack
- 3D U-Net (MONAI / PyTorch)
- Dice + CrossEntropy loss
- Sliding-window inference
- Streamlit demo app

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Files
- app.py — Streamlit web demo
- model.py — 3D U-Net architecture
- train.py — Training pipeline
- infer.py — Inference script
- generate_synthetic_data.py — Synthetic MRI data generator
