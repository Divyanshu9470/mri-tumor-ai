"""
model.py
---------------------------------------------------------
3D U-Net for multi-class segmentation (background / organ / tumor),
built on MONAI's UNet implementation -- the same architecture family
used in most published MR-guided radiotherapy auto-contouring papers.
"""

from monai.networks.nets import UNet
from monai.networks.layers import Norm


def build_unet(in_channels=1, out_channels=3, spatial_dims=3):
    """
    out_channels = 3 -> background, organ-at-risk, tumor
    """
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    )
    return model
