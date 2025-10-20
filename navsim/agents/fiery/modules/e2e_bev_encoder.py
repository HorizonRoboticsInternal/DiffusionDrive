from typing import Dict, List
import torch.nn as nn
from einops import rearrange

from .convolutions import InvertedResidual


class E2EBEVEncoder(nn.Module):
    """
    Encoder BEV feature from sensor encoder.
    """
    def __init__(
        self,
        raster_num_input_channels: int,
        embed_dims: int,
        freeze: bool = False,
    ):
        super().__init__()
        self.bev_conv = nn.Sequential(
            InvertedResidual(raster_num_input_channels, 80, 2),
            InvertedResidual(80, 112, 2),
            InvertedResidual(112, 192, 2),
            # InvertedResidual(192, 320, 2),
            # InvertedResidual(320, embed_dims, 1),
            InvertedResidual(192, embed_dims, 1),
        )

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, bev_feature) -> None:

        bev_embed = self.bev_conv(bev_feature)

        return bev_embed
