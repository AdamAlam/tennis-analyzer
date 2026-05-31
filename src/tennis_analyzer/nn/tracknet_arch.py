"""The TrackNet encoder/decoder used by the ball and court models.

This is a faithful reimplementation of the architecture from
``github.com/yastrebksv/TrackNet`` (ball) and ``github.com/yastrebksv/TennisCourtDetector``
(court). Both use the exact same module/layer names, so the published pretrained weights load
directly via :meth:`torch.nn.Module.load_state_dict`:

- Ball:  ``in_channels=9``  (3 stacked RGB frames), ``out_channels=256`` (heatmap levels).
- Court: ``in_channels=3``,  ``out_channels=14`` (one heatmap per court keypoint).

The network is a symmetric VGG-style encoder/decoder: three max-pool downsamples and three
nearest-neighbour upsamples, restoring the input spatial resolution (typically 360x640).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> ReLU -> BatchNorm, the repeated unit of the network."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 pad: int = 1, stride: int = 1, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad,
                      bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BallTrackerNet(nn.Module):
    """VGG-style heatmap network shared by the ball (Step 5) and court (Step 4) models.

    Parameters
    ----------
    in_channels:
        Number of input channels (9 for the 3-frame ball model, 3 for the court model).
    out_channels:
        Number of output channels (256 heatmap levels for the ball, 14 keypoints for the court).
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw ``(B, out_channels, H, W)`` feature map.

        Callers decode this differently: the ball tracker takes ``argmax`` over the channel
        dim to recover a single intensity heatmap; the court detector applies a per-channel
        sigmoid to get 14 keypoint heatmaps.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
