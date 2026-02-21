"""
ResNet50 Feature Extraction Backbone for HCN-TA (Section 2.2).

    F_res(t, f) = σ(W * S(t, f) + b)  ∈  R^{T × F × C}

Extracts both local and global audio patterns from mel-spectrograms.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Backbone(nn.Module):
    """
    ResNet50-based feature extractor for mel-spectrograms.

    Removes the final FC/avgpool layers and returns intermediate feature maps.

    Args:
        pretrained (bool): Use ImageNet pretrained weights.
        freeze_layers (int): Freeze first N layers (0 = train all).
    """

    def __init__(self, pretrained: bool = True, freeze_layers: int = 0):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        # Modify first conv to accept 3-channel mel spectrogram images
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Channel reduction for capsule input
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        if freeze_layers > 0:
            self._freeze(freeze_layers)

    def _freeze(self, n):
        layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer in enumerate(layers):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224) — mel-spectrogram images.
        Returns:
            features: (batch, 512, 7, 7) — feature maps F_res(t, f).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)
        x = self.channel_reduce(x)

        return x
