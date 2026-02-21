"""
Hierarchical Capsule Network (HCN) for HCN-TA (Section 2.3).

Lower-level capsules: capture local dependencies (phonetic transitions, tonal continuity)
    C_low(t) = CapsuleLayer(F_res(t, f))

Higher-level capsules: model global patterns (prosody, sentence coherence)
    Dynamic routing: v̂_j|i = W_ij · v_i
    Output: v_j = squash(Σ c_ij · v̂_j|i)
"""

import torch
import torch.nn as nn

from .capsule_layers import PrimaryCapsuleLayer, HigherCapsuleLayer, squash


class HierarchicalCapsuleNetwork(nn.Module):
    """
    Hierarchical Capsule Network with two levels.

    Level 1 (Lower): Captures local time-frequency patterns
    Level 2 (Higher): Aggregates into global representations via dynamic routing

    Args:
        in_channels (int): Input channels from backbone.
        lower_num_caps (int): Number of lower-level capsule types.
        lower_cap_dim (int): Dimension of lower capsule vectors.
        lower_kernel (int): Kernel size for lower capsule conv.
        lower_stride (int): Stride for lower capsule conv.
        higher_num_caps (int): Number of higher capsule types (2 = real/fake).
        higher_cap_dim (int): Dimension of higher capsule vectors.
        routing_iterations (int): Dynamic routing iterations.
    """

    def __init__(
        self,
        in_channels: int = 512,
        lower_num_caps: int = 8,
        lower_cap_dim: int = 32,
        lower_kernel: int = 9,
        lower_stride: int = 2,
        higher_num_caps: int = 2,
        higher_cap_dim: int = 16,
        routing_iterations: int = 3,
    ):
        super().__init__()

        # Pre-capsule conv layers
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Lower-level capsules: C_low(t)
        self.lower_capsules = PrimaryCapsuleLayer(
            in_channels=256,
            num_capsules=lower_num_caps,
            capsule_dim=lower_cap_dim,
            kernel_size=lower_kernel,
            stride=lower_stride,
        )

        self._higher_num_caps = higher_num_caps
        self._higher_cap_dim = higher_cap_dim
        self._lower_cap_dim = lower_cap_dim
        self._routing_iterations = routing_iterations
        self.higher_capsules = None
        self._initialized = False

    def _init_higher(self, num_routes, device):
        self.higher_capsules = HigherCapsuleLayer(
            num_capsules=self._higher_num_caps,
            num_routes=num_routes,
            in_dim=self._lower_cap_dim,
            out_dim=self._higher_cap_dim,
            routing_iterations=self._routing_iterations,
        ).to(device)
        self._initialized = True

    def forward(self, features):
        """
        Args:
            features: (batch, in_channels, H, W) — from ResNet50 backbone.
        Returns:
            lower_caps: (batch, num_lower_total, lower_cap_dim) — for temporal attention.
            higher_caps: (batch, num_classes, higher_cap_dim) — for classification.
        """
        x = self.pre_conv(features)
        lower_caps = self.lower_capsules(x)

        if not self._initialized:
            self._init_higher(lower_caps.size(1), features.device)

        higher_caps = self.higher_capsules(lower_caps)

        return lower_caps, higher_caps
