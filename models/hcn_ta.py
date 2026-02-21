"""
HCN-TA: Hierarchical Capsule Network with Temporal Attention
for a Generalizable Approach to Audio Deepfake Detection.

Full pipeline (Section 2):
    1. Mel-Spectrogram preprocessing
    2. ResNet50 feature extraction
    3. Hierarchical Capsule Network (lower + higher capsules)
    4. Multi-Resolution Temporal Attention + Locality Awareness
    5. Classification via capsule norms + Margin Loss

Paper: Wani et al., ACM SAC 2025
"""

import torch
import torch.nn as nn

from .resnet_backbone import ResNet50Backbone
from .hierarchical_capsule import HierarchicalCapsuleNetwork
from .temporal_attention import TemporalAttentionModule
from .capsule_layers import squash


class HCNTA(nn.Module):
    """
    HCN-TA: Hierarchical Capsule Network with Temporal Attention.

    Args:
        num_classes (int): Number of classes (2 = real/fake).
        pretrained_backbone (bool): Use pretrained ResNet50.
        lower_num_caps (int): Number of lower-level capsule types.
        lower_cap_dim (int): Lower capsule vector dimension.
        higher_cap_dim (int): Higher capsule vector dimension.
        num_resolutions (int): Number of temporal attention resolutions.
        attention_hidden_dim (int): Hidden dim for attention heads.
        routing_iterations (int): Dynamic routing iterations.
        locality_awareness (bool): Use temporal locality awareness.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        lower_num_caps: int = 8,
        lower_cap_dim: int = 32,
        higher_cap_dim: int = 16,
        num_resolutions: int = 3,
        attention_hidden_dim: int = 256,
        routing_iterations: int = 3,
        locality_awareness: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: ResNet50 backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)

        # Stage 2: Hierarchical Capsule Network
        self.hcn = HierarchicalCapsuleNetwork(
            in_channels=512,
            lower_num_caps=lower_num_caps,
            lower_cap_dim=lower_cap_dim,
            higher_num_caps=num_classes,
            higher_cap_dim=higher_cap_dim,
            routing_iterations=routing_iterations,
        )

        # Stage 3: Multi-Resolution Temporal Attention
        self.temporal_attention = TemporalAttentionModule(
            capsule_dim=lower_cap_dim,
            num_resolutions=num_resolutions,
            hidden_dim=attention_hidden_dim,
            locality_awareness=locality_awareness,
        )

        # Projection: attended capsule summary → class capsule refinement
        self.attention_projection = nn.Sequential(
            nn.Linear(lower_cap_dim, higher_cap_dim * num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_attention=False):
        """
        Full forward pass.

        Args:
            x: (batch, 3, 224, 224) — mel-spectrogram images.
            return_attention (bool): Also return attention weights for visualization.

        Returns:
            v_final: (batch, num_classes, higher_cap_dim) — final capsule vectors.
            attention_weights: (batch, T) — optional, for visualization.
        """
        # Stage 1: Feature extraction
        features = self.backbone(x)
        # (batch, 512, 7, 7)

        # Stage 2: Hierarchical capsules
        lower_caps, higher_caps = self.hcn(features)
        # lower_caps: (batch, N_lower, lower_cap_dim)
        # higher_caps: (batch, num_classes, higher_cap_dim)

        # Stage 3: Temporal attention on lower capsules
        attended, attn_weights = self.temporal_attention(lower_caps)
        # attended: (batch, lower_cap_dim)

        # Project attended features and add to higher capsule output
        attn_projected = self.attention_projection(attended)
        attn_projected = attn_projected.view(-1, self.num_classes, higher_caps.size(-1))
        # (batch, num_classes, higher_cap_dim)

        # Fuse: combine capsule output with attention-refined features
        v_final = squash(higher_caps + attn_projected)
        # (batch, num_classes, higher_cap_dim)

        if return_attention:
            return v_final, attn_weights
        return v_final

    def predict(self, x):
        """Predict class and confidence."""
        v = self.forward(x)
        confs = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)
        preds = confs.argmax(dim=1)
        return preds, confs

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
