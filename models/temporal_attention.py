"""
Multi-Resolution Temporal Attention with Temporal Locality Awareness (Section 2.4).

This is the key novel component of HCN-TA that differentiates it from prior work.

Multi-Resolution Attention:
    e_t^(r) = W_e^(r) · C_low(t)
    Captures anomalies at different time scales (fine, medium, coarse).

Temporal Locality Awareness:
    L_t = ||C_low(t) - C_low(t-1)||
    Prioritizes regions with abrupt changes (likely deepfake artifacts).

Combined Attention Weights:
    α_t = softmax(e_t · L_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLocalityAwareness(nn.Module):
    """
    Temporal Locality Awareness (Section 2.4).

    Computes locality scores by measuring the frame-to-frame difference
    in capsule representations. Abrupt changes indicate potential artifacts.

        L_t = ||C_low(t) - C_low(t-1)||

    Args:
        normalize (bool): Whether to L2-normalize locality scores.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, capsule_sequence):
        """
        Args:
            capsule_sequence: (batch, T, capsule_dim) — temporal capsule outputs.
        Returns:
            locality_scores: (batch, T) — how much each frame differs from previous.
        """
        # Compute frame-to-frame differences
        # Pad the first frame with zeros (no previous frame)
        shifted = torch.cat([
            torch.zeros_like(capsule_sequence[:, :1, :]),
            capsule_sequence[:, :-1, :]
        ], dim=1)

        # L_t = ||C_low(t) - C_low(t-1)||
        diff = capsule_sequence - shifted
        locality_scores = torch.norm(diff, p=2, dim=-1)
        # (batch, T)

        if self.normalize:
            # Normalize to [0, 1] range per sample
            min_val = locality_scores.min(dim=-1, keepdim=True)[0]
            max_val = locality_scores.max(dim=-1, keepdim=True)[0]
            locality_scores = (locality_scores - min_val) / (max_val - min_val + 1e-8)

        return locality_scores


class MultiResolutionAttention(nn.Module):
    """
    Multi-Resolution Temporal Attention (Section 2.4).

    Uses multiple attention heads at different resolutions to capture
    anomalies across different time scales.

        e_t^(r) = W_e^(r) · C_low(t)

    Each resolution has a different learned attention weight matrix.

    Args:
        capsule_dim (int): Dimension of input capsule vectors.
        num_resolutions (int): Number of temporal resolutions.
        hidden_dim (int): Hidden dimension of attention networks.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        capsule_dim: int = 32,
        num_resolutions: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions

        # Attention networks for each resolution r
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(capsule_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_resolutions)
        ])

        # Pooling at different resolutions (avg pool with different kernel sizes)
        self.pool_sizes = [1, 2, 4]  # Fine, medium, coarse

        # Resolution fusion
        self.fusion = nn.Linear(num_resolutions, 1)

    def forward(self, capsule_sequence):
        """
        Args:
            capsule_sequence: (batch, T, capsule_dim)
        Returns:
            attention_scores: (batch, T) — multi-resolution attention scores.
        """
        batch, T, D = capsule_sequence.shape
        all_scores = []

        for r, (head, pool_size) in enumerate(zip(self.attention_heads, self.pool_sizes)):
            if pool_size > 1 and T >= pool_size:
                # Downsample for coarser resolution
                pooled = F.avg_pool1d(
                    capsule_sequence.permute(0, 2, 1),
                    kernel_size=pool_size, stride=pool_size,
                ).permute(0, 2, 1)
                # Compute attention at this resolution
                scores_r = head(pooled).squeeze(-1)  # (batch, T_r)
                # Upsample back to original resolution
                scores_r = F.interpolate(
                    scores_r.unsqueeze(1), size=T, mode="linear", align_corners=False
                ).squeeze(1)
            else:
                scores_r = head(capsule_sequence).squeeze(-1)  # (batch, T)

            all_scores.append(scores_r)

        # Stack and fuse: (batch, T, num_resolutions) → (batch, T)
        stacked = torch.stack(all_scores, dim=-1)
        fused = self.fusion(stacked).squeeze(-1)

        return fused


class TemporalAttentionModule(nn.Module):
    """
    Complete Temporal Attention Module combining multi-resolution attention
    and temporal locality awareness (Section 2.4).

    Final attention weights:
        α_t = softmax(e_t · L_t)

    Args:
        capsule_dim (int): Dimension of capsule vectors from HCN.
        num_resolutions (int): Number of attention resolutions.
        hidden_dim (int): Hidden dimension for attention heads.
        dropout (float): Dropout rate.
        locality_awareness (bool): Whether to use temporal locality awareness.
    """

    def __init__(
        self,
        capsule_dim: int = 32,
        num_resolutions: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        locality_awareness: bool = True,
    ):
        super().__init__()

        self.multi_res_attention = MultiResolutionAttention(
            capsule_dim=capsule_dim,
            num_resolutions=num_resolutions,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.locality_awareness = locality_awareness
        if locality_awareness:
            self.temporal_locality = TemporalLocalityAwareness(normalize=True)

    def forward(self, capsule_sequence):
        """
        Args:
            capsule_sequence: (batch, T, capsule_dim) — lower capsule outputs.
        Returns:
            attended: (batch, capsule_dim) — attention-weighted summary.
            attention_weights: (batch, T) — for visualization.
        """
        # Multi-resolution attention scores: e_t
        e_t = self.multi_res_attention(capsule_sequence)
        # (batch, T)

        if self.locality_awareness:
            # Temporal locality scores: L_t
            L_t = self.temporal_locality(capsule_sequence)
            # (batch, T)

            # Combined: α_t = softmax(e_t · L_t)
            combined = e_t * L_t
        else:
            combined = e_t

        # Softmax to get attention weights
        alpha_t = F.softmax(combined, dim=-1)
        # (batch, T)

        # Weighted sum: context = Σ α_t · C_low(t)
        attended = (alpha_t.unsqueeze(-1) * capsule_sequence).sum(dim=1)
        # (batch, capsule_dim)

        return attended, alpha_t
