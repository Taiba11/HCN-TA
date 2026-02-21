"""
Capsule Layer Primitives for HCN-TA.

Implements squash, PrimaryCapsuleLayer, HigherCapsuleLayer with dynamic routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(tensor, dim=-1):
    """Squash activation: normalizes vector length to [0, 1]."""
    sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(sq_norm + 1e-8)
    return (sq_norm / (1.0 + sq_norm)) * tensor / norm


class PrimaryCapsuleLayer(nn.Module):
    """Primary capsule layer converting conv features to capsule vectors."""

    def __init__(self, in_channels, num_capsules=8, capsule_dim=32,
                 kernel_size=9, stride=2):
        super().__init__()
        self.capsule_dim = capsule_dim
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, capsule_dim, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2)
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        outputs = torch.stack([c(x) for c in self.capsules], dim=1)
        batch = outputs.size(0)
        outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()
        outputs = outputs.view(batch, -1, self.capsule_dim)
        return squash(outputs)


class HigherCapsuleLayer(nn.Module):
    """Higher capsule layer with dynamic routing (Section 2.3)."""

    def __init__(self, num_capsules=2, num_routes=-1, in_dim=32,
                 out_dim=16, routing_iterations=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, out_dim, in_dim) * 0.01
        )

    def forward(self, x):
        batch = x.size(0)
        x_exp = x.unsqueeze(2).unsqueeze(4)
        W = self.W.expand(batch, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x_exp).squeeze(-1)

        b_ij = torch.zeros(batch, self.num_routes, self.num_capsules, 1, device=x.device)

        for i in range(self.routing_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j, dim=-1)
            if i < self.routing_iterations - 1:
                b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)
