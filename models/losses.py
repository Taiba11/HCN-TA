"""
Margin Loss for HCN-TA (Section 2.5).

    L_k = T_k · max(0, m+ - ||v_k||)² + λ(1-T_k) · max(0, ||v_k|| - m-)²

Classification: Class = argmax(||v_real||, ||v_fake||)
"""

import torch
import torch.nn as nn


class MarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val

    def forward(self, v_k, targets):
        v_norm = torch.sqrt((v_k ** 2).sum(dim=-1) + 1e-8)
        T_k = torch.zeros(v_k.size(0), v_k.size(1), device=v_k.device)
        T_k.scatter_(1, targets.unsqueeze(1), 1.0)

        left = T_k * torch.clamp(self.m_plus - v_norm, min=0.0) ** 2
        right = self.lambda_val * (1 - T_k) * torch.clamp(v_norm - self.m_minus, min=0.0) ** 2
        return (left + right).sum(dim=-1).mean()
