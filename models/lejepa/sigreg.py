"""SIGReg adapted from the downloaded LeJEPA reference implementation."""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """用随机投影的特征函数约束表示分布。"""

    def __init__(self, knots: int = 17, projections: int = 256) -> None:
        """预计算高斯目标特征函数与积分权重。"""
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.projections = projections
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        """Estimate the characteristic-function distance to N(0, I).

        ``projected`` is ``[views, batch, dimension]``. As in mnist.py,
        compute the empirical characteristic function over the batch for
        each view separately, multiply by batch size, then average views
        and random directions.
        """
        if projected.ndim != 3:
            raise ValueError("projected must have shape [views, batch, dimension]")
        directions = torch.randn(
            projected.size(-1), self.projections, device=projected.device, dtype=projected.dtype
        )
        directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
        x_t = (projected @ directions).unsqueeze(-1) * self.t.to(projected.dtype)
        empirical_cos = x_t.cos().mean(dim=-3)
        empirical_sin = x_t.sin().mean(dim=-3)
        error = (empirical_cos - self.phi.to(projected.dtype)).square()
        error = error + empirical_sin.square()
        statistic = (error @ self.weights.to(projected.dtype)) * projected.size(-2)
        return statistic.mean()
