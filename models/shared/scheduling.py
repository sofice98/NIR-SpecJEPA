"""Common per-optimizer-step learning-rate scheduling for neural experiments."""

from __future__ import annotations

import math


def build_step_scheduler(
    optimizer,
    scheduler_name: str = "warmup_cosine",
    total_steps: int = 1,
    warmup_steps: int = 1,
    eta_min: float = 1e-6,
):
    """Warm up from 1% for one epoch, then decay to an absolute LR floor.

    Call ``step`` after each optimizer update. A one-epoch run performs
    warmup only. ``cosine`` is a backwards-compatible alias for this policy.
    """
    if scheduler_name == "constant" or total_steps == 0:
        return None
    if scheduler_name not in {"warmup_cosine", "cosine"}:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    if total_steps < 0 or warmup_steps < 0:
        raise ValueError("Scheduler step counts must be non-negative")
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if any(lr <= 0 for lr in base_lrs) or not 0 <= eta_min <= min(base_lrs):
        raise ValueError("eta_min must lie between zero and every positive base learning rate")
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = min(warmup_steps, total_steps)

    def scale(step: int, base_lr: float) -> float:
        """计算当前步的预热或余弦衰减倍率。"""
        if step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        if total_steps == warmup_steps:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        minimum = eta_min / base_lr
        return minimum + (1.0 - minimum) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return LambdaLR(optimizer, [lambda step, lr=lr: scale(step, lr) for lr in base_lrs])
