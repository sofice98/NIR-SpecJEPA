"""Prediction from the detached online LeJEPA probe."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
def predict_saved_lejepa_probe(
    embeddings: np.ndarray,
    model_state: dict[str, torch.Tensor],
) -> np.ndarray:
    """Apply the probe learned during one-stage LeJEPA training.

    The probe is saved together with the joint LeJEPA checkpoint.  Rebuilding
    its tiny module here keeps the downstream evaluation path free of a second
    supervised fit while preserving the exact normalization used in training.
    """
    probe_state = {
        key.removeprefix("probe."): value
        for key, value in model_state.items()
        if key.startswith("probe.")
    }
    try:
        output_weight = probe_state["1.weight"]
        target_mean = model_state["target_mean"].detach().cpu().numpy()
        target_scale = model_state["target_scale"].detach().cpu().numpy()
    except KeyError as exc:
        raise ValueError("LeJEPA checkpoint does not contain a complete probe") from exc
    embedding_dim = int(output_weight.shape[1])
    output_dim = int(output_weight.shape[0])
    probe = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, output_dim))
    probe.load_state_dict(probe_state)
    probe.eval()
    with torch.no_grad():
        normalized = probe(torch.as_tensor(embeddings, dtype=torch.float32)).cpu().numpy()
    return normalized * target_scale[None, :] + target_mean[None, :]
