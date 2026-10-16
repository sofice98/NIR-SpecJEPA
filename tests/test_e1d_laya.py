from __future__ import annotations

import torch

from models.lejepa.laya_spectral import SpectralLaya, SpectralLayaEncoder, spectral_block_mask


def test_block_mask_has_exact_rate_and_non_wrapping_runs() -> None:
    mask = spectral_block_mask(
        4, 67, 0.6, 5, 10,
        generator=torch.Generator().manual_seed(7),
    )
    assert mask.shape == (4, 67)
    assert mask.sum(dim=1).tolist() == [40, 40, 40, 40]
    for row in mask:
        runs = []
        start = None
        for index, value in enumerate(row.tolist() + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append(index - start)
                start = None
        assert all(1 <= length <= 10 for length in runs)


def test_laya_outputs_masked_targets_and_weighted_loss_with_gradients() -> None:
    torch.manual_seed(3)
    model = SpectralLaya(SpectralLayaEncoder(41, patch_size=6))
    spectra = torch.randn(3, 41)
    mask = spectral_block_mask(3, model.encoder.n_patches, 0.6, 2, 4)
    output = model(spectra, mask)
    assert output["targets"].shape == (3, model.encoder.n_patches, 48)
    assert output["prediction"].shape == output["targets"].shape
    expected = 0.95 * output["invariance_mse"] + 0.05 * output["sigreg"]
    assert torch.allclose(output["loss"], expected)
    output["loss"].backward()
    assert model.encoder.patch_projection.weight.grad is not None
    assert model.mask_token.grad is not None


def test_encoder_frozen_representation_is_deterministic_in_eval_mode() -> None:
    model = SpectralLaya(SpectralLayaEncoder(41, patch_size=6)).eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    spectra = torch.randn(2, 41)
    with torch.no_grad():
        first = model.encoder(spectra)[1]
        second = model.encoder(spectra)[1]
    assert torch.allclose(first, second)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
