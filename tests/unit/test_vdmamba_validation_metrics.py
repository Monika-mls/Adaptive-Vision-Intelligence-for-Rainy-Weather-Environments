import math
import pytest
import torch
from vision.deraining.vdmamba.validation_metrics import (
    ssim_per_image, evaluate_batch, summarize
)


def test_identical_images_psnr_infinite_ssim_one():
    x = torch.rand(1, 3, 16, 16)
    row = evaluate_batch(x, x, x)[0]
    assert math.isinf(row["rainy_psnr"])
    assert row["psnr_gain"] is None
    assert row["rainy_ssim"] == pytest.approx(1, abs=1e-5)
    assert row["restored_ssim"] == pytest.approx(1, abs=1e-5)
    assert row["clean_like_change_mae"] == 0
    assert not row["improved_mse"]


def test_restoration_improves_metrics():
    clean = torch.zeros(1, 3, 16, 16)
    rainy = torch.full_like(clean, 0.5)
    restored = torch.full_like(clean, 0.1)
    row = evaluate_batch(rainy, restored, clean)[0]
    assert row["psnr_gain"] > 0
    assert row["improved_mse"]
    assert row["ssim_gain"] > 0
    result = summarize([row])
    assert result["pooled_psnr_gain"] > 0
    assert result["improved_mse_count"] == 1


def test_clean_like_region_quantifies_unnecessary_change():
    clean = torch.full((1, 3, 8, 8), 0.5)
    rainy = clean.clone()
    rainy[:, :, :4, :] += 0.2
    restored = rainy.clone()
    restored[:, :, 4:, :] += 0.05
    row = evaluate_batch(rainy, restored, clean)[0]
    assert row["clean_like_fraction"] == pytest.approx(0.5)
    assert row["clean_like_change_mae"] == pytest.approx(0.05, abs=1e-5)


def test_summarize_includes_perfect_input_without_negative_infinity():
    x = torch.zeros(1, 3, 12, 12)
    perfect = evaluate_batch(x, x+0.01, x)[0]
    changed = evaluate_batch(x+0.2, x+0.1, x)[0]
    result = summarize([perfect, changed])
    assert result["perfect_input_count"] == 1
    assert math.isfinite(result["pooled_psnr_gain"])
    assert result["finite_pair_count"] == 1


def test_output_clamped_for_psnr():
    clean = torch.zeros(1, 3, 8, 8)
    rainy = torch.ones_like(clean)
    restored = torch.full_like(clean, -0.2)
    row = evaluate_batch(rainy, restored, clean)[0]
    assert row["restored_mse"] == 0


def test_wrong_shape_rejected():
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError):
        evaluate_batch(x, torch.zeros(1, 3, 4, 4), x)


def test_nonfinite_rejected():
    x = torch.zeros(1, 3, 8, 8)
    z = x.clone()
    z[0,0,0,0] = float('nan')
    with pytest.raises(ValueError):
        evaluate_batch(x, z, x)


def test_small_image_ssim():
    x = torch.rand(1, 3, 1, 1)
    assert torch.isfinite(ssim_per_image(x, x)).all()
