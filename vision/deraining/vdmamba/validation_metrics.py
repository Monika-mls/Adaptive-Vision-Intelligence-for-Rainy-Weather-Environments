"""Reference metrics for paired RGB deraining experiments.

Images are float tensors [B,3,H,W] in [0,1]. SSIM uses an 11x11,
sigma=1.5 Gaussian window, reflect padding, and population local moments.
Metrics are calculated at the current input resolution, not full image size.
"""
import math
import torch
import torch.nn.functional as F


def check_images(rainy, restored, clean):
    if not all(isinstance(x, torch.Tensor) for x in (rainy, restored, clean)):
        raise TypeError("Expected PyTorch tensors")
    if rainy.shape != restored.shape or rainy.shape != clean.shape:
        raise ValueError("Rainy, restored and clean shapes must match")
    if rainy.ndim != 4 or rainy.shape[1] != 3 or min(rainy.shape) < 1:
        raise ValueError("Expected nonempty [B,3,H,W] images")
    if not all(x.is_floating_point() and torch.isfinite(x).all().item()
               for x in (rainy, restored, clean)):
        raise ValueError("Images must contain finite floating point values")
    for name, x in (("rainy", rainy), ("clean", clean)):
        if x.min().item() < 0 or x.max().item() > 1:
            raise ValueError(f"{name} must be in [0,1]")


def ssim_per_image(x, y, window=11, sigma=1.5):
    """Gaussian-window SSIM; returns a [B] tensor (higher is better).

    For small images, window adapts to the smallest odd spatial extent.
    """
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError("SSIM requires matching [B,C,H,W] images")
    if not torch.isfinite(x).all().item() or not torch.isfinite(y).all().item():
        raise ValueError("Nonfinite SSIM input")
    x, y = x.float(), y.float()
    height, width = x.shape[-2:]
    k = min(window, height, width)
    if k % 2 == 0:
        k -= 1
    if k < 1:
        raise ValueError("Invalid image spatial size")
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel.expand(x.shape[1], 1, k, k).contiguous()
    pad = k // 2
    # Replicate padding also works for one-pixel dimensions.
    def blur(z):
        return F.conv2d(F.pad(z, (pad, pad, pad, pad), mode="replicate"),
                        kernel, groups=x.shape[1])
    mu_x, mu_y = blur(x), blur(y)
    var_x = (blur(x * x) - mu_x * mu_x).clamp_min(0)
    var_y = (blur(y * y) - mu_y * mu_y).clamp_min(0)
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    return (num / den).mean(dim=(1, 2, 3))


def psnr_from_mse(value):
    return float("inf") if value == 0 else -10 * math.log10(value)


@torch.inference_mode()
def evaluate_batch(rainy, restored, clean, clean_threshold=1/255):
    """Return per-image metrics and weighted pooled-MSE sufficient statistics.

    A 'clean-like' pixel means max-channel |rainy-clean| <= threshold.
    This is an oracle diagnostic using clean targets, NOT an inference mask.
    """
    check_images(rainy, restored, clean)
    if not math.isfinite(clean_threshold) or clean_threshold < 0:
        raise ValueError("clean_threshold must be finite and nonnegative")
    r = rainy.float()
    c = clean.float()
    p = restored.float().clamp(0, 1)
    baseline_error = (r - c).square()
    output_error = (p - c).square()
    in_ssim = ssim_per_image(r, c).cpu().tolist()
    out_ssim = ssim_per_image(p, c).cpu().tolist()
    rows = []
    for i in range(r.shape[0]):
        mse_r = baseline_error[i].mean().item()
        mse_p = output_error[i].mean().item()
        region_mask = ((r[i] - c[i]).abs().amax(dim=0) <= clean_threshold)
        fraction = region_mask.float().mean().item()
        if region_mask.any().item():
            changed = (p[i] - r[i]).abs().mean(dim=0)[region_mask]
            clean_region_mae = changed.mean().item()
        else:
            clean_region_mae = None
        rows.append({
            "rainy_mse": mse_r,
            "restored_mse": mse_p,
            "rainy_psnr": psnr_from_mse(mse_r),
            "restored_psnr": psnr_from_mse(mse_p),
            "psnr_gain": (psnr_from_mse(mse_p) - psnr_from_mse(mse_r)
                          if mse_r > 0 else None),
            "rainy_ssim": in_ssim[i],
            "restored_ssim": out_ssim[i],
            "ssim_gain": out_ssim[i] - in_ssim[i],
            "improved_mse": mse_p < mse_r,
            "perfect_input_crop": mse_r == 0,
            "clean_like_fraction": fraction,
            "clean_like_change_mae": clean_region_mae,
            "pixel_count": c[i].numel(),
        })
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    total = sum(r["pixel_count"] for r in rows)
    rainy_mse = sum(r["rainy_mse"] * r["pixel_count"] for r in rows) / total
    restored_mse = sum(r["restored_mse"] * r["pixel_count"] for r in rows) / total
    import statistics
    finite_gain = [r["psnr_gain"] for r in rows if r["psnr_gain"] is not None
                   and math.isfinite(r["psnr_gain"])]
    region = [r["clean_like_change_mae"] for r in rows
              if r["clean_like_change_mae"] is not None]
    return {
        "count": len(rows),
        "pooled_rainy_psnr": psnr_from_mse(rainy_mse),
        "pooled_restored_psnr": psnr_from_mse(restored_mse),
        "pooled_psnr_gain": psnr_from_mse(restored_mse) - psnr_from_mse(rainy_mse)
                            if rainy_mse > 0 else None,
        "mean_rainy_ssim": statistics.mean(r["rainy_ssim"] for r in rows),
        "mean_restored_ssim": statistics.mean(r["restored_ssim"] for r in rows),
        "mean_ssim_gain": statistics.mean(r["ssim_gain"] for r in rows),
        "improved_mse_count": sum(r["improved_mse"] for r in rows),
        "perfect_input_count": sum(r["perfect_input_crop"] for r in rows),
        "finite_pair_mean_psnr_gain": statistics.mean(finite_gain) if finite_gain else None,
        "finite_pair_count": len(finite_gain),
        "mean_clean_like_change_mae": statistics.mean(region) if region else None,
        "clean_like_threshold": "1/255 RGB max-channel difference (default)",
    }
