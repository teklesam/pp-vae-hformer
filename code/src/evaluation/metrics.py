"""
Quantitative evaluation metrics for denoising and uncertainty assessment.

Referenced metrics (require clean reference):
    PSNR, SSIM, MS-SSIM, FSIM (via piq), LPIPS (via lpips)

Clinical utility metrics (domain-specific):
    CNR, Edge Preservation Index, Noise Power Spectrum

Uncertainty calibration metrics:
    ECE (Expected Calibration Error), sharpness
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import numpy as np

try:
    import piq
    _HAS_PIQ = True
except ImportError:
    _HAS_PIQ = False

try:
    import lpips as lpips_module
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


# ── Basic pixel metrics ────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Single-scale SSIM, averaged over the batch."""
    from ..losses.ms_ssim_loss import _gaussian_kernel, _ssim_components
    kernel = _gaussian_kernel(11, 1.5).to(pred.device)
    lc, _ = _ssim_components(pred, target, kernel)
    return lc.item()


def ms_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if _HAS_PIQ:
        return piq.multi_scale_ssim(pred, target, data_range=1.0).item()
    # Fallback: single-scale SSIM
    return ssim(pred, target)


def fsim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    FSIM via piq library.
    Zhang et al. (2012): FSIM ranks #1 of 11 FR-IQA metrics.
    Expects grayscale single-channel images in [0,1].
    piq.fsim requires 3-channel images; we replicate the channel.
    """
    if not _HAS_PIQ:
        return float("nan")
    pred3 = pred.expand(-1, 3, -1, -1)
    target3 = target.expand(-1, 3, -1, -1)
    return piq.fsim(pred3, target3, data_range=1.0, reduction="mean").item()


# ── Perceptual ─────────────────────────────────────────────────────────────────

_lpips_model = None


def lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    LPIPS (1 - value, so higher is better).
    Lazy-loads the AlexNet LPIPS model on first call.
    """
    if not _HAS_LPIPS:
        return float("nan")
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips_module.LPIPS(net="alex", verbose=False).to(pred.device)
        _lpips_model.eval()
    pred3 = pred.expand(-1, 3, -1, -1) * 2 - 1    # LPIPS expects [-1, 1]
    target3 = target.expand(-1, 3, -1, -1) * 2 - 1
    with torch.no_grad():
        d = _lpips_model(pred3, target3).mean().item()
    return 1.0 - d  # flip to higher=better


# ── Clinical utility metrics ───────────────────────────────────────────────────

def contrast_to_noise_ratio(
    image: torch.Tensor,
    roi_mask: torch.Tensor,
    bg_mask: torch.Tensor,
) -> float:
    """
    CNR = |mean(ROI) - mean(background)| / std(background)

    Parameters
    ----------
    image    : (B, 1, H, W) single image (B=1 expected)
    roi_mask : (H, W) binary mask for the region of interest
    bg_mask  : (H, W) binary mask for background noise region
    """
    img = image[0, 0]
    roi_vals = img[roi_mask.bool()]
    bg_vals = img[bg_mask.bool()]
    if roi_vals.numel() == 0 or bg_vals.numel() == 0:
        return float("nan")
    return abs(roi_vals.mean().item() - bg_vals.mean().item()) / (bg_vals.std().item() + 1e-8)


def edge_preservation_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: float = 1.5,
) -> float:
    """
    EPI: SSIM between edge maps of predicted and target.

    Sobel filter extracts edges; SSIM then measures structural
    similarity specifically at diagnostic boundaries.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    def edges(x):
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return (gx.pow(2) + gy.pow(2)).sqrt()

    return ssim(edges(pred), edges(target))


def noise_power_spectrum_1d(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 32,
) -> np.ndarray:
    """
    1D radial NPS of the residual (pred - target).
    Returns mean power per radial frequency bin (shape: n_bins).
    """
    residual = (pred - target).squeeze()  # (H, W)
    H, W = residual.shape
    freq_map = torch.fft.fftshift(torch.fft.fft2(residual))
    power = freq_map.abs().pow(2)

    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = ((yy - cy).float().pow(2) + (xx - cx).float().pow(2)).sqrt()

    r_max = r.max().item()
    bins = np.linspace(0, r_max, n_bins + 1)
    nps = np.zeros(n_bins)
    r_np = r.cpu().numpy()
    p_np = power.cpu().numpy()
    for i in range(n_bins):
        mask = (r_np >= bins[i]) & (r_np < bins[i + 1])
        if mask.sum() > 0:
            nps[i] = p_np[mask].mean()
    return nps


# ── Calibration ────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    levels: list[float] | None = None,
) -> dict[str, float]:
    """
    Regression ECE: fraction of true values falling within nominal intervals.

    For a perfectly calibrated model, a 95% credible interval should
    contain 95% of true pixel values. This function checks multiple
    nominal coverage levels and returns the ECE (mean deviation) and
    a reliability dict.

    Parameters
    ----------
    y_true : (B, 1, H, W) clean ground truth
    mu     : (B, 1, H, W) predicted mean
    sigma  : (B, 1, H, W) predicted std (sqrt of aleatoric variance)
    levels : nominal coverage levels to check (default: 0.5, 0.8, 0.9, 0.95)
    """
    if levels is None:
        levels = [0.5, 0.8, 0.9, 0.95]

    results: dict[str, float] = {}
    ece_values = []

    for p in levels:
        z = torch.erfinv(torch.tensor(p)) * math.sqrt(2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        covered = ((y_true >= lower) & (y_true <= upper)).float().mean().item()
        results[f"coverage_{int(p*100)}"] = covered
        ece_values.append(abs(covered - p))

    results["ece"] = float(np.mean(ece_values))
    results["sharpness"] = (2 * 1.96 * sigma).mean().item()  # mean 95% interval width
    return results


# ── Aggregate eval ─────────────────────────────────────────────────────────────

def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_sig2a: torch.Tensor | None = None,
    include_perceptual: bool = True,
) -> dict[str, float]:
    """
    Compute the full referenced metric set for one batch.
    Perceptual metrics (FSIM, LPIPS) require piq/lpips; gracefully skipped.
    """
    metrics: dict[str, float] = {}
    metrics["psnr"] = psnr(pred, target)
    metrics["ssim"] = ssim(pred, target)

    if include_perceptual:
        metrics["ms_ssim"] = ms_ssim(pred, target)
        metrics["fsim"] = fsim(pred, target)
        metrics["lpips"] = lpips(pred, target)

    if log_sig2a is not None:
        sigma = torch.exp(0.5 * log_sig2a.clamp(-10, 10))
        cal = expected_calibration_error(target, pred, sigma)
        metrics.update(cal)

    return metrics
