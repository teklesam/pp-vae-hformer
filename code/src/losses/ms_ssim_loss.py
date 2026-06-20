"""
Multi-Scale SSIM loss.

Zhao et al. (2017) demonstrated that MS-SSIM+L1 outperforms L2 for
image restoration. We use 1 - MS-SSIM as a perceptual loss term,
combined with the NLL primary objective.

MS-SSIM operates at 5 spatial scales by progressive 2x downsampling.
The scale weights (Simoncelli et al. 2003 Table I) are:
    beta = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

Reference
---------
Wang, Z., Simoncelli, E.P., & Bovik, A.C. (2003). Multi-scale structural
    similarity for image quality assessment. ACSSC 2003.

Zhao, H. et al. (2017). Loss functions for image restoration with neural
    networks. IEEE TCI, 3(1), 47-57.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).view(1, 1, size, size)


def _ssim_components(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (luminance*contrast, structural) SSIM components."""
    pad = kernel.shape[-1] // 2
    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    # Clamp to non-negative: floating-point subtraction can yield tiny negatives
    sig_x2 = (F.conv2d(x.pow(2), kernel, padding=pad) - mu_x2).clamp(min=0.0)
    sig_y2 = (F.conv2d(y.pow(2), kernel, padding=pad) - mu_y2).clamp(min=0.0)
    sig_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy
    cs = (2 * sig_xy + C2) / (sig_x2 + sig_y2 + C2)
    lc = (2 * mu_xy + C1) * (2 * sig_xy + C2) / ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return lc.mean(), cs.mean()


class MSSSIMLoss(nn.Module):
    """
    1 - MS-SSIM loss term.

    Expects inputs in [0, 1], single channel (greyscale).
    """

    WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def __init__(self, num_scales: int = 5, kernel_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.num_scales = num_scales
        kernel = _gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel", kernel)

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_hat : (B, 1, H, W) reconstruction (will be clamped to [0,1])
        x     : (B, 1, H, W) clean target
        """
        # Clamp predictions to valid image range -- untrained networks can
        # produce arbitrary values; cs^(fractional power) is only real for cs>0.
        x_hat = x_hat.clamp(0.0, 1.0)
        x = x.clamp(0.0, 1.0)

        # Ensure enough resolution for all scales
        min_size = 2 ** self.num_scales
        if x.shape[-1] < min_size or x.shape[-2] < min_size:
            lc, _ = _ssim_components(x_hat, x, self.kernel)
            return 1.0 - lc.clamp(0.0, 1.0)

        cs_values: list[torch.Tensor] = []
        lc_final = torch.tensor(0.0, device=x.device)

        xp, xhp = x, x_hat
        for j in range(self.num_scales):
            lc, cs = _ssim_components(xhp, xp, self.kernel)
            # Clamp cs to (0, 1] -- values <= 0 indicate anti-correlated
            # signals and should be treated as the worst case (cs=0 floor),
            # but 0^fractional = 0 which zero-kills the product; use eps floor.
            cs = cs.clamp(1e-8, 1.0)
            if j == self.num_scales - 1:
                lc_final = lc.clamp(0.0, 1.0)
            else:
                cs_values.append(cs)
            if j < self.num_scales - 1:
                xp = F.avg_pool2d(xp, 2)
                xhp = F.avg_pool2d(xhp, 2)

        ms_ssim = lc_final ** self.WEIGHTS[-1]
        for j, cs in enumerate(cs_values):
            ms_ssim = ms_ssim * (cs ** self.WEIGHTS[j])

        return 1.0 - ms_ssim.nan_to_num(nan=1.0)
