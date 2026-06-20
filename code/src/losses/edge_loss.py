"""
Edge-Preservation Loss via Sobel spatial gradients.

    L_Edge = mean |∇x̂ - ∇x|₁

where ∇ denotes Sobel gradient (Gx, Gy stacked along dim=1).
Uses kornia.filters.spatial_gradient when available; falls back to
finite-difference when not.

Ported from ppvae_project_9_6 and adapted for 256×256 CXR inputs.

References
----------
Rudin, L. I. et al. (1992). Nonlinear total variation based noise removal.
    Physica D, 60(1-4), 259-268. https://doi.org/10.1016/0167-2789(92)90242-F
Riba, E. et al. (2020). Kornia. WACV 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import kornia.filters as KF
    _KORNIA = True
except ImportError:
    _KORNIA = False


def _sobel(x: torch.Tensor) -> torch.Tensor:
    """Return Sobel [Gx, Gy] stacked along dim=1 → (B, 2, H, W)."""
    if _KORNIA:
        g = KF.spatial_gradient(x, mode="sobel", order=1)  # (B,1,2,H,W)
        return g[:, 0, :, :, :]                             # (B,2,H,W)
    # Finite-difference fallback
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F.pad(gx, (0, 1))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.cat([gx, gy], dim=1)


class EdgePreservationLoss(nn.Module):
    """L1 loss on Sobel gradient maps."""

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(_sobel(x_hat) - _sobel(x)))
