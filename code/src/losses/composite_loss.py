"""
Composite PP-VAE-Hformer training objective — expanded ablation v2.

Arm string format ('+'-separated terms):
    Reconstruction base (exactly one):
        l2          — MSE
        l1          — L1 / MAE
        charb       — Charbonnier (smooth-L1 approx: sqrt(r^2 + eps^2), eps=1e-3)
        nll         — heteroscedastic NLL (Kendall & Gal 2017)
    If both 'l1' and 'nll' appear  → L1+NLL hybrid

    Spatial/frequency regularisers (any combination):
        ssim        — 1 − MS-SSIM (Zhao 2017)
        ffl         — Focal Frequency Loss (Jiang 2021)
        edge        — Sobel edge L1
        perc        — VGG-16 perceptual (Johnson 2016)

    VAE regulariser (at most one KL term):
        kl          — KL divergence, linear beta annealing
        kl_cyc      — KL with cyclical annealing (beta set externally)
        kl_fb       — KL with free-bits constraint
        kl_cyc_fb   — cyclical annealing + free-bits

KL beta is always updated externally via  loss_fn.beta = new_beta  each epoch.
The annealing schedule (linear / cyclical) is implemented in train_proposed.py.
Free-bits applies inside the KLDivergenceFreeBits module.

Original 5 arms (backward-compatible):
    arm_a_l2           → "l2"
    arm_b_nll          → "nll"
    arm_c_nll_ssim     → "nll+ssim"
    arm_d_nll_ssim_ffl → "nll+ssim+ffl"
    arm_e_ppvae        → "nll+ssim+ffl+kl"

New arms F–O:
    arm_f_kl_cyc       → "nll+ssim+ffl+kl_cyc"
    arm_g_kl_fb        → "nll+ssim+ffl+kl_fb"
    arm_h_kl_cyc_fb    → "nll+ssim+ffl+kl_cyc_fb"
    arm_i_l1           → "l1"
    arm_j_l1_ssim_ffl  → "l1+ssim+ffl"
    arm_k_nll_l1       → "nll+l1"
    arm_l_nll_edge_ffl → "nll+edge+ffl"
    arm_m_full_det     → "nll+ssim+edge+ffl"
    arm_n_perc         → "nll+perc+ssim+ffl"
    arm_o_prelu        → "nll+ssim+ffl+kl_cyc_fb"  (same loss; PReLU set in model config)

Arm Q (Charbonnier):
    arm_q_charb        → "charb"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nll_loss import HeteroscedasticNLL
from .ms_ssim_loss import MSSSIMLoss
from .focal_frequency import FocalFrequencyLoss
from .kl_loss import KLDivergence, KLDivergenceFreeBits
from .edge_loss import EdgePreservationLoss
from .perceptual_loss import PerceptualLoss


def _parse(arm: str) -> set[str]:
    """Return the set of '+'-separated tokens in the arm string."""
    return set(arm.lower().split("+"))


class PPVAEHformerLoss(nn.Module):
    """
    Composite loss for any PP-VAE-Hformer ablation arm.

    Parameters
    ----------
    arm          : arm descriptor string (see module docstring)
    beta         : KL weight (updated externally each epoch via .beta =)
    lambda_ssim  : MS-SSIM weight
    lambda_ffl   : FFL weight
    lambda_edge  : Edge loss weight
    lambda_perc  : Perceptual loss weight
    lambda_l1    : L1 auxiliary weight when using L1+NLL hybrid
    lambda_fb    : Free-bits threshold per latent channel (nats)
    device       : for PerceptualLoss VGG weights
    """

    def __init__(
        self,
        arm: str = "nll+ssim+ffl+kl",
        beta: float = 0.001,
        lambda_ssim: float = 0.5,
        lambda_ffl: float = 0.1,
        lambda_edge: float = 0.1,
        lambda_perc: float = 0.05,
        lambda_l1: float = 0.5,
        lambda_fb: float = 0.25,
        device: str = "cpu",
    ):
        super().__init__()
        self.arm = arm
        self.beta = beta
        self.lambda_ssim = lambda_ssim
        self.lambda_ffl = lambda_ffl
        self.lambda_edge = lambda_edge
        self.lambda_perc = lambda_perc
        self.lambda_l1 = lambda_l1

        tokens = _parse(arm)

        # Reconstruction base
        self._use_l2 = "l2" in tokens
        self._use_nll = "nll" in tokens
        self._use_l1 = "l1" in tokens
        self._use_charb = "charb" in tokens
        self._charb_eps = 1e-3

        # Spatial / frequency terms
        self._use_ssim = "ssim" in tokens
        self._use_ffl = "ffl" in tokens
        self._use_edge = "edge" in tokens
        self._use_perc = "perc" in tokens

        # KL variants
        self._use_kl = any(t.startswith("kl") for t in tokens)
        self._use_fb = "kl_fb" in tokens or "kl_cyc_fb" in tokens

        # Modules
        self.nll = HeteroscedasticNLL()
        self.ms_ssim = MSSSIMLoss() if self._use_ssim else None
        self.ffl = FocalFrequencyLoss(alpha=1.0) if self._use_ffl else None
        self.edge = EdgePreservationLoss() if self._use_edge else None
        self.perc = PerceptualLoss(device=device) if self._use_perc else None

        if self._use_kl:
            if self._use_fb:
                self.kl_fn = KLDivergenceFreeBits(lambda_fb=lambda_fb)
            else:
                self.kl_fn = KLDivergence()
        else:
            self.kl_fn = None

    def forward(
        self,
        y_true: torch.Tensor,
        mu_rec: torch.Tensor,
        log_sig2a: torch.Tensor,
        z_mu: torch.Tensor | None = None,
        z_logvar: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        y_true    : (B, 1, H, W) clean target
        mu_rec    : (B, 1, H, W) reconstructed mean
        log_sig2a : (B, 1, H, W) log aleatoric variance
        z_mu      : (B, C, h, w) VAE latent mean  (None for det. arms)
        z_logvar  : (B, C, h, w) VAE latent logvar (None for det. arms)

        Returns
        -------
        dict: 'loss' (total scalar) + individual component scalars.
        """
        losses: dict[str, torch.Tensor] = {}
        zero = torch.tensor(0.0, device=mu_rec.device)

        # ── Reconstruction ────────────────────────────────────────────────
        if self._use_nll:
            losses["l_rec"] = self.nll(y_true, mu_rec, log_sig2a)
        elif self._use_l2:
            losses["l_rec"] = F.mse_loss(mu_rec, y_true)
        elif self._use_charb:
            # Charbonnier: mean( sqrt((pred - target)^2 + eps^2) ) - eps
            # Subtract eps so that loss = 0 when pred == target (exact).
            diff = mu_rec - y_true
            losses["l_rec"] = (torch.sqrt(diff * diff + self._charb_eps ** 2) - self._charb_eps).mean()
        elif self._use_l1:
            losses["l_rec"] = F.l1_loss(mu_rec, y_true)
        else:
            losses["l_rec"] = F.mse_loss(mu_rec, y_true)  # safe default

        # L1 auxiliary when hybrid (nll+l1)
        if self._use_nll and self._use_l1:
            losses["l_l1"] = self.lambda_l1 * F.l1_loss(mu_rec, y_true)
            losses["l_rec"] = losses["l_rec"] + losses["l_l1"]
        else:
            losses["l_l1"] = zero

        # ── KL ────────────────────────────────────────────────────────────
        if self._use_kl:
            losses["l_kl"] = self.beta * self.kl_fn(z_mu, z_logvar)
        else:
            losses["l_kl"] = zero

        # ── Spatial / frequency ───────────────────────────────────────────
        losses["l_ssim"] = (
            self.lambda_ssim * self.ms_ssim(mu_rec, y_true)
            if self._use_ssim else zero
        )
        losses["l_ffl"] = (
            self.lambda_ffl * self.ffl(mu_rec, y_true)
            if self._use_ffl else zero
        )
        losses["l_edge"] = (
            self.lambda_edge * self.edge(mu_rec, y_true)
            if self._use_edge else zero
        )
        losses["l_perc"] = (
            self.lambda_perc * self.perc(mu_rec, y_true)
            if self._use_perc else zero
        )

        losses["loss"] = (
            losses["l_rec"]
            + losses["l_kl"]
            + losses["l_ssim"]
            + losses["l_ffl"]
            + losses["l_edge"]
            + losses["l_perc"]
        )
        return losses

    # Convenience alias used by training loop
    @property
    def _use_nll_flag(self) -> bool:
        return self._use_nll
