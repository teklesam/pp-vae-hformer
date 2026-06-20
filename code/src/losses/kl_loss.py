"""
KL divergence loss for the VAE bottleneck.

KL(q(z|x) || p(z)) where q(z|x) = N(z_mu, exp(z_logvar)) and p(z) = N(0, I).

Closed form (Kingma & Welling 2014, Appendix B):
    L_KL = -0.5 * mean(1 + z_logvar - z_mu^2 - exp(z_logvar))

This is averaged over all latent dimensions and batch.

KL Annealing:
    We apply a linear KL warmup schedule (beta annealing) to prevent
    posterior collapse, where the decoder learns to ignore the latent
    code and the KL term collapses to zero. Starting with beta=0 and
    gradually increasing it to beta_target over warmup_epochs allows
    the reconstruction objective to dominate early training.

Reference
---------
Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes.
    ICLR 2014.
"""

import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    def forward(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_mu     : (B, C, h, w) VAE latent mean
        z_logvar : (B, C, h, w) VAE latent log-variance

        Returns
        -------
        Scalar KL divergence, averaged over batch and spatial dimensions.
        """
        z_mu = z_mu.float()
        z_logvar = z_logvar.float()
        return -0.5 * torch.mean(1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp())


class KLDivergenceFreeBits(nn.Module):
    """
    Free-bits KL divergence (Kingma et al. 2016).

    Enforces a minimum information content per latent channel:

        L_KL_FB = mean_C  max( KL_c, lambda_fb )

    where KL_c = -0.5 * mean_{B,H,W}(1 + logvar_c - mu_c² - exp(logvar_c))
    is the per-channel KL averaged over batch and spatial dimensions.

    Setting lambda_fb > 0 prevents the encoder from collapsing any latent
    channel below the threshold while leaving the gradient signal intact for
    channels that are actively used (KL_c > lambda_fb).

    Parameters
    ----------
    lambda_fb : float
        Minimum KL per channel in nats (0.1–0.5 typical; default 0.25).

    References
    ----------
    Kingma, D. P. et al. (2016). Improving Variational Inference with
        Inverse Autoregressive Flow. NeurIPS 2016.
    """

    def __init__(self, lambda_fb: float = 0.25):
        super().__init__()
        self.lambda_fb = lambda_fb

    def forward(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        z_mu = z_mu.float()
        z_logvar = z_logvar.float()
        # Per-element KL: (B, C, H, W)
        kl_elem = -0.5 * (1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        # Average over batch and spatial dims → (C,)
        kl_per_channel = kl_elem.mean(dim=[0, 2, 3])
        # Clamp: free bits ensures each channel ≥ lambda_fb
        kl_clamped = torch.clamp(kl_per_channel, min=self.lambda_fb)
        return kl_clamped.mean()
