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
        # Cast to float32: z_logvar.exp() and z_mu.pow(2) can overflow float16
        # if z_mu grows large under long KL-driven gradient updates.
        z_mu = z_mu.float()
        z_logvar = z_logvar.float()
        return -0.5 * torch.mean(1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp())
