"""
Heteroscedastic Negative Log-Likelihood loss.

Kendall & Gal (2017, NeurIPS) showed that for regression tasks, the
optimal Bayesian loss is the heteroscedastic NLL, which simultaneously
learns a point estimate (mu) and a pixel-wise uncertainty (sigma^2):

    L_NLL = (1/2) * [(y - mu)^2 * exp(-s) + s]

where s = log(sigma^2) is the log aleatoric variance predicted by the
network's second output head. This form is numerically stable because
we regress on log-variance rather than variance directly.

The key advantage over L2: when sigma^2 is high (the input pixel is
very noisy), the first term is down-weighted by exp(-s). The model
learns to "trust its uncertainty" and not try to force an exact
reconstruction in regions where the data cannot support one.

Compared to L2 (MSE):
    L_MSE = (y - mu)^2

    L_NLL = (y - mu)^2 * exp(-s) + s

The NLL adds two things:
1. Automatic weighting of per-pixel reconstruction terms by uncertainty
2. A regularisation term (s) that prevents the model from predicting
   infinite uncertainty everywhere (which would zero out the first term)

Reference
---------
Kendall, A. & Gal, Y. (2017). What uncertainties do we need in Bayesian
    deep learning for computer vision? NeurIPS 2017.
"""

import torch
import torch.nn as nn


class HeteroscedasticNLL(nn.Module):
    """
    Pixel-averaged heteroscedastic NLL.

    Parameters
    ----------
    reduction : 'mean' (default) averages over all pixels and batch;
                'sum' for total loss (useful for monitoring).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def forward(
        self,
        y_true: torch.Tensor,
        mu: torch.Tensor,
        log_sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_true     : (B, 1, H, W) clean target
        mu         : (B, 1, H, W) predicted mean (reconstruction)
        log_sigma2 : (B, 1, H, W) predicted log aleatoric variance
        """
        # Cast to float32: exp(-log_sigma2) overflows float16 under AMP
        y_true = y_true.float()
        mu = mu.float()
        log_sigma2 = log_sigma2.float().clamp(-10.0, 10.0)
        loss = 0.5 * ((y_true - mu).pow(2) * torch.exp(-log_sigma2) + log_sigma2)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
