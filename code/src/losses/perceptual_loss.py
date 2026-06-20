"""
Perceptual Loss via frozen VGG-16 feature maps (Johnson et al. 2016).

    L_Perc = Σ_l mean ||φ_l(x̂) - φ_l(x)||²

Layers: relu1_2, relu2_2, relu3_3  (shallow to mid).
Single greyscale channel is broadcast to 3 channels before VGG.
VGG weights are frozen; no fine-tuning.

References
----------
Johnson, J. et al. (2016). Perceptual losses for real-time style transfer.
    ECCV 2016. https://doi.org/10.1007/978-3-319-46475-6_43
Simonyan, K. & Zisserman, A. (2015). VGG. ICLR 2015.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

_LAYER_IDX = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15}


class PerceptualLoss(nn.Module):
    """
    Parameters
    ----------
    layers : list of str, default ['relu1_2', 'relu2_2', 'relu3_3']
    device : str
    """

    def __init__(
        self,
        layers: list[str] | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.layers = layers or ["relu1_2", "relu2_2", "relu3_3"]
        max_idx = max(_LAYER_IDX[l] for l in self.layers)
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*list(vgg.features.children())[: max_idx + 1]).to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self._idxs = {l: _LAYER_IDX[l] for l in self.layers}
        self._max = max_idx

    def _features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        feats: dict[str, torch.Tensor] = {}
        h = x
        for i, layer in enumerate(self.net):
            h = layer(h)
            for name, idx in self._idxs.items():
                if i == idx:
                    feats[name] = h
        return feats

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        f_hat = self._features(x_hat)
        f_x = self._features(x)
        loss = torch.tensor(0.0, device=x.device)
        for name in self.layers:
            loss = loss + torch.mean((f_hat[name] - f_x[name]) ** 2)
        return loss
