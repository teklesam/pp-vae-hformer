"""
PP-VAE-Hformer: Pathology-Preserving VAE with Hformer-inspired backbone.

Architecture overview
---------------------
The model combines three distinct components:

1. Hformer-inspired hybrid encoder-decoder
   Based on Zhang et al. (2023) "Hformer" (hybrid CNN-Transformer for CT
   denoising). Each scale uses alternating CNN residual blocks (local feature
   extraction) and lightweight window attention blocks (non-local context).
   Skip connections preserve multi-scale anatomical detail.

2. VAE probabilistic bottleneck
   A spatial VAE at the bottleneck (following Kingma & Welling 2014)
   introduces a learnable prior over the compressed latent space.
   This is the source of epistemic uncertainty: K stochastic forward
   passes through the bottleneck yield K slightly different reconstructions
   from which we compute pixel-wise variance (sigma^2_epistemic).
   The KL divergence regularises the bottleneck to stay close to N(0,I).

3. Dual output head (aleatoric + reconstruction)
   Two parallel 1x1 conv heads on the final decoder feature map:
     mu_rec     (B, 1, H, W) -- the denoised image mean
     log_sig2_a (B, 1, H, W) -- log aleatoric variance (noise level map)

   Aleatoric uncertainty is the irreducible noise in the input image,
   modelled via the heteroscedastic NLL loss (Kendall & Gal 2017).
   It captures WHERE the input image is most noisy, grounded in the
   Foi signal-dependent noise model.

Ablation control
----------------
    use_vae=False  -> deterministic Hformer (Arm A, B, C, D)
    use_vae=True   -> full PP-VAE-Hformer with epistemic uncertainty (Arm E)

Three outputs are returned:
    mu_rec     -- the reconstructed denoised image (used for all arms)
    log_sig2_a -- log aleatoric variance (used in NLL loss, Arms B-E)
    z_mu       -- VAE latent mean (None for arms A-D)
    z_logvar   -- VAE latent logvar (None for arms A-D)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class WindowAttnBlock(nn.Module):
    """
    Lightweight local window self-attention (Swin-inspired).

    Using window_size=8 means each attention head sees 64 tokens, keeping
    the quadratic attention cost tractable even at 256x256 resolution.
    The learned position bias is omitted for simplicity; relative position
    information is captured by the surrounding CNN blocks.
    """

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.ws = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def _partition(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ws = self.ws
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape
        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)
        return x, (B, Hp, Wp, H, W, C)

    def _reverse(self, x: torch.Tensor, meta) -> torch.Tensor:
        B, Hp, Wp, H, W, C = meta
        ws = self.ws
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
        return x[:, :, :H, :W].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, meta = self._partition(x)
        t = self.norm1(tokens)
        attn_out, _ = self.attn(t, t, t)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return self._reverse(tokens, meta)


class HybridBlock(nn.Module):
    """One CNN ResBlock followed by one Transformer WindowAttnBlock."""

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.cnn = ResBlock(channels)
        self.attn = WindowAttnBlock(channels, num_heads, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.cnn(x))


class Downsample(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, 2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── VAE Bottleneck ─────────────────────────────────────────────────────────────

class VAEBottleneck(nn.Module):
    """
    Spatial VAE bottleneck operating on feature maps (not flattened vectors).

    Encodes: feature map -> (z_mu, z_logvar) via 1x1 convolutions
    Samples: z = z_mu + eps * exp(0.5 * z_logvar), eps ~ N(0, I)
    Decodes: z -> feature map via 1x1 convolution

    The spatial formulation preserves the spatial structure of the feature map,
    which is critical for image-to-image tasks (unlike the original VAE which
    maps to a global latent vector).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mu_proj = nn.Conv2d(channels, channels, 1)
        self.lv_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Disable autocast: Conv2d is whitelisted by AMP and would cast back to
        # float16 even after a manual .float() call. The reparametrisation and
        # the 1x1 projections need float32 to prevent NaN from accumulating over
        # ~100 epochs when KL gradients push z_mu to large magnitudes.
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float()
            z_mu = self.mu_proj(x)
            z_lv = self.lv_proj(x).clamp(-10.0, 4.0)
            if deterministic or not self.training:
                z = z_mu
            else:
                z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_lv)
            return self.out_proj(z), z_mu, z_lv


# ── Main model ─────────────────────────────────────────────────────────────────

class PPVAEHformer(nn.Module):
    """
    Full PP-VAE-Hformer model.

    Parameters
    ----------
    in_channels    : input image channels (1 for greyscale CXR)
    base_channels  : feature channels at the first encoder scale
    num_blocks     : HybridBlocks per encoder/decoder scale
    num_scales     : number of downsampling steps (spatial resolution halved each time)
    num_heads      : attention heads in WindowAttnBlock
    window_size    : attention window size (must divide spatial resolution at each scale)
    use_vae        : whether to include the VAE bottleneck (True = Arm E, False = Arms A-D)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 2,
        num_scales: int = 3,
        num_heads: int = 4,
        window_size: int = 8,
        use_vae: bool = True,
    ):
        super().__init__()
        self.use_vae = use_vae

        self.stem = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        skip_channels: list[int] = []
        for _ in range(num_scales):
            self.enc_blocks.append(
                nn.Sequential(*[HybridBlock(ch, num_heads, window_size) for _ in range(num_blocks)])
            )
            skip_channels.append(ch)
            self.downsamples.append(Downsample(ch, ch * 2))
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[HybridBlock(ch, num_heads, window_size) for _ in range(num_blocks)]
        )

        if use_vae:
            self.vae = VAEBottleneck(ch)

        # Decoder (mirror of encoder)
        self.upsamples = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for skip_ch in reversed(skip_channels):
            self.upsamples.append(Upsample(ch, ch))
            fused_ch = ch + skip_ch
            out_ch = ch // 2
            self.fuse_convs.append(nn.Conv2d(fused_ch, out_ch, 1, bias=False))
            self.dec_blocks.append(
                nn.Sequential(*[HybridBlock(out_ch, num_heads, window_size) for _ in range(num_blocks)])
            )
            ch = out_ch

        # Dual output head
        self.rec_head = nn.Conv2d(ch, in_channels, 3, 1, 1)   # mu_rec
        self.aleat_head = nn.Conv2d(ch, in_channels, 3, 1, 1) # log_sigma2_a
        nn.init.zeros_(self.aleat_head.bias)                   # log_sigma2≈0 at init → stable gradients

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Parameters
        ----------
        x            : (B, 1, H, W) noisy input
        deterministic: if True, use VAE mean without sampling (for evaluation)

        Returns
        -------
        mu_rec    : (B, 1, H, W) reconstructed denoised image
        log_sig2a : (B, 1, H, W) log aleatoric variance (noise level map)
        z_mu      : (B, C, h, w) VAE latent mean  -- None if use_vae=False
        z_logvar  : (B, C, h, w) VAE latent logvar -- None if use_vae=False
        """
        feat = self.stem(x)

        skips: list[torch.Tensor] = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.bottleneck(feat)

        z_mu, z_logvar = None, None
        if self.use_vae:
            feat, z_mu, z_logvar = self.vae(feat, deterministic=deterministic)

        for up, fuse, dec, skip in zip(
            self.upsamples, self.fuse_convs, self.dec_blocks, reversed(skips)
        ):
            feat = up(feat)
            # Handle odd spatial sizes from asymmetric padding
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = F.interpolate(feat, size=skip.shape[-2:], mode="nearest")
            feat = fuse(torch.cat([feat, skip], dim=1))
            feat = dec(feat)

        mu_rec = self.rec_head(feat)
        log_sig2a = self.aleat_head(feat)

        return mu_rec, log_sig2a, z_mu, z_logvar

    @torch.no_grad()
    def mc_epistemic_uncertainty(
        self, x: torch.Tensor, K: int = 20
    ) -> torch.Tensor:
        """
        Monte Carlo epistemic uncertainty via repeated VAE sampling.

        Forces training mode to activate stochastic VAE sampling even during
        inference. Returns pixel-wise variance across K reconstructions.
        This is NOT the aleatoric head -- it is a separate output that
        quantifies where the model's latent prior is contributing most to
        the reconstruction (high variance = reconstruction driven by prior,
        not by input data).

        Parameters
        ----------
        x : (B, 1, H, W) noisy input
        K : number of MC samples

        Returns
        -------
        sigma2_epistemic : (B, 1, H, W) pixel-wise epistemic variance
        """
        if not self.use_vae:
            return torch.zeros_like(x)
        was_training = self.training
        self.train()
        samples = torch.stack([self(x, deterministic=False)[0] for _ in range(K)], dim=0)
        self.train(was_training)
        return samples.var(dim=0)
