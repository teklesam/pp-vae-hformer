#!/usr/bin/env python
"""
train_baseline.py — Train comparison architectures on Kermany CXR with Foi noise.

All baselines are trained with:
  - Same Kermany dataset + Foi Poisson-Gaussian noise as the ablation arms
  - MSE (L2) loss — fair architectural comparison; loss improvements are our contribution
  - 200 epochs, cosine LR schedule, AdamW
  - Multi-GPU DataParallel if available

Supported architectures (--arch):
  dncnn     DnCNN-17 (Zhang et al. 2017)
  dncnn_b   DnCNN-B blind (Zhang et al. 2017)
  ffdnet    FFDNet with Foi sigma map (Zhang et al. 2018)
  ircnn     IRCNN (Zhang et al. 2017)
  drunet    DRUNet with Foi sigma conditioning (Zhang et al. 2021)
  swinir    SwinIR grayscale denoising (Liang et al. 2021)

Usage:
  python scripts/train_baseline.py --arch drunet --data_dir /path/to/chest_xray
  python scripts/train_baseline.py --arch swinir --data_dir /path/to/chest_xray --lr 2e-4
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAIR_ROOT = PROJECT_ROOT.parent / "KAIR"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(KAIR_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.data.kermany_dataset import make_loaders
from src.data.noise_simulation import PRESETS
from src.evaluation.metrics import psnr, ssim

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Architecture defaults ─────────────────────────────────────────────────────

ARCH_DEFAULTS = {
    "dncnn":   {"lr": 1e-3, "nb": 17, "nc": 64},
    "dncnn_b": {"lr": 1e-3, "nb": 20, "nc": 64},
    "ffdnet":  {"lr": 1e-3, "nb": 15, "nc": 64},
    "ircnn":   {"lr": 1e-3, "nb": 7,  "nc": 64},
    "drunet":  {"lr": 5e-5, "nc": [64, 128, 256, 512], "nb": 4},
    "swinir":  {"lr": 2e-4, "embed_dim": 180, "depths": [6,6,6,6,6,6],
                "num_heads": [6,6,6,6,6,6], "window_size": 8},
}

# Foi mid noise effective sigma (used for sigma-conditioned models at inference)
FOI_SIGMA_MID = 0.141  # sqrt(0.03*0.5 + 0.005)

# ── Model builders ────────────────────────────────────────────────────────────

def build_model(arch: str) -> nn.Module:
    d = ARCH_DEFAULTS[arch]

    if arch in ("dncnn", "dncnn_b"):
        from models.network_dncnn import DnCNN
        return DnCNN(in_nc=1, out_nc=1, nc=d["nc"], nb=d["nb"], act_mode="BR")

    elif arch == "ffdnet":
        from models.network_ffdnet import FFDNet
        return FFDNet(in_nc=1, out_nc=1, nc=d["nc"], nb=d["nb"], act_mode="BR")

    elif arch == "ircnn":
        from models.network_dncnn import IRCNN
        return IRCNN(in_nc=1, out_nc=1, nc=d["nc"])

    elif arch == "drunet":
        from models.network_unet import UNetRes
        return UNetRes(
            in_nc=2, out_nc=1, nc=d["nc"], nb=d["nb"],
            act_mode="L",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        )

    elif arch == "swinir":
        from models.network_swinir import SwinIR
        d = ARCH_DEFAULTS["swinir"]
        return SwinIR(
            upscale=1, in_chans=1, img_size=128,
            window_size=d["window_size"],
            img_range=1.0,
            depths=d["depths"],
            embed_dim=d["embed_dim"],
            num_heads=d["num_heads"],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )

    else:
        raise ValueError(f"Unknown arch: {arch}")


def model_forward(arch: str, model: nn.Module, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Run forward pass — handles sigma-conditioned models."""
    if arch in ("dncnn", "dncnn_b", "ircnn"):
        return model(noisy)

    elif arch == "ffdnet":
        # Foi sigma: sqrt(a*clean + b) — use clean signal for training (known)
        a, b = PRESETS["mid"]["a"], PRESETS["mid"]["b"]
        sigma = (a * clean.clamp(min=0) + b).sqrt()
        # FFDNet expects (B, 1, 1, 1) uniform sigma per image; use batch mean
        sigma_scalar = sigma.mean(dim=[1, 2, 3], keepdim=True)
        return model(noisy, sigma_scalar)

    elif arch == "drunet":
        # DRUNet: 2-channel input [noisy | sigma_map].
        # Sigma estimated from noisy (not clean) so training and inference are consistent —
        # using clean here leaks per-pixel ground truth via the sigma channel, causing
        # the model to memorise clean from sigma_map instead of learning to denoise.
        a, b = PRESETS["mid"]["a"], PRESETS["mid"]["b"]
        sigma_map = (a * noisy.clamp(min=0) + b).sqrt()
        x_in = torch.cat([noisy, sigma_map], dim=1)
        return model(x_in)

    elif arch == "swinir":
        # Pad to window_size multiple
        ws = ARCH_DEFAULTS["swinir"]["window_size"]
        _, _, h, w = noisy.shape
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(x)
        return out[:, :, :h, :w]

    else:
        raise ValueError(f"Unknown arch: {arch}")


def model_infer(arch: str, model: nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    """Inference-only forward (no clean reference) — uses fixed sigma."""
    if arch in ("dncnn", "dncnn_b", "ircnn"):
        return model(noisy).clamp(0, 1)

    elif arch == "ffdnet":
        sigma = torch.full((noisy.shape[0], 1, 1, 1), FOI_SIGMA_MID,
                           dtype=noisy.dtype, device=noisy.device)
        return model(noisy, sigma).clamp(0, 1)

    elif arch == "drunet":
        sigma_map = torch.full_like(noisy, FOI_SIGMA_MID)
        x_in = torch.cat([noisy, sigma_map], dim=1)
        return model(x_in).clamp(0, 1)

    elif arch == "swinir":
        ws = ARCH_DEFAULTS["swinir"]["window_size"]
        _, _, h, w = noisy.shape
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(x)
        return out[:, :, :h, :w].clamp(0, 1)

    else:
        raise ValueError(f"Unknown arch: {arch}")


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=True, choices=list(ARCH_DEFAULTS.keys()))
parser.add_argument("--data_dir", required=True)
parser.add_argument("--output_dir", default="./results")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--noise_level", default="random",
                    choices=["low", "mid", "high", "random"])
parser.add_argument("--lr", type=float, default=None,
                    help="Override default LR for architecture")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_gpus", type=int, default=0,
                    help="GPUs for DataParallel (0 = auto)")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


# ── Setup ─────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if args.num_gpus == 0 else min(args.num_gpus, torch.cuda.device_count())
N_GPUS = max(1, N_GPUS)
# DRUNet is deep enough that AMP FP16 underflows cause output collapse — use FP32
USE_AMP = (DEVICE.type == "cuda") and (args.arch != "drunet")

effective_batch = args.batch_size * N_GPUS
effective_lr = (args.lr or ARCH_DEFAULTS[args.arch]["lr"]) * N_GPUS

out_dir = Path(args.output_dir) / args.arch
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "train_log.csv"
ckpt_path = out_dir / "checkpoint.pth"
best_path = out_dir / "best_model.pth"

LOG_COLS = ["epoch", "loss", "val_psnr", "val_ssim", "lr", "time_s"]


def run() -> None:
    print(f"\n{'='*60}")
    print(f"  Baseline: {args.arch.upper()}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {effective_batch}  LR: {effective_lr:.2e}")
    print(f"  GPUs    : {N_GPUS}")
    print(f"  Out     : {out_dir}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = make_loaders(
        root_dir=args.data_dir,
        batch_size=effective_batch,
        target_size=args.image_size,
        noise_level=args.noise_level,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    _base_model = build_model(args.arch).to(DEVICE)
    n_params = sum(p.numel() for p in _base_model.parameters())
    print(f"Parameters: {n_params:,}")

    if N_GPUS > 1:
        model = nn.DataParallel(_base_model, device_ids=list(range(N_GPUS)))
    else:
        model = _base_model

    # ── Optimiser + LR ────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        _base_model.parameters(), lr=effective_lr, weight_decay=1e-4
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    start_epoch = 0
    best_psnr = 0.0

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        _base_model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimizer"])
        lr_sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # ── Log file ──────────────────────────────────────────────────────────────
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLS)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        _base_model.train()
        ep_loss = 0.0
        t0 = time.time()

        for noisy, clean, _ in loaders["train"]:
            noisy = noisy.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=USE_AMP):
                pred = model_forward(args.arch, model, noisy, clean)
                loss = F.mse_loss(pred, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(_base_model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            ep_loss += loss.item()

        ep_loss /= max(len(loaders["train"]), 1)
        lr_sched.step()

        # ── Validation every 10 epochs ─────────────────────────────────────
        val_psnr, val_ssim_val = 0.0, 0.0
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            _base_model.eval()
            p_vals, s_vals = [], []
            with torch.no_grad():
                for noisy, clean, _ in loaders["val"]:
                    noisy = noisy.to(DEVICE, non_blocking=True)
                    clean = clean.to(DEVICE, non_blocking=True)
                    pred = model_infer(args.arch, _base_model, noisy)
                    for i in range(noisy.shape[0]):
                        p_vals.append(psnr(pred[i:i+1], clean[i:i+1]))
                        s_vals.append(ssim(pred[i:i+1], clean[i:i+1]))
            val_psnr = sum(p_vals) / len(p_vals)
            val_ssim_val = sum(s_vals) / len(s_vals)

            elapsed = time.time() - t0
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"  Ep {epoch:>3}/{args.epochs} | loss={ep_loss:.6f} | "
                  f"val_psnr={val_psnr:.3f} | val_ssim={val_ssim_val:.4f} | "
                  f"lr={lr_now:.1e} | {elapsed:.1f}s")

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, round(ep_loss, 6),
                    round(val_psnr, 3), round(val_ssim_val, 4),
                    round(lr_now, 8), round(elapsed, 1)
                ])

            # ── Checkpoint ────────────────────────────────────────────────
            torch.save({
                "epoch": epoch,
                "model": _base_model.state_dict(),
                "optimizer": optimiser.state_dict(),
                "scheduler": lr_sched.state_dict(),
                "best_psnr": best_psnr,
            }, ckpt_path)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(_base_model.state_dict(), best_path)
                print(f"  *** New best: {best_psnr:.3f} dB ***")

    print(f"\nTraining complete. Best val PSNR: {best_psnr:.3f} dB")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    run()
