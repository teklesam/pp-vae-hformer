#!/usr/bin/env python
"""
train_dncnn.py — DnCNN baseline trained on the same Kermany Poisson-Gaussian setup.

Reference: Zhang et al. (2017) "Beyond a Gaussian Denoiser: Residual Learning of Deep
CNN for Image Denoising." IEEE TIP 26(7):3142-3155.

Architecture: 17-layer feed-forward CNN, residual (predict noise, subtract from input).
Loss: MSE on the residual (identical to the original paper — fair comparison).
This provides an apples-to-apples baseline: same data, same noise model, same metrics,
only the architecture and loss differ from Arms A-E.

Usage:
    python scripts/train_dncnn.py --data_dir /home/stm43/chest_xray
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.data.kermany_dataset import make_loaders
from src.evaluation.metrics import psnr, ssim

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── DnCNN Architecture ────────────────────────────────────────────────────────

class DnCNN(nn.Module):
    """
    Zhang et al. (2017) DnCNN-B: blind denoiser, depth=20, 64 channels.
    Predicts the noise residual; clean = noisy - predicted_noise.
    Uses BN + ReLU in hidden layers (no BN in first/last layers).
    """

    def __init__(self, depth: int = 20, channels: int = 64, in_ch: int = 1):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(channels, in_ch, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.body(x)   # residual subtraction: output = clean estimate


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--epochs",     type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--noise_level",type=str, default="random",
                    choices=["low", "mid", "high", "random"])
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--num_workers",type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

RUN_NAME = "dncnn_baseline"
out_dir  = Path(args.output_dir) / RUN_NAME
ckpt_dir = out_dir / "checkpoints"
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(exist_ok=True)
log_path = out_dir / "train_log.csv"

# ── Data ──────────────────────────────────────────────────────────────────────

loaders = make_loaders(
    root_dir=args.data_dir,
    batch_size=args.batch_size,
    target_size=args.image_size,
    noise_level=args.noise_level,
    num_workers=args.num_workers,
)
print(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)} | Test: {len(loaders['test'].dataset)}")

# ── Model ─────────────────────────────────────────────────────────────────────

model = DnCNN(depth=20, channels=64, in_ch=1).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"DnCNN parameters: {n_params:,}")

loss_fn  = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
# DnCNN paper uses MultiStepLR: halve at epoch 30 and 60
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimiser, milestones=[50, 100, 150], gamma=0.5
)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# ── Log header ────────────────────────────────────────────────────────────────

with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "lr", "time_s"])

# ── Training loop ─────────────────────────────────────────────────────────────

best_val_psnr = 0.0

for epoch in range(args.epochs):
    t0 = time.time()
    model.train()
    epoch_loss = []

    for noisy, clean, _ in loaders["train"]:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        optimiser.zero_grad()
        with autocast(enabled=(DEVICE.type == "cuda")):
            pred = model(noisy)
            loss = loss_fn(pred, clean)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser)
        scaler.update()
        epoch_loss.append(loss.item())

    scheduler.step()
    train_loss = sum(epoch_loss) / len(epoch_loss)

    # Validation every 5 epochs
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        model.eval()
        val_psnr_list, val_ssim_list = [], []
        with torch.no_grad():
            for noisy, clean, _ in loaders["val"]:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                pred = model(noisy).clamp(0, 1)
                val_psnr_list.append(psnr(pred, clean))
                val_ssim_list.append(ssim(pred, clean))
        val_psnr_mean = sum(val_psnr_list) / len(val_psnr_list)
        val_ssim_mean = sum(val_ssim_list) / len(val_ssim_list)
    else:
        val_psnr_mean = float("nan")
        val_ssim_mean = float("nan")

    elapsed = time.time() - t0
    lr_now  = optimiser.param_groups[0]["lr"]

    if val_psnr_mean > best_val_psnr:
        best_val_psnr = val_psnr_mean
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "best_val_psnr": best_val_psnr},
                   out_dir / "best_model.pth")

    if epoch % 10 == 0:
        torch.save({"epoch": epoch, "model": model.state_dict()},
                   ckpt_dir / f"epoch_{epoch:04d}.pth")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch, f"{train_loss:.6f}",
            f"{val_psnr_mean:.3f}", f"{val_ssim_mean:.4f}",
            f"{lr_now:.2e}", f"{elapsed:.1f}",
        ])

    if epoch % 10 == 0:
        print(f"  Ep {epoch:4d}/{args.epochs} | loss={train_loss:.6f} "
              f"| val_psnr={val_psnr_mean:.3f} | lr={lr_now:.1e} | {elapsed:.1f}s")
        sys.stdout.flush()

print(f"\nDnCNN done. Best val PSNR: {best_val_psnr:.3f} dB")
