#!/usr/bin/env python
"""
evaluate_local.py — Run on your Mac to generate all figures and per-image metrics.

Loads best_model.pth from trained_models/, runs test-set evaluation at three
noise levels, saves qualitative figures and per-image CSV (with class labels)
for downstream statistical analysis.

Usage:
    python scripts/evaluate_local.py \
        --data_dir "/Users/sam/Documents/PPVAE Dissertation Project/chest_xray" \
        --models_dir "/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/trained_models" \
        --output_dir "/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/evaluation_local"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.ppvae_hformer import PPVAEHformer
from src.data.kermany_dataset import KermanyDataset, make_loaders
from src.evaluation.metrics import psnr, ssim
from src.training.config import ABLATION_ARMS, ExperimentConfig

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Device: prefer MPS (Apple Silicon) → CUDA → CPU ──────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   type=str,
    default="/Users/sam/Documents/PPVAE Dissertation Project/chest_xray")
parser.add_argument("--models_dir", type=str,
    default="/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/trained_models")
parser.add_argument("--output_dir", type=str,
    default="/Users/sam/Documents/PPVAE Dissertation Project/PpCNN/evaluation_local")
parser.add_argument("--arms",  nargs="+", default=None)
parser.add_argument("--noise_levels", nargs="+", default=["low", "mid", "high"])
parser.add_argument("--image_size",   type=int, default=256)
parser.add_argument("--num_workers",  type=int, default=0)   # 0 = main process (safe on Mac)
parser.add_argument("--mc_samples",   type=int, default=20)
parser.add_argument("--qual_indices", type=int, nargs="+", default=[2, 15, 30, 50, 65])
args = parser.parse_args()

models_dir = Path(args.models_dir)
output_dir = Path(args.output_dir)
fig_dir    = output_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(exist_ok=True)

arm_names = args.arms or list(ABLATION_ARMS.keys())
CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(arm_name: str) -> PPVAEHformer | None:
    ckpt_path = models_dir / arm_name / "best_model.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] {arm_name}: no best_model.pth at {ckpt_path}")
        return None
    cfg = ExperimentConfig()
    for k, v in ABLATION_ARMS[arm_name].get("model", {}).items():
        setattr(cfg.model, k, v)
    model = PPVAEHformer(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        num_blocks=cfg.model.num_blocks,
        num_scales=cfg.model.num_scales,
        num_heads=cfg.model.num_heads,
        window_size=cfg.model.window_size,
        use_vae=cfg.model.use_vae,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded {arm_name} (ep={ckpt.get('epoch','?')}, best_val_psnr={ckpt.get('best_val_psnr',float('nan')):.3f})")
    return model


def t2np(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) float [0,1] → (H,W) float32"""
    return t.squeeze().detach().cpu().float().numpy()


def save_qualitative_panel(arm_name: str, noise_level: str,
                            samples: list[dict], out_path: Path):
    """Save a N-row × 5-col panel: noisy / clean / denoised / aleat / diff."""
    n = len(samples)
    use_vae = ABLATION_ARMS[arm_name].get("model", {}).get("use_vae", False)
    n_cols = 6 if use_vae else 5
    fig, axes = plt.subplots(n, n_cols, figsize=(3.5 * n_cols, 3.5 * n))

    col_titles = ["Noisy input", "Clean target", "Denoised (μ)",
                  "Aleatoric σ_a", "|Error|"]
    if use_vae:
        col_titles.append("Epistemic σ_e")

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9, fontweight="bold")

    for row, s in enumerate(samples):
        class_str = CLASS_NAMES.get(s["label"], "?")
        p_val = s["psnr"]
        axes[row, 0].set_ylabel(f"{class_str}\nPSNR={p_val:.2f}", fontsize=8)

        # Noisy
        axes[row, 0].imshow(t2np(s["noisy"]), cmap="gray", vmin=0, vmax=1)
        # Clean
        axes[row, 1].imshow(t2np(s["clean"]), cmap="gray", vmin=0, vmax=1)
        # Denoised
        axes[row, 2].imshow(t2np(s["mu"]), cmap="gray", vmin=0, vmax=1)
        # Aleatoric sigma
        sigma_a = torch.exp(0.5 * s["lsa"]).clamp(0, 1)
        im = axes[row, 3].imshow(t2np(sigma_a), cmap="inferno", vmin=0, vmax=0.3)
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)
        # Error
        err = (s["mu"] - s["clean"]).abs().clamp(0, 1)
        im2 = axes[row, 4].imshow(t2np(err), cmap="RdBu_r", vmin=0, vmax=0.3)
        plt.colorbar(im2, ax=axes[row, 4], fraction=0.046)
        # Epistemic (Arm E only)
        if use_vae and "epistemic" in s:
            ep = s["epistemic"]
            ep_norm = (ep - ep.min()) / (ep.max() - ep.min() + 1e-8)
            im3 = axes[row, 5].imshow(t2np(ep_norm), cmap="plasma", vmin=0, vmax=1)
            plt.colorbar(im3, ax=axes[row, 5], fraction=0.046)

    for ax in axes.flat:
        ax.axis("off")
    axes.flat[0].axis("on")  # keep ylabel visible

    fig.suptitle(f"{arm_name} | noise={noise_level}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure saved: {out_path.name}")


# ── Main evaluation loop ───────────────────────────────────────────────────────

all_rows = []   # per-image rows across all arms and noise levels

for arm_name in arm_names:
    print(f"\n{'='*60}")
    print(f" Arm: {arm_name}")
    print(f"{'='*60}")

    model = load_model(arm_name)
    if model is None:
        continue

    use_vae   = ABLATION_ARMS[arm_name].get("model", {}).get("use_vae", False)
    arm_dir   = fig_dir / arm_name
    arm_dir.mkdir(exist_ok=True)

    for noise_level in args.noise_levels:
        print(f"\n  Noise level: {noise_level}")

        test_ds = KermanyDataset(
            root_dir=args.data_dir,
            split="test",
            target_size=args.image_size,
            noise_level=noise_level,
            augment=False,
            seed=42,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=False,
        )

        psnr_list, ssim_list = [], []
        qual_samples = []

        with torch.no_grad():
            for idx, (noisy, clean, label) in enumerate(test_loader):
                noisy = noisy.to(DEVICE)
                clean = clean.to(DEVICE)

                mu, lsa, _, _ = model(noisy, deterministic=True)
                mu_c = mu.clamp(0, 1)

                p = psnr(mu_c, clean)
                s = ssim(mu_c, clean)
                psnr_list.append(p)
                ssim_list.append(s)

                all_rows.append({
                    "arm": arm_name,
                    "noise_level": noise_level,
                    "img_idx": idx,
                    "class": CLASS_NAMES.get(label.item(), str(label.item())),
                    "psnr": f"{p:.4f}",
                    "ssim": f"{s:.4f}",
                })

                # Collect qualitative samples
                if idx in args.qual_indices:
                    sample = {
                        "noisy": noisy.cpu(), "clean": clean.cpu(),
                        "mu": mu_c.cpu(),     "lsa": lsa.cpu(),
                        "label": label.item(), "psnr": p,
                    }
                    if use_vae:
                        model.train()
                        ep = model.mc_epistemic_uncertainty(noisy, K=args.mc_samples)
                        model.eval()
                        sample["epistemic"] = ep.cpu()
                    qual_samples.append(sample)

        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        print(f"  PSNR={mean_psnr:.3f} dB  SSIM={mean_ssim:.4f}  (n={len(psnr_list)})")

        # Save qualitative figure
        fig_path = arm_dir / f"qualitative_{noise_level}.png"
        save_qualitative_panel(arm_name, noise_level, qual_samples, fig_path)

# ── Write per-image CSV ───────────────────────────────────────────────────────
csv_path = output_dir / "per_image_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["arm","noise_level","img_idx","class","psnr","ssim"])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nPer-image CSV: {csv_path}")
print(f"Figures:       {fig_dir}/")
print("Run statistical_analysis.py next to get the LaTeX stats table.")
