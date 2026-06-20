#!/usr/bin/env python
"""
evaluate_all.py — Comprehensive GPU evaluation for all PP-VAE-Hformer arms.

Metrics computed per image:
  PSNR    — pixel fidelity (dB, higher better)
  SSIM    — structural similarity (higher better)
  MS-SSIM — multi-scale SSIM, more robust than SSIM (higher better)
  LPIPS   — learned perceptual similarity via AlexNet (lower better)
  NLL     — test Gaussian NLL per pixel (lower better; arms with σ_a only)
  Sharpness — mean predicted σ_a (lower = more confident; arms with σ_a only)

Why these metrics?
  PSNR/SSIM alone are insufficient for a probabilistic model:
    - PSNR = MSE in log scale; anti-correlated with perceptual quality at
      high values (Blau & Michaeli 2018).
    - SSIM is single-scale and cannot assess long-range anatomy.
  LPIPS correlates far better with human perceptual judgement (Zhang et al. 2018).
  MS-SSIM is more robust across spatial scales (Wang et al. 2003).
  NLL directly measures how well σ_a calibrates to actual reconstruction error.
  Sharpness measures how confident (tight) the uncertainty predictions are.

Produces (in output_dir/):
  per_image_metrics.csv        per image × arm × noise_level
  metrics_summary.csv          mean ± SD + bootstrap 95% CI per arm × noise
  pairwise_stats.csv           Bonferroni t-tests + Cohen's d (mid noise, PSNR)
  table_recon_mid.tex          ready-to-paste LaTeX table
  figures/
    qualitative_<arm>_pneumonia.png
    qualitative_<arm>_normal.png
    ablation_barchart.png
    dose_curve.png
    training_curves.png
    uncertainty_calibration.png

Usage (CSD3):
    python scripts/evaluate_all.py \
        --data_dir  /rds/user/stm43/hpc-work/chest_xray \
        --results_dir /rds/user/stm43/hpc-work/ppvae_results \
        --output_dir  /rds/user/stm43/hpc-work/ppvae_results/evaluation

Usage (local Mac):
    python scripts/evaluate_all.py \
        --data_dir "/path/to/chest_xray" \
        --results_dir "/path/to/trained_models" \
        --output_dir  "/path/to/evaluation_local" \
        --num_workers 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import math
from itertools import combinations
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAIR_ROOT    = PROJECT_ROOT.parent / "KAIR"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(KAIR_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.ppvae_hformer import PPVAEHformer
from src.data.kermany_dataset import KermanyDataset
from src.evaluation.metrics import psnr, ssim
from src.training.config import ABLATION_ARMS, ExperimentConfig

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Optional perceptual metrics ───────────────────────────────────────────────
try:
    import lpips as lpips_lib
    _lpips_fn = None  # initialised after DEVICE is known
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("WARNING: lpips not installed — LPIPS metric will be NaN. "
          "Run: pip install lpips")

try:
    import piq
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False
    print("WARNING: piq not installed — MS-SSIM will be NaN. "
          "Run: pip install piq")

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",    type=str, required=True)
parser.add_argument("--results_dir", type=str, required=True,
                    help="Parent dir containing arm_*/ subdirs with best_model.pth")
parser.add_argument("--output_dir",  type=str, default=None)
parser.add_argument("--image_size",  type=int, default=256)
parser.add_argument("--mc_samples",  type=int, default=20)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--arms",        nargs="+", default=None)
parser.add_argument("--noise_levels",nargs="+", default=["low","mid","high"])
parser.add_argument("--no_figures",  action="store_true")
parser.add_argument("--bootstrap_n", type=int, default=1000,
                    help="Bootstrap resamples for 95%% CI (set 0 to skip)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

if HAS_LPIPS:
    _lpips_fn = lpips_lib.LPIPS(net='alex').to(DEVICE)
    _lpips_fn.eval()
    print("LPIPS  : AlexNet backbone loaded")

results_dir = Path(args.results_dir)
output_dir  = Path(args.output_dir) if args.output_dir else results_dir / "evaluation"
fig_dir     = output_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(exist_ok=True)

ARM_ORDER = [
    # ── Trained baselines (fair comparison: same data, same epochs) ───────────
    "ffdnet", "ircnn", "drunet", "swinir",
    # ── Original ablation arms A–E ────────────────────────────────────────────
    "arm_a_l2", "arm_b_nll", "arm_c_nll_ssim",
    "arm_d_nll_ssim_ffl", "arm_e_ppvae",
    # ── Extended ablation arms F–P ────────────────────────────────────────────
    "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb",
    "arm_i_l1", "arm_j_l1_ssim_ffl",
    "arm_k_nll_l1", "arm_l_nll_edge_ffl", "arm_m_full_det",
    "arm_n_perc", "arm_o_prelu", "arm_p_best",
]
ARM_LABELS = {
    # baselines
    "ffdnet":  "FFDNet (Zhang et al., 2018)",
    "ircnn":   "IRCNN (Zhang et al., 2017)",
    "drunet":  "DRUNet (Zhang et al., 2021)",
    "swinir":  "SwinIR (Liang et al., 2021)",
    # original arms
    "arm_a_l2":           "Arm A: $\\mathcal{L}_{2}$ (MSE)",
    "arm_b_nll":          "Arm B: NLL",
    "arm_c_nll_ssim":     "Arm C: NLL + SSIM",
    "arm_d_nll_ssim_ffl": "Arm D: NLL + SSIM + FFL",
    "arm_e_ppvae":        "Arm E: PP-VAE (NLL+SSIM+FFL+KL, linear)",
    # new arms
    "arm_f_kl_cyc":       "Arm F: PP-VAE + cyclical KL",
    "arm_g_kl_fb":        "Arm G: PP-VAE + free bits",
    "arm_h_kl_cyc_fb":    "Arm H: PP-VAE + cyc KL + free bits",
    "arm_i_l1":           "Arm I: $\\mathcal{L}_{1}$ (MAE)",
    "arm_j_l1_ssim_ffl":  "Arm J: $\\mathcal{L}_{1}$ + SSIM + FFL",
    "arm_k_nll_l1":       "Arm K: NLL + $\\mathcal{L}_{1}$",
    "arm_l_nll_edge_ffl": "Arm L: NLL + Edge + FFL",
    "arm_m_full_det":     "Arm M: NLL + SSIM + Edge + FFL",
    "arm_n_perc":         "Arm N: NLL + Perceptual + SSIM + FFL",
    "arm_o_prelu":        "Arm O: NLL + SSIM + FFL (PReLU)",
    "arm_p_best":         "Arm P: PP-VAE best (NLL+SSIM+Edge+FFL+KL, PReLU)",
}
# Arms that output aleatoric uncertainty σ_a (have NLL in loss)
UNCERTAINTY_ARMS = {
    "arm_b_nll", "arm_c_nll_ssim", "arm_d_nll_ssim_ffl",
    "arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb",
    "arm_k_nll_l1", "arm_l_nll_edge_ffl", "arm_m_full_det",
    "arm_n_perc", "arm_o_prelu", "arm_p_best",
}
# Arms with epistemic uncertainty via MC dropout / VAE sampling
VAE_ARMS_SET = {"arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb", "arm_p_best"}
# Trained baselines (KAIR architectures, raw state_dict checkpoint)
TRAINED_BASELINES = {"ffdnet", "ircnn", "drunet", "swinir"}
FOI_SIGMA_MID = 0.141  # sqrt(0.03*0.5 + 0.005) — used for sigma-conditioned baselines

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}

arm_names = args.arms if args.arms else ARM_ORDER


# ── DnCNN architecture (Zhang et al. 2017, depth=20) ──────────────────────────

class DnCNN(nn.Module):
    def __init__(self, depth=20, channels=64, in_ch=1):
        super().__init__()
        layers = [nn.Conv2d(in_ch, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, in_ch, 3, padding=1))
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        return x - self.body(x)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_ppvae(arm_name: str) -> PPVAEHformer | None:
    ckpt_path = results_dir / arm_name / "best_model.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] {arm_name}: no checkpoint at {ckpt_path}")
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
        activation=getattr(cfg.model, "activation", "gelu"),
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  {arm_name}: ep={ckpt.get('epoch','?')}  "
          f"best_val_psnr={ckpt.get('best_val_psnr',float('nan')):.3f}  params={n:,}")
    return model


def load_dncnn() -> DnCNN | None:
    ckpt_path = results_dir / "dncnn_baseline" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] dncnn_baseline: no checkpoint at {ckpt_path}")
        return None
    model = DnCNN(depth=20, channels=64, in_ch=1).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  dncnn_baseline: ep={ckpt.get('epoch','?')}  "
          f"best_val_psnr={ckpt.get('best_val_psnr',float('nan')):.3f}  params={n:,}")
    return model


def load_trained_baseline(arch: str) -> nn.Module | None:
    """Load a KAIR baseline trained on our CXR data (raw state_dict checkpoint)."""
    ckpt_path = results_dir / "baselines" / arch / "best_model.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] {arch}: no checkpoint at {ckpt_path}")
        return None
    try:
        if arch == "ffdnet":
            from models.network_ffdnet import FFDNet
            model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode="BR")
        elif arch == "ircnn":
            from models.network_dncnn import IRCNN
            model = IRCNN(in_nc=1, out_nc=1, nc=64)
        elif arch == "drunet":
            from models.network_unet import UNetRes
            model = UNetRes(
                in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4,
                act_mode="L", downsample_mode="strideconv",
                upsample_mode="convtranspose",
            )
        elif arch == "swinir":
            from models.network_swinir import SwinIR
            model = SwinIR(
                upscale=1, in_chans=1, img_size=128, window_size=8,
                img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180,
                num_heads=[6,6,6,6,6,6], mlp_ratio=2,
                upsampler="", resi_connection="1conv",
            )
        else:
            print(f"  [SKIP] {arch}: unknown baseline architecture")
            return None
        # Baselines save raw state_dict (not wrapped in a dict)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model = model.to(DEVICE).eval()
        n = sum(p.numel() for p in model.parameters())
        print(f"  {arch}: params={n:,}")
        return model
    except Exception as e:
        print(f"  [ERROR] {arch}: {e}")
        return None


def infer_baseline(arch: str, model: nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    """Inference-time forward for trained KAIR baselines."""
    if arch == "ffdnet":
        sigma = torch.full((noisy.shape[0], 1, 1, 1), FOI_SIGMA_MID,
                           dtype=noisy.dtype, device=noisy.device)
        return model(noisy, sigma).clamp(0, 1)
    elif arch in ("ircnn",):
        return model(noisy).clamp(0, 1)
    elif arch == "drunet":
        sigma_map = torch.full_like(noisy, FOI_SIGMA_MID)
        return model(torch.cat([noisy, sigma_map], dim=1)).clamp(0, 1)
    elif arch == "swinir":
        ws = 8
        _, _, h, w = noisy.shape
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")
        return model(x)[:, :, :h, :w].clamp(0, 1)
    return model(noisy).clamp(0, 1)


# ── Forward pass (uniform interface) ──────────────────────────────────────────

def forward_model(model, noisy, arm_name):
    """Returns (mu_clamped, log_sigma2_aleat_or_zeros)."""
    if arm_name == "dncnn_baseline" or arm_name in TRAINED_BASELINES:
        if arm_name in TRAINED_BASELINES:
            mu = infer_baseline(arm_name, model, noisy)
        else:
            mu = model(noisy).clamp(0, 1)
        lsa = torch.zeros_like(mu)
        return mu, lsa
    else:
        mu, lsa, _, _ = model(noisy, deterministic=True)
        return mu.clamp(0, 1), lsa


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_lpips(mu: torch.Tensor, clean: torch.Tensor) -> float:
    """LPIPS with AlexNet. Expects [B,1,H,W] in [0,1]; returns scalar (lower better)."""
    if not HAS_LPIPS or _lpips_fn is None:
        return float("nan")
    # LPIPS expects [-1,1] and 3-channel
    mu3    = mu.repeat(1, 3, 1, 1) * 2 - 1
    clean3 = clean.repeat(1, 3, 1, 1) * 2 - 1
    with torch.no_grad():
        val = _lpips_fn(mu3, clean3)
    return float(val.mean().item())


def compute_ms_ssim(mu: torch.Tensor, clean: torch.Tensor) -> float:
    """Multi-scale SSIM via piq. Expects [B,1,H,W] in [0,1]."""
    if not HAS_PIQ:
        return float("nan")
    with torch.no_grad():
        val = piq.multi_scale_ssim(mu.clamp(0, 1), clean.clamp(0, 1), data_range=1.0)
    return float(val.item())


def compute_nll(mu: torch.Tensor, log_sigma2: torch.Tensor,
                clean: torch.Tensor) -> float:
    """Mean per-pixel Gaussian NLL. Lower = better calibrated uncertainty."""
    # NLL = 0.5*(log(2π) + log σ² + (y - μ)² / σ²)
    residual2 = (clean - mu) ** 2
    nll_map   = 0.5 * (math.log(2 * math.pi) + log_sigma2 + residual2 / (log_sigma2.exp() + 1e-8))
    return float(nll_map.mean().item())


def compute_sharpness(log_sigma2: torch.Tensor) -> float:
    """Mean predicted σ_a per image. Lower = tighter/more confident predictions."""
    return float(log_sigma2.exp().sqrt().mean().item())


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n: int = 1000, alpha: float = 0.95) -> tuple[float, float]:
    if n == 0 or len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values)
    # Filter NaN values before bootstrap
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")
    means = np.array([arr[np.random.choice(len(arr), len(arr), replace=True)].mean()
                      for _ in range(n)])
    lo = float(np.percentile(means, 100*(1-alpha)/2))
    hi = float(np.percentile(means, 100*(1+alpha)/2))
    return lo, hi


# ── Cohen's d ─────────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)
                     / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


# ── Qualitative figure helpers ─────────────────────────────────────────────────

def t2np(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().detach().cpu().float().numpy()


def save_qualitative_panel(arm_name: str, samples: list[dict], noise_level: str,
                           case_class: str, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figures")
        return

    use_vae = (arm_name in VAE_ARMS_SET)
    n_rows  = len(samples)
    n_cols  = 6 if use_vae else 5   # noisy/clean/denoised/sigma_a/error[/sigma_e]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Noisy input", "Clean target", r"Denoised ($\hat{\mu}$)",
                  r"Aleat. $\sigma_a$", r"$|\hat{\mu} - y|$"]
    if use_vae:
        col_titles.append(r"Epist. $\sigma_e$")

    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=9, fontweight="bold")

    for ri, s in enumerate(samples):
        sigma_a = torch.exp(0.5 * s["lsa"]).clamp(0, 0.5)
        err     = (s["mu"] - s["clean"]).abs().clamp(0, 0.3)
        p_val   = psnr(s["mu"].to(DEVICE), s["clean"].to(DEVICE))
        l_val   = s.get("lpips", float("nan"))
        label   = f"PSNR={p_val:.2f} dB"
        if not math.isnan(l_val):
            label += f"  LPIPS={l_val:.3f}"
        axes[ri, 0].set_ylabel(label, fontsize=7)
        axes[ri, 0].imshow(t2np(s["noisy"]),  cmap="gray", vmin=0, vmax=1)
        axes[ri, 1].imshow(t2np(s["clean"]),  cmap="gray", vmin=0, vmax=1)
        axes[ri, 2].imshow(t2np(s["mu"]),     cmap="gray", vmin=0, vmax=1)
        im3 = axes[ri, 3].imshow(t2np(sigma_a), cmap="inferno", vmin=0, vmax=0.3)
        im4 = axes[ri, 4].imshow(t2np(err),   cmap="RdBu_r",  vmin=0, vmax=0.3)
        plt.colorbar(im3, ax=axes[ri, 3], fraction=0.046, pad=0.04)
        plt.colorbar(im4, ax=axes[ri, 4], fraction=0.046, pad=0.04)
        if use_vae and "epistemic" in s:
            ep = s["epistemic"]
            ep_n = (ep - ep.min()) / (ep.max() - ep.min() + 1e-8)
            im5 = axes[ri, 5].imshow(t2np(ep_n), cmap="plasma", vmin=0, vmax=1)
            plt.colorbar(im5, ax=axes[ri, 5], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis("off")
        ax.set_aspect("equal")

    fig.suptitle(
        f"{ARM_LABELS.get(arm_name, arm_name)} — {case_class} cases ({noise_level} noise)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure: {out_path.name}")


# ── Main evaluation loop ───────────────────────────────────────────────────────

per_img_rows = []
all_models   = {}

print("\n── Loading models ────────────────────────────────────────────")
for arm_name in arm_names:
    if arm_name == "dncnn_baseline":
        m = load_dncnn()
    elif arm_name in TRAINED_BASELINES:
        m = load_trained_baseline(arm_name)
    elif arm_name in ABLATION_ARMS:
        m = load_ppvae(arm_name)
    else:
        print(f"  [SKIP] {arm_name}: not in ABLATION_ARMS and not a known baseline")
        m = None
    if m is not None:
        all_models[arm_name] = m

print(f"\n{len(all_models)} models loaded: {list(all_models.keys())}\n")

QUAL_NOISE   = "mid"
QUAL_TARGETS = {"Pneumonia": [5, 15, 40], "Normal": [3, 10, 25]}

t_start = time.time()

for noise_level in args.noise_levels:
    print(f"\n{'='*65}")
    print(f" Noise level: {noise_level}")
    print(f"{'='*65}")

    test_ds = KermanyDataset(
        root_dir=args.data_dir, split="test",
        target_size=args.image_size, noise_level=noise_level,
        augment=False, seed=42,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"  Test images: {len(test_ds)}")

    class_indices: dict[str, list[int]] = {"Normal": [], "Pneumonia": []}
    for i in range(len(test_ds)):
        _, _, lbl = test_ds[i]
        class_indices[CLASS_NAMES.get(int(lbl), "?")].append(i)

    for arm_name, model in all_models.items():
        psnr_list, ssim_list, ms_ssim_list = [], [], []
        lpips_list, nll_list, sharp_list   = [], [], []
        labels_list                        = []
        qual_samples: dict[str, list[dict]] = {"Pneumonia": [], "Normal": []}

        with torch.no_grad():
            for idx, (noisy, clean, label) in enumerate(test_loader):
                noisy = noisy.to(DEVICE)
                clean = clean.to(DEVICE)
                mu, lsa = forward_model(model, noisy, arm_name)

                p  = psnr(mu, clean)
                s  = ssim(mu, clean)
                ms = compute_ms_ssim(mu, clean)
                lp = compute_lpips(mu, clean)
                cls = CLASS_NAMES.get(int(label.item()), "?")

                psnr_list.append(p)
                ssim_list.append(s)
                ms_ssim_list.append(ms)
                lpips_list.append(lp)
                labels_list.append(cls)

                # Uncertainty metrics only for arms that output σ_a
                if arm_name in UNCERTAINTY_ARMS:
                    nll_list.append(compute_nll(mu, lsa, clean))
                    sharp_list.append(compute_sharpness(lsa))
                else:
                    nll_list.append(float("nan"))
                    sharp_list.append(float("nan"))

                per_img_rows.append({
                    "arm": arm_name, "noise_level": noise_level,
                    "img_idx": idx, "class": cls,
                    "psnr":     f"{p:.4f}",
                    "ssim":     f"{s:.4f}",
                    "ms_ssim":  f"{ms:.4f}" if not math.isnan(ms) else "nan",
                    "lpips":    f"{lp:.4f}" if not math.isnan(lp) else "nan",
                    "nll":      f"{nll_list[-1]:.4f}" if not math.isnan(nll_list[-1]) else "nan",
                    "sharpness":f"{sharp_list[-1]:.4f}" if not math.isnan(sharp_list[-1]) else "nan",
                })

                # Qualitative samples (mid noise only)
                if noise_level == QUAL_NOISE and not args.no_figures:
                    within_class_idx = class_indices[cls].index(idx) if idx in class_indices[cls] else -1
                    if within_class_idx in QUAL_TARGETS.get(cls, []) and \
                       len(qual_samples[cls]) < len(QUAL_TARGETS[cls]):
                        sample = {
                            "noisy": noisy.cpu(), "clean": clean.cpu(),
                            "mu": mu.cpu(), "lsa": lsa.cpu(),
                            "lpips": lp,
                        }
                        if arm_name in VAE_ARMS_SET:
                            model.train()
                            ep = model.mc_epistemic_uncertainty(noisy, K=args.mc_samples)
                            model.eval()
                            sample["epistemic"] = ep.cpu()
                        qual_samples[cls].append(sample)

        valid_nll   = [v for v in nll_list   if not math.isnan(v)]
        valid_sharp = [v for v in sharp_list if not math.isnan(v)]
        valid_lpips = [v for v in lpips_list if not math.isnan(v)]
        valid_ms    = [v for v in ms_ssim_list if not math.isnan(v)]
        print(f"  {arm_name:30s}  PSNR={np.mean(psnr_list):.3f}  "
              f"SSIM={np.mean(ssim_list):.4f}  "
              f"MS-SSIM={np.mean(valid_ms):.4f}" if valid_ms else
              f"  {arm_name:30s}  PSNR={np.mean(psnr_list):.3f}  "
              f"SSIM={np.mean(ssim_list):.4f}")
        if valid_lpips:
            print(f"  {'':30s}  LPIPS={np.mean(valid_lpips):.4f}  "
                  f"NLL={np.mean(valid_nll):.4f}  "
                  f"Sharpness={np.mean(valid_sharp):.4f}" if valid_nll else "")

        if noise_level == QUAL_NOISE and not args.no_figures:
            for case_class, samples in qual_samples.items():
                if samples:
                    save_qualitative_panel(
                        arm_name, samples, noise_level, case_class,
                        fig_dir / f"qualitative_{arm_name}_{case_class.lower()}.png"
                    )

# ── Compute summary statistics ─────────────────────────────────────────────────

print("\n── Computing summary statistics ──────────────────────────────")
grouped      : dict[tuple, list] = defaultdict(list)
grouped_ssim : dict[tuple, list] = defaultdict(list)
grouped_ms   : dict[tuple, list] = defaultdict(list)
grouped_lpips: dict[tuple, list] = defaultdict(list)
grouped_nll  : dict[tuple, list] = defaultdict(list)
grouped_sharp: dict[tuple, list] = defaultdict(list)
grouped_cls  : dict[tuple, dict] = defaultdict(lambda: {"Normal": [], "Pneumonia": []})

for r in per_img_rows:
    key = (r["arm"], r["noise_level"])
    grouped[key].append(float(r["psnr"]))
    grouped_ssim[key].append(float(r["ssim"]))
    ms  = float(r["ms_ssim"])  if r["ms_ssim"]   != "nan" else float("nan")
    lp  = float(r["lpips"])    if r["lpips"]      != "nan" else float("nan")
    nl  = float(r["nll"])      if r["nll"]        != "nan" else float("nan")
    sh  = float(r["sharpness"])if r["sharpness"]  != "nan" else float("nan")
    grouped_ms[key].append(ms)
    grouped_lpips[key].append(lp)
    grouped_nll[key].append(nl)
    grouped_sharp[key].append(sh)
    grouped_cls[key][r["class"]].append(float(r["psnr"]))

def nanmean(lst):
    a = np.array([v for v in lst if not math.isnan(v)])
    return float(a.mean()) if len(a) > 0 else float("nan")

def nanstd(lst):
    a = np.array([v for v in lst if not math.isnan(v)])
    return float(a.std()) if len(a) > 1 else float("nan")

summary_rows = []
for arm in [a for a in ARM_ORDER if a in all_models]:
    for nl in args.noise_levels:
        key   = (arm, nl)
        pvals = np.array(grouped[key])
        svals = np.array(grouped_ssim[key])
        if len(pvals) == 0:
            continue
        p_lo, p_hi = bootstrap_ci(pvals.tolist(), args.bootstrap_n)
        s_lo, s_hi = bootstrap_ci(svals.tolist(), args.bootstrap_n)

        ms_mean  = nanmean(grouped_ms[key])
        lp_mean  = nanmean(grouped_lpips[key])
        nll_mean = nanmean(grouped_nll[key])
        sh_mean  = nanmean(grouped_sharp[key])

        def fmt(v, dec=3):
            return f"{v:.{dec}f}" if not math.isnan(v) else "nan"

        summary_rows.append({
            "arm": arm, "noise_level": nl,
            "psnr_mean":    f"{pvals.mean():.3f}",
            "psnr_sd":      f"{pvals.std():.3f}",
            "psnr_ci_lo":   f"{p_lo:.3f}",
            "psnr_ci_hi":   f"{p_hi:.3f}",
            "ssim_mean":    f"{svals.mean():.4f}",
            "ssim_sd":      f"{svals.std():.4f}",
            "ssim_ci_lo":   f"{s_lo:.4f}",
            "ssim_ci_hi":   f"{s_hi:.4f}",
            "ms_ssim_mean": fmt(ms_mean, 4),
            "lpips_mean":   fmt(lp_mean, 4),
            "nll_mean":     fmt(nll_mean, 4),
            "sharpness_mean": fmt(sh_mean, 4),
            "n": len(pvals),
        })

# ── ANOVA + Bonferroni + Cohen's d (mid noise, PSNR) ─────────────────────────

from scipy import stats as scipy_stats

EVAL_NL   = "mid"
mid_arms  = [a for a in ARM_ORDER if a in all_models and (a, EVAL_NL) in grouped]
psnr_groups = {a: np.array(grouped[(a, EVAL_NL)]) for a in mid_arms}

F_stat, p_anova = scipy_stats.f_oneway(*psnr_groups.values())
print(f"\nOne-way ANOVA (PSNR, {EVAL_NL}): F={F_stat:.2f}, p={p_anova:.4e}")

pairs   = list(combinations(mid_arms, 2))
n_comp  = len(pairs)
pairwise_rows = []
for a1, a2 in pairs:
    t, p_raw = scipy_stats.ttest_ind(psnr_groups[a1], psnr_groups[a2])
    p_bonf   = min(1.0, p_raw * n_comp)
    d        = cohen_d(psnr_groups[a1], psnr_groups[a2])
    sig = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "n.s."
    pairwise_rows.append({
        "arm1": a1, "arm2": a2,
        "t": f"{t:.3f}", "p_raw": f"{p_raw:.4e}",
        "p_bonf": f"{p_bonf:.4e}", "cohens_d": f"{d:.3f}", "sig": sig,
    })
    print(f"  {a1:28s} vs {a2:28s}: p_bonf={p_bonf:.3e} {sig}  d={d:+.2f}")

# ── Subgroup (Pneumonia vs Normal) ────────────────────────────────────────────

print(f"\nSubgroup analysis — PSNR (mid noise):")
for arm in mid_arms:
    norm = np.array(grouped_cls[(arm, EVAL_NL)]["Normal"])
    pneu = np.array(grouped_cls[(arm, EVAL_NL)]["Pneumonia"])
    if len(norm) > 1 and len(pneu) > 1:
        t, p = scipy_stats.ttest_ind(norm, pneu)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {arm:30s}  Normal={norm.mean():.3f}±{norm.std():.3f}  "
              f"Pneumonia={pneu.mean():.3f}±{pneu.std():.3f}  "
              f"Δ={norm.mean()-pneu.mean():+.3f}  {sig}")

# ── Write CSVs ────────────────────────────────────────────────────────────────

per_img_path = output_dir / "per_image_metrics.csv"
with open(per_img_path, "w", newline="") as f:
    fields = ["arm","noise_level","img_idx","class",
              "psnr","ssim","ms_ssim","lpips","nll","sharpness"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(per_img_rows)

summary_path = output_dir / "metrics_summary.csv"
with open(summary_path, "w", newline="") as f:
    fields = ["arm","noise_level",
              "psnr_mean","psnr_sd","psnr_ci_lo","psnr_ci_hi",
              "ssim_mean","ssim_sd","ssim_ci_lo","ssim_ci_hi",
              "ms_ssim_mean","lpips_mean","nll_mean","sharpness_mean","n"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(summary_rows)

pairwise_path = output_dir / "pairwise_stats.csv"
with open(pairwise_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["arm1","arm2","t","p_raw","p_bonf","cohens_d","sig"])
    w.writeheader(); w.writerows(pairwise_rows)

print(f"\nCSVs written:")
print(f"  {per_img_path}")
print(f"  {summary_path}")
print(f"  {pairwise_path}")

# ── LaTeX table (mid noise) — PSNR, SSIM, MS-SSIM, LPIPS ─────────────────────

mid_summary = {r["arm"]: r for r in summary_rows if r["noise_level"] == EVAL_NL}
best_psnr   = max((float(r["psnr_mean"])  for r in mid_summary.values()), default=0)
best_ssim   = max((float(r["ssim_mean"])  for r in mid_summary.values()), default=0)
best_ms     = max((float(r["ms_ssim_mean"]) for r in mid_summary.values()
                   if r["ms_ssim_mean"] != "nan"), default=0)
best_lpips  = min((float(r["lpips_mean"]) for r in mid_summary.values()
                   if r["lpips_mean"] != "nan"), default=1)

latex_lines = [
    r"% ── Paste into §4.3.1 ──────────────────────────────────────",
    r"\begin{longtable}{lcccccc}",
    r"\caption{Test-set reconstruction performance at \textit{mid} noise level"
    r" ($n=624$). Mean\,$\pm$\,SD. \textbf{Bold} = best per column."
    r" $\downarrow$ = lower is better. UQ = outputs aleatoric $\hat{\sigma}_a$."
    r"} \label{tab:results:recon:mid} \\",
    r"\toprule",
    r"\textbf{Method} & \textbf{VAE} & \textbf{UQ} & \textbf{PSNR (dB)} & "
    r"\textbf{SSIM} & \textbf{MS-SSIM} & \textbf{LPIPS $\downarrow$} \\",
    r"\midrule \endfirsthead",
    r"\toprule",
    r"\textbf{Method} & \textbf{VAE} & \textbf{UQ} & \textbf{PSNR (dB)} & "
    r"\textbf{SSIM} & \textbf{MS-SSIM} & \textbf{LPIPS $\downarrow$} \\",
    r"\midrule \endhead",
    r"\midrule \multicolumn{7}{r}{\textit{Continued\ldots}} \\ \endfoot",
    r"\bottomrule \endlastfoot",
    r"\multicolumn{7}{l}{\textit{Trained baselines}} \\",
    r"\midrule",
]

group_markers = {
    "arm_a_l2":    r"\midrule\multicolumn{7}{l}{\textit{Ablation arms A--E (original)}} \\\midrule",
    "arm_f_kl_cyc":r"\midrule\multicolumn{7}{l}{\textit{Ablation arms F--H (VAE posterior collapse)}} \\\midrule",
    "arm_i_l1":    r"\midrule\multicolumn{7}{l}{\textit{Ablation arms I--P (loss combinations)}} \\\midrule",
}

for arm in ARM_ORDER:
    if arm in group_markers:
        latex_lines.append(group_markers[arm])
    if arm not in mid_summary:
        continue
    r   = mid_summary[arm]
    lbl = ARM_LABELS.get(arm, arm)
    vae = r"\checkmark" if arm in VAE_ARMS_SET else "---"
    uq  = r"\checkmark" if arm in UNCERTAINTY_ARMS else "---"
    pm  = r"$\pm$"

    p_str  = f"{r['psnr_mean']} {pm} {r['psnr_sd']}"
    s_str  = f"{r['ssim_mean']} {pm} {r['ssim_sd']}"
    ms_str = r["ms_ssim_mean"] if r["ms_ssim_mean"] != "nan" else "---"
    lp_str = r["lpips_mean"]   if r["lpips_mean"]   != "nan" else "---"

    if abs(float(r["psnr_mean"]) - best_psnr) < 0.001:
        p_str = r"\textbf{" + p_str + r"}"
    if abs(float(r["ssim_mean"]) - best_ssim) < 0.0001:
        s_str = r"\textbf{" + s_str + r"}"
    if r["ms_ssim_mean"] != "nan" and abs(float(r["ms_ssim_mean"]) - best_ms) < 0.0001:
        ms_str = r"\textbf{" + ms_str + r"}"
    if r["lpips_mean"] != "nan" and abs(float(r["lpips_mean"]) - best_lpips) < 0.0001:
        lp_str = r"\textbf{" + lp_str + r"}"

    latex_lines.append(f"  {lbl} & {vae} & {uq} & {p_str} & {s_str} & {ms_str} & {lp_str} \\\\")

latex_lines += [r"\end{longtable}"]
latex_str  = "\n".join(latex_lines)
latex_path = output_dir / "table_recon_mid.tex"
with open(latex_path, "w") as f:
    f.write(latex_str)
print(f"\nLaTeX table → {latex_path}")

# ── Figures ───────────────────────────────────────────────────────────────────

if not args.no_figures:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_arms  = [a for a in ARM_ORDER if a in mid_summary]
        xlabels    = [ARM_LABELS.get(a, a).split(": ")[-1] for a in eval_arms]
        psnr_means = [float(mid_summary[a]["psnr_mean"]) for a in eval_arms]
        psnr_sds   = [float(mid_summary[a]["psnr_sd"])   for a in eval_arms]
        ssim_means = [float(mid_summary[a]["ssim_mean"]) for a in eval_arms]
        ssim_sds   = [float(mid_summary[a]["ssim_sd"])   for a in eval_arms]

        # ── Ablation bar chart (PSNR + SSIM) ───────────────────────────────
        x, w = np.arange(len(eval_arms)), 0.38
        fig, ax1 = plt.subplots(figsize=(max(10, 2.2*len(eval_arms)), 5))
        ax2 = ax1.twinx()
        b1 = ax1.bar(x-w/2, psnr_means, w, yerr=psnr_sds,
                     color="steelblue", alpha=0.85, capsize=4, label="PSNR (dB)")
        b2 = ax2.bar(x+w/2, ssim_means, w, yerr=ssim_sds,
                     color="coral",     alpha=0.85, capsize=4, label="SSIM")
        for bar, val in zip(b1, psnr_means):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="steelblue")
        for bar, val in zip(b2, ssim_means):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0008,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="coral")
        ax1.set_ylabel("PSNR (dB)", color="steelblue", fontsize=12)
        ax2.set_ylabel("SSIM",      color="coral",     fontsize=12)
        ax1.set_xticks(x); ax1.set_xticklabels(xlabels, fontsize=9, rotation=15, ha="right")
        ax1.set_title("Ablation Study: Incremental Effect of Each Loss Term (mid noise)", fontsize=12)
        ax1.legend([b1, b2], ["PSNR (dB)", "SSIM"], loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(fig_dir / "ablation_barchart.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: ablation_barchart.png")

        # ── Dose curve ─────────────────────────────────────────────────────
        noise_order = ["low", "mid", "high"]
        colors = plt.cm.tab10(np.linspace(0, 0.7, len(eval_arms)))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for arm, col in zip(eval_arms, colors):
            for ax, key in zip(axes, ["psnr", "ssim"]):
                means2, stds2 = [], []
                for nl in noise_order:
                    matches = [r for r in summary_rows if r["arm"]==arm and r["noise_level"]==nl]
                    if matches:
                        means2.append(float(matches[0][f"{key}_mean"]))
                        stds2.append(float(matches[0][f"{key}_sd"]))
                    else:
                        means2.append(float("nan")); stds2.append(0)
                ax.errorbar(noise_order, means2, yerr=stds2,
                            label=ARM_LABELS.get(arm, arm), marker="o",
                            linewidth=2, capsize=4, color=col)
        for ax, label in zip(axes, ["PSNR (dB)", "SSIM"]):
            ax.set_xlabel("Noise level", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f"Reconstruction {label} vs Noise Severity", fontsize=12)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(fig_dir / "dose_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: dose_curve.png")

        # ── Training curves ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors2 = plt.cm.tab10(np.linspace(0, 0.7, len(eval_arms)))
        for arm, col in zip(eval_arms, colors2):
            log_path = results_dir / arm / "train_log.csv"
            if not log_path.exists():
                continue
            epochs, psnr_vals, kl_vals = [], [], []
            with open(log_path) as lf:
                for row in csv.DictReader(lf):
                    try:
                        vp = float(row.get("val_psnr","nan"))
                        kl = float(row.get("l_kl", 0) or 0)
                        if not math.isnan(vp) and vp > 0:
                            epochs.append(int(row["epoch"]))
                            psnr_vals.append(vp)
                            kl_vals.append(kl)
                    except Exception:
                        pass
            if epochs:
                axes[0].plot(epochs, psnr_vals, label=ARM_LABELS.get(arm,arm),
                             color=col, linewidth=2)
            if any(k > 0 for k in kl_vals):
                axes[1].plot(epochs, kl_vals, label=ARM_LABELS.get(arm,arm),
                             color=col, linewidth=2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val PSNR (dB)")
        axes[0].set_title("Validation PSNR During Training")
        axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel(r"$\beta \cdot L_{\mathrm{KL}}$")
        axes[1].set_title("KL Divergence Trajectory (Arm E)")
        axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: training_curves.png")

        # ── Uncertainty calibration: σ_a vs |error| scatter ────────────────
        # For each uncertainty arm at mid noise, plot mean predicted σ_a vs mean
        # absolute error per image to assess whether uncertainty is well-calibrated.
        uncert_arms = [a for a in eval_arms if a in UNCERTAINTY_ARMS]
        if uncert_arms:
            fig, axes = plt.subplots(1, len(uncert_arms),
                                     figsize=(5*len(uncert_arms), 4), squeeze=False)
            for ax, arm in zip(axes[0], uncert_arms):
                rows_arm = [r for r in per_img_rows
                            if r["arm"]==arm and r["noise_level"]==EVAL_NL
                            and r["sharpness"] != "nan"]
                if not rows_arm:
                    continue
                sharps = np.array([float(r["sharpness"]) for r in rows_arm])
                # proxy for error: use 1/PSNR relationship
                psnrs  = np.array([float(r["psnr"])      for r in rows_arm])
                errors = 10 ** (- psnrs / 20)   # approx normalised RMSE from PSNR
                ax.scatter(sharps, errors, alpha=0.3, s=8,
                           c=["crimson" if r["class"]=="Pneumonia" else "steelblue"
                              for r in rows_arm])
                # perfect calibration line
                lim = max(sharps.max(), errors.max())
                ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect cal.")
                corr = np.corrcoef(sharps, errors)[0, 1]
                ax.set_title(f"{arm}\nr={corr:.3f}", fontsize=9)
                ax.set_xlabel(r"Predicted $\sigma_a$ (sharpness)", fontsize=9)
                ax.set_ylabel("Approx. RMSE", fontsize=9)
                ax.legend(fontsize=7)
            from matplotlib.lines import Line2D
            legend_els = [Line2D([0],[0], marker='o', color='w', markerfacecolor='crimson',
                                 markersize=6, label='Pneumonia'),
                          Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',
                                 markersize=6, label='Normal')]
            fig.legend(handles=legend_els, loc="upper right", fontsize=8)
            fig.suptitle("Uncertainty Calibration: Predicted σ_a vs Reconstruction Error",
                         fontsize=11)
            plt.tight_layout()
            fig.savefig(fig_dir / "uncertainty_calibration.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  Saved: uncertainty_calibration.png")

    except Exception as e:
        import traceback
        print(f"  Figure generation error: {e}")
        traceback.print_exc()

# ── Final summary ──────────────────────────────────────────────────────────────

elapsed = time.time() - t_start
print(f"\n{'='*65}")
print(f" Evaluation complete in {elapsed/60:.1f} min")
print(f" per_image_metrics.csv  → {per_img_path}")
print(f" metrics_summary.csv    → {summary_path}")
print(f" pairwise_stats.csv     → {pairwise_path}")
print(f" table_recon_mid.tex    → {latex_path}")
print(f" Figures                → {fig_dir}/")
print(f"{'='*65}")
print(f"\nMetrics computed:")
print(f"  PSNR, SSIM          — pixel fidelity and structural similarity")
print(f"  MS-SSIM             — {'enabled' if HAS_PIQ else 'DISABLED (pip install piq)'}")
print(f"  LPIPS               — {'enabled (AlexNet)' if HAS_LPIPS else 'DISABLED (pip install lpips)'}")
print(f"  NLL + Sharpness     — uncertainty calibration (arms B/C/D/E only)")
