"""
generate_roi_panels_v2.py — Publication-quality ROI figures for the 16-arm PP-VAE ablation study.

Generates four figures:
  Fig A  — Normal case,    mid-lung vascular ROI   (shows fine vessel margin preservation)
  Fig B  — Pneumonia case, consolidation ROI        (shows pathological feature preservation)
  Fig C  — Normal case,    costophrenic angle ROI   (shows sharp pleural boundary + FFL effect)
  Fig D  — Uncertainty panel: Arm D & H with σ_a maps for both cases

Models shown (9 columns in A/B/C):
  Noisy | Ref | Arm A (L2) | Arm I (L1) | Arm D (NLL+SSIM+FFL) | Arm H (VAE cyc+fb)
  | IRCNN | SwinIR | Arm O (PReLU fail)

Usage:
  # Interactive (GPU node):
  srun --account=mrc-bsu2-sl2-gpu --partition=ampere --gres=gpu:1 --time=1:00:00 --pty bash
  cd ~/hpc-work/ppvae_hformer
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate ppvae
  python scripts/generate_roi_panels_v2.py

  # Or submit via SLURM:
  sbatch slurm/roi_panels.sh
"""

from __future__ import annotations
import os, sys, glob, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT   = "/rds/user/stm43/hpc-work/ppvae_hformer"
RESULTS   = "/rds/user/stm43/hpc-work/ppvae_results"
DATA      = "/rds/user/stm43/hpc-work/chest_xray/test"
KAIR_ROOT = "/rds/user/stm43/hpc-work/KAIR"
OUT_DIR   = "/rds/user/stm43/hpc-work/ppvae_results/roi_panels_v2"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, PROJECT)
sys.path.insert(0, KAIR_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Sigma for FFDNet / DRUNet (Foi mid-noise level: sqrt(0.03*0.5 + 0.005))
FOI_SIGMA_MID = 0.141

# ── Case & ROI definitions ─────────────────────────────────────────────────────
# Each entry: (label, class, sorted_index, roi_xywh, arrow_offset, anatomical_note)
PANELS = [
    {
        "fig_id": "A",
        "label": "Normal — right mid-lung vascular margins",
        "cls": "NORMAL",
        "idx": 9,
        "roi": (90, 65, 75, 75),
        "arrow": (-14, -14),
        "anatomy": "right mid-lung vascular margin",
        "noise_eta": 200,
    },
    {
        "fig_id": "B",
        "label": "Pneumonia — alveolar consolidation (lower lobe)",
        "cls": "PNEUMONIA",
        "idx": 10,
        "roi": (55, 125, 85, 75),
        "arrow": (14, -14),
        "anatomy": "alveolar consolidation margin",
        "noise_eta": 200,
    },
    {
        "fig_id": "C",
        "label": "Normal — costophrenic angle / pleural boundary",
        "cls": "NORMAL",
        "idx": 0,
        "roi": (125, 155, 70, 65),
        "arrow": (-14, 14),
        "anatomy": "right costophrenic angle",
        "noise_eta": 200,
    },
]

# Models for comparison panels A/B/C
METHODS = [
    ("noisy",  "Noisy\nInput",                  "#666666"),
    ("clean",  "Reference",                      "#222222"),
    ("arm_a",  "Arm A\n(L₂, 34.62 dB)",         "#1f77b4"),
    ("arm_i",  "Arm I\n(L₁, 34.44 dB)",         "#aec7e8"),
    ("arm_d",  "Arm D\n(NLL+SSIM+FFL)",          "#ff7f0e"),
    ("arm_h",  "Arm H\n(VAE cyc+fb)",            "#9467bd"),
    ("ircnn",  "IRCNN\n(CNN base, 32.66 dB)",    "#2ca02c"),
    ("swinir", "SwinIR\n(Transf., 31.18 dB)",    "#d62728"),
    ("arm_o",  "Arm O\n(PReLU, 29.07 dB)",       "#8c564b"),
]

# Uncertainty panel uses these arms only
UQ_METHODS = [
    ("arm_b", "Arm B\n(NLL)"),
    ("arm_d", "Arm D\n(NLL+SSIM+FFL)"),
    ("arm_h", "Arm H\n(VAE cyc+fb)"),
]

# ── Model loading ─────────────────────────────────────────────────────────────

def _load_ckpt(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


class DnCNN(torch.nn.Module):
    """DnCNN (Zhang et al. 2017, depth=20) — residual noise predictor."""
    def __init__(self, depth=20, channels=64, in_ch=1):
        super().__init__()
        layers = [torch.nn.Conv2d(in_ch, channels, 3, padding=1), torch.nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                       torch.nn.BatchNorm2d(channels), torch.nn.ReLU(inplace=True)]
        layers.append(torch.nn.Conv2d(channels, in_ch, 3, padding=1))
        self.body = torch.nn.Sequential(*layers)
    def forward(self, x):
        return x - self.body(x)


def load_all_models():
    from src.models.ppvae_hformer import PPVAEHformer
    from src.training.config import ABLATION_ARMS, ExperimentConfig

    models = {}

    # ── PP-VAE-Hformer arms ───────────────────────────────────────────────────
    ppvae_arms = {
        "arm_a": "arm_a_l2",
        "arm_b": "arm_b_nll",
        "arm_d": "arm_d_nll_ssim_ffl",
        "arm_h": "arm_h_kl_cyc_fb",
        "arm_i": "arm_i_l1",
        "arm_o": "arm_o_prelu",
    }
    for key, arm_name in ppvae_arms.items():
        ckpt_path = os.path.join(RESULTS, arm_name, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {key}: {ckpt_path} not found")
            models[key] = None
            continue
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
        )
        model.load_state_dict(_load_ckpt(ckpt_path))
        model.to(DEVICE).eval()
        models[key] = model
        print(f"  ✓ {key} ({arm_name})")

    # ── DnCNN ────────────────────────────────────────────────────────────────
    dncnn_path = os.path.join(RESULTS, "dncnn_baseline", "best_model.pth")
    if os.path.exists(dncnn_path):
        m = DnCNN(depth=20, channels=64, in_ch=1).to(DEVICE)
        m.load_state_dict(_load_ckpt(dncnn_path))
        m.eval()
        models["dncnn"] = m
        print(f"  ✓ dncnn")

    # ── KAIR baselines ────────────────────────────────────────────────────────
    from models.network_ffdnet import FFDNet
    from models.network_dncnn import IRCNN
    from models.network_unet import UNetRes
    from models.network_swinir import SwinIR

    kair_specs = {
        "ffdnet": (FFDNet, dict(in_nc=1, out_nc=1, nc=64, nb=15, act_mode="BR")),
        "ircnn":  (IRCNN,  dict(in_nc=1, out_nc=1, nc=64)),
        "drunet": (UNetRes, dict(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4,
                                 act_mode="L", downsample_mode="strideconv",
                                 upsample_mode="convtranspose")),
        "swinir": (SwinIR, dict(upscale=1, in_chans=1, img_size=128, window_size=8,
                                img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180,
                                num_heads=[6,6,6,6,6,6], mlp_ratio=2,
                                upsampler="", resi_connection="1conv")),
    }
    for arch, (cls, kwargs) in kair_specs.items():
        ckpt_path = os.path.join(RESULTS, "baselines", arch, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {arch}: not found")
            models[arch] = None
            continue
        m = cls(**kwargs)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        m.load_state_dict(state, strict=True)
        m.to(DEVICE).eval()
        models[arch] = m
        print(f"  ✓ {arch}")

    return models


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, noisy_t, key):
    """Returns (recon [0,1], log_sigma2_aleat or None)."""
    x = noisy_t.unsqueeze(0).to(DEVICE)
    if key in ("arm_a", "arm_b", "arm_d", "arm_h", "arm_i", "arm_o"):
        mu, lsa, _, _ = model(x, deterministic=True)
        return mu.squeeze().clamp(0, 1).cpu().float().numpy(), lsa.squeeze().cpu().float().numpy()
    elif key == "dncnn":
        out = model(x).clamp(0, 1)
        return out.squeeze().cpu().float().numpy(), None
    elif key == "ffdnet":
        sigma = torch.full((1, 1, 1, 1), FOI_SIGMA_MID, dtype=x.dtype, device=DEVICE)
        out = model(x, sigma).clamp(0, 1)
        return out.squeeze().cpu().float().numpy(), None
    elif key == "ircnn":
        out = model(x).clamp(0, 1)
        return out.squeeze().cpu().float().numpy(), None
    elif key == "drunet":
        sigma_map = torch.full_like(x, FOI_SIGMA_MID)
        out = model(torch.cat([x, sigma_map], dim=1)).clamp(0, 1)
        return out.squeeze().cpu().float().numpy(), None
    elif key == "swinir":
        ws = 8
        _, _, h, w = x.shape
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(xp)[:, :, :h, :w].clamp(0, 1)
        return out.squeeze().cpu().float().numpy(), None
    else:
        out = model(x).clamp(0, 1)
        return out.squeeze().cpu().float().numpy(), None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_image(cls: str, idx: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(DATA, cls, "*.jpeg")) +
                   glob.glob(os.path.join(DATA, cls, "*.png")))
    img = np.array(Image.open(files[idx]).convert("L").resize((256, 256))) / 255.0
    return img.astype(np.float32)


def add_noise(clean: np.ndarray, eta: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poisson = rng.poisson(clean * eta) / eta
    gaussian = rng.normal(0, 0.01, clean.shape)
    return np.clip(poisson + gaussian, 0, 1).astype(np.float32)


def psnr(clean, recon):
    mse = np.mean((clean - recon) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


# ── Figure helpers ────────────────────────────────────────────────────────────

def add_roi_rect(ax, x, y, w, h, lw=1.8):
    ax.add_patch(patches.Rectangle((x, y), w, h, lw=lw,
                 edgecolor="red", facecolor="none", zorder=10))


def add_arrow(ax, cx, cy, dx, dy):
    ax.annotate("", xy=(cx, cy), xytext=(cx + dx, cy + dy),
                arrowprops=dict(arrowstyle="-|>", color="red",
                                lw=1.8, mutation_scale=14), zorder=11)


def crop(img, x, y, w, h):
    return img[y:y+h, x:x+w]


# ── Panel A/B/C: comparison figure ────────────────────────────────────────────

def make_comparison_panel(panel_cfg, models):
    fig_id    = panel_cfg["fig_id"]
    cls       = panel_cfg["cls"]
    idx       = panel_cfg["idx"]
    roi_x, roi_y, roi_w, roi_h = panel_cfg["roi"]
    arr_dx, arr_dy = panel_cfg["arrow"]
    eta       = panel_cfg["noise_eta"]
    anatomy   = panel_cfg["anatomy"]

    print(f"\n=== Panel {fig_id}: {panel_cfg['label']} ===")

    clean = load_image(cls, idx)
    noisy = add_noise(clean, eta, seed=42 + idx)
    noisy_t = torch.tensor(noisy).float().unsqueeze(0)

    # Run all models
    recons = {}
    lsas   = {}
    for key, _, _ in METHODS:
        if key in ("noisy", "clean"):
            continue
        m = models.get(key)
        if m is None:
            print(f"  skip {key} (model not loaded)")
            continue
        recon, lsa = infer(m, noisy_t, key)
        recons[key] = recon
        lsas[key]   = lsa
        p = psnr(clean, recon)
        print(f"  {key:8s}: {p:.2f} dB")

    n_cols = len(METHODS)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5.4),
                             gridspec_kw={"hspace": 0.03, "wspace": 0.02})

    imgs = {"noisy": noisy, "clean": clean, **recons}

    for col, (key, label, color) in enumerate(METHODS):
        img = imgs.get(key)
        if img is None:
            axes[0, col].axis("off"); axes[1, col].axis("off"); continue

        # ── Row 0: full image ──
        ax0 = axes[0, col]
        ax0.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        add_roi_rect(ax0, roi_x, roi_y, roi_w, roi_h)
        ax0.axis("off")
        ax0.set_title(label, fontsize=7.2, pad=2, color=color,
                      fontweight="bold" if key not in ("noisy","clean") else "normal")

        if key not in ("noisy", "clean"):
            p = psnr(clean, img)
            ax0.text(0.03, 0.04, f"{p:.2f} dB",
                     transform=ax0.transAxes, color="yellow", fontsize=6.5,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))

        # ── Row 1: zoomed ROI ──
        ax1 = axes[1, col]
        c = crop(img, roi_x, roi_y, roi_w, roi_h)
        ax1.imshow(c, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")

        # Arrow tip at 40% of the crop
        tip_x, tip_y = roi_w * 0.40, roi_h * 0.40
        add_arrow(ax1, tip_x, tip_y, arr_dx, arr_dy)

        for spine in ax1.spines.values():
            spine.set_edgecolor("red"); spine.set_linewidth(1.8)
        ax1.set_xticks([]); ax1.set_yticks([])

    axes[0, 0].set_ylabel("Full image", fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel(f"ROI: {anatomy}", fontsize=7, labelpad=4)

    fig.suptitle(
        f"Figure {fig_id} — {panel_cfg['label']}  (η = {eta})",
        fontsize=9.5, y=1.01, fontweight="bold",
    )

    stem = os.path.join(OUT_DIR, f"fig_{fig_id.lower()}_roi_comparison")
    fig.savefig(stem + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {stem}.pdf / .png")


# ── Panel D: Uncertainty maps ─────────────────────────────────────────────────

def make_uncertainty_panel(models):
    """For each UQ arm × 2 cases: show Noisy | Clean | Recon | σ_a | |Error|"""
    print("\n=== Panel D: Uncertainty maps ===")

    cases = [
        {"cls": "NORMAL",    "idx": 9,  "label": "Normal"},
        {"cls": "PNEUMONIA", "idx": 10, "label": "Pneumonia"},
    ]
    # ROI for uncertainty panel: mid-lung for normal, consolidation for pneumonia
    rois = [(90, 65, 75, 75), (55, 125, 85, 75)]

    n_cases = len(cases)
    n_arms  = len(UQ_METHODS)
    n_cols  = 6  # Noisy | Clean | Recon | σ_a | |Error| | ROI crop
    n_rows  = n_cases * n_arms

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.3, n_rows * 2.5),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.03})

    col_titles = ["Noisy Input", "Reference", "Reconstruction",
                  r"Aleatoric $\hat{\sigma}_a$", r"Abs. Error $|\hat{\mu}-y|$",
                  "ROI crop"]

    for ci, (case_cfg, roi) in enumerate(zip(cases, rois)):
        clean = load_image(case_cfg["cls"], case_cfg["idx"])
        noisy = add_noise(clean, 200, seed=42 + case_cfg["idx"])
        noisy_t = torch.tensor(noisy).float().unsqueeze(0)
        roi_x, roi_y, roi_w, roi_h = roi

        for ai, (arm_key, arm_label) in enumerate(UQ_METHODS):
            row = ci * n_arms + ai
            m = models.get(arm_key)

            if m is None:
                for c in range(n_cols):
                    axes[row, c].axis("off")
                continue

            recon, lsa = infer(m, noisy_t, arm_key)
            sigma_a = np.exp(0.5 * lsa).clip(0, 0.5) if lsa is not None else np.zeros_like(recon)
            error   = np.abs(recon - clean)
            roi_crop = crop(recon, roi_x, roi_y, roi_w, roi_h)

            imgs_row = [noisy, clean, recon, sigma_a, error, roi_crop]
            cmaps    = ["gray", "gray", "gray", "inferno", "hot", "gray"]
            vmaxs    = [1.0,    1.0,    1.0,    0.30,     0.30,  1.0]

            for c, (img, cm, vmax) in enumerate(zip(imgs_row, cmaps, vmaxs)):
                ax = axes[row, c]
                ax.imshow(img, cmap=cm, vmin=0, vmax=vmax, interpolation="lanczos")

                # Red ROI rect on full images (cols 0-4)
                if c < 5:
                    add_roi_rect(ax, roi_x, roi_y, roi_w, roi_h, lw=1.5)

                # PSNR on reconstruction col
                if c == 2:
                    p = psnr(clean, recon)
                    ax.text(0.03, 0.04, f"{p:.2f} dB",
                            transform=ax.transAxes, color="yellow", fontsize=7,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))

                # NLL on σ_a col
                if c == 3 and lsa is not None:
                    nll_val = 0.5 * (np.log(2 * math.pi) + lsa + (recon - clean)**2 / (np.exp(lsa) + 1e-8))
                    ax.text(0.03, 0.04, f"NLL={nll_val.mean():.3f}",
                            transform=ax.transAxes, color="white", fontsize=6.5,
                            bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.6))

                # Red border on ROI crop
                if c == 5:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red"); spine.set_linewidth(1.8)

                ax.set_xticks([]); ax.set_yticks([])

                # Column titles (top row only)
                if row == 0:
                    ax.set_title(col_titles[c], fontsize=8.5, pad=3)

            # Row label
            axes[row, 0].set_ylabel(
                f"{case_cfg['label']}\n{arm_label}",
                fontsize=7.5, labelpad=4, rotation=0,
                ha="right", va="center",
            )

            p = psnr(clean, recon)
            print(f"  {case_cfg['label']} / {arm_key}: PSNR={p:.2f} dB")

    fig.suptitle(
        "Figure D — Aleatoric uncertainty maps: NLL and VAE arms\n"
        r"Columns: Noisy | Reference | Reconstruction | $\hat{\sigma}_a$ | $|\hat{\mu}-y|$ | ROI",
        fontsize=10, y=1.01, fontweight="bold",
    )

    stem = os.path.join(OUT_DIR, "fig_d_uncertainty_maps")
    fig.savefig(stem + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  → {stem}.pdf / .png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading models...")
    models = load_all_models()
    print(f"Loaded: {[k for k,v in models.items() if v is not None]}\n")

    for panel_cfg in PANELS:
        make_comparison_panel(panel_cfg, models)

    make_uncertainty_panel(models)

    print(f"\n✓ All figures saved to {OUT_DIR}")
    print("Download with:")
    print(f"  ssh -S /tmp/ssh_mux_login.hpc.cam.ac.uk_22_stm43 stm43@login.hpc.cam.ac.uk "
          f'"tar czf - -C {OUT_DIR} . | cat" > /tmp/roi_panels_v2.tar.gz')
