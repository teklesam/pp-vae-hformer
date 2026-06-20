"""
generate_figures_v2.py — Regenerate all dissertation evaluation figures with fixes:

  1. Qualitative panels: remove flat σ_a for non-NLL arms; n×m layout
  2. Dose curve: legend outside the plot area
  3. Ablation barchart: horizontal bars, PSNR-only, color-coded
  4. Subgroup quality panels: Bacterial / Viral / Normal × {A, D, H, IRCNN}
  5. Latent space: UMAP colored by class + KMeans Adjusted Rand Index
  6. Pixel-level uncertainty comparison panel (fixed)

Output directory: /rds/user/stm43/hpc-work/ppvae_results/figures_v2/

Usage:
  sbatch slurm/gen_figures_v2.sh
"""

from __future__ import annotations
import os, sys, glob, math, csv, itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

PROJECT   = "/rds/user/stm43/hpc-work/ppvae_hformer"
RESULTS   = "/rds/user/stm43/hpc-work/ppvae_results"
DATA      = "/rds/user/stm43/hpc-work/chest_xray/test"
KAIR_ROOT = "/rds/user/stm43/hpc-work/KAIR"
OUT_DIR   = Path("/rds/user/stm43/hpc-work/ppvae_results/figures_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, PROJECT)
sys.path.insert(0, KAIR_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

FOI_SIGMA_MID = 0.141
NOISE_ETA = 200  # mid noise level

# ── Arm taxonomy ──────────────────────────────────────────────────────────────
# Arms that produce valid σ_a (have NLL loss head)
NLL_ARMS = {
    "arm_b_nll", "arm_c_nll_ssim", "arm_d_nll_ssim_ffl",
    "arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb",
    "arm_k_nll_l1", "arm_l_nll_edge_ffl", "arm_m_full_det",
    "arm_n_perc", "arm_o_prelu", "arm_p_best",
}
VAE_ARMS = {"arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb", "arm_p_best"}
KAIR_BASELINES = {"ffdnet", "ircnn", "drunet", "swinir"}

ARM_LABELS = {
    "ffdnet":             "FFDNet",
    "ircnn":              "IRCNN",
    "drunet":             "DRUNet",
    "swinir":             "SwinIR",
    "dncnn_baseline":     "DnCNN",
    "arm_a_l2":           "A: L₂ (MSE)",
    "arm_b_nll":          "B: NLL",
    "arm_c_nll_ssim":     "C: NLL+SSIM",
    "arm_d_nll_ssim_ffl": "D: NLL+SSIM+FFL",
    "arm_e_ppvae":        "E: PP-VAE (linear KL)",
    "arm_f_kl_cyc":       "F: PP-VAE cyc KL",
    "arm_g_kl_fb":        "G: PP-VAE free bits",
    "arm_h_kl_cyc_fb":    "H: PP-VAE cyc+fb",
    "arm_i_l1":           "I: L₁ (MAE)",
    "arm_j_l1_ssim_ffl":  "J: L₁+SSIM+FFL",
    "arm_k_nll_l1":       "K: NLL+L₁",
    "arm_l_nll_edge_ffl": "L: NLL+Edge+FFL",
    "arm_m_full_det":     "M: NLL+SSIM+Edge+FFL",
    "arm_n_perc":         "N: NLL+Perceptual+SSIM+FFL",
    "arm_o_prelu":        "O: NLL+SSIM+FFL (PReLU)",
    "arm_p_best":         "P: PP-VAE best",
}

# Color palette by arm type
def arm_color(k):
    if k in KAIR_BASELINES or k == "dncnn_baseline": return "#2ca02c"
    if k in VAE_ARMS:                                 return "#9467bd"
    if k in NLL_ARMS:                                 return "#ff7f0e"
    return "#1f77b4"  # pixel-norm arms

ARM_ORDER = [
    "ffdnet", "ircnn", "drunet", "swinir",
    "arm_a_l2", "arm_b_nll", "arm_c_nll_ssim", "arm_d_nll_ssim_ffl", "arm_e_ppvae",
    "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb",
    "arm_i_l1", "arm_j_l1_ssim_ffl", "arm_k_nll_l1", "arm_l_nll_edge_ffl",
    "arm_m_full_det", "arm_n_perc", "arm_o_prelu", "arm_p_best",
]

# ── Data helpers ──────────────────────────────────────────────────────────────

def get_test_files():
    """Returns list of (path, label) where label in {Normal, Bacterial, Viral}."""
    files = []
    for f in sorted(glob.glob(os.path.join(DATA, "NORMAL", "*.jpeg")) +
                    glob.glob(os.path.join(DATA, "NORMAL", "*.png"))):
        files.append((f, "Normal"))
    for f in sorted(glob.glob(os.path.join(DATA, "PNEUMONIA", "*.jpeg")) +
                    glob.glob(os.path.join(DATA, "PNEUMONIA", "*.png"))):
        name = os.path.basename(f)
        if name.startswith("BACTERIA"):
            files.append((f, "Bacterial"))
        else:
            files.append((f, "Viral"))
    return files


def load_img(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L").resize((256, 256))) / 255.0


def add_noise(clean: np.ndarray, eta: int = NOISE_ETA, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poisson = rng.poisson(clean * eta) / eta
    gaussian = rng.normal(0, 0.01, clean.shape)
    return np.clip(poisson + gaussian, 0, 1).astype(np.float32)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def t2np(t): return t.squeeze().detach().cpu().float().numpy()

# ── Model loading ──────────────────────────────────────────────────────────────

class DnCNN(torch.nn.Module):
    def __init__(self, depth=20, channels=64, in_ch=1):
        super().__init__()
        layers = [torch.nn.Conv2d(in_ch, channels, 3, padding=1), torch.nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                       torch.nn.BatchNorm2d(channels), torch.nn.ReLU(inplace=True)]
        layers.append(torch.nn.Conv2d(channels, in_ch, 3, padding=1))
        self.body = torch.nn.Sequential(*layers)
    def forward(self, x): return x - self.body(x)


def load_models(arm_list):
    from src.models.ppvae_hformer import PPVAEHformer
    from src.training.config import ABLATION_ARMS, ExperimentConfig
    from models.network_ffdnet import FFDNet
    from models.network_dncnn import IRCNN
    from models.network_unet import UNetRes
    from models.network_swinir import SwinIR

    models = {}
    for arm in arm_list:
        if arm in KAIR_BASELINES:
            ckpt_path = os.path.join(RESULTS, "baselines", arm, "best_model.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {arm}: {ckpt_path}"); continue
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if arm == "ffdnet":
                m = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode="BR")
            elif arm == "ircnn":
                m = IRCNN(in_nc=1, out_nc=1, nc=64)
            elif arm == "drunet":
                m = UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4,
                            act_mode="L", downsample_mode="strideconv",
                            upsample_mode="convtranspose")
            else:  # swinir
                m = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
                           img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180,
                           num_heads=[6,6,6,6,6,6], mlp_ratio=2,
                           upsampler="", resi_connection="1conv")
            m.load_state_dict(state, strict=True)
            m.to(DEVICE).eval(); models[arm] = m
        elif arm == "dncnn_baseline":
            ckpt_path = os.path.join(RESULTS, "dncnn_baseline", "best_model.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {arm}"); continue
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            m = DnCNN().to(DEVICE)
            m.load_state_dict(ckpt["model"]); m.eval(); models[arm] = m
        else:
            ckpt_path = os.path.join(RESULTS, arm, "best_model.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {arm}: no checkpoint"); continue
            cfg = ExperimentConfig()
            for k, v in ABLATION_ARMS[arm].get("model", {}).items():
                setattr(cfg.model, k, v)
            m = PPVAEHformer(
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
            m.load_state_dict(ckpt["model"]); m.eval(); models[arm] = m
        print(f"  ✓ {arm}")
    return models


@torch.no_grad()
def infer(arm, model, noisy_t):
    """Returns (recon np, lsa np or None, z_mu np or None)."""
    x = noisy_t.unsqueeze(0).to(DEVICE)
    if arm in KAIR_BASELINES:
        if arm == "ffdnet":
            sig = torch.full((1,1,1,1), FOI_SIGMA_MID, device=DEVICE)
            out = model(x, sig).clamp(0, 1)
        elif arm == "drunet":
            sig = torch.full_like(x, FOI_SIGMA_MID)
            out = model(torch.cat([x, sig], 1)).clamp(0, 1)
        elif arm == "swinir":
            ws = 8; _, _, h, w = x.shape
            ph, pw = (ws - h%ws)%ws, (ws - w%ws)%ws
            xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
            out = model(xp)[:, :, :h, :w].clamp(0, 1)
        else:
            out = model(x).clamp(0, 1)
        return t2np(out), None, None
    elif arm == "dncnn_baseline":
        return t2np(model(x).clamp(0, 1)), None, None
    else:
        mu, lsa, z_mu, _ = model(x, deterministic=True)
        return t2np(mu.clamp(0, 1)), t2np(lsa), (t2np(z_mu) if z_mu is not None else None)


# ── 1. Fixed qualitative panels ───────────────────────────────────────────────

def make_qualitative_panels(models):
    """3-case × n_col panels per arm, correct σ_a handling."""
    ALL_FILES = get_test_files()

    # Pick representative cases: 3 per class, evenly spaced
    def pick(label, n=3):
        pool = [(p, l) for p, l in ALL_FILES if l == label]
        step = max(1, len(pool) // (n + 1))
        return [pool[step * (i+1)] for i in range(n)]

    cases = {
        "Normal":    pick("Normal",    3),
        "Bacterial": pick("Bacterial", 3),
        "Viral":     pick("Viral",     3),
    }

    qual_dir = OUT_DIR / "qualitative"
    qual_dir.mkdir(exist_ok=True)

    for arm, model in models.items():
        has_uq  = arm in NLL_ARMS
        has_vae = arm in VAE_ARMS
        # column layout
        if has_vae:
            cols = ["Noisy", "Reference", r"$\hat{\mu}$", r"$\hat{\sigma}_a$",
                    r"$|\hat{\mu}-y|$", r"Epist. $\hat{\sigma}_e$"]
        elif has_uq:
            cols = ["Noisy", "Reference", r"$\hat{\mu}$", r"$\hat{\sigma}_a$",
                    r"$|\hat{\mu}-y|$"]
        else:
            cols = ["Noisy", "Reference", r"$\hat{\mu}$", r"$|\hat{\mu}-y|$"]

        for cls_name, cls_cases in cases.items():
            n_rows = len(cls_cases)
            n_cols = len(cols)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(2.9 * n_cols, 3.0 * n_rows),
                                     gridspec_kw={"hspace": 0.06, "wspace": 0.04})
            if n_rows == 1: axes = axes[np.newaxis, :]

            for ci, title in enumerate(cols):
                axes[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=3)

            for ri, (path, _) in enumerate(cls_cases):
                clean = load_img(path).astype(np.float32)
                noisy = add_noise(clean, seed=42 + ri)
                noisy_t = torch.tensor(noisy).float().unsqueeze(0)
                recon, lsa, z_mu = infer(arm, model, noisy_t)
                p = psnr(clean, recon)
                axes[ri, 0].set_ylabel(f"PSNR {p:.2f} dB", fontsize=7.5, labelpad=3)

                axes[ri, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
                axes[ri, 1].imshow(clean, cmap="gray", vmin=0, vmax=1)
                axes[ri, 2].imshow(recon, cmap="gray", vmin=0, vmax=1)

                col = 3
                if has_uq and lsa is not None:
                    sigma_a = np.exp(0.5 * lsa)
                    vmax_sa = max(0.02, float(np.percentile(sigma_a, 99)))
                    im = axes[ri, col].imshow(sigma_a.clip(0, vmax_sa),
                                              cmap="inferno", vmin=0, vmax=vmax_sa)
                    div = make_axes_locatable(axes[ri, col])
                    cax = div.append_axes("right", size="6%", pad=0.04)
                    plt.colorbar(im, cax=cax)
                    col += 1

                err = np.abs(recon - clean)
                vmax_err = max(0.05, float(np.percentile(err, 99)))
                im2 = axes[ri, col].imshow(err.clip(0, vmax_err),
                                           cmap="hot", vmin=0, vmax=vmax_err)
                div2 = make_axes_locatable(axes[ri, col])
                cax2 = div2.append_axes("right", size="6%", pad=0.04)
                plt.colorbar(im2, cax=cax2)
                col += 1

                if has_vae and col < n_cols:
                    x4d = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        try:
                            ep = model.epistemic_map(x4d, n_samples=8)
                            ep_np = ep.squeeze().detach().cpu().float().numpy()
                        except Exception:
                            ep_np = None
                    if ep_np is not None:
                        vmax_ep = max(0.01, float(np.percentile(ep_np, 99)))
                        im_ep = axes[ri, col].imshow(ep_np.clip(0, vmax_ep),
                                                     cmap="plasma", vmin=0, vmax=vmax_ep)
                        div_ep = make_axes_locatable(axes[ri, col])
                        cax_ep = div_ep.append_axes("right", size="6%", pad=0.04)
                        plt.colorbar(im_ep, cax=cax_ep)
                    else:
                        axes[ri, col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                           transform=axes[ri, col].transAxes,
                                           fontsize=9, color="white")
                        axes[ri, col].set_facecolor("#333333")

            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            lbl = ARM_LABELS.get(arm, arm)
            fig.suptitle(f"{lbl} — {cls_name} cases (η={NOISE_ETA}, mid noise)",
                         fontsize=10, fontweight="bold", y=1.01)
            out_path = qual_dir / f"qualitative_{arm}_{cls_name.lower()}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ {out_path.name}")


# ── 2. Fixed dose curve ────────────────────────────────────────────────────────

def make_dose_curve():
    csv_path = Path(RESULTS) / "evaluation" / "metrics_summary.csv"
    if not csv_path.exists():
        print("  [SKIP] metrics_summary.csv not found"); return

    rows = list(csv.DictReader(open(csv_path)))

    def get_vals(arm, metric):
        out = {}
        for nl in ["low", "mid", "high"]:
            r = [x for x in rows if x["arm"] == arm and x["noise_level"] == nl]
            out[nl] = (float(r[0][f"{metric}_mean"]), float(r[0][f"{metric}_sd"])) if r else (np.nan, 0)
        return out

    noise_order = ["low", "mid", "high"]
    arms_present = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]

    # Separate baselines vs ablation arms for two panels
    baselines = [a for a in arms_present if a in KAIR_BASELINES or a == "dncnn_baseline"]
    ablation  = [a for a in arms_present if a not in KAIR_BASELINES and a != "dncnn_baseline"]

    # colour by type
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(ablation), 5)))

    for metric, ylabel in [("psnr", "PSNR (dB)"), ("ssim", "SSIM")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        legend_handles = []

        for i, arm in enumerate(ablation):
            vals = get_vals(arm, metric)
            means = [vals[nl][0] for nl in noise_order]
            stds  = [vals[nl][1] for nl in noise_order]
            col   = arm_color(arm)
            ls    = "--" if arm in VAE_ARMS else "-"
            lw    = 1.4
            h = ax.errorbar(noise_order, means, yerr=stds, color=col,
                            linestyle=ls, linewidth=lw, marker="o", markersize=5,
                            capsize=3, label=ARM_LABELS.get(arm, arm))
            legend_handles.append(h)

        for arm in baselines:
            vals = get_vals(arm, metric)
            means = [vals[nl][0] for nl in noise_order]
            stds  = [vals[nl][1] for nl in noise_order]
            h = ax.errorbar(noise_order, means, yerr=stds, color=arm_color(arm),
                            linestyle=":", linewidth=2.2, marker="s", markersize=6,
                            capsize=3, label=ARM_LABELS.get(arm, arm))
            legend_handles.append(h)

        ax.set_xlabel("Noise severity (η = 100 / 200 / 300)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Reconstruction {ylabel} vs Noise Severity — all 21 models", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Legend outside, two columns
        legend = ax.legend(
            handles=legend_handles,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
            fontsize=7.5, ncol=1, framealpha=0.9,
            title="Model", title_fontsize=8,
        )

        fig.savefig(OUT_DIR / f"dose_curve_{metric}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ dose_curve_{metric}.png")


# ── 3. Fixed ablation barchart ─────────────────────────────────────────────────

def make_ablation_barchart():
    csv_path = Path(RESULTS) / "evaluation" / "metrics_summary.csv"
    if not csv_path.exists():
        print("  [SKIP] metrics_summary.csv not found"); return

    rows = list(csv.DictReader(open(csv_path)))
    mid_rows = [r for r in rows if r["noise_level"] == "mid"]

    arms_present = [a for a in ARM_ORDER if any(r["arm"] == a for r in mid_rows)]
    # Sort by PSNR descending
    def get_psnr(a):
        r = [x for x in mid_rows if x["arm"] == a]
        return float(r[0]["psnr_mean"]) if r else 0.0

    arms_sorted = sorted(arms_present, key=get_psnr)  # ascending for horizontal
    labels  = [ARM_LABELS.get(a, a) for a in arms_sorted]
    psnr_v  = [get_psnr(a) for a in arms_sorted]
    psnr_sd = [float(next(r["psnr_sd"] for r in mid_rows if r["arm"]==a)) for a in arms_sorted]
    colors  = [arm_color(a) for a in arms_sorted]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(arms_sorted) + 1.5))
    y = np.arange(len(arms_sorted))
    bars = ax.barh(y, psnr_v, xerr=psnr_sd, color=colors, alpha=0.85,
                   capsize=3, height=0.7)
    for bar, val, sd in zip(bars, psnr_v, psnr_sd):
        ax.text(val + sd + 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("PSNR (dB) — mid noise (η = 200)", fontsize=11)
    ax.set_title("Ablation Study: Reconstruction Quality at Mid Noise\nAll 21 models (16 ablation arms + 5 retrained baselines)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(left=max(0, min(psnr_v) - 1.5), right=max(psnr_v) + 1.5)

    # Legend for colour coding
    legend_elems = [
        plt.Rectangle((0,0),1,1, color="#1f77b4", label="Pixel-norm (L₁/L₂/Charb)"),
        plt.Rectangle((0,0),1,1, color="#ff7f0e", label="NLL arms"),
        plt.Rectangle((0,0),1,1, color="#9467bd", label="VAE arms"),
        plt.Rectangle((0,0),1,1, color="#2ca02c", label="Retrained baselines"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ablation_barchart_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ ablation_barchart_v2.png")

    # Second chart: SSIM
    def get_ssim(a):
        r = [x for x in mid_rows if x["arm"] == a]
        return float(r[0]["ssim_mean"]) if r else 0.0

    arms_ssim = sorted(arms_present, key=get_ssim)
    ssim_v  = [get_ssim(a) for a in arms_ssim]
    ssim_sd = [float(next(r["ssim_sd"] for r in mid_rows if r["arm"]==a)) for a in arms_ssim]
    colors2 = [arm_color(a) for a in arms_ssim]
    labels2 = [ARM_LABELS.get(a, a) for a in arms_ssim]

    fig2, ax2 = plt.subplots(figsize=(7, 0.45 * len(arms_ssim) + 1.5))
    bars2 = ax2.barh(np.arange(len(arms_ssim)), ssim_v, xerr=ssim_sd,
                     color=colors2, alpha=0.85, capsize=3, height=0.7)
    for bar, val in zip(bars2, ssim_v):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)
    ax2.set_yticks(np.arange(len(arms_ssim)))
    ax2.set_yticklabels(labels2, fontsize=9)
    ax2.set_xlabel("SSIM — mid noise (η = 200)", fontsize=11)
    ax2.set_title("Ablation Study: SSIM at Mid Noise", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.legend(handles=legend_elems, loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "ablation_barchart_ssim.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ ablation_barchart_ssim.png")


# ── 4. Subgroup comparison (Bacterial / Viral / Normal) ───────────────────────

def make_subgroup_panels(models):
    """3×4 panel: 3 classes × 4 best models, one row per subgroup case."""
    SHOW_ARMS = ["arm_a_l2", "arm_d_nll_ssim_ffl", "arm_h_kl_cyc_fb", "ircnn"]
    SHOW_LABELS = ["Arm A\n(L₂)", "Arm D\n(NLL+SSIM+FFL)", "Arm H\n(VAE cyc+fb)", "IRCNN"]

    avail = [a for a in SHOW_ARMS if a in models]
    if not avail:
        print("  [SKIP] no subgroup models available"); return

    ALL_FILES = get_test_files()

    def pick_cases(label, n=3):
        pool = [(p, l) for p, l in ALL_FILES if l == label]
        step = max(1, len(pool) // (n + 1))
        return [pool[step * (i+1)] for i in range(n)]

    subgroups = [
        ("Normal",    pick_cases("Normal",    3)),
        ("Bacterial", pick_cases("Bacterial", 3)),
        ("Viral",     pick_cases("Viral",     3)),
    ]

    for sg_label, sg_cases in subgroups:
        n_rows = len(sg_cases)
        # columns: Noisy | Clean | [arm1 recon | arm2 recon | ...]
        n_arm_cols = len(avail)
        n_cols = 2 + n_arm_cols

        col_titles = ["Noisy\nInput", "Reference"] + \
                     [SHOW_LABELS[SHOW_ARMS.index(a)] for a in avail]

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.8 * n_cols, 3.0 * n_rows),
                                 gridspec_kw={"hspace": 0.06, "wspace": 0.04})
        if n_rows == 1: axes = axes[np.newaxis, :]

        for ci, title in enumerate(col_titles):
            axes[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=3)

        for ri, (path, _) in enumerate(sg_cases):
            clean = load_img(path).astype(np.float32)
            noisy = add_noise(clean, seed=100 + ri)

            axes[ri, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
            axes[ri, 1].imshow(clean, cmap="gray", vmin=0, vmax=1)

            for ci, arm in enumerate(avail):
                noisy_t = torch.tensor(noisy).float().unsqueeze(0)
                recon, lsa, _ = infer(arm, models[arm], noisy_t)
                p = psnr(clean, recon)
                axes[ri, 2 + ci].imshow(recon, cmap="gray", vmin=0, vmax=1)
                axes[ri, 2 + ci].text(0.03, 0.04, f"{p:.2f} dB",
                    transform=axes[ri, 2+ci].transAxes,
                    color="yellow", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))

        # Row label: filename stem (shows patient ID)
        for ri, (path, _) in enumerate(sg_cases):
            axes[ri, 0].set_ylabel(os.path.basename(path)[:18], fontsize=6.5,
                                   labelpad=3, rotation=0, ha="right", va="center")

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            f"Subgroup: {sg_label} — Reconstruction Comparison (η={NOISE_ETA})\n"
            f"Columns: Noisy | Reference | {' | '.join(SHOW_LABELS[:n_arm_cols])}",
            fontsize=10, fontweight="bold", y=1.01,
        )
        out_path = OUT_DIR / f"subgroup_{sg_label.lower()}_comparison.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out_path.name}")

    # ── Quantitative subgroup bar chart from CSV ──────────────────────────────
    csv_path = Path(RESULTS) / "evaluation" / "metrics_summary.csv"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(open(csv_path)))

    # Check if subgroup columns exist
    if rows and "normal_psnr_mean" in rows[0]:
        mid_rows = [r for r in rows if r["noise_level"] == "mid"]
        arms_p = [a for a in ARM_ORDER if any(r["arm"]==a for r in mid_rows)]

        labels_sg = ["Normal", "Bacterial", "Viral"]
        col_keys  = ["normal_psnr_mean", "bacteria_psnr_mean", "viral_psnr_mean"]

        x = np.arange(len(arms_p))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(10, 0.7*len(arms_p)), 5))
        colors3 = ["#1f77b4", "#d62728", "#ff7f0e"]
        for bi, (sg, ck) in enumerate(zip(labels_sg, col_keys)):
            vals = []
            for a in arms_p:
                r = next((x for x in mid_rows if x["arm"]==a), None)
                vals.append(float(r[ck]) if r and ck in r and r[ck] else np.nan)
            ax.bar(x + bi*w, vals, w, color=colors3[bi], alpha=0.85, label=sg)

        ax.set_xticks(x + w); ax.set_xticklabels(
            [ARM_LABELS.get(a, a) for a in arms_p], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.set_title("Subgroup PSNR: Normal vs Bacterial vs Viral Pneumonia (mid noise)", fontsize=10)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "subgroup_psnr_barchart.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ subgroup_psnr_barchart.png")
    else:
        print("  [INFO] subgroup columns not in CSV — skipping bar chart")


# ── 5. Latent space analysis ──────────────────────────────────────────────────

def make_latent_space_analysis(models):
    """UMAP/t-SNE of VAE latent codes + ARI score."""
    try:
        import umap.umap_ as umap_mod
        HAS_UMAP = True
    except ImportError:
        try:
            import umap
            umap_mod = umap
            HAS_UMAP = True
        except ImportError:
            HAS_UMAP = False
            print("  [WARN] umap-learn not installed, will use t-SNE only")

    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    ALL_FILES = get_test_files()
    # Subsample for speed: max 200 per class
    rng = np.random.default_rng(0)
    def subsample(label, n=150):
        pool = [(p, l) for p, l in ALL_FILES if l == label]
        idx = rng.choice(len(pool), min(n, len(pool)), replace=False)
        return [pool[i] for i in sorted(idx)]

    sample_files = (subsample("Normal", 150) +
                    subsample("Bacterial", 150) +
                    subsample("Viral", 150))
    true_labels = [{"Normal":0, "Bacterial":1, "Viral":2}[l] for _, l in sample_files]
    label_names = {0:"Normal", 1:"Bacterial", 2:"Viral"}
    label_colors = {0:"#1f77b4", 1:"#d62728", 2:"#ff7f0e"}

    vae_models = {k: v for k, v in models.items() if k in VAE_ARMS}
    if not vae_models:
        print("  [SKIP] no VAE models loaded"); return

    latent_dir = OUT_DIR / "latent_space"
    latent_dir.mkdir(exist_ok=True)

    ari_results = {}

    for arm, model in vae_models.items():
        print(f"  Extracting latent codes: {arm} ({len(sample_files)} images)...")
        latent_codes = []
        for path, _ in sample_files:
            clean = load_img(path).astype(np.float32)
            noisy = add_noise(clean)
            # model expects [B, C, H, W]; add both channel and batch dims
            noisy_t = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, _, z_mu, _ = model(noisy_t, deterministic=True)
            if z_mu is None:
                # fallback: use reconstruction features
                latent_codes.append(np.zeros(64))
                continue
            # Global average pool the spatial latent map → 1D vector
            z_flat = z_mu.squeeze(0).mean(dim=(1,2)).detach().cpu().float().numpy()
            latent_codes.append(z_flat)

        Z = np.array(latent_codes)
        if Z.ndim == 1 or Z.shape[0] < 10:
            continue
        y = np.array(true_labels[:len(Z)])

        # ── k-means ARI ──
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km_labels = km.fit_predict(Z)
        ari = adjusted_rand_score(y, km_labels)
        sil = silhouette_score(Z, y) if len(set(y)) > 1 else float("nan")
        ari_results[arm] = {"ARI": ari, "Silhouette": sil, "n": len(Z)}
        print(f"    {arm}: ARI={ari:.4f}  Silhouette={sil:.4f}")

        # ── t-SNE ──
        print(f"    Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=40, random_state=42,
                    max_iter=1000, learning_rate="auto", init="pca")
        Z_2d = tsne.fit_transform(Z)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, embed, title in zip(axes, [Z_2d, None], ["t-SNE", "UMAP"]):
            if embed is None:
                if not HAS_UMAP:
                    ax.axis("off"); continue
                print(f"    Running UMAP...")
                reducer = umap_mod.UMAP(n_components=2, n_neighbors=30,
                                        min_dist=0.1, random_state=42)
                embed = reducer.fit_transform(Z)

            for class_id, cls_name in label_names.items():
                mask = y == class_id
                ax.scatter(embed[mask, 0], embed[mask, 1],
                           c=label_colors[class_id], label=cls_name,
                           alpha=0.65, s=18, linewidths=0)
            ax.set_title(f"{title} — {ARM_LABELS.get(arm, arm)}\n"
                         f"ARI={ari:.3f}  Silhouette={sil:.3f}",
                         fontsize=10)
            ax.legend(fontsize=9, framealpha=0.9)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            f"Latent Space Visualisation: {ARM_LABELS.get(arm, arm)}\n"
            f"n={len(Z)} images  •  k-means ARI vs Normal/Bacterial/Viral ground truth",
            fontsize=10, fontweight="bold",
        )
        fig.savefig(latent_dir / f"latent_{arm}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    ✓ latent_{arm}.png")

    # ── ARI comparison bar chart ──
    if ari_results:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        arms_r  = list(ari_results.keys())
        ari_v   = [ari_results[a]["ARI"] for a in arms_r]
        sil_v   = [ari_results[a]["Silhouette"] for a in arms_r]
        x = np.arange(len(arms_r))
        w = 0.38
        ax.bar(x - w/2, ari_v, w, color="#9467bd", alpha=0.85, label="Adj. Rand Index")
        ax.bar(x + w/2, sil_v, w, color="#2ca02c", alpha=0.85, label="Silhouette Score")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms_r], fontsize=9)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Latent Space Clustering Quality vs Ground Truth\n"
                     "(3 classes: Normal / Bacterial / Viral)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(-0.3, 1.0)
        for i, (av, sv) in enumerate(zip(ari_v, sil_v)):
            ax.text(i - w/2, av + 0.02, f"{av:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, sv + 0.02, f"{sv:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(latent_dir / "latent_ari_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ latent_ari_comparison.png")

        # Write ARI table
        with open(latent_dir / "ari_results.csv", "w") as f:
            f.write("arm,ARI,Silhouette,n\n")
            for a, v in ari_results.items():
                f.write(f"{a},{v['ARI']:.6f},{v['Silhouette']:.6f},{v['n']}\n")
        print("  ✓ ari_results.csv")


# ── 6. Fixed uncertainty calibration panel ────────────────────────────────────

def make_uncertainty_panel_fixed(models):
    """NLL arms only: 4-row × 6-col panel sorted by NLL quality."""
    NLL_show = ["arm_b_nll", "arm_d_nll_ssim_ffl", "arm_h_kl_cyc_fb",
                "arm_k_nll_l1", "arm_l_nll_edge_ffl", "arm_n_perc"]
    avail = [a for a in NLL_show if a in models]
    if not avail:
        print("  [SKIP] no NLL models for uncertainty panel"); return

    ALL_FILES = get_test_files()
    cases = [(p, l) for p, l in ALL_FILES if l == "Bacterial"]
    step = max(1, len(cases) // 5)
    show_cases = [cases[step * (i+1)] for i in range(4)]

    n_rows = len(show_cases)
    n_cols = 5  # Noisy | Clean | Recon | σ_a | |Error|

    fig, big_axes = plt.subplots(n_rows, n_cols * len(avail),
        figsize=(2.5 * n_cols * len(avail), 3.2 * n_rows))

    # Simpler: one figure per arm, rows = cases
    for arm in avail:
        fig2, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(2.8 * n_cols, 3.0 * n_rows),
                                  gridspec_kw={"hspace": 0.05, "wspace": 0.04})
        if n_rows == 1: axes = axes[np.newaxis, :]
        col_titles = ["Noisy", "Reference", r"$\hat{\mu}$",
                      r"$\hat{\sigma}_a$", r"$|\hat{\mu}-y|$"]
        for ci, t in enumerate(col_titles):
            axes[0, ci].set_title(t, fontsize=9, fontweight="bold")

        for ri, (path, _) in enumerate(show_cases):
            clean = load_img(path).astype(np.float32)
            noisy = add_noise(clean, seed=200 + ri)
            noisy_t = torch.tensor(noisy).float().unsqueeze(0)
            recon, lsa, _ = infer(arm, models[arm], noisy_t)
            sigma_a = np.exp(0.5 * lsa)
            err     = np.abs(recon - clean)
            vmax_sa  = max(0.02, float(np.percentile(sigma_a, 99)))
            vmax_err = max(0.05, float(np.percentile(err, 99)))
            p = psnr(clean, recon)
            axes[ri, 0].set_ylabel(f"PSNR {p:.2f} dB", fontsize=7.5)
            axes[ri, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
            axes[ri, 1].imshow(clean, cmap="gray", vmin=0, vmax=1)
            axes[ri, 2].imshow(recon, cmap="gray", vmin=0, vmax=1)
            im3 = axes[ri, 3].imshow(sigma_a.clip(0, vmax_sa),
                                     cmap="inferno", vmin=0, vmax=vmax_sa)
            im4 = axes[ri, 4].imshow(err.clip(0, vmax_err),
                                     cmap="hot", vmin=0, vmax=vmax_err)
            for ax_cb, im_cb in [(axes[ri, 3], im3), (axes[ri, 4], im4)]:
                div_cb = make_axes_locatable(ax_cb)
                cax_cb = div_cb.append_axes("right", size="6%", pad=0.04)
                plt.colorbar(im_cb, cax=cax_cb)

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])

        fig2.suptitle(f"Uncertainty maps: {ARM_LABELS.get(arm, arm)} — Bacterial Pneumonia cases",
                      fontsize=10, fontweight="bold", y=1.01)
        fig2.savefig(OUT_DIR / f"uncertainty_{arm}.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  ✓ uncertainty_{arm}.png")

    plt.close(fig)  # close the empty placeholder


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Fixed ablation barchart + dose curve (no model loading)")
    print("=" * 60)
    make_ablation_barchart()
    make_dose_curve()

    print("\n" + "=" * 60)
    print("Step 2: Loading models...")
    print("=" * 60)
    ALL_ARMS = ARM_ORDER + ["dncnn_baseline"]
    models = load_models(ALL_ARMS)
    print(f"Loaded: {list(models.keys())}\n")

    print("\n" + "=" * 60)
    print("Step 3: Fixed qualitative panels")
    print("=" * 60)
    make_qualitative_panels(models)

    print("\n" + "=" * 60)
    print("Step 4: Subgroup panels (Bacterial / Viral / Normal)")
    print("=" * 60)
    make_subgroup_panels(models)

    print("\n" + "=" * 60)
    print("Step 5: Fixed uncertainty calibration panels")
    print("=" * 60)
    make_uncertainty_panel_fixed(models)

    print("\n" + "=" * 60)
    print("Step 6: Latent space analysis (VAE arms)")
    print("=" * 60)
    make_latent_space_analysis(models)

    print(f"\n✓ All figures saved to {OUT_DIR}")
