"""
gen_vae_qual_only.py — Regenerate ONLY VAE arm qualitative panels with correct
epistemic uncertainty (MC variance over 8 stochastic samples).

Fixes the N/A epistemic column bug: replaces model.epistemic_map() call with
manual Monte Carlo variance across deterministic=False forward passes.

Output: /rds/user/stm43/hpc-work/ppvae_results/figures_v2/qualitative/
"""
from __future__ import annotations
import os, sys, glob, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

PROJECT   = "/rds/user/stm43/hpc-work/ppvae_hformer"
RESULTS   = "/rds/user/stm43/hpc-work/ppvae_results"
DATA      = "/rds/user/stm43/hpc-work/chest_xray/test"
OUT_DIR   = Path(RESULTS) / "figures_v2" / "qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, PROJECT)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

FOI_A_MID, FOI_B_MID = 0.03, 0.005
NOISE_ETA = 200

VAE_ARMS = ["arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb", "arm_p_best"]
ARM_LABELS = {
    "arm_e_ppvae":      "E: PP-VAE (Linear KL)",
    "arm_f_kl_cyc":     "F: PP-VAE (Cyclical KL)",
    "arm_g_kl_fb":      "G: PP-VAE (Free-Bits KL)",
    "arm_h_kl_cyc_fb":  "H: PP-VAE (Cyclical + Free-Bits KL)",
    "arm_p_best":       "P: PP-VAE (Full Model, PReLU)",
}

def load_img(path):
    return np.array(Image.open(path).convert("L").resize((256, 256))) / 255.0

def add_noise(clean, seed=42):
    rng = np.random.default_rng(seed)
    variance = FOI_A_MID * np.clip(clean, 0, None) + FOI_B_MID
    return np.clip(clean + rng.standard_normal(clean.shape) * np.sqrt(variance), 0, 1).astype(np.float32)

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))

def t2np(t): return t.squeeze().detach().cpu().float().numpy()

def get_test_files():
    files = []
    for f in sorted(glob.glob(os.path.join(DATA, "NORMAL", "*.jpeg")) +
                    glob.glob(os.path.join(DATA, "NORMAL", "*.png"))):
        files.append((f, "Normal"))
    for f in sorted(glob.glob(os.path.join(DATA, "PNEUMONIA", "*.jpeg")) +
                    glob.glob(os.path.join(DATA, "PNEUMONIA", "*.png"))):
        name = os.path.basename(f)
        files.append((f, "Bacterial" if name.startswith("BACTERIA") else "Viral"))
    return files

def load_vae_model(arm):
    from src.models.ppvae_hformer import PPVAEHformer
    from src.training.config import ABLATION_ARMS, ExperimentConfig
    ckpt_path = os.path.join(RESULTS, arm, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {arm}: no checkpoint")
        return None
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
    m.load_state_dict(ckpt["model"]); m.eval()
    print(f"  ✓ loaded {arm}")
    return m

def make_panel(arm, model, all_files):
    cols = ["Noisy", "Reference", r"$\hat{\mu}$", r"$\hat{\sigma}_a$",
            r"$|\hat{\mu}-y|$", r"Epist. $\hat{\sigma}_e$"]
    n_cols = len(cols)

    def pick(label, n=3):
        pool = [(p, l) for p, l in all_files if l == label]
        step = max(1, len(pool) // (n + 1))
        return [pool[step * (i + 1)] for i in range(n)]

    cases = {
        "Normal":    pick("Normal", 3),
        "Bacterial": pick("Bacterial", 3),
        "Viral":     pick("Viral", 3),
    }

    for cls_name, cls_cases in cases.items():
        n_rows = len(cls_cases)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.9 * n_cols, 3.0 * n_rows),
                                 gridspec_kw={"hspace": 0.06, "wspace": 0.04})
        if n_rows == 1: axes = axes[np.newaxis, :]
        for ci, title in enumerate(cols):
            axes[0, ci].set_title(title, fontsize=9, fontweight="bold", pad=3)

        for ri, (path, _) in enumerate(cls_cases):
            clean  = load_img(path).astype(np.float32)
            noisy  = add_noise(clean, seed=42 + ri)
            noisy_t = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                mu, lsa, _, _ = model(noisy_t, deterministic=True)
                recon = t2np(mu.clamp(0, 1))

            p = psnr(clean, recon)
            axes[ri, 0].set_ylabel(f"PSNR {p:.2f} dB", fontsize=7.5, labelpad=3)
            axes[ri, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
            axes[ri, 1].imshow(clean, cmap="gray", vmin=0, vmax=1)
            axes[ri, 2].imshow(recon, cmap="gray", vmin=0, vmax=1)

            # σ_a (aleatoric)
            sigma_a = np.exp(0.5 * t2np(lsa))
            vmax_sa = max(0.02, float(np.percentile(sigma_a, 99)))
            im_sa = axes[ri, 3].imshow(sigma_a.clip(0, vmax_sa),
                                       cmap="inferno", vmin=0, vmax=vmax_sa)
            div = make_axes_locatable(axes[ri, 3])
            plt.colorbar(im_sa, cax=div.append_axes("right", size="6%", pad=0.04))

            # |error|
            err = np.abs(recon - clean)
            vmax_err = max(0.05, float(np.percentile(err, 99)))
            im_err = axes[ri, 4].imshow(err.clip(0, vmax_err),
                                        cmap="hot", vmin=0, vmax=vmax_err)
            div2 = make_axes_locatable(axes[ri, 4])
            plt.colorbar(im_err, cax=div2.append_axes("right", size="6%", pad=0.04))

            # Epistemic (Monte Carlo variance over 8 stochastic samples)
            ep_np = None
            with torch.no_grad():
                try:
                    preds = []
                    for _ in range(16):
                        mu_s, _, _, _ = model(noisy_t, deterministic=False)
                        preds.append(mu_s.squeeze().detach().cpu().float())
                    ep_np = torch.stack(preds).std(dim=0).numpy()
                except Exception as exc:
                    print(f"    [WARN] epistemic failed ({arm}, {cls_name} row{ri}): {exc}")

            if ep_np is not None:
                vmax_ep = max(float(np.percentile(ep_np, 95)), 1e-12)
                im_ep = axes[ri, 5].imshow(ep_np.clip(0, vmax_ep),
                                           cmap="plasma", vmin=0, vmax=vmax_ep)
                div_ep = make_axes_locatable(axes[ri, 5])
                plt.colorbar(im_ep, cax=div_ep.append_axes("right", size="6%", pad=0.04))
            else:
                axes[ri, 5].text(0.5, 0.5, "N/A", ha="center", va="center",
                                 transform=axes[ri, 5].transAxes, fontsize=9, color="white")
                axes[ri, 5].set_facecolor("#333333")

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])

        lbl = ARM_LABELS.get(arm, arm)
        fig.suptitle(f"{lbl} — {cls_name} cases", fontsize=10, fontweight="bold")
        fig.subplots_adjust(top=0.92, hspace=0.06, wspace=0.04)
        out_path = OUT_DIR / f"qualitative_{arm}_{cls_name.lower()}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"    ✓ {out_path.name}")

if __name__ == "__main__":
    all_files = get_test_files()
    print(f"Test images: {len(all_files)}")
    for arm in VAE_ARMS:
        print(f"\n[{arm}]")
        model = load_vae_model(arm)
        if model is None:
            continue
        make_panel(arm, model, all_files)
        del model
        torch.cuda.empty_cache()
    print(f"\n✓ Done. Figures in {OUT_DIR}")
