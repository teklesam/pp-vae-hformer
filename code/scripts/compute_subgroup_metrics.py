"""
compute_subgroup_metrics.py — Per-subgroup PSNR/SSIM for all 21 models.

Reads per-image denoised outputs from each model checkpoint and computes:
  - normal_psnr_mean/sd, bacterial_psnr_mean/sd, viral_psnr_mean/sd
  - normal_ssim_mean/sd, bacterial_ssim_mean/sd, viral_ssim_mean/sd

Output:
  /rds/user/stm43/hpc-work/ppvae_results/evaluation/subgroup_metrics.csv

Usage:
  sbatch slurm/compute_subgroup.sh
"""

from __future__ import annotations
import os, sys, glob, math, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT   = "/rds/user/stm43/hpc-work/ppvae_hformer"
RESULTS   = "/rds/user/stm43/hpc-work/ppvae_results"
DATA      = "/rds/user/stm43/hpc-work/chest_xray/test"
KAIR_ROOT = "/rds/user/stm43/hpc-work/KAIR"
OUT_CSV   = Path(RESULTS) / "evaluation" / "subgroup_metrics.csv"

sys.path.insert(0, PROJECT)
sys.path.insert(0, KAIR_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

FOI_SIGMA_MID = 0.141
# FOI mid preset — must match evaluate_all.py (a=0.03, b=0.005)
FOI_A_MID = 0.03
FOI_B_MID = 0.005

NLL_ARMS = {
    "arm_b_nll", "arm_c_nll_ssim", "arm_d_nll_ssim_ffl",
    "arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb",
    "arm_k_nll_l1", "arm_l_nll_edge_ffl", "arm_m_full_det",
    "arm_n_perc", "arm_o_prelu", "arm_p_best",
}
VAE_ARMS = {"arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb", "arm_p_best"}
KAIR_BASELINES = {"ffdnet", "ircnn", "drunet", "swinir"}

ARM_ORDER = [
    "arm_a_l2", "arm_b_nll", "arm_c_nll_ssim", "arm_d_nll_ssim_ffl",
    "arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb",
    "arm_i_l1", "arm_j_l1_ssim_ffl", "arm_k_nll_l1", "arm_l_nll_edge_ffl",
    "arm_m_full_det", "arm_n_perc", "arm_o_prelu", "arm_p_best",
    "dncnn_baseline", "ffdnet", "ircnn", "drunet", "swinir",
]


def get_test_files():
    files = []
    for f in sorted(glob.glob(os.path.join(DATA, "NORMAL", "*.jpeg")) +
                    glob.glob(os.path.join(DATA, "NORMAL", "*.png"))):
        files.append((f, "normal"))
    for f in sorted(glob.glob(os.path.join(DATA, "PNEUMONIA", "*.jpeg")) +
                    glob.glob(os.path.join(DATA, "PNEUMONIA", "*.png"))):
        name = os.path.basename(f)
        label = "bacterial" if name.startswith("BACTERIA") else "viral"
        files.append((f, label))
    return files


def load_img(path):
    return np.array(Image.open(path).convert("L").resize((256, 256))) / 255.0


def add_noise(clean, seed=42, a=FOI_A_MID, b=FOI_B_MID):
    """FOI Poisson-Gaussian noise — matches evaluate_all.py mid preset."""
    rng = np.random.default_rng(seed)
    variance = a * np.clip(clean, 0, None) + b
    noise = rng.standard_normal(clean.shape) * np.sqrt(variance)
    return np.clip(clean + noise, 0, 1).astype(np.float32)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def ssim_np(a, b, win=11):
    from scipy.ndimage import uniform_filter
    mu_a = uniform_filter(a, win); mu_b = uniform_filter(b, win)
    sig_a = uniform_filter(a**2, win) - mu_a**2
    sig_b = uniform_filter(b**2, win) - mu_b**2
    sig_ab = uniform_filter(a*b, win) - mu_a*mu_b
    C1, C2 = (0.01)**2, (0.03)**2
    num = (2*mu_a*mu_b + C1) * (2*sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)
    return float(np.mean(num / den))


def t2np(t):
    return t.squeeze().detach().cpu().float().numpy()


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


def load_model(arm):
    from src.models.ppvae_hformer import PPVAEHformer
    from src.training.config import ABLATION_ARMS, ExperimentConfig
    from models.network_ffdnet import FFDNet
    from models.network_dncnn import IRCNN
    from models.network_unet import UNetRes
    from models.network_swinir import SwinIR

    if arm in KAIR_BASELINES:
        ckpt_path = os.path.join(RESULTS, "baselines", arm, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {arm}"); return None
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
        else:
            m = SwinIR(upscale=1, in_chans=1, img_size=128, window_size=8,
                       img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180,
                       num_heads=[6,6,6,6,6,6], mlp_ratio=2,
                       upsampler="", resi_connection="1conv")
        m.load_state_dict(state, strict=True)
        return m.to(DEVICE).eval()

    elif arm == "dncnn_baseline":
        ckpt_path = os.path.join(RESULTS, "dncnn_baseline", "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {arm}"); return None
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        m = DnCNN().to(DEVICE)
        m.load_state_dict(ckpt["model"]); return m.eval()

    else:
        ckpt_path = os.path.join(RESULTS, arm, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {arm}"); return None
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
        m.load_state_dict(ckpt["model"]); return m.eval()


@torch.no_grad()
def infer(arm, model, x):
    """x: [1,1,H,W] tensor. Returns recon np array."""
    if arm in KAIR_BASELINES:
        if arm == "ffdnet":
            sig = torch.full((1,1,1,1), FOI_SIGMA_MID, device=DEVICE)
            return t2np(model(x, sig).clamp(0, 1))
        elif arm == "drunet":
            sig = torch.full_like(x, FOI_SIGMA_MID)
            return t2np(model(torch.cat([x, sig], 1)).clamp(0, 1))
        elif arm == "swinir":
            ws = 8; _, _, h, w = x.shape
            ph, pw = (ws - h%ws)%ws, (ws - w%ws)%ws
            xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
            return t2np(model(xp)[:, :, :h, :w].clamp(0, 1))
        else:
            return t2np(model(x).clamp(0, 1))
    elif arm == "dncnn_baseline":
        return t2np(model(x).clamp(0, 1))
    else:
        mu, _, _, _ = model(x, deterministic=True)
        return t2np(mu.clamp(0, 1))


if __name__ == "__main__":
    all_files = get_test_files()
    print(f"Total test images: {len(all_files)}")
    counts = {"normal": 0, "bacterial": 0, "viral": 0}
    for _, l in all_files: counts[l] += 1
    print(f"  Normal: {counts['normal']}, Bacterial: {counts['bacterial']}, Viral: {counts['viral']}")

    results = []
    for arm in ARM_ORDER:
        print(f"\n[{arm}]")
        model = load_model(arm)
        if model is None:
            continue

        per_label = {"normal": {"psnr": [], "ssim": []},
                     "bacterial": {"psnr": [], "ssim": []},
                     "viral": {"psnr": [], "ssim": []}}

        for i, (path, label) in enumerate(all_files):
            if i % 50 == 0:
                print(f"  {i}/{len(all_files)} ...", flush=True)
            clean = load_img(path).astype(np.float32)
            noisy = add_noise(clean, seed=i + 1000)  # FOI mid preset
            x = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            recon = infer(arm, model, x)
            per_label[label]["psnr"].append(psnr(clean, recon))
            per_label[label]["ssim"].append(ssim_np(clean, recon))

        row = {"arm": arm}
        for label in ["normal", "bacterial", "viral"]:
            ps = per_label[label]["psnr"]
            ss = per_label[label]["ssim"]
            row[f"{label}_psnr_mean"] = float(np.mean(ps)) if ps else float("nan")
            row[f"{label}_psnr_sd"]   = float(np.std(ps))  if ps else float("nan")
            row[f"{label}_ssim_mean"] = float(np.mean(ss)) if ss else float("nan")
            row[f"{label}_ssim_sd"]   = float(np.std(ss))  if ss else float("nan")
            row[f"{label}_n"]         = len(ps)
        results.append(row)
        print(f"  normal={row['normal_psnr_mean']:.3f}  "
              f"bacterial={row['bacterial_psnr_mean']:.3f}  "
              f"viral={row['viral_psnr_mean']:.3f}")

        # free GPU memory between models
        del model
        torch.cuda.empty_cache()

    fieldnames = ["arm",
                  "normal_psnr_mean", "normal_psnr_sd", "normal_n",
                  "bacterial_psnr_mean", "bacterial_psnr_sd", "bacterial_n",
                  "viral_psnr_mean", "viral_psnr_sd", "viral_n",
                  "normal_ssim_mean", "normal_ssim_sd",
                  "bacterial_ssim_mean", "bacterial_ssim_sd",
                  "viral_ssim_mean", "viral_ssim_sd"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\n✓ Subgroup metrics saved to {OUT_CSV}")
    print(f"  {len(results)} arms processed")
