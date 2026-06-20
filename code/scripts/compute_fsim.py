"""
compute_fsim.py — FSIM & LPIPS evaluation for all trained models at mid noise.

FSIM (Feature Similarity Index Measure, Zhang et al. 2011) via piq — uses
phase congruency + gradient magnitude. No training arm optimised for it,
making it a true held-out perceptual quality metric.

LPIPS (Zhang et al. 2018) via piq AlexNet — secondary perceptual metric.
Note: Arm N used VGG perceptual loss, so LPIPS is not fully held-out for it.

Output: /rds/user/stm43/hpc-work/ppvae_results/evaluation/fsim_results.csv
"""
from __future__ import annotations
import os, sys, glob, math, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import piq

PROJECT   = "/rds/user/stm43/hpc-work/ppvae_hformer"
RESULTS   = "/rds/user/stm43/hpc-work/ppvae_results"
DATA      = "/rds/user/stm43/hpc-work/chest_xray/test"
KAIR_ROOT = "/rds/user/stm43/hpc-work/KAIR"
OUT_CSV   = Path(RESULTS) / "evaluation" / "fsim_results.csv"

sys.path.insert(0, PROJECT)
sys.path.insert(0, KAIR_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}  |  piq {piq.__version__}")

FOI_A_MID, FOI_B_MID, FOI_SIGMA_MID = 0.03, 0.005, 0.141
KAIR_BASELINES = {"ffdnet", "ircnn", "drunet", "swinir"}

ARM_ORDER = [
    "arm_a_l2", "arm_b_nll", "arm_c_nll_ssim", "arm_d_nll_ssim_ffl",
    "arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb",
    "arm_i_l1", "arm_j_l1_ssim_ffl", "arm_k_nll_l1", "arm_l_nll_edge_ffl",
    "arm_m_full_det", "arm_n_perc", "arm_o_prelu", "arm_p_best",
    "dncnn_baseline", "ffdnet", "ircnn", "drunet", "swinir",
]

lpips_fn = piq.LPIPS(reduction="none").to(DEVICE)


def load_img(path):
    return np.array(Image.open(path).convert("L").resize((256, 256))) / 255.0


def add_noise(clean, seed=42):
    rng = np.random.default_rng(seed)
    var = FOI_A_MID * np.clip(clean, 0, None) + FOI_B_MID
    return np.clip(clean + rng.standard_normal(clean.shape) * np.sqrt(var), 0, 1).astype(np.float32)


def psnr_np(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-14 else 20 * math.log10(1.0 / math.sqrt(mse))


def t2np(t): return t.squeeze().detach().cpu().float().numpy()


class DnCNN(torch.nn.Module):
    def __init__(self, depth=20, ch=64, in_ch=1):
        super().__init__()
        L = [torch.nn.Conv2d(in_ch, ch, 3, padding=1), torch.nn.ReLU(True)]
        for _ in range(depth - 2):
            L += [torch.nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                  torch.nn.BatchNorm2d(ch), torch.nn.ReLU(True)]
        L.append(torch.nn.Conv2d(ch, in_ch, 3, padding=1))
        self.body = torch.nn.Sequential(*L)
    def forward(self, x): return x - self.body(x)


def load_model(arm):
    from src.models.ppvae_hformer import PPVAEHformer
    from src.training.config import ABLATION_ARMS, ExperimentConfig
    from models.network_ffdnet import FFDNet
    from models.network_dncnn import IRCNN
    from models.network_unet import UNetRes
    from models.network_swinir import SwinIR

    if arm in KAIR_BASELINES:
        p = os.path.join(RESULTS, "baselines", arm, "best_model.pth")
        if not os.path.exists(p):
            print(f"  [SKIP] {arm}"); return None
        state = torch.load(p, map_location=DEVICE, weights_only=False)
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
                       img_range=1., depths=[6]*6, embed_dim=180,
                       num_heads=[6]*6, mlp_ratio=2, upsampler="", resi_connection="1conv")
        m.load_state_dict(state, strict=True)
        return m.to(DEVICE).eval()

    elif arm == "dncnn_baseline":
        p = os.path.join(RESULTS, "dncnn_baseline", "best_model.pth")
        if not os.path.exists(p):
            print(f"  [SKIP] {arm}"); return None
        ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
        m = DnCNN().to(DEVICE)
        m.load_state_dict(ckpt["model"]); return m.eval()

    else:
        # Check for fine-tuned arms too
        p = os.path.join(RESULTS, arm, "best_model.pth")
        if not os.path.exists(p):
            print(f"  [SKIP] {arm}"); return None
        cfg = ExperimentConfig()
        arm_key = arm if arm in ABLATION_ARMS else None
        if arm_key:
            for k, v in ABLATION_ARMS[arm_key].get("model", {}).items():
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
        ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ckpt["model"]); return m.eval()


@torch.no_grad()
def infer(arm, model, x):
    if arm in KAIR_BASELINES:
        if arm == "ffdnet":
            return model(x, torch.full((1,1,1,1), FOI_SIGMA_MID, device=DEVICE)).clamp(0,1)
        elif arm == "drunet":
            return model(torch.cat([x, torch.full_like(x, FOI_SIGMA_MID)], 1)).clamp(0,1)
        elif arm == "swinir":
            ws=8; _,_,h,w=x.shape; ph,pw=(ws-h%ws)%ws,(ws-w%ws)%ws
            return model(F.pad(x,(0,pw,0,ph),"reflect"))[:,:,:h,:w].clamp(0,1)
        else:
            return model(x).clamp(0,1)
    elif arm == "dncnn_baseline":
        return model(x).clamp(0,1)
    else:
        mu, _, _, _ = model(x, deterministic=True)
        return mu.clamp(0,1)


def get_test_files():
    files = []
    for f in sorted(glob.glob(os.path.join(DATA,"NORMAL","*.jpeg")) +
                    glob.glob(os.path.join(DATA,"NORMAL","*.png"))):
        files.append((f, "Normal"))
    for f in sorted(glob.glob(os.path.join(DATA,"PNEUMONIA","*.jpeg")) +
                    glob.glob(os.path.join(DATA,"PNEUMONIA","*.png"))):
        name = os.path.basename(f)
        files.append((f, "Bacterial" if name.startswith("BACTERIA") else "Viral"))
    return files


if __name__ == "__main__":
    all_files = get_test_files()
    n = len(all_files)
    print(f"Test images: {n}")

    results = []
    for arm in ARM_ORDER:
        print(f"\n[{arm}]")
        model = load_model(arm)
        if model is None:
            continue

        fsim_vals, lpips_vals, psnr_vals = [], [], []
        for i, (path, _) in enumerate(all_files):
            if i % 100 == 0:
                print(f"  {i}/{n} ...", flush=True)
            clean = load_img(path).astype(np.float32)
            noisy = add_noise(clean, seed=i + 42)

            x = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            y = torch.tensor(clean).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                recon = infer(arm, model, x)

            psnr_vals.append(psnr_np(t2np(recon), clean))
            fsim_vals.append(piq.fsim(recon, y, data_range=1.0, chromatic=False).item())
            lpips_vals.append(lpips_fn(recon.repeat(1,3,1,1), y.repeat(1,3,1,1)).item())

        row = dict(arm=arm,
                   fsim_mean=float(np.mean(fsim_vals)), fsim_sd=float(np.std(fsim_vals)),
                   lpips_mean=float(np.mean(lpips_vals)), lpips_sd=float(np.std(lpips_vals)),
                   psnr_mean=float(np.mean(psnr_vals)), n=len(fsim_vals))
        results.append(row)
        print(f"  FSIM={row['fsim_mean']:.4f} ± {row['fsim_sd']:.4f}  "
              f"LPIPS={row['lpips_mean']:.4f} ± {row['lpips_sd']:.4f}  "
              f"PSNR={row['psnr_mean']:.2f}")

        del model; torch.cuda.empty_cache()

    fields = ["arm","fsim_mean","fsim_sd","lpips_mean","lpips_sd","psnr_mean","n"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)

    print(f"\n✓ FSIM results → {OUT_CSV}")
    top5 = sorted(results, key=lambda r: -r["fsim_mean"])[:5]
    print("  Top 5 by FSIM:")
    for r in top5:
        print(f"    {r['arm']:35s}: FSIM={r['fsim_mean']:.4f}  LPIPS={r['lpips_mean']:.4f}")
