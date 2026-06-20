"""
generate_roi_panels.py — Publication-quality multi-panel ROI comparison figure.

Creates a two-row figure for each selected test case:
  Row 1: Full images (Noisy | Clean | Arm A | Arm D | Arm H | IRCNN | SwinIR | Arm O)
          with a red ROI rectangle overlaid on each.
  Row 2: Zoomed ROI crops for each method, with a red annotation arrow
          pointing to the key diagnostic feature.

Panel design rationale (16-arm study):
  • Noisy / Clean  — baseline reference pair
  • Arm A (L2)     — best PSNR (34.62 dB); MSE-optimal point
  • Arm I (L1)     — statistically tied with A (d=0.164, n.s.); norm equivalence
  • Arm D (NLL+SSIM+FFL) — best NLL arm (33.44 dB + uncertainty output)
  • Arm H (VAE cyc+fb)   — best VAE arm (32.76 dB + aleatoric + epistemic UQ)
  • IRCNN          — best retrained CNN baseline (32.66 dB)
  • SwinIR         — worst baseline (31.18 dB); Transformer underperformance
  • Arm O (PReLU)  — catastrophic failure (29.07 dB); activation ablation story

Output: one PDF + one PNG per selected case, saved to --output_dir.

Usage (CSD3):
  srun --account=mrc-bsu2-sl2-gpu --partition=ampere \
       --gres=gpu:1 --time=0:45:00 --pty bash
  cd ~/hpc-work/ppvae_hformer
  python scripts/generate_roi_panels.py \
    --results_dir ~/hpc-work/ppvae_results \
    --data_dir    ~/hpc-work \
    --output_dir  ~/hpc-work/ppvae_results/roi_panels \
    --noise_eta   200 \
    --case_indices 0 3 \
    --roi 90 70 70 70
"""

import argparse
import os
import sys
import glob
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Panel columns — chosen to tell the full 16-arm story in 8 slots:
#   best PSNR (A), L1 equivalence (I), best NLL arm (D), best VAE (H),
#   best CNN baseline (IRCNN), Transformer underperformance (SwinIR),
#   catastrophic PReLU failure (O).
METHODS = [
    ("noisy",   "Noisy\nInput",                "#888888"),
    ("clean",   "Reference\n(Clean)",          "#222222"),
    ("arm_a",   "Arm A\n(L₂, 34.62 dB)",      "#1f77b4"),
    ("arm_i",   "Arm I\n(L1, 34.44 dB)",      "#aec7e8"),
    ("arm_d",   "Arm D\n(NLL+SSIM+FFL)",       "#ff7f0e"),
    ("arm_h",   "Arm H\n(VAE cyc+fb)",         "#9467bd"),
    ("ircnn",   "IRCNN\n(best CNN base)",       "#2ca02c"),
    ("swinir",  "SwinIR\n(Transf., 31.18 dB)", "#d62728"),
    ("arm_o",   "Arm O\n(PReLU, 29.07 dB)",    "#8c564b"),
]

ARM_CHECKPOINT_DIRS = {
    "arm_a":  "arm_a_l2",
    "arm_i":  "arm_i_l1",
    "arm_d":  "arm_d_nll_ssim_ffl",
    "arm_h":  "arm_h_kl_cyc_fb",
    "ircnn":  "ircnn",
    "swinir": "swinir",
    "arm_o":  "arm_o_prelu",
}


def load_model(method_key, results_dir, device):
    """Load a trained model from its checkpoint directory."""
    if method_key in ("noisy", "clean"):
        return None

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ckpt_dir = os.path.join(results_dir, ARM_CHECKPOINT_DIRS[method_key], "checkpoints")
    best = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.exists(best):
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        best = candidates[-1]
        print(f"  [warn] {method_key}: best_model.pth missing, using {os.path.basename(best)}")

    ckpt = torch.load(best, map_location=device)

    # ── Determine which model class to instantiate ────────────────────────
    if method_key in ("ircnn", "swinir", "ffdnet", "drunet"):
        # KAIR baselines — loaded via their own model classes
        kair_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "..", "KAIR")
        sys.path.insert(0, kair_root)
        if method_key == "ircnn":
            from models.network_ircnn import IRCNN as Net
            model = Net(in_nc=1, out_nc=1, nc=64)
        elif method_key == "swinir":
            from models.network_swinir import SwinIR as Net
            model = Net(upscale=1, in_chans=1, img_size=64, window_size=8,
                        img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180,
                        num_heads=[6,6,6,6,6,6], mlp_ratio=2, upsampler='',
                        resi_connection='1conv')
        elif method_key == "ffdnet":
            from models.network_ffdnet import FFDNet as Net
            model = Net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
        elif method_key == "drunet":
            from models.network_unet import UNetRes as Net
            model = Net(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4,
                        act_mode='R', downsample_mode='strideconv',
                        upsample_mode='convtranspose')
    elif method_key == "dncnn":
        from src.models.dncnn import DnCNN
        model = DnCNN(num_layers=20, num_features=64)
    else:
        # PP-VAE-Hformer arms — VAE active for arm_h only
        use_vae = method_key in ("arm_h", "arm_e", "arm_f", "arm_g", "arm_p")
        activation = "prelu" if method_key in ("arm_o", "arm_p") else "gelu"
        from src.models.ppvae_hformer import PPVAEHformer
        model = PPVAEHformer(use_vae=use_vae, activation=activation)

    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def run_inference(model, noisy_tensor, method_key, device):
    """Run a forward pass and return a numpy image [0,1]."""
    if method_key in ("noisy", "clean"):
        raise ValueError("Should not call run_inference for noisy/clean")

    x = noisy_tensor.unsqueeze(0).to(device)  # [1, 1, H, W]
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]   # take mean output if (mean, logvar, ...) tuple

        # KAIR FFDNet requires a noise-level map as second input; inference
        # sends a zero map (blind denoising mode since model was retrained blind).
        if method_key == "ffdnet":
            sigma_map = torch.zeros(1, 1, x.shape[2] // 2, x.shape[3] // 2, device=device)
            out = model(x, sigma_map)

        out = out.squeeze().cpu().float()

    out = torch.clamp(out, 0.0, 1.0)
    return out.numpy()


def add_roi_rect(ax, roi_x, roi_y, roi_w, roi_h, color="red", lw=1.5):
    rect = patches.Rectangle(
        (roi_x, roi_y), roi_w, roi_h,
        linewidth=lw, edgecolor=color, facecolor="none", zorder=10
    )
    ax.add_patch(rect)


def add_red_arrow(ax, tip_x, tip_y, dx=-10, dy=-10):
    """Add a red arrow pointing to (tip_x, tip_y) from offset (dx, dy)."""
    ax.annotate(
        "", xy=(tip_x, tip_y),
        xytext=(tip_x + dx, tip_y + dy),
        arrowprops=dict(
            arrowstyle="-|>",
            color="red",
            lw=1.8,
            mutation_scale=14,
        ),
        zorder=11,
    )


def crop_roi(img_np, roi_x, roi_y, roi_w, roi_h):
    return img_np[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]


def psnr_np(clean, recon):
    mse = np.mean((clean.astype(float) - recon.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test_pairs(data_dir, noise_eta, case_indices):
    """
    Load (noisy, clean) pairs from pre-saved numpy arrays or PNG files.
    Adjust the paths below to match your actual evaluation output layout.
    """
    base = os.path.join(data_dir, "ppvae_results", "evaluation_final")

    noisy_list, clean_list = [], []
    for idx in case_indices:
        # Try numpy first
        n_path = os.path.join(base, f"noisy_eta{noise_eta}_idx{idx:04d}.npy")
        c_path = os.path.join(base, f"clean_eta{noise_eta}_idx{idx:04d}.npy")
        if os.path.exists(n_path) and os.path.exists(c_path):
            noisy_list.append(np.load(n_path))
            clean_list.append(np.load(c_path))
            continue

        # Fall back: try loading from the blind_eval panels output
        from PIL import Image
        n_path = os.path.join(data_dir, "ppvae_results", "blind_eval",
                              "source", f"noisy_idx{idx:04d}.png")
        c_path = os.path.join(data_dir, "ppvae_results", "blind_eval",
                              "source", f"clean_idx{idx:04d}.png")
        if os.path.exists(n_path) and os.path.exists(c_path):
            noisy_list.append(np.array(Image.open(n_path).convert("L")) / 255.0)
            clean_list.append(np.array(Image.open(c_path).convert("L")) / 255.0)
            continue

        # Final fallback: load directly from the Kermany HDF5 / PNG dataset
        dataset_dir = os.path.join(data_dir, "kermany_cxr", "test")
        files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.jpeg"), recursive=True) +
                       glob.glob(os.path.join(dataset_dir, "**", "*.png"),  recursive=True))
        if idx < len(files):
            from PIL import Image
            import torchvision.transforms.functional as TF
            img = TF.to_tensor(Image.open(files[idx]).convert("L"))  # [1, H, W]
            clean_np = img.squeeze().numpy()
            # Simulate noise
            rng = np.random.default_rng(42 + idx)
            poisson = rng.poisson(clean_np * noise_eta) / noise_eta
            gaussian = rng.normal(0, 0.01, clean_np.shape)
            noisy_np = np.clip(poisson + gaussian, 0, 1).astype(np.float32)
            noisy_list.append(noisy_np)
            clean_list.append(clean_np.astype(np.float32))
        else:
            raise IndexError(f"Case index {idx} out of range (only {len(files)} test files found)")

    return noisy_list, clean_list


# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_roi_figure(
    case_idx, noisy_np, clean_np, reconstructions,
    roi_x, roi_y, roi_w, roi_h,
    arrow_tip_offset=(-8, -8),
    output_dir=".",
    dpi=300,
    noise_eta=200,
):
    """
    reconstructions: dict  method_key → np.ndarray [H, W] float32 [0,1]
    """
    n_cols = len(METHODS)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(2.1 * n_cols, 5.2),
        gridspec_kw={"hspace": 0.04, "wspace": 0.02},
    )

    images = {
        "noisy": noisy_np,
        "clean": clean_np,
        **reconstructions,
    }

    for col, (key, label, _) in enumerate(METHODS):
        img = images.get(key)
        if img is None:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        # ── Row 0: full image with ROI rectangle ──
        ax0 = axes[0, col]
        ax0.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        add_roi_rect(ax0, roi_x, roi_y, roi_w, roi_h)
        ax0.axis("off")
        ax0.set_title(label, fontsize=7.5, pad=3, fontfamily="DejaVu Sans")

        # PSNR annotation (skip noisy/clean)
        if key not in ("noisy", "clean"):
            p = psnr_np(clean_np, img)
            ax0.text(
                0.03, 0.03, f"{p:.2f} dB",
                transform=ax0.transAxes,
                color="yellow", fontsize=6.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
            )

        # ── Row 1: zoomed ROI crop ──
        ax1 = axes[1, col]
        crop = crop_roi(img, roi_x, roi_y, roi_w, roi_h)
        ax1.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")

        # Red arrow pointing to the diagnostic feature in the crop
        # Arrow tip is placed at 40% from left, 40% from top of crop by default
        tip_cx = roi_w * 0.40
        tip_cy = roi_h * 0.40
        dx, dy = arrow_tip_offset
        add_red_arrow(ax1, tip_cx, tip_cy, dx=dx, dy=dy)

        # Red border on zoomed crops to match the rectangle above
        for spine in ax1.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)
        ax1.set_xticks([])
        ax1.set_yticks([])

    # Row labels on the left
    axes[0, 0].set_ylabel("Full image", fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel("ROI ×zoom", fontsize=8, labelpad=4)

    fig.suptitle(
        f"Case {case_idx + 1} — Reconstruction comparison (η = {noise_eta})",
        fontsize=9, y=1.01
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"roi_panel_case{case_idx + 1:02d}_eta{noise_eta}")
    fig.savefig(stem + ".pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(stem + ".png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {stem}.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate ROI multi-panel comparison figures")
    p.add_argument("--results_dir", default=os.path.expanduser("~/hpc-work/ppvae_results"),
                   help="Root results directory (contains arm_a_l2/, dncnn_baseline/, etc.)")
    p.add_argument("--data_dir",    default=os.path.expanduser("~/hpc-work"),
                   help="Root data directory (contains kermany_cxr/)")
    p.add_argument("--output_dir",  default=os.path.expanduser("~/hpc-work/ppvae_results/roi_panels"),
                   help="Where to save output figures")
    p.add_argument("--noise_eta",   type=int, default=200,
                   help="Noise level η used during evaluation")
    p.add_argument("--case_indices", type=int, nargs="+", default=[0, 1, 2],
                   help="Test-set indices to include (0-based). Use 3–5 for pneumonia cases.")
    p.add_argument("--roi", type=int, nargs=4, default=[90, 70, 70, 70],
                   metavar=("X", "Y", "W", "H"),
                   help="ROI rectangle: top-left x y, then width height (pixels in 256×256 image). "
                        "Default targets the right mid-lung zone.")
    p.add_argument("--arrow_dx", type=float, default=-12,
                   help="Arrow horizontal offset from tip (negative = arrow comes from left)")
    p.add_argument("--arrow_dy", type=float, default=-12,
                   help="Arrow vertical offset from tip (negative = arrow comes from above)")
    p.add_argument("--dpi",  type=int, default=300, help="Output DPI")
    p.add_argument("--cpu",  action="store_true",   help="Force CPU inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"ROI: x={args.roi[0]}, y={args.roi[1]}, w={args.roi[2]}, h={args.roi[3]}")
    print(f"Cases: {args.case_indices}")

    # ── Load all models ─────────────────────────────────────────────────────
    print("\nLoading models...")
    models = {}
    for key, _, _ in METHODS:
        if key in ("noisy", "clean"):
            continue
        try:
            models[key] = load_model(key, args.results_dir, device)
            print(f"  ✓ {key}")
        except FileNotFoundError as e:
            print(f"  ✗ {key}: {e} — will skip this method")
            models[key] = None

    # ── Load test pairs and run inference ───────────────────────────────────
    print("\nLoading test images...")
    noisy_list, clean_list = load_test_pairs(args.data_dir, args.noise_eta, args.case_indices)

    for i, (case_idx, noisy_np, clean_np) in enumerate(
            zip(args.case_indices, noisy_list, clean_list)):
        print(f"\n--- Case {case_idx} (list position {i}) ---")

        recons = {}
        noisy_t = torch.tensor(noisy_np).float().unsqueeze(0)  # [1, H, W]
        for key, model in models.items():
            if model is None:
                continue
            recon = run_inference(model, noisy_t, key, device)
            p = psnr_np(clean_np, recon)
            print(f"  {key:8s}: PSNR = {p:.3f} dB")
            recons[key] = recon

        build_roi_figure(
            case_idx=case_idx,
            noisy_np=noisy_np,
            clean_np=clean_np,
            reconstructions=recons,
            roi_x=args.roi[0], roi_y=args.roi[1],
            roi_w=args.roi[2], roi_h=args.roi[3],
            arrow_tip_offset=(args.arrow_dx, args.arrow_dy),
            output_dir=args.output_dir,
            dpi=args.dpi,
            noise_eta=args.noise_eta,
        )

    print(f"\nAll done. Figures saved to: {args.output_dir}")
    print("\nTo copy to local Mac:")
    print(f"  rsync -avz stm43@login.hpc.cam.ac.uk:{args.output_dir}/ "
          f"~/Documents/PPVAE\\ Dissertation\\ Project/PpCNN/03_Attachments/roi_panels/")
