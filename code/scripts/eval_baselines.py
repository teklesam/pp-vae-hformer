"""
Zero-shot inference evaluation of comparison baselines on the Kermany CXR test set.

All methods are loaded with pretrained weights (trained on natural images / AWGN)
and tested AS-IS on our Foi Poisson-Gaussian CXR noise — no retraining.
This demonstrates the domain gap and situates our method in the literature.

Noise level tested: mid (a=0.03, b=0.005, ~σ=0.141 at mean intensity)
This aligns with BSD68 σ=25 for contextual comparison (our effective σ is slightly
higher due to signal-dependent Poisson term).

Usage:
    # All available methods
    python scripts/eval_baselines.py --data_root /path/to/chest_xray

    # Specific methods only
    python scripts/eval_baselines.py --data_root /path/to/chest_xray --methods bm3d dncnn swinir

    # All three noise levels
    python scripts/eval_baselines.py --data_root /path/to/chest_xray --noise_levels low mid high

    # On CSD3 (weights in model_zoo/baselines/):
    python scripts/eval_baselines.py \\
        --data_root /rds/user/stm43/hpc-work/ppvae_hformer/data/chest_xray \\
        --output_dir /rds/user/stm43/hpc-work/ppvae_results/baselines/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
KAIR_ROOT = ROOT.parent / "KAIR"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(KAIR_ROOT))

from data.kermany_dataset import KermanyDataset
from evaluation.metrics import psnr, ssim, ms_ssim, lpips


WEIGHTS_DIR = ROOT / "model_zoo" / "baselines"


# ─── Wrapper protocol ────────────────────────────────────────────────────────

class BaselineModel(Protocol):
    name: str
    def eval(self) -> "BaselineModel": ...
    def to(self, device: torch.device) -> "BaselineModel": ...
    def __call__(self, noisy: torch.Tensor) -> torch.Tensor: ...


# ─── BM3D ────────────────────────────────────────────────────────────────────

class BM3DWrapper:
    """Non-neural BM3D — runs on CPU numpy, no GPU needed."""
    name = "BM3D"

    def eval(self) -> "BM3DWrapper":
        return self

    def to(self, device: torch.device) -> "BM3DWrapper":
        return self  # always CPU

    def __call__(self, noisy: torch.Tensor) -> torch.Tensor:
        import bm3d
        # Foi mid: effective σ ≈ sqrt(0.03*0.5 + 0.005) ≈ 0.141
        sigma_psd = 0.141
        results = []
        for i in range(noisy.shape[0]):
            img_np = noisy[i, 0].cpu().numpy().astype(np.float64)
            denoised = bm3d.bm3d(img_np, sigma_psd=sigma_psd)
            results.append(torch.from_numpy(denoised.astype(np.float32)))
        return torch.stack(results).unsqueeze(1).clamp(0.0, 1.0)


# ─── DnCNN ───────────────────────────────────────────────────────────────────

class DnCNNWrapper(nn.Module):
    name = "DnCNN"

    def __init__(self, weights_path: Path, nb: int = 17) -> None:
        super().__init__()
        from models.network_dncnn import DnCNN
        self.net = DnCNN(in_nc=1, out_nc=1, nc=64, nb=nb, act_mode="R")
        state = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state, strict=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.net(noisy).clamp(0.0, 1.0)


class DnCNNBlindWrapper(DnCNNWrapper):
    """Blind DnCNN (dncnn_gray_blind) — handles unknown noise level."""
    name = "DnCNN-B"

    def __init__(self, weights_path: Path) -> None:
        super().__init__(weights_path, nb=20)


# ─── FFDNet ──────────────────────────────────────────────────────────────────

class FFDNetWrapper(nn.Module):
    """FFDNet with Foi effective sigma passed as the scalar sigma map.

    FFDNet accepts sigma as (B, 1, 1, 1) and replicates it spatially.
    We pass the effective Foi mid-noise sigma (sqrt(a*x_mean + b)) as a
    uniform scalar — the best approximation without modifying the network.
    Note: this is still zero-shot (AWGN-trained weights on Foi-noisy CXR).
    """
    name = "FFDNet"

    # Foi mid: sqrt(0.03*0.5 + 0.005) ≈ 0.141
    SIGMA_MID = 0.141

    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        from models.network_ffdnet import FFDNet
        self.net = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode="R")
        state = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state, strict=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        # FFDNet expects (B, 1, 1, 1) scalar sigma map
        sigma = torch.full((noisy.shape[0], 1, 1, 1), self.SIGMA_MID,
                           dtype=noisy.dtype, device=noisy.device)
        return self.net(noisy, sigma).clamp(0.0, 1.0)


# ─── IRCNN ───────────────────────────────────────────────────────────────────

class IRCNNWrapper(nn.Module):
    """IRCNN denoiser (plug-and-play prior). Zero-shot on our Foi noise."""
    name = "IRCNN"

    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        from models.network_dncnn import IRCNN
        self.net = IRCNN(in_nc=1, out_nc=1, nc=64)
        state = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state, strict=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.net(noisy).clamp(0.0, 1.0)


# ─── DRUNet ──────────────────────────────────────────────────────────────────

class DRUNetWrapper(nn.Module):
    """DRUNet: deep U-Net + explicit sigma conditioning.

    Accepts a 2-channel input [noisy | sigma_map]. We use the Foi mid-noise
    effective sigma (≈0.141) as a uniform map. Running with the Foi sigma
    is a zero-shot adaptation — DRUNet was trained on AWGN but its sigma
    conditioning makes it more transferable.
    """
    name = "DRUNet"

    SIGMA_MID = 0.141  # sqrt(0.03*0.5 + 0.005)

    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        from models.network_unet import UNetRes
        self.net = UNetRes(
            in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4,
            act_mode="R", downsample_mode="strideconv",
            upsample_mode="convtranspose",
        )
        state = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state, strict=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        sigma_map = torch.full_like(noisy, self.SIGMA_MID)
        x_in = torch.cat([noisy, sigma_map], dim=1)
        return self.net(x_in).clamp(0.0, 1.0)


# ─── SwinIR ──────────────────────────────────────────────────────────────────

class SwinIRWrapper(nn.Module):
    """SwinIR grayscale denoising model (noise25, DFWB training set)."""
    name = "SwinIR"

    WINDOW_SIZE = 8

    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        from models.network_swinir import SwinIR
        self.net = SwinIR(
            upscale=1, in_chans=1, img_size=128,
            window_size=self.WINDOW_SIZE,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )
        state = torch.load(weights_path, map_location="cpu")
        # SwinIR checkpoints sometimes wrap state under 'params'
        if "params" in state:
            state = state["params"]
        self.net.load_state_dict(state, strict=True)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        _, _, h, w = noisy.shape
        ws = self.WINDOW_SIZE
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x = F.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")
        out = self.net(x)
        return out[:, :, :h, :w].clamp(0.0, 1.0)


# ─── Registry ────────────────────────────────────────────────────────────────

def build_model(name: str) -> BaselineModel | None:
    """Instantiate a baseline model by name. Returns None if weights missing."""
    w = WEIGHTS_DIR

    def weights(filename: str) -> Path | None:
        p = w / filename
        if not p.exists():
            print(f"  [SKIP] {name}: weights not found at {p}")
            print(f"         Run:  python scripts/download_baseline_weights.py")
            return None
        return p

    if name == "bm3d":
        try:
            import bm3d  # noqa
            return BM3DWrapper()
        except ImportError:
            print("  [SKIP] bm3d not installed. Run: pip install bm3d")
            return None

    elif name == "dncnn":
        p = weights("dncnn_25.pth")
        return DnCNNWrapper(p) if p else None

    elif name == "dncnn_blind":
        p = weights("dncnn_gray_blind.pth")
        return DnCNNBlindWrapper(p) if p else None

    elif name == "ffdnet":
        p = weights("ffdnet_gray.pth")
        return FFDNetWrapper(p) if p else None

    elif name == "ircnn":
        p = weights("ircnn_gray.pth")
        return IRCNNWrapper(p) if p else None

    elif name == "drunet":
        p = weights("drunet_gray.pth")
        return DRUNetWrapper(p) if p else None

    elif name == "swinir":
        p = weights("004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth")
        return SwinIRWrapper(p) if p else None

    else:
        print(f"  [SKIP] Unknown method: {name}")
        return None


ALL_METHODS = ["bm3d", "dncnn", "dncnn_blind", "ffdnet", "ircnn", "drunet", "swinir"]


# ─── Evaluation loop ─────────────────────────────────────────────────────────

LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def evaluate_model(
    model: BaselineModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_level: str,
) -> dict:
    """Run inference on the full test set and return per-class metrics."""
    model.eval()
    model.to(device)

    per_class: dict[int, dict[str, list]] = {
        0: {"psnr": [], "ssim": [], "ms_ssim": [], "lpips": []},
        1: {"psnr": [], "ssim": [], "ms_ssim": [], "lpips": []},
    }

    t0 = time.time()
    with torch.no_grad():
        for noisy, clean, labels in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            pred = model(noisy)

            for i in range(noisy.shape[0]):
                p = pred[i:i+1]
                c = clean[i:i+1]
                lbl = int(labels[i])

                per_class[lbl]["psnr"].append(psnr(p, c))
                per_class[lbl]["ssim"].append(ssim(p, c))
                per_class[lbl]["ms_ssim"].append(ms_ssim(p, c))
                per_class[lbl]["lpips"].append(lpips(p, c))

    elapsed = time.time() - t0
    n_total = sum(len(v["psnr"]) for v in per_class.values())

    results = {
        "method": model.name,
        "noise_level": noise_level,
        "n_total": n_total,
        "elapsed_s": round(elapsed, 1),
    }

    for lbl, metrics in per_class.items():
        prefix = LABEL_NAMES[lbl].lower()
        n = len(metrics["psnr"])
        results[f"n_{prefix}"] = n
        if n == 0:
            continue
        results[f"psnr_{prefix}"] = round(float(np.mean(metrics["psnr"])), 3)
        results[f"ssim_{prefix}"] = round(float(np.mean(metrics["ssim"])), 4)
        results[f"ms_ssim_{prefix}"] = round(float(np.mean(metrics["ms_ssim"])), 4)
        results[f"lpips_{prefix}"] = round(float(np.mean(metrics["lpips"])), 4)

    # Overall averages
    all_psnr  = [v for m in per_class.values() for v in m["psnr"]]
    all_ssim  = [v for m in per_class.values() for v in m["ssim"]]
    all_ms    = [v for m in per_class.values() for v in m["ms_ssim"]]
    all_lpips = [v for m in per_class.values() for v in m["lpips"]]
    results["psnr_mean"]    = round(float(np.mean(all_psnr)), 3)
    results["ssim_mean"]    = round(float(np.mean(all_ssim)), 4)
    results["ms_ssim_mean"] = round(float(np.mean(all_ms)), 4)
    results["lpips_mean"]   = round(float(np.mean(all_lpips)), 4)

    return results


CSV_FIELDS = [
    "method", "noise_level", "n_total",
    "psnr_mean", "ssim_mean", "ms_ssim_mean", "lpips_mean",
    "n_normal", "psnr_normal", "ssim_normal", "ms_ssim_normal", "lpips_normal",
    "n_pneumonia", "psnr_pneumonia", "ssim_pneumonia", "ms_ssim_pneumonia", "lpips_pneumonia",
    "elapsed_s",
]


def print_row(r: dict) -> None:
    print(f"  {r['method']:<12s} | noise={r['noise_level']:<4s} | "
          f"PSNR={r.get('psnr_mean', float('nan')):6.3f} "
          f"SSIM={r.get('ssim_mean', float('nan')):.4f} "
          f"LPIPS={r.get('lpips_mean', float('nan')):.4f}  "
          f"({r['elapsed_s']:.0f}s)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation on Kermany CXR test set")
    parser.add_argument("--data_root", required=True,
                        help="Path to chest_xray/ (contains train/ and test/)")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS + ["all"],
                        help="Baselines to evaluate (default: all available)")
    parser.add_argument("--noise_levels", nargs="+", default=["mid"],
                        choices=["low", "mid", "high"],
                        help="Foi noise presets to evaluate (default: mid)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results CSV (default: ppvae_results/baselines/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    methods = ALL_METHODS if "all" in args.methods else args.methods

    output_dir = Path(args.output_dir) if args.output_dir else ROOT.parent.parent / "ppvae_results" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_metrics.csv"

    existing_rows: list[dict] = []
    done_keys: set[tuple] = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                done_keys.add((row["method"], row["noise_level"]))

    new_rows: list[dict] = []

    print()
    print("=== Baseline evaluation ===")
    print(f"  Methods : {methods}")
    print(f"  Noise   : {args.noise_levels}")
    print(f"  Data    : {args.data_root}")
    print(f"  Output  : {csv_path}")
    print()

    for noise_level in args.noise_levels:
        dataset = KermanyDataset(
            root_dir=args.data_root,
            split="test",
            noise_level=noise_level,
            augment=False,
            seed=args.seed,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Noise={noise_level}: {len(dataset)} test images")

        for method_name in methods:
            key = (method_name.upper().replace("_", "-") if method_name != "dncnn_blind"
                   else "DnCNN-B", noise_level)
            # Quick check using display name
            display_key = (method_name, noise_level)
            if display_key in done_keys:
                print(f"  [skip] {method_name} @ {noise_level} already in CSV")
                continue

            print(f"  Evaluating {method_name} ...")
            model = build_model(method_name)
            if model is None:
                continue

            try:
                row = evaluate_model(model, loader, device, noise_level)
                new_rows.append(row)
                print_row(row)
            except Exception as exc:
                print(f"  [ERROR] {method_name}: {exc}")

            # Free GPU memory between models
            if hasattr(model, "to"):
                model.to("cpu")
            torch.cuda.empty_cache()

    # Write CSV (append to existing)
    all_rows = existing_rows + new_rows
    if new_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print()
        print(f"Results saved → {csv_path}")

    # Print summary table
    print()
    print("=" * 80)
    print(f"{'Method':<14} {'Noise':<6} {'PSNR':>7} {'SSIM':>7} {'MS-SSIM':>8} {'LPIPS':>7}")
    print("-" * 80)
    for r in all_rows:
        print(f"{r.get('method','?'):<14} {r.get('noise_level','?'):<6} "
              f"{float(r.get('psnr_mean',0)):>7.3f} "
              f"{float(r.get('ssim_mean',0)):>7.4f} "
              f"{float(r.get('ms_ssim_mean',0)):>8.4f} "
              f"{float(r.get('lpips_mean',0)):>7.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
