#!/usr/bin/env python
"""
smoke_test.py -- verify every code path before HPC submission.

Runs in under 60 seconds on CPU. Does NOT require CUDA.
Does NOT require full Kermany dataset (falls back to synthetic images).
Does NOT require piq or lpips (skipped gracefully).

What this test verifies:
    1. Imports: all src modules load without error
    2. Noise simulation: Foi noise applied to synthetic images
    3. Dataset: KermanyDataset loads (if chest_xray/ is present)
    4. Forward pass: all 5 ablation arms produce the right output shapes
    5. Loss: all loss terms compute without NaN or inf
    6. Backward: gradients flow through the full graph
    7. Optimiser: one parameter update step completes
    8. Uncertainty: MC epistemic sampling runs for Arm E
    9. Metrics: PSNR and SSIM compute correctly

Usage (from project root):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --data_dir /path/to/chest_xray --n_images 8

Pass/Fail: exits 0 on success, non-zero on any failure.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# ── Allow running from any CWD ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# ── Parse args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None,
                    help="Path to chest_xray/ directory (optional)")
parser.add_argument("--n_images", type=int, default=4,
                    help="Number of images to use (smaller = faster)")
parser.add_argument("--image_size", type=int, default=128,
                    help="Spatial size (smaller = faster; 64 works fine)")
args = parser.parse_args()

DEVICE = torch.device("cpu")
B = 2              # batch size
C = 1              # channels (greyscale)
H = W = args.image_size
PASS_MARKER = "[PASS]"
FAIL_MARKER = "[FAIL]"
failures: list[str] = []


def check(name: str, fn):
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"  {PASS_MARKER} {name:50s} ({elapsed:.2f}s)")
        return result
    except Exception:
        elapsed = time.time() - t0
        print(f"  {FAIL_MARKER} {name:50s} ({elapsed:.2f}s)")
        traceback.print_exc()
        failures.append(name)
        return None


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== PP-VAE-Hformer Smoke Test ===")
print(f"    Device: {DEVICE}")
print(f"    Image size: {H}x{W}, batch: {B}")
print()

# ── 1. IMPORTS ─────────────────────────────────────────────────────────────────
print("[1] Module imports")

src_noise  = check("src.data.noise_simulation",  lambda: __import__("src.data.noise_simulation",  fromlist=["."]))
src_ds     = check("src.data.kermany_dataset",   lambda: __import__("src.data.kermany_dataset",   fromlist=["."]))
src_model  = check("src.models.ppvae_hformer",   lambda: __import__("src.models.ppvae_hformer",   fromlist=["."]))
src_nll    = check("src.losses.nll_loss",         lambda: __import__("src.losses.nll_loss",         fromlist=["."]))
src_ssim   = check("src.losses.ms_ssim_loss",     lambda: __import__("src.losses.ms_ssim_loss",     fromlist=["."]))
src_ffl    = check("src.losses.focal_frequency",  lambda: __import__("src.losses.focal_frequency",  fromlist=["."]))
src_kl     = check("src.losses.kl_loss",          lambda: __import__("src.losses.kl_loss",          fromlist=["."]))
src_comp   = check("src.losses.composite_loss",   lambda: __import__("src.losses.composite_loss",   fromlist=["."]))
src_met    = check("src.evaluation.metrics",      lambda: __import__("src.evaluation.metrics",      fromlist=["."]))

if failures:
    print(f"\nCRITICAL: {len(failures)} import(s) failed. Fix before proceeding.")
    sys.exit(1)

# ── Re-import properly ─────────────────────────────────────────────────────────
from src.data.noise_simulation import add_foi_noise, PRESETS
from src.data.kermany_dataset import KermanyDataset
from src.models.ppvae_hformer import PPVAEHformer
from src.losses.composite_loss import PPVAEHformerLoss
from src.evaluation.metrics import psnr, ssim

print()

# ── 2. NOISE SIMULATION ────────────────────────────────────────────────────────
print("[2] Foi noise simulation")

clean = torch.rand(B, C, H, W)  # synthetic clean image

for level, params in PRESETS.items():
    noisy = check(
        f"Foi noise ({level}: a={params['a']}, b={params['b']})",
        lambda p=params: add_foi_noise(clean.clone(), a=p["a"], b=p["b"]),
    )
    if noisy is not None:
        assert noisy.shape == clean.shape, "shape mismatch"
        assert noisy.min() >= 0 and noisy.max() <= 1, "values out of [0,1]"

print()

# ── 3. DATASET (optional -- only if data_dir provided or default path exists) ──
print("[3] KermanyDataset")

DEFAULT_PATHS = [
    "/Users/sam/Documents/PPVAE Dissertation Project/chest_xray",
    "/rds/user/stm43/hpc-work/chest_xray",
]
data_dir = args.data_dir
if data_dir is None:
    for p in DEFAULT_PATHS:
        if Path(p).exists():
            data_dir = p
            break

if data_dir and Path(data_dir).exists():
    def _load_dataset():
        ds = KermanyDataset(
            root_dir=data_dir,
            split="train",
            target_size=H,
            noise_level="mid",
            augment=False,
        )
        noisy, clean_img, label = ds[0]
        assert noisy.shape == (1, H, H), f"Expected (1,{H},{H}), got {noisy.shape}"
        return f"{len(ds)} images, first label={label}"
    result = check("KermanyDataset (real images)", _load_dataset)
    if result:
        print(f"         --> {result}")
else:
    print(f"  [SKIP] No data dir found (pass --data_dir to test). Using synthetic data.")
    check("Synthetic images (no data_dir)", lambda: torch.rand(B, C, H, W))

print()

# ── 4. MODEL FORWARD PASS ──────────────────────────────────────────────────────
print("[4] Model forward passes (all 5 ablation arms)")

ARM_CONFIGS = [
    ("Arm A: L2  (no VAE)", False, "l2"),
    ("Arm B: NLL (no VAE)", False, "nll"),
    ("Arm C: NLL+SSIM",     False, "nll+ssim"),
    ("Arm D: NLL+SSIM+FFL", False, "nll+ssim+ffl"),
    ("Arm E: Full PP-VAE",  True,  "nll+ssim+ffl+kl"),
]

noisy_in = add_foi_noise(torch.rand(B, C, H, W), a=0.03, b=0.005)
clean_ref = torch.rand(B, C, H, W)

model_e = None  # save Arm E model for uncertainty test

for arm_name, use_vae, arm_str in ARM_CONFIGS:
    def _run_arm(uv=use_vae, astr=arm_str, aname=arm_name):
        model = PPVAEHformer(
            in_channels=C,
            base_channels=16,   # small for smoke test speed
            num_blocks=1,
            num_scales=2,
            num_heads=2,
            window_size=min(8, H // 4),
            use_vae=uv,
        ).to(DEVICE)
        model.train()

        mu, lsa, z_mu, z_lv = model(noisy_in)

        assert mu.shape  == (B, C, H, W), f"mu shape wrong: {mu.shape}"
        assert lsa.shape == (B, C, H, W), f"log_sig2a shape wrong: {lsa.shape}"
        if uv:
            assert z_mu is not None, "z_mu should not be None for VAE arm"
            assert z_lv is not None, "z_lv should not be None for VAE arm"
        else:
            assert z_mu is None, "z_mu should be None for non-VAE arm"

        loss_fn = PPVAEHformerLoss(arm=astr, beta=0.001, lambda_ssim=0.5, lambda_ffl=0.1)
        losses = loss_fn(clean_ref, mu, lsa, z_mu, z_lv)

        for k, v in losses.items():
            assert not torch.isnan(v), f"NaN in {k}"
            assert not torch.isinf(v), f"Inf in {k}"

        return model, losses

    result = check(arm_name, _run_arm)
    if result is not None:
        model_out, loss_dict = result
        loss_str = " | ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
        print(f"         --> {loss_str}")
        if "Full PP-VAE" in arm_name:
            model_e = model_out

print()

# ── 5. BACKWARD PASS ──────────────────────────────────────────────────────────
print("[5] Backward pass + optimiser step")

def _run_backward():
    model = PPVAEHformer(
        in_channels=C, base_channels=16, num_blocks=1,
        num_scales=2, num_heads=2, window_size=min(8, H // 4), use_vae=True,
    ).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = PPVAEHformerLoss(arm="nll+ssim+ffl+kl", beta=0.001)

    model.train()
    optimiser.zero_grad()
    mu, lsa, z_mu, z_lv = model(noisy_in)
    losses = loss_fn(clean_ref, mu, lsa, z_mu, z_lv)
    losses["loss"].backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = total_norm ** 0.5
    assert grad_norm > 0, "Zero gradient -- backward may have failed"

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()
    return grad_norm

gn = check("Backward + AdamW step (full PP-VAE-Hformer)", _run_backward)
if gn is not None:
    print(f"         --> grad norm: {gn:.4f}")

print()

# ── 6. EPISTEMIC UNCERTAINTY ───────────────────────────────────────────────────
print("[6] MC epistemic uncertainty (Arm E)")

if model_e is not None:
    def _mc_uncertainty():
        model_e.train()
        sigma2_e = model_e.mc_epistemic_uncertainty(noisy_in[:1], K=5)
        assert sigma2_e.shape == (1, C, H, W), f"Wrong shape: {sigma2_e.shape}"
        assert (sigma2_e >= 0).all(), "Negative variance!"
        return sigma2_e.mean().item()

    epist = check("MC epistemic uncertainty (K=5 samples)", _mc_uncertainty)
    if epist is not None:
        print(f"         --> mean sigma^2_e: {epist:.6f}")
else:
    print("  [SKIP] Arm E model not available")

print()

# ── 7. METRICS ────────────────────────────────────────────────────────────────
print("[7] Evaluation metrics")

def _psnr():
    v = psnr(clean_ref, clean_ref)
    assert v == float("inf") or v > 50, f"Self-PSNR should be inf, got {v}"
    v2 = psnr(noisy_in, clean_ref)
    return f"PSNR(clean|clean)=inf, PSNR(noisy|clean)={v2:.2f}dB"

def _ssim():
    v = ssim(clean_ref, clean_ref)
    assert abs(v - 1.0) < 0.01, f"Self-SSIM should be ~1.0, got {v}"
    v2 = ssim(noisy_in, clean_ref)
    return f"SSIM(clean|clean)={v:.4f}, SSIM(noisy|clean)={v2:.4f}"

r_psnr = check("PSNR sanity check", _psnr)
r_ssim = check("SSIM sanity check", _ssim)
if r_psnr: print(f"         --> {r_psnr}")
if r_ssim: print(f"         --> {r_ssim}")

print()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("=" * 60)
if failures:
    print(f"SMOKE TEST FAILED -- {len(failures)} check(s) failed:")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
else:
    print("SMOKE TEST PASSED -- all checks OK")
    print("Ready to submit to CSD3 HPC.")
    sys.exit(0)
