#!/usr/bin/env python
"""
train_proposed.py -- PP-VAE-Hformer ablation training, expanded v2.

Usage:
    python scripts/train_proposed.py --arm arm_e_ppvae --data_dir /path/to/chest_xray
    python scripts/train_proposed.py --arm arm_f_kl_cyc --data_dir /path/to/chest_xray
    python scripts/train_proposed.py --arm all          --data_dir /path/to/chest_xray
    python scripts/train_proposed.py --arm new          --data_dir /path/to/chest_xray

Arms:
    Original (A–E):
        arm_a_l2, arm_b_nll, arm_c_nll_ssim, arm_d_nll_ssim_ffl, arm_e_ppvae
    Posterior collapse fixes (F–H):
        arm_f_kl_cyc    -- cyclical KL annealing
        arm_g_kl_fb     -- free-bits KL
        arm_h_kl_cyc_fb -- cyclical + free-bits
    New reconstruction losses (I–K):
        arm_i_l1        -- L1 base
        arm_j_l1_ssim_ffl  -- L1 + SSIM + FFL
        arm_k_nll_l1    -- NLL + L1 hybrid
    Edge / perceptual (L–N):
        arm_l_nll_edge_ffl  -- NLL + Sobel edge + FFL
        arm_m_full_det      -- NLL + SSIM + Edge + FFL
        arm_n_perc          -- NLL + VGG perceptual + SSIM + FFL
    Architecture (O–P):
        arm_o_prelu     -- Arm D with PReLU
        arm_p_best      -- best full VAE (NLL+SSIM+Edge+FFL+cyclic+fb, PReLU)
    Aliases:
        all  -- all 16 arms sequentially
        new  -- arms F–P only (skip already-trained A–E)

KL schedule modes (set per arm in config):
    "linear"    -- ramp 0 → beta_end over beta_warmup_epochs, hold
    "cyclical"  -- Fu et al. (2019) cosine reset every kl_cycle_period epochs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.data.kermany_dataset import make_loaders
from src.models.ppvae_hformer import PPVAEHformer
from src.losses.composite_loss import PPVAEHformerLoss
from src.evaluation.metrics import psnr, ssim
from src.training.config import ExperimentConfig, ABLATION_ARMS, NEW_ARMS, ORIGINAL_ARMS

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--arm", type=str, default="arm_e_ppvae",
                    choices=list(ABLATION_ARMS.keys()) + ["all", "new"])
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--noise_level", type=str, default="random",
                    choices=["low", "mid", "high", "random"])
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_gpus", type=int, default=0,
                    help="GPUs to use via DataParallel (0 = auto-detect all available)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if args.num_gpus == 0 else min(args.num_gpus, torch.cuda.device_count())
N_GPUS = max(1, N_GPUS)
print(f"Device: {DEVICE}  |  GPUs available: {torch.cuda.device_count()}  |  GPUs used: {N_GPUS}")

# All log columns (union of all possible loss keys)
LOG_COLS = ["epoch", "train_loss", "l_rec", "l_kl", "l_ssim", "l_ffl",
            "l_edge", "l_perc", "l_l1", "val_psnr", "val_ssim", "beta", "lr", "time_s"]


def get_beta(epoch: int, cfg_loss, cfg_model) -> float:
    """Compute KL beta for this epoch according to the arm's schedule."""
    if not cfg_model.use_vae:
        return 0.0
    sched = cfg_loss.kl_schedule
    beta_end = cfg_loss.beta_end
    warmup = cfg_loss.beta_warmup_epochs

    if sched == "cyclical":
        period = cfg_loss.kl_cycle_period
        ratio = cfg_loss.kl_cycle_ratio
        cycle_epoch = epoch % period
        ramp_epochs = max(1, int(period * ratio))
        if cycle_epoch < ramp_epochs:
            return beta_end * cycle_epoch / ramp_epochs
        return beta_end
    else:
        # Linear warmup (default, covers "linear" and free-bits arms)
        return min(beta_end, beta_end * epoch / max(1, warmup))


def train_arm(arm_name: str):
    arm_cfg = ABLATION_ARMS[arm_name]
    cfg = ExperimentConfig()

    for k, v in arm_cfg.get("model", {}).items():
        setattr(cfg.model, k, v)
    for k, v in arm_cfg.get("loss", {}).items():
        setattr(cfg.loss, k, v)

    # Scale batch size linearly with GPU count (linear scaling rule)
    effective_batch = args.batch_size * N_GPUS
    # Scale LR proportionally (linear scaling rule; Goyal et al. 2017)
    effective_lr = args.lr * N_GPUS

    cfg.data.root_dir = args.data_dir
    cfg.data.batch_size = effective_batch
    cfg.data.target_size = args.image_size
    cfg.data.noise_level = args.noise_level
    cfg.data.num_workers = args.num_workers
    cfg.train.epochs = args.epochs
    cfg.train.lr = effective_lr
    cfg.train.output_dir = args.output_dir
    cfg.train.run_name = arm_name

    out_dir = Path(args.output_dir) / arm_name
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    log_path = out_dir / "train_log.csv"

    print(f"\n{'='*60}")
    print(f"Training {arm_name}")
    print(f"  Loss arm     : {cfg.loss.arm}")
    print(f"  KL sched     : {cfg.loss.kl_schedule}")
    print(f"  VAE          : {cfg.model.use_vae}")
    print(f"  Activation   : {cfg.model.activation}")
    print(f"  GPUs         : {N_GPUS}  (batch={effective_batch}, lr={effective_lr:.2e})")
    print(f"  Output dir   : {out_dir}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = make_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        target_size=args.image_size,
        noise_level=args.noise_level,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    _base_model = PPVAEHformer(
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        num_blocks=cfg.model.num_blocks,
        num_scales=cfg.model.num_scales,
        num_heads=cfg.model.num_heads,
        window_size=cfg.model.window_size,
        use_vae=cfg.model.use_vae,
        activation=cfg.model.activation,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in _base_model.parameters() if p.requires_grad)

    if N_GPUS > 1:
        import torch.nn as nn
        model = nn.DataParallel(_base_model, device_ids=list(range(N_GPUS)))
        print(f"Parameters: {n_params:,}  — DataParallel across {N_GPUS} GPUs")
    else:
        model = _base_model
        print(f"Parameters: {n_params:,}  (single GPU)")

    # Checkpointing always uses the underlying module (unwrap DataParallel)
    _model_core = _base_model

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = PPVAEHformerLoss(
        arm=cfg.loss.arm,
        beta=cfg.loss.beta_end,
        lambda_ssim=cfg.loss.lambda_ssim,
        lambda_ffl=cfg.loss.lambda_ffl,
        lambda_edge=cfg.loss.lambda_edge,
        lambda_perc=cfg.loss.lambda_perc,
        lambda_l1=cfg.loss.lambda_l1,
        lambda_fb=cfg.loss.lambda_fb,
        device=str(DEVICE),
    ).to(DEVICE)

    # ── Optimiser + LR scheduler ──────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.train.epochs, eta_min=1e-6
    )
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_psnr = 0.0
    if args.resume:
        latest = sorted(ckpt_dir.glob("epoch_*.pth"))
        if latest:
            ckpt = torch.load(latest[-1], map_location=DEVICE)
            _model_core.load_state_dict(ckpt["model"])
            optimiser.load_state_dict(ckpt["optimiser"])
            lr_sched.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_psnr = ckpt.get("best_val_psnr", 0.0)
            print(f"Resumed from epoch {start_epoch-1}, best_val_psnr={best_val_psnr:.3f}")

    # ── Log header ────────────────────────────────────────────────────────────
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLS)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        beta = get_beta(epoch, cfg.loss, cfg.model)
        loss_fn.beta = beta

        # Train
        model.train()
        epoch_losses: dict[str, list[float]] = {k: [] for k in
            ["loss", "l_rec", "l_kl", "l_ssim", "l_ffl", "l_edge", "l_perc", "l_l1"]}

        for noisy, clean, _ in loaders["train"]:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            optimiser.zero_grad()

            with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                mu, lsa, z_mu, z_lv = model(noisy)
                losses = loss_fn(clean, mu, lsa, z_mu, z_lv)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            for k in epoch_losses:
                epoch_losses[k].append(losses.get(k, torch.tensor(0.0)).item())

        lr_sched.step()
        avgs = {k: sum(v) / max(1, len(v)) for k, v in epoch_losses.items()}

        # Validate — use _model_core directly so DataParallel kwargs don't break
        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs - 1:
            _model_core.eval()
            vp, vs = [], []
            with torch.no_grad():
                for noisy, clean, _ in loaders["val"]:
                    noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                    mu, _, _, _ = _model_core(noisy, deterministic=True)
                    mu_c = mu.clamp(0, 1)
                    vp.append(psnr(mu_c, clean))
                    vs.append(ssim(mu_c, clean))
            _model_core.train()
            val_psnr_mean = sum(vp) / len(vp)
            val_ssim_mean = sum(vs) / len(vs)
        else:
            val_psnr_mean = float("nan")
            val_ssim_mean = float("nan")

        elapsed = time.time() - t0
        lr_now = optimiser.param_groups[0]["lr"]

        # Save best — always save _model_core weights (without DataParallel wrapper)
        if val_psnr_mean > best_val_psnr:
            best_val_psnr = val_psnr_mean
            torch.save({
                "epoch": epoch,
                "model": _model_core.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": lr_sched.state_dict(),
                "best_val_psnr": best_val_psnr,
            }, out_dir / "best_model.pth")

        # Periodic checkpoint
        if epoch % cfg.train.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model": _model_core.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": lr_sched.state_dict(),
                "best_val_psnr": best_val_psnr,
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")

        # Log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{avgs['loss']:.5f}", f"{avgs['l_rec']:.5f}",
                f"{avgs['l_kl']:.6f}", f"{avgs['l_ssim']:.5f}", f"{avgs['l_ffl']:.5f}",
                f"{avgs['l_edge']:.5f}", f"{avgs['l_perc']:.5f}", f"{avgs['l_l1']:.5f}",
                f"{val_psnr_mean:.3f}", f"{val_ssim_mean:.4f}",
                f"{beta:.6f}", f"{lr_now:.2e}", f"{elapsed:.1f}",
            ])

        if epoch % 10 == 0:
            kl_str = f"kl={avgs['l_kl']:.5f}(β={beta:.4f})" if cfg.model.use_vae else ""
            edge_str = f" edge={avgs['l_edge']:.4f}" if cfg.loss.lambda_edge and "edge" in cfg.loss.arm else ""
            print(
                f"  Ep {epoch:4d}/{cfg.train.epochs} | loss={avgs['loss']:.4f} "
                f"(rec={avgs['l_rec']:.4f} ssim={avgs['l_ssim']:.4f} ffl={avgs['l_ffl']:.4f}"
                f"{edge_str} {kl_str}) | val_psnr={val_psnr_mean:.3f} | "
                f"lr={lr_now:.1e} | {elapsed:.1f}s"
            )
            sys.stdout.flush()

    print(f"\nFinished {arm_name}. Best val PSNR: {best_val_psnr:.3f} dB")
    return best_val_psnr


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if args.arm == "all":
        arm_list = list(ABLATION_ARMS.keys())
    elif args.arm == "new":
        arm_list = NEW_ARMS
    else:
        arm_list = [args.arm]

    results = {}
    for arm_name in arm_list:
        results[arm_name] = train_arm(arm_name)

    if len(results) > 1:
        print("\n=== Ablation Summary ===")
        for k, v in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {k:30s}: {v:.3f} dB")
