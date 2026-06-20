"""
finetune_from_l2.py — Two-stage curriculum fine-tuning.

Stage 1 (already done): Train with L2 loss → Arm A converged at 34.49 dB.
Stage 2 (this script): Load Arm A weights → fine-tune with richer loss.

Arms:
  arm_r_ft_j : Arm A → L1+SSIM+FFL   (safe: no variance head change)
  arm_s_ft_d : Arm A → NLL+SSIM+FFL  (sigma head warms up from random init)

For Arm S, the first --blend_epochs epochs use a linear blend:
  loss = (1 - alpha) * L2 + alpha * (NLL+SSIM+FFL)
so the sigma head initialises gradually without causing NLL explosion.

Usage:
  python scripts/finetune_from_l2.py --arm arm_r_ft_j --ft_lr 5e-5 --ft_epochs 100
  python scripts/finetune_from_l2.py --arm arm_s_ft_d --ft_lr 2e-5 --ft_epochs 100 --blend_epochs 20
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.amp import GradScaler, autocast

from src.data.kermany_dataset import make_loaders
from src.models.ppvae_hformer import PPVAEHformer
from src.losses.composite_loss import PPVAEHformerLoss
from src.evaluation.metrics import psnr, ssim
from src.training.config import ExperimentConfig, ABLATION_ARMS

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True,
                    choices=["arm_r_ft_j", "arm_s_ft_d"])
parser.add_argument("--pretrain_arm", default="arm_a_l2",
                    help="Which arm checkpoint to load as starting weights")
parser.add_argument("--data_dir", default="/rds/user/stm43/hpc-work/chest_xray")
parser.add_argument("--output_dir", default="/rds/user/stm43/hpc-work/ppvae_results")
parser.add_argument("--ft_lr", type=float, default=5e-5)
parser.add_argument("--ft_epochs", type=int, default=100)
parser.add_argument("--blend_epochs", type=int, default=0,
                    help="For NLL arms: blend L2+target loss for first N epochs")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LOG_COLS = ["epoch","train_loss","l_rec","l_kl","l_ssim","l_ffl",
            "l_edge","l_perc","l_l1","val_psnr","val_ssim","beta","lr","time_s"]

# ── Arm configs ────────────────────────────────────────────────────────────────
FT_ARMS = {
    "arm_r_ft_j": {
        "model": {"use_vae": False},
        "loss":  {"arm": "l1+ssim+ffl"},
    },
    "arm_s_ft_d": {
        "model": {"use_vae": False},
        "loss":  {"arm": "nll+ssim+ffl"},
    },
}

arm_cfg = FT_ARMS[args.arm]
cfg = ExperimentConfig()
for k, v in arm_cfg["model"].items():
    setattr(cfg.model, k, v)
for k, v in arm_cfg["loss"].items():
    setattr(cfg.loss, k, v)

out_dir  = Path(args.output_dir) / args.arm
ckpt_dir = out_dir / "checkpoints"
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(exist_ok=True)
log_path = out_dir / "train_log.csv"

print(f"\n{'='*60}")
print(f"Fine-tuning: {args.pretrain_arm} → {args.arm}")
print(f"  New loss    : {cfg.loss.arm}")
print(f"  LR          : {args.ft_lr:.1e}")
print(f"  Epochs      : {args.ft_epochs}")
print(f"  Blend epochs: {args.blend_epochs}")
print(f"  Output      : {out_dir}")
print(f"{'='*60}\n")

# ── Data ───────────────────────────────────────────────────────────────────────
loaders = make_loaders(
    root_dir=args.data_dir,
    batch_size=args.batch_size,
    target_size=256,
    noise_level="random",
    num_workers=args.num_workers,
)
print(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)}")

# ── Model: build from pretrain arch, load pretrain weights ────────────────────
model = PPVAEHformer(
    in_channels=cfg.model.in_channels,
    base_channels=cfg.model.base_channels,
    num_blocks=cfg.model.num_blocks,
    num_scales=cfg.model.num_scales,
    num_heads=cfg.model.num_heads,
    window_size=cfg.model.window_size,
    use_vae=cfg.model.use_vae,
    activation=cfg.model.activation,
).to(DEVICE)

pretrain_ckpt = Path(args.output_dir) / args.pretrain_arm / "best_model.pth"
ckpt = torch.load(pretrain_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded pretrain weights from epoch {ckpt.get('epoch','?')} "
      f"(best val PSNR {ckpt.get('best_val_psnr', 0):.3f} dB)")

# Save the pretrain starting point so we can plot before/after
torch.save(ckpt, out_dir / "pretrain_start.pth")

# ── Target loss function ───────────────────────────────────────────────────────
loss_fn = PPVAEHformerLoss(
    arm=cfg.loss.arm,
    beta=0.0,
    lambda_ssim=cfg.loss.lambda_ssim,
    lambda_ffl=cfg.loss.lambda_ffl,
    lambda_edge=cfg.loss.lambda_edge,
    lambda_perc=cfg.loss.lambda_perc,
    lambda_l1=cfg.loss.lambda_l1,
    lambda_fb=cfg.loss.lambda_fb,
    device=str(DEVICE),
).to(DEVICE)

# L2 blend loss (for NLL arm sigma head warmup)
l2_loss_fn = PPVAEHformerLoss(arm="l2", beta=0.0, device=str(DEVICE)).to(DEVICE)

# ── Optimiser + cosine schedule ────────────────────────────────────────────────
optimiser = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=1e-4)
lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=args.ft_epochs, eta_min=1e-7)
scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

best_val_psnr = 0.0
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(LOG_COLS)

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(args.ft_epochs):
    t0 = time.time()

    # Blend coefficient: 1.0 → pure L2, 0.0 → pure target loss
    if args.blend_epochs > 0 and epoch < args.blend_epochs:
        l2_alpha = 1.0 - epoch / args.blend_epochs
    else:
        l2_alpha = 0.0

    model.train()
    epoch_losses = {k: [] for k in ["loss","l_rec","l_kl","l_ssim","l_ffl","l_edge","l_perc","l_l1"]}

    for noisy, clean, _ in loaders["train"]:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        optimiser.zero_grad()

        with autocast("cuda", enabled=(DEVICE.type == "cuda")):
            mu, lsa, z_mu, z_lv = model(noisy)
            target_losses = loss_fn(clean, mu, lsa, z_mu, z_lv)

            if l2_alpha > 0:
                l2_losses = l2_loss_fn(clean, mu, lsa, z_mu, z_lv)
                total = (1 - l2_alpha) * target_losses["loss"] + l2_alpha * l2_losses["loss"]
                combined = dict(target_losses)
                combined["loss"] = total
            else:
                combined = target_losses

        scaler.scale(combined["loss"]).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimiser)
        scaler.update()

        for k in epoch_losses:
            epoch_losses[k].append(combined.get(k, torch.tensor(0.0)).item())

    lr_sched.step()
    avgs = {k: sum(v)/max(1,len(v)) for k,v in epoch_losses.items()}

    # Validate every 5 epochs and at the last
    if epoch % 5 == 0 or epoch == args.ft_epochs - 1:
        model.eval()
        vp, vs = [], []
        with torch.no_grad():
            for noisy, clean, _ in loaders["val"]:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                mu, _, _, _ = model(noisy, deterministic=True)
                mu_c = mu.clamp(0, 1)
                vp.append(psnr(mu_c, clean))
                vs.append(ssim(mu_c, clean))
        model.train()
        val_psnr_mean = sum(vp)/len(vp)
        val_ssim_mean = sum(vs)/len(vs)
    else:
        val_psnr_mean = val_ssim_mean = float("nan")

    elapsed = time.time() - t0
    lr_now  = optimiser.param_groups[0]["lr"]

    if val_psnr_mean > best_val_psnr:
        best_val_psnr = val_psnr_mean
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": lr_sched.state_dict(),
                    "best_val_psnr": best_val_psnr},
                   out_dir / "best_model.pth")

    if epoch % 10 == 0:
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": lr_sched.state_dict(),
                    "best_val_psnr": best_val_psnr},
                   ckpt_dir / f"epoch_{epoch:04d}.pth")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch, f"{avgs['loss']:.5f}", f"{avgs['l_rec']:.5f}",
            f"{avgs['l_kl']:.6f}", f"{avgs['l_ssim']:.5f}", f"{avgs['l_ffl']:.5f}",
            f"{avgs['l_edge']:.5f}", f"{avgs['l_perc']:.5f}", f"{avgs['l_l1']:.5f}",
            f"{val_psnr_mean:.3f}", f"{val_ssim_mean:.4f}",
            "0.000000", f"{lr_now:.2e}", f"{elapsed:.1f}",
        ])

    if epoch % 5 == 0:
        blend_str = f" [blend α={l2_alpha:.2f}]" if l2_alpha > 0 else ""
        print(f"  Ep {epoch:3d}/{args.ft_epochs} | loss={avgs['loss']:.4f} "
              f"(rec={avgs['l_rec']:.4f} ssim={avgs['l_ssim']:.4f} "
              f"ffl={avgs['l_ffl']:.4f} l1={avgs['l_l1']:.4f}){blend_str} | "
              f"val_psnr={val_psnr_mean:.3f} | lr={lr_now:.1e} | {elapsed:.1f}s")
        sys.stdout.flush()

print(f"\nFinished {args.arm}. Best val PSNR: {best_val_psnr:.3f} dB")
