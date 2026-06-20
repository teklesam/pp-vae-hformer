# PP-VAE-Hformer: Pathology-Preserving Variational Autoencoder for Paediatric Chest X-Ray Denoising

**MPhil Dissertation — University of Cambridge**  
**Samuel Tekle | Department of Applied Mathematics and Theoretical Physics / MRC Biostatistics Unit**

---

## Overview

This repository contains the full research artefacts for my MPhil dissertation on **pathology-preserving image denoising of paediatric chest X-rays (CXR)**. The central contribution is **PP-VAE-Hformer**: a hybrid architecture combining a Variational Autoencoder (VAE) with a Hierarchical Swin Transformer (Hformer) encoder-decoder that jointly denoises X-ray images while estimating both *aleatoric* (noise-inherent) and *epistemic* (model) uncertainty.

The motivation: standard deep denoising networks (DnCNN, FFDNet, DRUNet) optimise for PSNR/SSIM, which rewards smooth reconstructions. On CXR, this erases clinically critical fine-structure — vascular markings, consolidation margins, costophrenic angles. PP-VAE-Hformer uses a **heteroscedastic NLL loss + structural regularisers** to penalise pathology-destroying smoothness directly.

---

## Key Results

| Arm | Loss | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Notes |
|-----|------|------------|--------|---------|-------|
| A | L2 (MSE) | **34.62** | 0.913 | 0.201 | Smoothest; loses vascular detail |
| D | NLL+SSIM+FFL | 33.44 | 0.909 | **0.158** | Best LPIPS; best uncertainty calibration |
| J | L1+SSIM+FFL | 34.32 | **0.926** | **0.127** | Best perceptual quality |
| H | PP-VAE (cyc+fb) | 32.76 | 0.898 | 0.182 | Best VAE arm; ARI=0.187 |
| F | PP-VAE (cyc KL) | 32.73 | 0.900 | 0.179 | Highest latent class sep. ARI=0.310 |
| O | NLL+SSIM+FFL (PReLU) | 29.07 | 0.795 | 0.418 | PReLU catastrophic failure |

**Key finding:** PSNR-optimal (Arm A, L2) produces plastically smooth images that destroy vascular markings. Perceptual-metric-optimal arms (J, D) achieve lower PSNR but preserve diagnostic detail — demonstrating the PSNR/clinical-quality dissociation in medical imaging.

---

## Architecture

```
Input (noisy CXR) ──▶ Swin-Transformer Encoder ──▶ VAE Bottleneck ──▶ Swin-Transformer Decoder ──▶ μ̂ (reconstruction)
                                                    ↕ z ~ q(z|x)           └──▶ log σ̂²_a (aleatoric uncertainty)
                       Monte Carlo sampling ────────▶ σ̂_e (epistemic uncertainty)
```

- **Encoder**: 3-scale hierarchical Swin-Transformer blocks with shifted windows
- **VAE bottleneck**: Reparameterised latent z with KL regularisation (cyclical annealing + free-bits)
- **Decoder**: Symmetric Swin-Transformer with skip connections
- **Outputs**: Reconstruction μ̂, aleatoric log-variance log σ̂²_a, (optionally) epistemic σ̂_e via MC sampling

---

## 16-Arm Ablation

The study systematically ablates loss functions across 16 arms (A–P) + 5 retrained baselines:

| Group | Arms | Hypothesis tested |
|-------|------|-------------------|
| Pixel norms | A (L2), I (L1), Q (Charbonnier) | L1 vs L2 vs smooth-L1 |
| Uncertainty | B (NLL), C (NLL+SSIM), D (NLL+SSIM+FFL) | Heteroscedastic calibration |
| VAE variants | E–H, P | Posterior collapse fixes (cyclical KL, free bits) |
| Frequency | D, J, L | Focal Frequency Loss for texture preservation |
| Perceptual | N | VGG-16 feature matching |
| Architecture | O | PReLU vs GELU in decoder |
| Curriculum FT | R, S | L2 pretrain → loss switch (this work) |

---

## Repository Structure

```
pp-vae-hformer/
├── dissertation/           ← Full LaTeX source (synced with Overleaf)
│   ├── main.tex            ← Master document
│   ├── figures/            ← All evaluation figures (PNGs)
│   └── References.bib      ← Bibliography
│
├── code/
│   ├── src/
│   │   ├── models/         ← PPVAEHformer architecture
│   │   ├── losses/         ← Composite loss (NLL, SSIM, FFL, KL, Edge, Perceptual)
│   │   ├── data/           ← Kermany CXR dataset loader with FOI noise model
│   │   ├── training/       ← Config, arm registry, training utilities
│   │   └── evaluation/     ← PSNR, SSIM, NLL metrics
│   │
│   ├── scripts/
│   │   ├── train_proposed.py         ← Main 16-arm ablation training
│   │   ├── finetune_from_l2.py       ← Two-stage curriculum fine-tuning
│   │   ├── evaluate_all.py           ← Full evaluation (PSNR/SSIM/NLL)
│   │   ├── compute_fsim.py           ← FSIM + LPIPS held-out metrics
│   │   ├── compute_subgroup_metrics.py ← Bacterial/Viral subgroup PSNR
│   │   ├── generate_figures_v2.py    ← All dissertation figures
│   │   └── generate_roi_panels_v2.py ← ROI comparison panels
│   │
│   └── slurm/              ← CSD3 (Cambridge HPC) SLURM job scripts
│
└── results/                ← Evaluation CSVs (no model weights)
    ├── metrics_summary.csv
    ├── pairwise_stats.csv
    ├── per_image_metrics.csv
    ├── subgroup_metrics.csv
    └── fsim_results.csv
```

---

## Noise Model

The **FOI (Flat-panel detector Optical Imaging) Poisson-Gaussian model** is used throughout:

```
Var(z | y) = a·y + b
```

where `a = 0.03` (Poisson coefficient) and `b = 0.005` (Gaussian floor), giving effective `σ_eff ≈ 0.122` at mid preset. Three noise levels (low/mid/high) are evaluated.

---

## Training

Models trained on [Kermany et al. (2018)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) paediatric CXR dataset (5,216 train / 624 test images, Normal + Bacterial/Viral Pneumonia) on **Cambridge CSD3 A100 GPU** via SLURM.

```bash
# Train a single arm
python code/scripts/train_proposed.py --arm arm_d_nll_ssim_ffl --data_dir /path/to/chest_xray

# Two-stage curriculum fine-tuning (L2 → NLL+SSIM+FFL)
python code/scripts/finetune_from_l2.py --arm arm_s_ft_d --ft_lr 2e-5 --ft_epochs 100 --blend_epochs 20

# FSIM + LPIPS evaluation (held-out perceptual metrics)
python code/scripts/compute_fsim.py
```

---

## Dependencies

```
torch>=2.0
torchvision
numpy
scipy
Pillow
matplotlib
piq>=0.8.0      # FSIM + LPIPS
scikit-learn    # ARI (latent space evaluation)
umap-learn      # UMAP projections
```

---

## Citation

If you use this work, please cite:

```
Tekle, S. (2026). Pathology-Preserving Variational Autoencoder for Paediatric 
Chest X-Ray Denoising. MPhil Dissertation, University of Cambridge.
```

---

## Status (Updated Daily)

| Component | Status |
|-----------|--------|
| 16-arm ablation training | ✅ Complete (Arms A–P) |
| Arm Q (Charbonnier) | 🔄 Training |
| Baseline retraining (FFDNet, IRCNN, DRUNet, SwinIR) | 🔄 Training |
| FSIM/LPIPS evaluation | 🔄 Running |
| Curriculum fine-tuning (Arms R, S) | 🔄 Running |
| Dissertation write-up | 🔄 Active (§1–6 complete, figures updating) |
| Epistemic uncertainty figures | 🔄 Regenerating (fixed MC variance) |
