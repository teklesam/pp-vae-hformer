"""
Training configuration for PP-VAE-Hformer experiments — expanded ablation v2.

Arms A–E: original 5 arms (unchanged for reproducibility)
Arms F–O: new arms (posterior collapse fixes, new losses, PReLU)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    in_channels: int = 1
    base_channels: int = 64
    num_blocks: int = 2
    num_scales: int = 3
    num_heads: int = 4
    window_size: int = 8
    use_vae: bool = True
    activation: str = "gelu"   # "gelu" | "prelu"


@dataclass
class DataConfig:
    root_dir: str = "/rds/user/stm43/hpc-work/chest_xray"
    target_size: int = 256
    noise_level: str = "mid"   # 'low', 'mid', 'high', 'random'
    batch_size: int = 16
    num_workers: int = 8
    val_fraction: float = 0.15
    seed: int = 42


@dataclass
class LossConfig:
    arm: str = "nll+ssim+ffl+kl"
    # KL schedule: "linear" | "cyclical" | (free-bits encoded in arm string via kl_fb/kl_cyc_fb)
    kl_schedule: str = "linear"
    # KL cyclical params (used when kl_schedule="cyclical")
    kl_cycle_period: int = 50    # epochs per cycle
    kl_cycle_ratio: float = 0.5  # fraction of cycle spent ramping
    # Beta range
    beta_start: float = 0.0
    beta_end: float = 0.001
    beta_warmup_epochs: int = 20
    # Loss weights
    lambda_ssim: float = 0.5
    lambda_ffl: float = 0.1
    lambda_edge: float = 0.1
    lambda_perc: float = 0.05
    lambda_l1: float = 0.5
    lambda_fb: float = 0.25     # free-bits threshold per channel (nats)


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    checkpoint_every: int = 10
    eval_every: int = 5
    output_dir: str = "/rds/user/stm43/hpc-work/ppvae_results"
    run_name: str = "arm_e_ppvae"
    mixed_precision: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ── Ablation arm registry ─────────────────────────────────────────────────────
# Each entry overrides only the fields that differ from ExperimentConfig defaults.

ABLATION_ARMS: dict[str, dict] = {

    # ── Original 5 arms (A–E) — unchanged ───────────────────────────────────
    "arm_a_l2": {
        "model": {"use_vae": False},
        "loss": {"arm": "l2"},
        "train": {"run_name": "arm_a_l2"},
    },
    "arm_b_nll": {
        "model": {"use_vae": False},
        "loss": {"arm": "nll"},
        "train": {"run_name": "arm_b_nll"},
    },
    "arm_c_nll_ssim": {
        "model": {"use_vae": False},
        "loss": {"arm": "nll+ssim"},
        "train": {"run_name": "arm_c_nll_ssim"},
    },
    "arm_d_nll_ssim_ffl": {
        "model": {"use_vae": False},
        "loss": {"arm": "nll+ssim+ffl"},
        "train": {"run_name": "arm_d_nll_ssim_ffl"},
    },
    "arm_e_ppvae": {
        "model": {"use_vae": True},
        "loss": {"arm": "nll+ssim+ffl+kl", "kl_schedule": "linear"},
        "train": {"run_name": "arm_e_ppvae"},
    },

    # ── Arm F: VAE + cyclical KL annealing (posterior collapse fix #1) ──────
    # Fu et al. (2019): periodically resetting β allows the posterior to
    # "breathe" and recover from incipient collapse. Period=50 → 4 cycles
    # over 200 epochs. The final cycle converges with a fully annealed β.
    "arm_f_kl_cyc": {
        "model": {"use_vae": True},
        "loss": {
            "arm": "nll+ssim+ffl+kl_cyc",
            "kl_schedule": "cyclical",
            "kl_cycle_period": 50,
            "kl_cycle_ratio": 0.5,
        },
        "train": {"run_name": "arm_f_kl_cyc"},
    },

    # ── Arm G: VAE + free bits (posterior collapse fix #2) ──────────────────
    # Kingma et al. (2016): enforce KL per channel ≥ λ_fb nats, preventing
    # complete collapse of individual latent channels.
    "arm_g_kl_fb": {
        "model": {"use_vae": True},
        "loss": {
            "arm": "nll+ssim+ffl+kl_fb",
            "kl_schedule": "linear",
            "lambda_fb": 0.25,
        },
        "train": {"run_name": "arm_g_kl_fb"},
    },

    # ── Arm H: VAE + cyclical KL + free bits (collapse fix #1 + #2) ─────────
    # Combines both: cyclical annealing prevents global collapse while free
    # bits prevents per-channel collapse during high-β phases.
    "arm_h_kl_cyc_fb": {
        "model": {"use_vae": True},
        "loss": {
            "arm": "nll+ssim+ffl+kl_cyc_fb",
            "kl_schedule": "cyclical",
            "kl_cycle_period": 50,
            "kl_cycle_ratio": 0.5,
            "lambda_fb": 0.25,
        },
        "train": {"run_name": "arm_h_kl_cyc_fb"},
    },

    # ── Arm I: L1 base (compare robustness with L2/NLL) ─────────────────────
    # L1/MAE is more robust than L2 and avoids over-smoothing slightly
    # better due to its heavy-tailed gradient. Direct comparison to Arm A.
    "arm_i_l1": {
        "model": {"use_vae": False},
        "loss": {"arm": "l1"},
        "train": {"run_name": "arm_i_l1"},
    },

    # ── Arm J: L1 + SSIM + FFL (L1-based version of Arm D) ─────────────────
    # Tests whether switching the base from NLL (Arm D) to L1 improves
    # the frequency-domain and structural regularisation combination.
    "arm_j_l1_ssim_ffl": {
        "model": {"use_vae": False},
        "loss": {"arm": "l1+ssim+ffl"},
        "train": {"run_name": "arm_j_l1_ssim_ffl"},
    },

    # ── Arm K: NLL + L1 hybrid (robustness + uncertainty) ───────────────────
    # Combines heteroscedastic NLL (aleatoric uncertainty) with L1 as
    # an auxiliary robustness term. Hypothesis: L1 gradient stabilises
    # early training while NLL drives calibration.
    "arm_k_nll_l1": {
        "model": {"use_vae": False},
        "loss": {"arm": "nll+l1", "lambda_l1": 0.5},
        "train": {"run_name": "arm_k_nll_l1"},
    },

    # ── Arm L: NLL + Edge + FFL (Sobel supervision instead of SSIM) ─────────
    # Hypothesis: direct Sobel gradient supervision preserves clinical
    # boundary detail (cardiac silhouette, consolidation margins) better
    # than SSIM, which operates on patch statistics.
    "arm_l_nll_edge_ffl": {
        "model": {"use_vae": False},
        "loss": {"arm": "nll+edge+ffl", "lambda_edge": 0.1},
        "train": {"run_name": "arm_l_nll_edge_ffl"},
    },

    # ── Arm M: NLL + SSIM + Edge + FFL (Arm D + edge supervision) ───────────
    # Full deterministic kitchen sink: tests whether adding explicit
    # Sobel gradient loss on top of Arm D adds orthogonal information.
    "arm_m_full_det": {
        "model": {"use_vae": False},
        "loss": {"arm": "nll+ssim+edge+ffl", "lambda_edge": 0.1},
        "train": {"run_name": "arm_m_full_det"},
    },

    # ── Arm N: NLL + Perceptual + SSIM + FFL (VGG-16 feature matching) ──────
    # Perceptual loss (Johnson 2016) via frozen VGG-16. Tests whether
    # high-level feature-space supervision improves perceived image quality
    # on CXR (cross-domain transfer from ImageNet as in Bredell 2023).
    "arm_n_perc": {
        "model": {"use_vae": False},
        "loss": {"arm": "nll+perc+ssim+ffl", "lambda_perc": 0.05},
        "train": {"run_name": "arm_n_perc"},
    },

    # ── Arm O: Arm D architecture with PReLU (NLL+SSIM+FFL, PReLU) ──────────
    # PReLU (He et al. 2015) has a learnable negative slope per channel,
    # potentially preserving subtle low-intensity features in CXR that
    # GELU's fixed activation curve clips. Direct comparison to Arm D.
    "arm_o_prelu": {
        "model": {"use_vae": False, "activation": "prelu"},
        "loss": {"arm": "nll+ssim+ffl"},
        "train": {"run_name": "arm_o_prelu"},
    },

    # ── Arm P: Best VAE + PReLU (collapse-fixed full model) ─────────────────
    # Combines best posterior collapse fix (cyclical + free bits, Arm H)
    # with PReLU activation. This is the proposed PP-VAE-Hformer v2.
    "arm_p_best": {
        "model": {"use_vae": True, "activation": "prelu"},
        "loss": {
            "arm": "nll+ssim+edge+ffl+kl_cyc_fb",
            "kl_schedule": "cyclical",
            "kl_cycle_period": 50,
            "kl_cycle_ratio": 0.5,
            "lambda_fb": 0.25,
            "lambda_edge": 0.1,
        },
        "train": {"run_name": "arm_p_best"},
    },

    # ── Arm Q: Charbonnier base loss ─────────────────────────────────────────
    # Charbonnier (sqrt(r^2 + eps^2) - eps, eps=1e-3) is a smooth L1
    # approximation used by SwinIR (Liang 2021) and cited in the lit review.
    # Because L1 ≈ L2 on this task (Arms I vs A: d=0.164, n.s.), Charbonnier
    # is expected to also be equivalent. This arm closes the loop on all three
    # pixel-norm baselines. Direct comparison to Arms A (L2) and I (L1).
    "arm_q_charb": {
        "model": {"use_vae": False},
        "loss": {"arm": "charb"},
        "train": {"run_name": "arm_q_charb"},
    },
}

# Quick lookup sets for SLURM array jobs
ORIGINAL_ARMS = ["arm_a_l2", "arm_b_nll", "arm_c_nll_ssim", "arm_d_nll_ssim_ffl", "arm_e_ppvae"]
NEW_ARMS = [k for k in ABLATION_ARMS if k not in ORIGINAL_ARMS]
VAE_ARMS = [k for k in ABLATION_ARMS if ABLATION_ARMS[k].get("model", {}).get("use_vae", False)]
DET_ARMS = [k for k in ABLATION_ARMS if not ABLATION_ARMS[k].get("model", {
    # ── Arm R: Fine-tuned A → L1+SSIM+FFL (curriculum: L2 pretrain → texture loss) ─
    # Tests Zhao 2017 two-stage curriculum: stable L2 pretraining followed by
    # loss-switch to L1+SSIM+FFL. Compare to Arm J (L1+SSIM+FFL from scratch)
    # to isolate the benefit of L2 initialisation.
    "arm_r_ft_j": {
        "model": {"use_vae": False},
        "loss":  {"arm": "l1+ssim+ffl"},
        "train": {"run_name": "arm_r_ft_j"},
    },

    # ── Arm S: Fine-tuned A → NLL+SSIM+FFL (curriculum: L2 → uncertainty loss) ──
    # As Arm R but switches to heteroscedastic NLL+SSIM+FFL (Arm D's loss).
    # First 20 fine-tuning epochs blend L2 and NLL to warm up the sigma head.
    # Compare to Arm D (NLL+SSIM+FFL from scratch).
    "arm_s_ft_d": {
        "model": {"use_vae": False},
        "loss":  {"arm": "nll+ssim+ffl"},
        "train": {"run_name": "arm_s_ft_d"},
    },
}
