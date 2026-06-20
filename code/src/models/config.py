"""
Training configuration for PP-VAE-Hformer experiments.
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
    arm: str = "nll+ssim+ffl+kl"  # see composite_loss.py
    beta_start: float = 0.0
    beta_end: float = 0.001
    beta_warmup_epochs: int = 20
    lambda_ssim: float = 0.5
    lambda_ffl: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    scheduler: str = "cosine"      # 'cosine' | 'step'
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


# Ablation arm presets
ABLATION_ARMS: dict[str, dict] = {
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
        "loss": {"arm": "nll+ssim+ffl+kl"},
        "train": {"run_name": "arm_e_ppvae"},
    },
}
