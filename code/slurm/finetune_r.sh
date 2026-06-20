#!/bin/bash
#SBATCH --job-name=ppvae_ft_r
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=40G
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/finetune_r_%j.log

source ~/.bashrc
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer
python scripts/finetune_from_l2.py \
    --arm arm_r_ft_j \
    --ft_lr 5e-5 \
    --ft_epochs 100 \
    --blend_epochs 0
