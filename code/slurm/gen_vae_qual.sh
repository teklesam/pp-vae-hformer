#!/bin/bash
#SBATCH --job-name=ppvae_vae_qual
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_hformer/logs/gen_vae_qual_%j.log

source ~/.bashrc
conda activate ppvae
cd /rds/user/stm43/hpc-work/ppvae_hformer
python scripts/gen_vae_qual_only.py
