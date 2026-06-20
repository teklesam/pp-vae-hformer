#!/bin/bash
#SBATCH --job-name=ppvae_subgroup
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/subgroup_%j.log
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/subgroup_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/compute_subgroup_metrics.py

echo "Subgroup metrics complete."
