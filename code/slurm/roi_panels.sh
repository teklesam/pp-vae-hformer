#!/bin/bash
#SBATCH --job-name=ppvae_roi_v2
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/roi_panels_v2_%j.log
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/roi_panels_v2_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/generate_roi_panels_v2.py

echo "ROI panels complete — saved to /rds/user/stm43/hpc-work/ppvae_results/roi_panels_v2/"
