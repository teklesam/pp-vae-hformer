#!/bin/bash
#SBATCH --job-name=ppvae_figs_v2
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/gen_figures_v2_%j.log
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/gen_figures_v2_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

# Install umap-learn and sklearn if not present
pip install umap-learn scikit-learn --quiet 2>/dev/null

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/generate_figures_v2.py

echo "Done — figures at /rds/user/stm43/hpc-work/ppvae_results/figures_v2/"
