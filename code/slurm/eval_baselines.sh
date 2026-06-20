#!/bin/bash
# SLURM job: evaluate all comparison baselines on Kermany CXR test set.
#
# Single GPU, single node — inference only, no training.
# Expected runtime: ~30–60 min depending on BM3D (CPU-only, slowest).
#
# Prerequisites (run on login node BEFORE submitting):
#   1. Download weights:
#      cd /rds/user/stm43/hpc-work/ppvae_hformer
#      python scripts/download_baseline_weights.py
#
#   2. Confirm data path:
#      ls /rds/user/stm43/hpc-work/ppvae_hformer/data/chest_xray/test/
#
# Submit:
#   sbatch slurm/eval_baselines.sh
#
# To add noise levels (low / mid / high):
#   sbatch slurm/eval_baselines.sh --noise_levels low mid high

#SBATCH -J ppvae_eval_baselines
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /rds/user/stm43/hpc-work/ppvae_results/baselines/eval_%j.out
#SBATCH -e /rds/user/stm43/hpc-work/ppvae_results/baselines/eval_%j.err

set -eo pipefail

module purge
module load rhel8/default-amp

PROJECT=/rds/user/stm43/hpc-work/ppvae_hformer
DATA=/rds/user/stm43/hpc-work/ppvae_hformer/data/chest_xray
OUT=/rds/user/stm43/hpc-work/ppvae_results/baselines

mkdir -p "$OUT"

# Activate conda environment
source /home/stm43/.bashrc
conda activate ppvae

cd "$PROJECT"

echo "=== Baseline Evaluation ==="
echo "Start: $(date)"
echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

# Parse optional extra args forwarded from sbatch command line
EXTRA_ARGS="${@:-}"

python scripts/eval_baselines.py \
    --data_root "$DATA" \
    --output_dir "$OUT" \
    --noise_levels mid \
    --batch_size 16 \
    --num_workers 8 \
    $EXTRA_ARGS

echo
echo "Done: $(date)"
