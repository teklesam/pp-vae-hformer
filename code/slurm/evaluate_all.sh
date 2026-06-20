#!/bin/bash
# Comprehensive evaluation: all 16 ablation arms + trained baselines on Kermany CXR test set.
#
# - Arms missing best_model.pth are silently skipped (safe to submit before all arms finish)
# - Run again after SwinIR completes to add swinir row to the table
#
# Submit:
#   sbatch slurm/evaluate_all.sh

#SBATCH -J ppvae_eval
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /rds/user/stm43/hpc-work/ppvae_results/evaluation/eval_%j.out
#SBATCH -e /rds/user/stm43/hpc-work/ppvae_results/evaluation/eval_%j.err

set -eo pipefail

module purge
module load rhel8/default-amp

PROJECT=/rds/user/stm43/hpc-work/ppvae_hformer
DATA=/rds/user/stm43/hpc-work/chest_xray
RESULTS=/rds/user/stm43/hpc-work/ppvae_results
OUT=${RESULTS}/evaluation

mkdir -p "$OUT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

cd "$PROJECT"

echo "=== Comprehensive Evaluation ==="
echo "Start : $(date)"
echo "Node  : $SLURMD_NODENAME"
echo "GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Output: $OUT"
echo

python scripts/evaluate_all.py \
    --data_dir    "$DATA"    \
    --results_dir "$RESULTS" \
    --output_dir  "$OUT"     \
    --noise_levels low mid high \
    --mc_samples  20         \
    --num_workers 8          \
    --bootstrap_n 1000

echo
echo "Done: $(date)"
