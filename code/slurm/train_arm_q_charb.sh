#!/bin/bash
# SLURM job — Arm Q: Charbonnier base loss ablation.
#
# Purpose: close the L2 vs L1 vs Charbonnier comparison.
# Expected outcome: Arm Q ≈ Arm A (L2) ≈ Arm I (L1), confirming pixel-norm
# choice is not a critical hyperparameter for Poisson-Gaussian CXR denoising.
# Charbonnier (eps=1e-3) = smooth L1 used by SwinIR and cited in the lit review.
#
# Submit:   sbatch slurm/train_arm_q_charb.sh
# Monitor:  squeue -u stm43
# Expected wall time: ~10 h on A100

#SBATCH -J ppvae_arm_q
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/slurm-%j_arm_q.out

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

PROJECT=/rds/user/stm43/hpc-work/ppvae_hformer
DATA=/rds/user/stm43/hpc-work/chest_xray
RESULTS=/rds/user/stm43/hpc-work/ppvae_results

mkdir -p ${RESULTS}/logs

echo "====================================="
echo "Job:   ${SLURM_JOB_ID}"
echo "Arm:   arm_q_charb"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time:  $(date)"
echo "====================================="

cd ${PROJECT}

python scripts/train_proposed.py \
    --arm        arm_q_charb    \
    --data_dir   ${DATA}        \
    --output_dir ${RESULTS}     \
    --epochs     200            \
    --batch_size 16             \
    --image_size 256            \
    --noise_level random        \
    --num_workers 8             \
    --num_gpus 1                \
    --resume

echo "Done arm_q_charb: $(date)"
