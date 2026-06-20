#!/bin/bash
# Train ablation arms F–P (new arms only; A–E must already be trained).
# Submit: sbatch slurm/train_new_arms.sh [arm_name]
# Default: runs all new arms sequentially (~9–11 h on A100 40 GB).
#
# To train a single arm:
#   sbatch slurm/train_new_arms.sh arm_f_kl_cyc
#
# To split across two jobs:
#   sbatch slurm/train_new_arms_1.sh   (F–J)
#   sbatch slurm/train_new_arms_2.sh   (K–P)

#SBATCH -J ppvae_new_arms
#SBATCH --time=30:00:00
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --gres=gpu:1

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

echo "Job:    ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Time:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python --version)"

cd ${PROJECT}

# Smoke test
echo ""
echo "=== Smoke test ==="
python scripts/smoke_test.py --data_dir ${DATA} --image_size 256
if [ $? -ne 0 ]; then echo "SMOKE TEST FAILED"; exit 1; fi
echo "=== Smoke test passed ==="
echo ""

# Select which arm(s) to run
ARM=${1:-new}

python scripts/train_proposed.py \
    --arm ${ARM} \
    --data_dir ${DATA} \
    --output_dir ${RESULTS} \
    --epochs 200 \
    --batch_size 16 \
    --image_size 256 \
    --noise_level random \
    --num_workers 8

echo ""
echo "Done: $(date)"
