#!/bin/bash
# SLURM job array — all 16 arms (A–P), each on 4 A100s (DataParallel).
# All 16 arms fire simultaneously; wall time ≈ 45 min per arm.
# Use this when you want to re-run everything, e.g. after an architecture change.
#
# Submit all 16:  sbatch slurm/array_all_arms.sh
# Submit new only (0-indexed 5–15): sbatch --array=5-15 slurm/array_all_arms.sh
# Submit single:  sbatch --array=7  slurm/array_all_arms.sh
#
# Note: arm_n_perc (index 13) downloads VGG-16 weights from the internet on
# first run. If compute nodes have no internet, pre-cache on login node:
#   python -c "from torchvision.models import vgg16, VGG16_Weights; vgg16(weights=VGG16_Weights.IMAGENET1K_V1)"

#SBATCH -J ppvae_all
#SBATCH --array=0-15          # 16 arms total
#SBATCH --time=12:00:00       # 12 h per arm; 1 GPU → ~9.6 h observed
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --gres=gpu:1          # 1 A100 per task (consistent with array_new_arms.sh)
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/slurm-%A_%a.out

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

# Ordered list matching ABLATION_ARMS insertion order in config.py
ARMS=(
  arm_a_l2          # 0
  arm_b_nll         # 1
  arm_c_nll_ssim    # 2
  arm_d_nll_ssim_ffl # 3
  arm_e_ppvae       # 4
  arm_f_kl_cyc      # 5
  arm_g_kl_fb       # 6
  arm_h_kl_cyc_fb   # 7
  arm_i_l1          # 8
  arm_j_l1_ssim_ffl # 9
  arm_k_nll_l1      # 10
  arm_l_nll_edge_ffl # 11
  arm_m_full_det    # 12
  arm_n_perc        # 13
  arm_o_prelu       # 14
  arm_p_best        # 15
)

ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "====================================="
echo "Job:   ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Arm:   ${ARM}"
echo "GPUs:  ${NGPU}  ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
echo "Node:  $(hostname)"
echo "Time:  $(date)"
echo "====================================="

cd ${PROJECT}

python scripts/train_proposed.py \
    --arm        ${ARM}          \
    --data_dir   ${DATA}         \
    --output_dir ${RESULTS}      \
    --epochs 200                 \
    --batch_size 16              \
    --image_size 256             \
    --noise_level random         \
    --num_workers 8              \
    --num_gpus 1                 \
    --resume

# --resume loads checkpoint.pth if it exists (safe to set even on fresh runs).

echo "Done ${ARM}: $(date)"
