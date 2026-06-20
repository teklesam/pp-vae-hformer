#!/bin/bash
# SLURM job array — 11 new arms (F–P), each on 1 A100.
# All arms fire simultaneously; wall time = time for one arm (~2 h).
#
# Submit:   sbatch slurm/array_new_arms.sh
# Monitor:  squeue -u stm43
# Logs:     /rds/.../ppvae_results/<arm_name>/train_log.csv
#           slurm-<jobid>_<taskid>.out

#SBATCH -J ppvae_new
#SBATCH --array=0-10          # 11 tasks: indices 0–10 → arms F–P
#SBATCH --time=12:00:00       # 12 h per arm (~172s/epoch × 200 = ~9.6 h observed)
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --gres=gpu:1          # 1 A100 per task
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

# Map task index → arm name (same order as NEW_ARMS in config.py)
ARMS=(
  arm_f_kl_cyc
  arm_g_kl_fb
  arm_h_kl_cyc_fb
  arm_i_l1
  arm_j_l1_ssim_ffl
  arm_k_nll_l1
  arm_l_nll_edge_ffl
  arm_m_full_det
  arm_n_perc
  arm_o_prelu
  arm_p_best
)

ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

echo "====================================="
echo "Job:   ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Arm:   ${ARM}"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time:  $(date)"
echo "====================================="

cd ${PROJECT}

python scripts/train_proposed.py \
    --arm   ${ARM}    \
    --data_dir  ${DATA}     \
    --output_dir ${RESULTS}  \
    --epochs 200             \
    --batch_size 16          \
    --image_size 256         \
    --noise_level random     \
    --num_workers 8          \
    --num_gpus 1             \
    --resume

echo "Done ${ARM}: $(date)"
