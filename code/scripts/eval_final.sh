#!/bin/bash
#SBATCH --job-name=eval_final
#SBATCH --account=TRAINING-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/%j_eval_final.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stm43@cam.ac.uk

module load cuda/12.1
source ~/.bashrc
conda activate ppvae

cd /rds/user/stm43/hpc-work/ppvae_hformer

python scripts/evaluate_all.py \
  --data_dir    /rds/user/stm43/hpc-work \
  --results_dir /rds/user/stm43/hpc-work/ppvae_results \
  --output_dir  /rds/user/stm43/hpc-work/ppvae_results/evaluation_final \
  --arms arm_a_l2 arm_b_nll arm_c_nll_ssim arm_d_nll_ssim_ffl arm_e_ppvae dncnn_baseline \
  --noise_levels 100 200 300 \
  --bootstrap_n 1000
