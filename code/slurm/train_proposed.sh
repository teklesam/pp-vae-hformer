#!/bin/bash
#SBATCH -J ppvae_hformer
#SBATCH --time=36:00:00
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=stm43@cam.ac.uk
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -o /rds/user/stm43/hpc-work/ppvae_results/logs/%j_ppvae.out
#SBATCH -e /rds/user/stm43/hpc-work/ppvae_results/logs/%j_ppvae.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/stm43/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

PROJECT=/rds/user/stm43/hpc-work/ppvae_hformer
DATA=/home/stm43/chest_xray
RESULTS=/rds/user/stm43/hpc-work/ppvae_results

mkdir -p ${RESULTS}/logs

echo "Job:    ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Time:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python --version)"

cd ${PROJECT}

# -- Run smoke test first (must pass before full training) --------------------
echo ""
echo "=== Running smoke test ==="
python scripts/smoke_test.py --data_dir ${DATA} --image_size 256
SMOKE_EXIT=$?

if [ ${SMOKE_EXIT} -ne 0 ]; then
    echo "SMOKE TEST FAILED -- aborting job"
    exit 1
fi
echo "=== Smoke test passed ==="
echo ""

# -- Full ablation training (all 5 arms) ------------------------------------
# To run a single arm: replace 'all' with e.g. 'arm_e_ppvae'
ARM=${1:-all}

python -u scripts/train_proposed.py \
    --arm ${ARM} \
    --data_dir ${DATA} \
    --output_dir ${RESULTS} \
    --epochs 200 \
    --batch_size 16 \
    --image_size 256 \
    --noise_level random \
    --num_workers 8 \
    ${@:2}

echo ""
echo "Done: $(date)"
