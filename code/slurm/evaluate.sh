#!/bin/bash
#SBATCH -J ppvae_eval
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=stm43@cam.ac.uk
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -o /rds/user/stm43/hpc-work/ppvae_results/logs/%j_eval.out
#SBATCH -e /rds/user/stm43/hpc-work/ppvae_results/logs/%j_eval.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/stm43/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

PROJECT=/rds/user/stm43/hpc-work/ppvae_hformer
DATA=/rds/user/stm43/hpc-work/chest_xray
RESULTS=/rds/user/stm43/hpc-work/ppvae_results
EVAL_OUT=${RESULTS}/evaluation_final
SITE_PKG=/rds/user/stm43/hpc-work/ppvae_site_packages

mkdir -p ${EVAL_OUT}/figures
mkdir -p ${RESULTS}/logs
mkdir -p ${SITE_PKG}

echo "================================================================"
echo "Job:    ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Time:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(python --version)"
echo "================================================================"

# ── Install dependencies to RDS (home disk is limited) ───────────────────────
pip install --quiet matplotlib scipy lpips piq --target ${SITE_PKG} 2>/dev/null
export PYTHONPATH=${SITE_PKG}:${PYTHONPATH}

# ── Verify checkpoints are present on RDS ────────────────────────────────────
echo ""
echo "Checkpoint inventory:"
for arm in arm_a_l2 arm_b_nll arm_c_nll_ssim arm_d_nll_ssim_ffl arm_e_ppvae dncnn_baseline; do
    if [ -f "${RESULTS}/${arm}/best_model.pth" ]; then
        echo "  ✓ ${arm}"
    else
        echo "  ✗ ${arm} — MISSING (will be skipped)"
    fi
done

# ── Determine which arms to evaluate ─────────────────────────────────────────
# Lists all arms; evaluate_all.py skips any whose best_model.pth is not found.
ARMS="arm_a_l2 arm_b_nll arm_c_nll_ssim arm_d_nll_ssim_ffl arm_e_ppvae dncnn_baseline"

echo ""
echo "=== Evaluating: ${ARMS} ==="
echo ""

cd ${PROJECT}

python -u scripts/evaluate_all.py \
    --data_dir    ${DATA} \
    --results_dir ${RESULTS} \
    --output_dir  ${EVAL_OUT} \
    --image_size  256 \
    --noise_levels low mid high \
    --mc_samples  20 \
    --num_workers 4 \
    --bootstrap_n 1000 \
    --arms        ${ARMS}

echo ""
echo "=== Running statistical analysis ==="
echo ""

# statistical_analysis.py reads the CSV we just wrote
python -u scripts/statistical_analysis.py \
    --csv         ${EVAL_OUT}/per_image_metrics.csv \
    --noise_level mid \
    --output_dir  ${EVAL_OUT}

echo ""
echo "================================================================"
echo "Evaluation complete: $(date)"
echo "Results at: ${EVAL_OUT}"
echo "================================================================"
