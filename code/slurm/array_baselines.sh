#!/bin/bash
# SLURM job array — train comparison architectures on Kermany CXR.
#
# One job per architecture; each gets 1 GPU on Ampere partition.
# All use the same Kermany + Foi noise dataset and MSE loss —
# a fully fair architectural comparison against our ablation arms.
#
# Array index → architecture mapping:
#   0 = dncnn      ~3h   (300K params)
#   1 = dncnn_b    ~3h   (300K params, blind)
#   2 = ffdnet     ~3h   (500K params, sigma-conditioned)
#   3 = ircnn      ~4h   (1.5M params)
#   4 = drunet     ~12h  (32.7M params, CNN ceiling)
#   5 = swinir     ~10h  (12M params,  Transformer baseline)
#
# Submit all 6:
#   sbatch slurm/array_baselines.sh
#
# Submit just the heavy ones (DRUNet + SwinIR):
#   sbatch --array=4-5 slurm/array_baselines.sh
#
# Submit a single arch (e.g. DRUNet):
#   sbatch --array=4 slurm/array_baselines.sh
#
# !! Run AFTER ablation array is complete so GPU queue is not saturated !!
# !! Monitor ablation with: squeue -u stm43                             !!

#SBATCH -J ppvae_baselines
#SBATCH --array=0-5
#SBATCH -A mrc-bsu2-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=35:00:00
#SBATCH -o /rds/user/stm43/hpc-work/ppvae_results/baselines/train_%a_%j.out
#SBATCH -e /rds/user/stm43/hpc-work/ppvae_results/baselines/train_%a_%j.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae

PROJECT=/rds/user/stm43/hpc-work/ppvae_hformer
DATA=/rds/user/stm43/hpc-work/chest_xray
OUT=/rds/user/stm43/hpc-work/ppvae_results/baselines

mkdir -p "$OUT"

cd "$PROJECT"

# Map SLURM_ARRAY_TASK_ID → architecture name
ARCHS=(dncnn dncnn_b ffdnet ircnn drunet swinir)
ARCH="${ARCHS[$SLURM_ARRAY_TASK_ID]}"

# SwinIR: smaller batch (OOM at 16) and fewer epochs (slow per-epoch, 36h wall limit)
if [ "$ARCH" = "swinir" ]; then
    BATCH_SIZE=4
    EPOCHS=150
else
    BATCH_SIZE=16
    EPOCHS=200
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Baseline Training: ${ARCH} ==="
echo "Task ID : $SLURM_ARRAY_TASK_ID"
echo "Batch   : $BATCH_SIZE"
echo "Start   : $(date)"
echo "Node    : $SLURMD_NODENAME"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

python scripts/train_baseline.py \
    --arch    "$ARCH" \
    --data_dir "$DATA" \
    --output_dir "$OUT" \
    --epochs  "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --noise_level random \
    --num_workers 8 \
    --num_gpus 1

echo
echo "Done: $(date)"
