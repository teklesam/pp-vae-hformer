#!/bin/bash
#SBATCH --job-name=ppvae_latent
#SBATCH --account=mrc-bsu2-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/rds/user/stm43/hpc-work/ppvae_results/logs/gen_latent_%j.log
#SBATCH --error=/rds/user/stm43/hpc-work/ppvae_results/logs/gen_latent_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ppvae
pip install umap-learn scikit-learn --quiet 2>/dev/null

cd /rds/user/stm43/hpc-work/ppvae_hformer

python - <<'PYEOF'
import sys
sys.path.insert(0, "/rds/user/stm43/hpc-work/ppvae_hformer")
sys.path.insert(0, "/rds/user/stm43/hpc-work/KAIR")

# Patch the script to only run latent space step
import importlib.util, types

spec = importlib.util.spec_from_file_location(
    "gfv2", "/rds/user/stm43/hpc-work/ppvae_hformer/scripts/generate_figures_v2.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("Loading VAE models...")
vae_arms = ["arm_e_ppvae", "arm_f_kl_cyc", "arm_g_kl_fb", "arm_h_kl_cyc_fb", "arm_p_best"]
models = mod.load_models(vae_arms)
print("Running latent space analysis...")
mod.make_latent_space_analysis(models)
print("Done.")
PYEOF

echo "Latent space figures complete."
