#!/usr/bin/env python
"""
statistical_analysis.py — ANOVA + Bonferroni + Cohen's d on per_image_metrics.csv.

Produces:
  1. One-way ANOVA across ablation arms (PSNR and SSIM)
  2. Bonferroni-corrected pairwise t-tests
  3. Cohen's d effect sizes for each pair
  4. Subgroup analysis: Pneumonia vs Normal
  5. Ready-to-paste LaTeX tables for Chapter 4

Usage:
    python scripts/statistical_analysis.py \
        --csv /path/to/per_image_metrics.csv \
        --noise_level mid \
        --output_dir /path/to/evaluation_local
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--csv",         type=str, required=True,
                    help="Path to per_image_metrics.csv from evaluate_local.py")
parser.add_argument("--noise_level", type=str, default="mid",
                    choices=["low", "mid", "high"])
parser.add_argument("--output_dir",  type=str, default=".")
args = parser.parse_args()

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

rows = []
with open(args.csv) as f:
    for r in csv.DictReader(f):
        if r.get("noise_level", args.noise_level) == args.noise_level:
            rows.append(r)

if not rows:
    print(f"No rows found for noise_level={args.noise_level}. Available levels:")
    with open(args.csv) as f:
        levels = {r.get("noise_level","?") for r in csv.DictReader(f)}
    print(levels)
    sys.exit(1)

# Group by arm
arms_data: dict[str, dict[str, list[float]]] = {}
for r in rows:
    arm = r["arm"]
    if arm not in arms_data:
        arms_data[arm] = {"psnr": [], "ssim": [], "psnr_normal": [], "psnr_pneumonia": [],
                          "ssim_normal": [], "ssim_pneumonia": []}
    p, s = float(r["psnr"]), float(r["ssim"])
    cls = r.get("class", "?")
    arms_data[arm]["psnr"].append(p)
    arms_data[arm]["ssim"].append(s)
    if cls == "Normal":
        arms_data[arm]["psnr_normal"].append(p)
        arms_data[arm]["ssim_normal"].append(s)
    elif cls == "Pneumonia":
        arms_data[arm]["psnr_pneumonia"].append(p)
        arms_data[arm]["ssim_pneumonia"].append(s)

arm_order = [a for a in
    ["arm_a_l2","arm_b_nll","arm_c_nll_ssim","arm_d_nll_ssim_ffl","arm_e_ppvae","dncnn_baseline"]
    if a in arms_data]

print(f"\nArms found: {arm_order}")
print(f"Images per arm: {[len(arms_data[a]['psnr']) for a in arm_order]}\n")


# ── Cohen's d ─────────────────────────────────────────────────────────────────

def cohen_d(a: list[float], b: list[float]) -> float:
    """Pooled Cohen's d."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)
                         / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (pooled_std + 1e-12))


# ── One-way ANOVA ─────────────────────────────────────────────────────────────

def run_anova(metric: str) -> tuple[float, float]:
    groups = [arms_data[a][metric] for a in arm_order]
    f, p = stats.f_oneway(*groups)
    return float(f), float(p)


# ── Pairwise Bonferroni t-tests ───────────────────────────────────────────────

def bonferroni_pairwise(metric: str) -> list[dict]:
    pairs = list(combinations(arm_order, 2))
    n_tests = len(pairs)
    results = []
    for a1, a2 in pairs:
        t, p_raw = stats.ttest_ind(arms_data[a1][metric], arms_data[a2][metric])
        p_corr = min(1.0, p_raw * n_tests)   # Bonferroni correction
        d = cohen_d(arms_data[a1][metric], arms_data[a2][metric])
        results.append({
            "arm1": a1, "arm2": a2,
            "t": t, "p_raw": p_raw, "p_bonf": p_corr, "cohens_d": d,
        })
    return results


# ── Summary table ─────────────────────────────────────────────────────────────

ARM_LABELS = {
    "arm_a_l2":           "Arm A — Hformer + L$_2$",
    "arm_b_nll":          "Arm B — Hformer + NLL",
    "arm_c_nll_ssim":     "Arm C — Hformer + NLL + SSIM",
    "arm_d_nll_ssim_ffl": "Arm D — Hformer + NLL + SSIM + FFL",
    "arm_e_ppvae":        "Arm E — PP-VAE-Hformer (full)",
    "dncnn_baseline":     "DnCNN (Zhang et al., 2017)",
}

print("=" * 70)
print(f"SUMMARY TABLE (noise_level={args.noise_level})")
print("=" * 70)
print(f"{'Arm':<40} {'PSNR mean±SD':>18} {'SSIM mean±SD':>18}")
print("-" * 70)
for arm in arm_order:
    psnr_arr = np.array(arms_data[arm]["psnr"])
    ssim_arr = np.array(arms_data[arm]["ssim"])
    label = ARM_LABELS.get(arm, arm)
    print(f"{label:<40} {psnr_arr.mean():.3f}±{psnr_arr.std():.3f}     "
          f"{ssim_arr.mean():.4f}±{ssim_arr.std():.4f}")

# ── ANOVA ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ONE-WAY ANOVA")
print("=" * 70)
for metric in ["psnr", "ssim"]:
    F, p = run_anova(metric)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {metric.upper():5s}:  F={F:.2f},  p={p:.4e}  {sig}")

# ── Bonferroni pairwise ───────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PAIRWISE BONFERRONI-CORRECTED t-TESTS (PSNR)")
print("=" * 70)
psnr_pairs = bonferroni_pairwise("psnr")
for r in psnr_pairs:
    sig = "***" if r["p_bonf"] < 0.001 else "**" if r["p_bonf"] < 0.01 else \
          "*" if r["p_bonf"] < 0.05 else "n.s."
    print(f"  {r['arm1']:30s} vs {r['arm2']:30s}: "
          f"t={r['t']:+.2f}, p_bonf={r['p_bonf']:.4e} {sig}, d={r['cohens_d']:+.3f}")

# ── Subgroup ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUBGROUP ANALYSIS: Normal vs Pneumonia (PSNR)")
print("=" * 70)
for arm in arm_order:
    n_n = len(arms_data[arm]["psnr_normal"])
    n_p = len(arms_data[arm]["psnr_pneumonia"])
    if n_n == 0 or n_p == 0:
        print(f"  {arm}: no class labels available")
        continue
    mn = np.mean(arms_data[arm]["psnr_normal"])
    mp = np.mean(arms_data[arm]["psnr_pneumonia"])
    t, p = stats.ttest_ind(arms_data[arm]["psnr_normal"],
                           arms_data[arm]["psnr_pneumonia"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    label = ARM_LABELS.get(arm, arm)
    print(f"  {label:<40}  Normal={mn:.3f} dB  Pneumonia={mp:.3f} dB  t={t:+.2f} {sig}")

# ── LaTeX output ──────────────────────────────────────────────────────────────

latex_main = r"""
% ── Table: Reconstruction performance ─────────────────────────────────────
% Paste into §4.3.1
\begin{table}[htbp]
\centering
\caption{Test-set reconstruction performance at \textit{mid} noise level.
         Values are mean\,$\pm$\,SD over 624 held-out test images.
         Best result per metric in \textbf{bold}.}
\label{tab:results:recon}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{VAE} & \textbf{PSNR (dB)} & \textbf{SSIM} \\
\midrule
"""

psnr_vals = {a: np.array(arms_data[a]["psnr"]) for a in arm_order}
ssim_vals = {a: np.array(arms_data[a]["ssim"]) for a in arm_order}
best_psnr = max(v.mean() for v in psnr_vals.values())
best_ssim = max(v.mean() for v in ssim_vals.values())

rows_latex = []
for arm in arm_order:
    label = ARM_LABELS.get(arm, arm)
    vae = r"\checkmark" if ABLATION_ARMS_VAE.get(arm, False) else "---"
    pm = "\\pm"
    pv = psnr_vals[arm]
    sv = ssim_vals[arm]
    psnr_str = f"{pv.mean():.3f}$\\,{pm}\\,${pv.std():.3f}"
    ssim_str = f"{sv.mean():.4f}$\\,{pm}\\,${sv.std():.4f}"
    if abs(pv.mean() - best_psnr) < 0.001:
        psnr_str = f"\\textbf{{{psnr_str}}}"
    if abs(sv.mean() - best_ssim) < 0.0001:
        ssim_str = f"\\textbf{{{ssim_str}}}"
    rows_latex.append(f"{label} & {vae} & {psnr_str} & {ssim_str} \\\\")

ABLATION_ARMS_VAE = {
    "arm_a_l2": False, "arm_b_nll": False, "arm_c_nll_ssim": False,
    "arm_d_nll_ssim_ffl": False, "arm_e_ppvae": True, "dncnn_baseline": False,
}

# Rebuild properly
rows_latex = []
for arm in arm_order:
    label = ARM_LABELS.get(arm, arm)
    vae   = r"\checkmark" if ABLATION_ARMS_VAE.get(arm, False) else "---"
    pv    = psnr_vals[arm]
    sv    = ssim_vals[arm]
    psnr_str = f"{pv.mean():.3f} $\\pm$ {pv.std():.3f}"
    ssim_str = f"{sv.mean():.4f} $\\pm$ {sv.std():.4f}"
    if abs(pv.mean() - best_psnr) < 0.001:
        psnr_str = r"\textbf{" + psnr_str + r"}"
    if abs(sv.mean() - best_ssim) < 0.0001:
        ssim_str = r"\textbf{" + ssim_str + r"}"
    rows_latex.append(f"  {label} & {vae} & {psnr_str} & {ssim_str} \\\\")

print("\n" + "=" * 70)
print("LATEX TABLE — paste into §4.3.1")
print("=" * 70)
print(latex_main)
for r in rows_latex:
    print(r)
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# ── Save CSV of pairwise stats ────────────────────────────────────────────────

stats_path = out_dir / f"pairwise_stats_{args.noise_level}.csv"
with open(stats_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["arm1","arm2","t","p_raw","p_bonf","cohens_d"])
    writer.writeheader()
    writer.writerows(bonferroni_pairwise("psnr"))
print(f"\nPairwise stats CSV: {stats_path}")

# ── Bonferroni table for SSIM ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PAIRWISE BONFERRONI-CORRECTED t-TESTS (SSIM)")
print("=" * 70)
ssim_pairs = bonferroni_pairwise("ssim")
for r in ssim_pairs:
    sig = "***" if r["p_bonf"] < 0.001 else "**" if r["p_bonf"] < 0.01 else \
          "*" if r["p_bonf"] < 0.05 else "n.s."
    print(f"  {r['arm1']:30s} vs {r['arm2']:30s}: "
          f"t={r['t']:+.2f}, p_bonf={r['p_bonf']:.4e} {sig}, d={r['cohens_d']:+.3f}")

print(f"\nAll done. Outputs in {out_dir}/")
