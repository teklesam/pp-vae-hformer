#!/bin/bash
# Training monitor — reads CSV headers dynamically to avoid column-offset bugs.
# Run with: watch -n 60 bash scripts/monitor.sh

RESULTS=/rds/user/stm43/hpc-work/ppvae_results
BL_RESULTS=/rds/user/stm43/hpc-work/ppvae_results/baselines

echo "==== TRAINING MONITOR ==== $(date '+%H:%M:%S')"

# last_val <csv_path> <show_kl: yes|no>
# Reads the CSV header to find val_psnr (and optionally l_kl) column positions.
# Outputs a tab-separated line: "EPOCH  PSNR  [KL]" or an error string.
last_val() {
    local f="$1"
    local want_kl="$2"
    [[ -f "$f" ]] || { printf "(no log yet)"; return; }
    awk -F, -v want_kl="$want_kl" '
        NR==1 {
            for (i=1; i<=NF; i++) {
                col=tolower($i)
                gsub(/[[:space:]]/, "", col)
                if (col=="epoch")    ep_c=i
                if (col=="val_psnr") psnr_c=i
                if (col=="l_kl")     kl_c=i
            }
            next
        }
        psnr_c > 0 && $psnr_c != "nan" && $psnr_c+0 > 1 {
            ep=$ep_c
            psnr=$psnr_c
            kl=(kl_c>0 ? $kl_c : "n/a")
        }
        END {
            if (psnr_c==0 || ep=="") { print "(no val yet)"; exit }
            if (want_kl=="yes") printf "%s\t%s\t%s", ep, psnr, kl
            else                printf "%s\t%s",    ep, psnr
        }
    ' "$f"
}

# ── ABLATION ARMS ─────────────────────────────────────────────────────────────
echo ""
echo "--- ABLATION ARMS ---"
printf "  %-30s %-8s %-12s %s\n" "Arm" "Epoch" "PSNR" "KL"

ARMS=(
    arm_a_l2
    arm_b_nll
    arm_c_nll_ssim
    arm_d_nll_ssim_ffl
    arm_e_ppvae
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

for ARM in "${ARMS[@]}"; do
    LOG="$RESULTS/$ARM/train_log.csv"
    ROW=$(last_val "$LOG" "yes")
    if [[ "$ROW" == "("* ]]; then
        printf "  %-30s %s\n" "$ARM" "$ROW"
    else
        EP=$(  printf '%s' "$ROW" | cut -f1)
        PSNR=$(printf '%s' "$ROW" | cut -f2)
        KL=$(  printf '%s' "$ROW" | cut -f3)
        printf "  %-30s %-8s %-12s %s\n" "$ARM" "$EP" "$PSNR" "$KL"
    fi
done

# ── BASELINES ─────────────────────────────────────────────────────────────────
echo ""
echo "--- BASELINES ---"
printf "  %-14s %-8s %s\n" "Arch" "Epoch" "PSNR"

for ARCH in dncnn dncnn_b ffdnet ircnn drunet swinir; do
    LOG="$BL_RESULTS/$ARCH/train_log.csv"
    ROW=$(last_val "$LOG" "no")
    if [[ "$ROW" == "("* ]]; then
        printf "  %-14s %s\n" "$ARCH" "$ROW"
    else
        EP=$(  printf '%s' "$ROW" | cut -f1)
        PSNR=$(printf '%s' "$ROW" | cut -f2)
        printf "  %-14s %-8s %s\n" "$ARCH" "$EP" "$PSNR"
    fi
done

# ── QUEUE ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- QUEUE ---"
printf "  %-12s %-22s %-12s %s\n" "JOBID" "NAME" "STATE" "TIME"
squeue -u stm43 --noheader --Format "JobID:12,Name:22,State:12,TimeUsed" 2>/dev/null \
    | awk '{printf "  %s\n", $0}' \
    || echo "  (squeue unavailable)"
