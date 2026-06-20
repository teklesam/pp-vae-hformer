#!/usr/bin/env python3
"""
generate_blind_panels.py — Blinded radiologist evaluation panel generator.

Design
------
EEDN-style forced-choice paradigm (Coupé et al. 2012):
  • Each panel: Reference (clean) image on left, Reconstruction on right.
  • 6 cases selected from the test set (3 Normal, 3 Pneumonia) spanning
    easy / medium / hard reconstruction difficulty (by Arm-A PSNR decile).
  • 6 methods (A, B, C, D, E, DnCNN) each assigned a random letter code
    (P/Q/R/S/T/U) — same across all radiologists for inter-rater comparison.
  • Ground truth always labelled "Reference" — not blinded.
  • A PI-only key file maps codes → real method names.

Output (output_dir/)
--------------------
  panels/
    case01_normal_easy/
      reference.png
      method_P.png  ...  method_U.png
    case02_normal_medium/ ...
    ...
  scoring_sheet.pdf        print-ready per-radiologist form
  pi_key.csv               CONFIDENTIAL — method code → arm name mapping
  case_manifest.csv        case IDs, class, PSNR-difficulty, file paths

Usage (CSD3 after all training is complete)
-------------------------------------------
  python scripts/generate_blind_panels.py \
      --data_dir   /rds/user/stm43/hpc-work/chest_xray \
      --results_dir /rds/user/stm43/hpc-work/ppvae_results \
      --output_dir  /rds/user/stm43/hpc-work/ppvae_results/blind_eval \
      --noise_eta  200 \
      --n_cases_per_class 3 \
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from src.models.ppvae_hformer import PPVAEHformer
from src.data.kermany_dataset import KermanyDataset
from src.training.config import ABLATION_ARMS, ExperimentConfig
from src.evaluation.metrics import psnr as compute_psnr

# ── Constants ──────────────────────────────────────────────────────────────────

ARMS = [
    "arm_a_l2",
    "arm_b_nll",
    "arm_c_nll_ssim",
    "arm_d_nll_ssim_ffl",
    "arm_e_ppvae",
    "dncnn_baseline",
]

ARM_LABELS = {
    "arm_a_l2":           "Arm A — L₂ (MSE)",
    "arm_b_nll":          "Arm B — NLL",
    "arm_c_nll_ssim":     "Arm C — NLL + SSIM",
    "arm_d_nll_ssim_ffl": "Arm D — NLL + SSIM + FFL",
    "arm_e_ppvae":        "Arm E — PP-VAE-Hformer",
    "dncnn_baseline":     "DnCNN (Zhang et al. 2017)",
}

BLIND_LETTERS = list("PQRSTU")   # 6 letters, one per method
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PANEL_W, PANEL_H = 512, 512     # px per image in panel
FONT_SIZE        = 20
LABEL_BAR_H      = 36           # pixels reserved for label below each image
BORDER_PX        = 4            # white border between panels


# ── Noise model ────────────────────────────────────────────────────────────────

def add_poisson_gaussian_noise(img_tensor: torch.Tensor, eta: int = 200) -> torch.Tensor:
    """Poisson-Gaussian noise matching the training degradation model."""
    # img_tensor: [1, H, W] float32 in [0, 1]
    scale = 1.0 / eta
    poisson = torch.poisson(img_tensor / scale) * scale
    gaussian = torch.randn_like(img_tensor) * 0.01
    return (poisson + gaussian).clamp(0.0, 1.0)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(arm_name: str, results_dir: Path) -> torch.nn.Module:
    ckpt_path = results_dir / arm_name / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = ExperimentConfig(arm=arm_name)
    model = PPVAEHformer(
        use_vae=cfg.use_vae,
        loss_arm=cfg.loss_arm,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Loaded {arm_name} (epoch {ckpt.get('epoch','?')}, "
          f"best_psnr={ckpt.get('best_val_psnr', 0):.3f})")
    return model


@torch.no_grad()
def reconstruct(model: torch.nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    """Run model and return mean reconstruction [1, H, W]."""
    x = noisy.unsqueeze(0).to(DEVICE)   # [1, 1, H, W]
    out = model(x)
    if isinstance(out, (tuple, list)):
        mu = out[0]
    else:
        mu = out
    return mu.squeeze(0).clamp(0.0, 1.0).cpu()  # [1, H, W]


# ── Case selection ─────────────────────────────────────────────────────────────

def select_cases(
    dataset,
    arm_a_model: torch.nn.Module,
    n_per_class: int,
    eta: int,
    rng: random.Random,
) -> list[dict]:
    """
    Select n_per_class cases per pathology class spanning easy/medium/hard.
    Difficulty = Arm-A PSNR on noisy image (lower PSNR → harder case).
    Returns list of dicts with index, class, psnr, image path.
    """
    torch.manual_seed(42)
    records_by_class: dict[str, list[dict]] = {"NORMAL": [], "PNEUMONIA": []}

    print("Scoring all test images with Arm A to select cases …")
    for idx in range(len(dataset)):
        clean, label = dataset[idx]          # [1, H, W], str
        noisy = add_poisson_gaussian_noise(clean, eta)
        recon = reconstruct(arm_a_model, noisy)
        p = compute_psnr(recon, clean).item()
        records_by_class[label].append({
            "idx": idx, "class": label, "arm_a_psnr": p,
            "clean": clean, "noisy": noisy,
        })

    selected = []
    for cls, recs in records_by_class.items():
        recs.sort(key=lambda r: r["arm_a_psnr"])
        n = len(recs)
        # Hard: bottom third; Medium: middle; Easy: top third
        zones = {
            "hard":   recs[: n // 3],
            "medium": recs[n // 3: 2 * n // 3],
            "easy":   recs[2 * n // 3:],
        }
        for difficulty, pool in zones.items():
            if len(pool) == 0:
                continue
            chosen = rng.choice(pool)
            chosen["difficulty"] = difficulty
            selected.append(chosen)
            if len([s for s in selected if s["class"] == cls]) >= n_per_class:
                break

    selected.sort(key=lambda r: (r["class"], r["difficulty"]))
    return selected[:n_per_class * 2]  # cap at 2 × n_per_class


# ── Image utilities ────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    """Convert [1, H, W] float tensor to resized PIL Image."""
    arr = (t.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img.resize(size, Image.LANCZOS)


def add_label(img: Image.Image, text: str, bar_h: int = LABEL_BAR_H) -> Image.Image:
    """Append a white label bar below the image."""
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (bar_h - th) // 2 + h), text, fill=(30, 30, 30), font=font)
    return canvas


def make_panel(ref_img: Image.Image, recon_img: Image.Image,
               ref_label: str, recon_label: str) -> Image.Image:
    """Side-by-side panel: Reference | Reconstruction."""
    w, h = ref_img.size[0], ref_img.size[1] + LABEL_BAR_H
    ref_labeled   = add_label(ref_img,   ref_label)
    recon_labeled = add_label(recon_img, recon_label)
    panel = Image.new("RGB", (2 * w + BORDER_PX, h), (200, 200, 200))
    panel.paste(ref_labeled,   (0, 0))
    panel.paste(recon_labeled, (w + BORDER_PX, 0))
    return panel


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "panels").mkdir(exist_ok=True)

    # ── Blinding key: randomly shuffle letters → arms ──────────────────────
    shuffled_letters = BLIND_LETTERS[:]
    rng.shuffle(shuffled_letters)
    # blind_key[letter] = arm_name
    blind_key = {letter: arm for letter, arm in zip(shuffled_letters, ARMS)}
    # reverse: arm_name → letter
    arm_to_letter = {arm: letter for letter, arm in blind_key.items()}

    # Save PI key (CONFIDENTIAL)
    key_path = output_dir / "pi_key.csv"
    with open(key_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_code", "arm_name", "arm_description"])
        for letter in sorted(blind_key):
            arm = blind_key[letter]
            w.writerow([letter, arm, ARM_LABELS[arm]])
    print(f"\n*** PI KEY saved to {key_path} — KEEP CONFIDENTIAL ***\n")
    print("Blinding assignment:")
    for letter in sorted(blind_key):
        print(f"  {letter} → {ARM_LABELS[blind_key[letter]]}")
    print()

    # ── Load dataset ────────────────────────────────────────────────────────
    test_dir = Path(args.data_dir) / "chest_xray" / "test"
    if not test_dir.exists():
        test_dir = Path(args.data_dir) / "test"
    dataset = KermanyDataset(str(test_dir), split="test", img_size=256)
    print(f"Test set: {len(dataset)} images")

    # ── Load all models ─────────────────────────────────────────────────────
    print("\nLoading models …")
    models = {}
    for arm in ARMS:
        try:
            models[arm] = load_model(arm, results_dir)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {arm}")

    if "arm_a_l2" not in models:
        raise RuntimeError("Arm A required for case selection but checkpoint missing.")

    # ── Select cases ────────────────────────────────────────────────────────
    print("\nSelecting cases …")
    cases = select_cases(
        dataset, models["arm_a_l2"],
        n_per_class=args.n_cases_per_class,
        eta=args.noise_eta, rng=rng,
    )
    print(f"Selected {len(cases)} cases:")
    for i, c in enumerate(cases, 1):
        print(f"  Case {i:02d}: idx={c['idx']:4d}  class={c['class']:9s}  "
              f"difficulty={c['difficulty']:6s}  arm_a_psnr={c['arm_a_psnr']:.2f} dB")

    # ── Generate panels ─────────────────────────────────────────────────────
    manifest_rows = []
    for case_idx, case in enumerate(cases, 1):
        cls_tag   = case["class"].lower()
        diff_tag  = case["difficulty"]
        case_name = f"case{case_idx:02d}_{cls_tag}_{diff_tag}"
        case_dir  = output_dir / "panels" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        clean = case["clean"]
        noisy = case["noisy"]

        # Save reference
        ref_img = tensor_to_pil(clean, (PANEL_W, PANEL_H))
        ref_path = case_dir / "reference.png"
        ref_img.save(ref_path)

        # Save noisy (for context)
        noisy_img = tensor_to_pil(noisy, (PANEL_W, PANEL_H))
        (case_dir / "noisy_input.png").save(str(case_dir / "noisy_input.png"))
        noisy_img.save(str(case_dir / "noisy_input.png"))

        # Generate reconstruction + panel for each arm
        arm_rows = []
        for arm in ARMS:
            if arm not in models:
                continue
            recon = reconstruct(models[arm], noisy)
            arm_psnr = compute_psnr(recon, clean).item()
            letter = arm_to_letter[arm]

            recon_img  = tensor_to_pil(recon, (PANEL_W, PANEL_H))
            blind_name = f"method_{letter}.png"
            recon_img.save(str(case_dir / blind_name))

            # Side-by-side panel
            panel = make_panel(ref_img, recon_img,
                               "REFERENCE (clean)", f"Method {letter}")
            panel.save(str(case_dir / f"panel_{letter}.png"))

            arm_rows.append({
                "case_id": case_name,
                "case_number": case_idx,
                "class": case["class"],
                "difficulty": diff_tag,
                "arm": arm,
                "blind_code": letter,
                "arm_a_psnr_case": case["arm_a_psnr"],
                "this_arm_psnr": arm_psnr,
                "panel_path": str(case_dir / f"panel_{letter}.png"),
            })
            print(f"  {case_name} | {letter} ({arm:25s}) | PSNR {arm_psnr:.2f} dB")

        manifest_rows.extend(arm_rows)
        print()

    # ── Save manifest ────────────────────────────────────────────────────────
    manifest_path = output_dir / "case_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest saved: {manifest_path}")

    # ── Generate LaTeX scoring sheet (per-radiologist) ───────────────────────
    _write_latex_scoring_sheet(output_dir, cases, arm_to_letter, args.noise_eta)

    print("\nDone. Summary:")
    print(f"  Panels   : {output_dir}/panels/")
    print(f"  Manifest : {manifest_path}")
    print(f"  PI key   : {key_path}  ← KEEP CONFIDENTIAL")
    print(f"  LaTeX    : {output_dir}/scoring_sheet.tex")


# ── LaTeX scoring sheet generator ─────────────────────────────────────────────

def _write_latex_scoring_sheet(
    output_dir: Path,
    cases: list[dict],
    arm_to_letter: dict[str, str],
    eta: int,
):
    """Write a print-ready per-radiologist scoring sheet with case table."""
    lines = [
        r"\documentclass[a4paper,10pt]{article}",
        r"\usepackage[top=1.8cm,bottom=1.8cm,left=2cm,right=2cm]{geometry}",
        r"\usepackage{booktabs,tabularx,multirow,graphicx,xcolor,fancyhdr,mdframed,parskip,array}",
        r"\usepackage[T1]{fontenc}",
        r"\definecolor{darkblue}{RGB}{0,61,100}",
        r"\definecolor{cambridgeblue}{RGB}{163,193,173}",
        r"\pagestyle{fancy}\fancyhf{}",
        r"\renewcommand{\headrulewidth}{0.4pt}",
        r"\fancyhead[L]{\small\textcolor{darkblue}{\textbf{CONFIDENTIAL — RESEARCH USE ONLY}}}",
        r"\fancyhead[R]{\small\textcolor{darkblue}{PP-VAE-Hformer Blinded Expert Evaluation}}",
        r"\fancyfoot[C]{\small Page \thepage}",
        r"\begin{document}",
        "",
        # Header
        r"\begin{minipage}[t]{0.28\textwidth}\vspace{0pt}",
        r"  \framebox[3.8cm]{\rule{0pt}{1.2cm}\quad\textit{Cambridge logo}\quad}",
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.40\textwidth}\vspace{0pt}\centering",
        r"  {\large\bfseries\textcolor{darkblue}{Blinded Radiological Assessment}}\\[3pt]",
        r"  {\small PP-VAE-Hformer Paediatric CXR Denoising}\\[2pt]",
        r"  {\footnotesize MPhil Dissertation Study, University of Cambridge, 2026}",
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.24\textwidth}\vspace{0pt}\raggedleft",
        r"  \framebox[3.2cm]{\rule{0pt}{1.2cm}\quad\textit{BSU logo}\quad}",
        r"\end{minipage}",
        r"\vspace{0.3cm}{\color{darkblue}\hrule height 1.2pt}\vspace{0.3cm}",
        "",
        # Background
        r"{\small\textbf{Background.} "
        r"This study evaluates deep-learning denoising of paediatric chest X-rays "
        r"(Kermany dataset~[1]) under a Poisson-Gaussian noise model ($\eta=" + str(eta) + r"$). "
        r"Six reconstruction methods are presented under blinded codes (P–U). "
        r"The reference image (clean, high-quality) is always shown on the \textbf{left}; "
        r"the reconstruction is shown on the \textbf{right}. "
        r"Score each pair using the rubric below. "
        r"The method identity will be revealed only after all assessments are complete.}",
        r"\vspace{0.2cm}",
        "",
        # Rubric
        r"\begin{mdframed}[backgroundcolor=cambridgeblue!25,linewidth=0.5pt,"
        r"innertopmargin=4pt,innerbottommargin=4pt]",
        r"\textbf{Scoring Rubric:}",
        r"\begin{tabular}{>{\bfseries\centering}p{0.6cm}p{14cm}}",
        r"  1   & \textbf{Diagnostically equivalent}: all features (lung markings, cardiac silhouette,"
        r" consolidation/infiltrates) preserved at same diagnostic confidence level. \\[2pt]",
        r"  0.5 & \textbf{Diagnostically acceptable}: minor degradation; would not change clinical management. \\[2pt]",
        r"  0   & \textbf{Diagnostically unacceptable}: blurring, artefacts, or feature loss that would"
        r" alter interpretation. \\",
        r"\end{tabular}",
        r"\end{mdframed}\vspace{0.25cm}",
        "",
        # Main assessment table
        r"\begin{tabularx}{\textwidth}{>{\centering}p{0.5cm} >{\centering}p{1.2cm}"
        r" >{\centering}p{2.0cm} >{\centering}p{1.4cm} >{\centering}p{1.8cm}"
        r" >{\centering}p{2.4cm} >{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Case ID} & \textbf{Class} & \textbf{Noise} &"
        r" \textbf{Method} & \textbf{Score (1~/~0.5~/~0)} & \textbf{Observations} \\",
        r"\midrule",
    ]

    row_num = 0
    prev_case = None
    for case_idx, case in enumerate(cases, 1):
        cls_tag   = case["class"].lower()
        diff_tag  = case["difficulty"]
        case_id   = f"C{case_idx:02d}"
        case_name = f"case{case_idx:02d}_{cls_tag}_{diff_tag}"

        if case_name != prev_case:
            prev_case = case_name
            label = r"\multicolumn{7}{l}{\small\textit{"
            label += f"Case {case_idx}: {case['class'].capitalize()} — {diff_tag} difficulty"
            label += r"}} \\"
            lines.append(label)

        for arm in ARMS:
            row_num += 1
            letter = arm_to_letter.get(arm, "?")
            cls_display = case["class"].capitalize()
            noise_display = f"$\\eta={eta}$"
            lines.append(
                f"{row_num} & {case_id} & {cls_display} & {noise_display} & "
                f"\\textbf{{{letter}}} & 1\\quad 0.5\\quad 0 & \\\\[8pt]"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\vspace{0.3cm}",
        "",
        # Reliability check notice
        r"\begin{mdframed}[linewidth=0.5pt,innertopmargin=3pt,innerbottommargin=3pt]",
        r"{\small\textbf{Note on repeat cases.} "
        r"Three cases from above will be repeated at the end of your assessment "
        r"(same images, different order) to measure intra-rater reliability. "
        r"Please score them independently without referring back to your earlier responses.}",
        r"\end{mdframed}",
        r"\vspace{0.25cm}",
        "",
        # Consent
        r"\begin{mdframed}[linewidth=0.8pt,innertopmargin=5pt,innerbottommargin=5pt]",
        r"{\small\textbf{Declaration of Consent.} "
        r"I confirm I am a qualified radiologist/radiographer providing this assessment "
        r"voluntarily for academic research only. Images are from the publicly available "
        r"Kermany paediatric CXR dataset (no identifiable patient data). Results will be "
        r"used solely in the MPhil dissertation of S.\,Tekle (stm43@cam.ac.uk), University of Cambridge, "
        r"and may be reported in aggregate in academic publications. "
        r"I may withdraw at any time. No NHS REC approval required (anonymised public data).",
        r"\vspace{6pt}",
        r"\begin{tabular}{p{5.8cm}p{4.2cm}p{4.8cm}}",
        r"\textbf{Name (print):}\dotfill & \textbf{GMC/GDC No.:}\dotfill & \textbf{Date:}\dotfill \\[14pt]",
        r"\textbf{Signature:}\dotfill & \textbf{Institution:}\dotfill & \textbf{Specialty:}\dotfill \\",
        r"\end{tabular}}",
        r"\end{mdframed}",
        "",
        r"{\footnotesize\textbf{References.}\quad"
        r"[1]~Kermany et al.\ (2018). \textit{Cell} 172(5):1122--1131.\quad"
        r"[2]~Coup\'e et al.\ (2012). EEDN. \textit{IEEE TMI}.\quad"
        r"[3]~Wang et al.\ (2004). SSIM. \textit{IEEE TIP}.}",
        r"\end{document}",
    ]

    tex_path = output_dir / "scoring_sheet.tex"
    tex_path.write_text("\n".join(lines))
    print(f"LaTeX scoring sheet saved: {tex_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blinded radiologist panels")
    parser.add_argument("--data_dir",        required=True)
    parser.add_argument("--results_dir",     required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--noise_eta",       type=int,   default=200)
    parser.add_argument("--n_cases_per_class", type=int, default=3)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()
    main(args)
