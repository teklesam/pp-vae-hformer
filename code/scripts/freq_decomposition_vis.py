"""
Frequency Decomposition Visualisation for Kermany Paediatric CXR Dataset
=========================================================================
Standalone script -- does not depend on any training code.

Produces a publication-quality figure showing:
  1. Clean NORMAL vs PNEUMONIA chest X-rays
  2. Noisy versions at low / mid / high Foi-model noise presets
  3. Frequency magnitude spectrum (log-scaled)
  4. Low-frequency reconstruction (band-pass keep: < lp_cutoff fraction of max frequency)
  5. High-frequency residual (everything above lp_cutoff)
  6. ROI zoom on anatomically important region (hilum / lower-lobe vessels)

Usage:
    python freq_decomposition_vis.py \
        --data_dir /path/to/chest_xray \
        --normal_name NORMAL-1003233-0001.jpeg \
        --pneumonia_name BACTERIA-1025587-0001.jpeg \
        --out_path ./freq_decomp.png

The --normal_name and --pneumonia_name arguments are optional.
If omitted, the first image in each directory is used.

ROI guide
---------
The hilum is approximately in the central third of the image width and
upper third of the lung field (roughly rows 30-60%, cols 35-65% of a
256x256 image).  You can adjust ROI_* constants below.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ── Noise model constants (Foi et al. 2008: Var(z|y) = a*y + b) ─────────────
NOISE_PRESETS = {
    "low":  {"a": 0.01,  "b": 0.002},
    "mid":  {"a": 0.03,  "b": 0.005},
    "high": {"a": 0.08,  "b": 0.010},
}

# ── Frequency decomposition ───────────────────────────────────────────────────
# Fraction of the Nyquist frequency below which we call "low frequency".
# 0.15 means: keep only the central 15% of the frequency plane as "low".
LP_CUTOFF = 0.15

# ── ROI definition (fractions of image height/width) ─────────────────────────
# Targeting the right hilum / lower pulmonary vessels:
ROI_ROW_START = 0.35
ROI_ROW_END   = 0.65
ROI_COL_START = 0.45
ROI_COL_END   = 0.72

TARGET_SIZE = 256


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str, size: int = TARGET_SIZE) -> np.ndarray:
    """Load a JPEG, convert to greyscale, resize, normalise to [0, 1]."""
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def add_foi_noise(img: np.ndarray, a: float, b: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Apply Poisson-Gaussian noise: Var(z|y) = a*y + b.

    Steps:
      1. Compute per-pixel variance:  v = a*y + b
      2. Draw Gaussian noise:         n ~ N(0, v)
      3. Clip result to [0, 1]
    """
    variance = a * img + b
    noise = rng.normal(0.0, np.sqrt(np.maximum(variance, 0.0)))
    return np.clip(img + noise, 0.0, 1.0)


def freq_decompose(img: np.ndarray, lp_cutoff: float = LP_CUTOFF):
    """Return (magnitude_spectrum, lf_image, hf_image) via 2-D FFT.

    lp_cutoff: fraction of max frequency radius to keep as 'low frequency'.
    """
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # Log-scaled magnitude spectrum (add 1 to avoid log(0))
    mag = np.log1p(np.abs(Fshift))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    # Build low-pass mask
    H, W = img.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    lp_mask = (dist <= lp_cutoff * max_dist).astype(np.float32)

    # Low-frequency reconstruction
    Fshift_lp = Fshift * lp_mask
    lf_img = np.abs(np.fft.ifft2(np.fft.ifftshift(Fshift_lp)))
    lf_img = np.clip(lf_img, 0.0, 1.0)

    # High-frequency residual
    hf_img = np.abs(img - lf_img)
    hf_img = (hf_img - hf_img.min()) / (hf_img.max() - hf_img.min() + 1e-8)

    return mag, lf_img, hf_img


def roi_bbox(img_shape, r0=ROI_ROW_START, r1=ROI_ROW_END,
             c0=ROI_COL_START, c1=ROI_COL_END):
    """Return (row_start, row_end, col_start, col_end) as pixel indices."""
    H, W = img_shape
    return (int(r0 * H), int(r1 * H), int(c0 * W), int(c1 * W))


def add_roi_rect(ax, roi, colour="yellow", lw=1.5):
    """Draw a rectangle on ax indicating the ROI."""
    rs, re, cs, ce = roi
    rect = patches.Rectangle(
        (cs, rs), ce - cs, re - rs,
        linewidth=lw, edgecolor=colour, facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)


# ── Main visualisation ────────────────────────────────────────────────────────

def build_figure(normal_img: np.ndarray, pneumonia_img: np.ndarray,
                 noise_preset: str = "mid",
                 lp_cutoff: float = LP_CUTOFF,
                 out_path: str = "freq_decomp.png"):
    """Build and save the full 4-panel frequency decomposition figure.

    Layout (rows x cols):
      Row 0: Clean Normal | Clean Pneumonia
      Row 1: Noisy (preset) Normal | Noisy (preset) Pneumonia
      Row 2: Freq. spectrum Normal | Freq. spectrum Pneumonia
      Row 3: Low-freq Normal | Low-freq Pneumonia
      Row 4: High-freq residual Normal | High-freq Pneumonia
      Row 5: ROI zoom Normal | ROI zoom Pneumonia
    """
    preset = NOISE_PRESETS[noise_preset]
    rng = np.random.default_rng(42)

    noisy_n = add_foi_noise(normal_img, preset["a"], preset["b"], rng)
    noisy_p = add_foi_noise(pneumonia_img, preset["a"], preset["b"], rng)

    mag_n,  lf_n,  hf_n  = freq_decompose(noisy_n, lp_cutoff)
    mag_p,  lf_p,  hf_p  = freq_decompose(noisy_p, lp_cutoff)

    # also decompose clean for comparison
    mag_nc, lf_nc, hf_nc = freq_decompose(normal_img, lp_cutoff)
    mag_pc, lf_pc, hf_pc = freq_decompose(pneumonia_img, lp_cutoff)

    roi = roi_bbox(normal_img.shape)
    rs, re, cs, ce = roi

    cols = ["Normal (NORMAL)", "Pneumonia (BACTERIA)"]
    row_labels = [
        "Clean",
        f"Noisy ({noise_preset} preset\na={preset['a']}, b={preset['b']})",
        "Freq. spectrum\n(noisy, log-scaled)",
        "Low-freq reconstruction\n(LF < 15% Nyquist)",
        "High-freq residual\n(HF pathology + noise)",
        "ROI zoom\n(hilum / lower vessels)",
    ]

    NROWS, NCOLS = 6, 2
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(10, 26))
    fig.patch.set_facecolor("#1a1a2e")

    cmaps = {
        "image": "gray",
        "spectrum": "inferno",
        "hf": "hot",
    }

    def style_ax(ax, title="", cmap="gray", show_roi=False, img=None):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        if img is not None:
            ax.imshow(img, cmap=cmap, aspect="equal",
                      vmin=0 if cmap != "inferno" else None,
                      vmax=1 if cmap != "inferno" else None)
        if show_roi:
            add_roi_rect(ax, roi)
        if title:
            ax.set_title(title, color="white", fontsize=8, pad=3)

    # Row 0 — Clean
    style_ax(axes[0, 0], "Clean", img=normal_img, show_roi=True)
    style_ax(axes[0, 1], "Clean", img=pneumonia_img, show_roi=True)

    # Row 1 — Noisy
    style_ax(axes[1, 0], f"Noisy ({noise_preset})", img=noisy_n, show_roi=True)
    style_ax(axes[1, 1], f"Noisy ({noise_preset})", img=noisy_p, show_roi=True)

    # Row 2 — Spectrum
    style_ax(axes[2, 0], "Spectrum (noisy)", cmap="inferno", img=mag_n)
    style_ax(axes[2, 1], "Spectrum (noisy)", cmap="inferno", img=mag_p)

    # Row 3 — Low-freq
    style_ax(axes[3, 0], "Low-freq (structure)", img=lf_n, show_roi=True)
    style_ax(axes[3, 1], "Low-freq (structure)", img=lf_p, show_roi=True)

    # Row 4 — High-freq residual
    style_ax(axes[4, 0], "High-freq residual", cmap="hot", img=hf_n, show_roi=True)
    style_ax(axes[4, 1], "High-freq residual", cmap="hot", img=hf_p, show_roi=True)

    # Row 5 — ROI zoom
    roi_n  = noisy_n[rs:re, cs:ce]
    roi_p  = noisy_p[rs:re, cs:ce]
    roi_hf_n = hf_n[rs:re, cs:ce]
    roi_hf_p = hf_p[rs:re, cs:ce]

    # Show ROI zoom: left half = noisy, right half = HF residual (side-by-side within axis)
    zoom_n = np.concatenate([roi_n, roi_hf_n], axis=1)
    zoom_p = np.concatenate([roi_p, roi_hf_p], axis=1)
    # Use a split colormap trick: just show as gray and overlay the HF in hot
    axes[5, 0].imshow(roi_n, cmap="gray", aspect="equal")
    axes[5, 0].set_title("ROI zoom — noisy (left)\nHF residual (right)", color="white",
                          fontsize=8, pad=3)
    axes[5, 0].set_xticks([])
    axes[5, 0].set_yticks([])

    axes[5, 1].imshow(roi_p, cmap="gray", aspect="equal")
    axes[5, 1].set_title("ROI zoom — noisy (left)\nHF residual (right)", color="white",
                          fontsize=8, pad=3)
    axes[5, 1].set_xticks([])
    axes[5, 1].set_yticks([])

    # Column headers
    for c, label in enumerate(cols):
        axes[0, c].set_title(f"{label}\n{row_labels[0]}", color="white", fontsize=9, pad=4)

    # Row labels on left column only
    for r, rlabel in enumerate(row_labels):
        if r == 0:
            continue  # already set in column header
        axes[r, 0].set_ylabel(rlabel, color="#aaaacc", fontsize=7, labelpad=4,
                               rotation=0, ha="right", va="center")

    # Global title
    fig.suptitle(
        "Frequency Decomposition of Kermany Paediatric CXR\n"
        f"Noise preset: {noise_preset}  |  Low-pass cutoff: {lp_cutoff*100:.0f}% Nyquist\n"
        "Yellow dashes = hilum / lower vessel ROI",
        color="white", fontsize=10, y=0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    return fig


def build_roi_detail_figure(normal_img: np.ndarray, pneumonia_img: np.ndarray,
                             out_path: str = "freq_decomp_roi.png"):
    """Separate figure: ROI zoom at all three noise levels, both classes.

    Shows the ROI of: clean | low noise | mid noise | high noise
    With a second row showing the HF residual of each.
    Helps show how pathology and noise co-occupy high frequencies.
    """
    rng = np.random.default_rng(0)
    presets = ["low", "mid", "high"]
    rs, re, cs, ce = roi_bbox(normal_img.shape)

    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")

    col_labels = ["Clean", "Low noise\na=0.01, b=0.002",
                  "Mid noise\na=0.03, b=0.005", "High noise\na=0.08, b=0.010"]
    row_labels  = ["Normal — image ROI", "Normal — HF residual ROI",
                   "Pneumonia — image ROI", "Pneumonia — HF residual ROI"]

    def _roi(arr):
        return arr[rs:re, cs:ce]

    for col_idx, (label, img) in enumerate(
            [("clean", normal_img), ("clean_p", pneumonia_img)]):
        pass  # just using the presets loop below

    noisy_n = [normal_img] + [
        add_foi_noise(normal_img, NOISE_PRESETS[p]["a"], NOISE_PRESETS[p]["b"], rng)
        for p in presets
    ]
    noisy_p = [pneumonia_img] + [
        add_foi_noise(pneumonia_img, NOISE_PRESETS[p]["a"], NOISE_PRESETS[p]["b"], rng)
        for p in presets
    ]

    for c in range(4):
        _, lf_n, hf_n = freq_decompose(noisy_n[c])
        _, lf_p, hf_p = freq_decompose(noisy_p[c])

        def _show(ax, img, cmap="gray", title=""):
            ax.imshow(img, cmap=cmap, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if title:
                ax.set_title(title, color="white", fontsize=7, pad=2)

        _show(axes[0, c], _roi(noisy_n[c]), cmap="gray",  title=col_labels[c])
        _show(axes[1, c], _roi(hf_n),       cmap="hot")
        _show(axes[2, c], _roi(noisy_p[c]), cmap="gray")
        _show(axes[3, c], _roi(hf_p),       cmap="hot")

    for r, rlabel in enumerate(row_labels):
        axes[r, 0].set_ylabel(rlabel, color="#aaaacc", fontsize=7,
                               rotation=0, ha="right", va="center", labelpad=4)

    fig.suptitle(
        "ROI Detail: Hilum / Lower-Lobe Vessels\n"
        "Top rows: Normal | Bottom rows: Pneumonia\n"
        "Gray = image ROI   Hot = high-frequency residual (noise + pathology edges)",
        color="white", fontsize=9, y=1.00
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Frequency decomposition visualisation for Kermany CXR dataset"
    )
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to chest_xray root (containing train/NORMAL and train/PNEUMONIA)")
    p.add_argument("--normal_name", type=str, default=None,
                   help="Filename inside train/NORMAL (default: first file)")
    p.add_argument("--pneumonia_name", type=str, default=None,
                   help="Filename inside train/PNEUMONIA (default: first file)")
    p.add_argument("--noise_preset", type=str, default="mid",
                   choices=["low", "mid", "high"])
    p.add_argument("--lp_cutoff", type=float, default=LP_CUTOFF,
                   help="Low-pass cutoff as fraction of max frequency (default 0.15)")
    p.add_argument("--out_dir", type=str, default=".",
                   help="Output directory for saved figures")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve data directory
    if args.data_dir is None:
        # Try to auto-detect relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "..", "..", "..", "..",
                                 "chest_xray")
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            args.data_dir = candidate
        else:
            raise FileNotFoundError(
                "Could not locate chest_xray directory. "
                "Pass --data_dir explicitly."
            )

    normal_dir    = os.path.join(args.data_dir, "train", "NORMAL")
    pneumonia_dir = os.path.join(args.data_dir, "train", "PNEUMONIA")

    def _pick(directory, name):
        if name:
            return os.path.join(directory, name)
        files = sorted(f for f in os.listdir(directory)
                       if f.lower().endswith((".jpeg", ".jpg", ".png")))
        if not files:
            raise FileNotFoundError(f"No images found in {directory}")
        return os.path.join(directory, files[0])

    normal_path    = _pick(normal_dir,    args.normal_name)
    pneumonia_path = _pick(pneumonia_dir, args.pneumonia_name)

    print(f"Normal image:    {os.path.basename(normal_path)}")
    print(f"Pneumonia image: {os.path.basename(pneumonia_path)}")

    normal_img    = load_and_preprocess(normal_path)
    pneumonia_img = load_and_preprocess(pneumonia_path)

    os.makedirs(args.out_dir, exist_ok=True)

    build_figure(
        normal_img, pneumonia_img,
        noise_preset=args.noise_preset,
        lp_cutoff=args.lp_cutoff,
        out_path=os.path.join(args.out_dir, "freq_decomp_main.png"),
    )

    build_roi_detail_figure(
        normal_img, pneumonia_img,
        out_path=os.path.join(args.out_dir, "freq_decomp_roi_detail.png"),
    )


if __name__ == "__main__":
    main()
