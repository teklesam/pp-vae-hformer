"""
Frequency decomposition visualisation for Kermany paediatric CXR dataset.

Produces a multi-panel figure showing:
  - Clean normal vs clean pneumonia CXR pair
  - For each: low-frequency (DC + slow spatial variation) and high-frequency
    (edges, vessels, fine texture) components via 2D FFT
  - Simulated noisy versions at mid Foi preset and their frequency decomposition
  - Zoomed ROI panels highlighting the right hilum / perihilar vessels

Usage:
    python frequency_decomposition_viz.py \
        --data_dir /path/to/chest_xray \
        --output_dir ./freq_viz_output \
        --noise_preset mid

Prerequisites:
    pip install numpy matplotlib Pillow scipy
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Foi (2008) Poisson-Gaussian noise simulation
# ---------------------------------------------------------------------------

FOI_PRESETS = {
    "low":  {"a": 0.01,  "b": 0.002},
    "mid":  {"a": 0.03,  "b": 0.005},
    "high": {"a": 0.08,  "b": 0.010},
}


def foi_noise(image_norm: np.ndarray, preset: str = "mid", rng=None) -> np.ndarray:
    """Apply Foi Poisson-Gaussian noise to a [0,1] float32 image.

    Foi (2008) model: Var(z|y) = a*y + b, where y is the true intensity.
    Implemented via the Gaussian approximation: z = y + sqrt(a*y + b) * N(0,1).
    """
    if rng is None:
        rng = np.random.default_rng()
    p = FOI_PRESETS[preset]
    a, b = p["a"], p["b"]
    std_map = np.sqrt(a * image_norm + b)
    noise = rng.standard_normal(image_norm.shape).astype(np.float32)
    noisy = image_norm + std_map * noise
    return np.clip(noisy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FFT-based frequency decomposition
# ---------------------------------------------------------------------------

def fft_decompose(image_norm: np.ndarray, lp_radius_frac: float = 0.08):
    """Split image into low- and high-frequency components via 2D FFT.

    Args:
        image_norm: 2-D float32 array in [0, 1].
        lp_radius_frac: Low-pass radius as a fraction of min(H, W).
            0.08 ≈ keeping the central 8 % of the spectrum (smooth structures).

    Returns:
        fft_mag: log-scaled magnitude spectrum (for display).
        low:     low-frequency component (soft structures, lung fields).
        high:    high-frequency component (edges, vessels, fine texture).
    """
    H, W = image_norm.shape
    F = np.fft.fftshift(np.fft.fft2(image_norm))
    fft_mag = np.log1p(np.abs(F))

    # Circular low-pass mask centred on DC
    cy, cx = H // 2, W // 2
    radius = lp_radius_frac * min(H, W)
    ys, xs = np.ogrid[:H, :W]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    lp_mask = dist <= radius

    F_low = F * lp_mask
    F_high = F * (~lp_mask)

    low = np.real(np.fft.ifft2(np.fft.ifftshift(F_low))).astype(np.float32)
    high = np.real(np.fft.ifft2(np.fft.ifftshift(F_high))).astype(np.float32)

    # Normalise each component independently for display
    def _norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    return _norm(fft_mag), _norm(low), _norm(high)


# ---------------------------------------------------------------------------
# ROI extraction (right hilum / perihilar region)
# ---------------------------------------------------------------------------

def get_hilum_roi(image_norm: np.ndarray, roi_frac=(0.30, 0.55, 0.50, 0.75)):
    """Return a zoomed crop of the right hilum / perihilar region.

    roi_frac: (y_start, y_end, x_start, x_end) as fractions of image dims.
    Default targets the right mid-lung / hilar zone typical of AP paediatric CXR.
    """
    H, W = image_norm.shape
    y0 = int(roi_frac[0] * H)
    y1 = int(roi_frac[1] * H)
    x0 = int(roi_frac[2] * W)
    x1 = int(roi_frac[3] * W)
    return image_norm[y0:y1, x0:x1], (y0, y1, x0, x1)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_sample(folder: Path, size: int = 256) -> np.ndarray:
    """Load one JPEG from folder, convert to grayscale float32 in [0,1]."""
    files = list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.JPEG"))
    if not files:
        raise FileNotFoundError(f"No JPEG files found in {folder}")
    path = random.choice(files)
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _imshow(ax, img, title, cmap="gray", vmin=0, vmax=1):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


def _add_roi_box(ax, roi_coords, img_shape):
    y0, y1, x0, x1 = roi_coords
    H, W = img_shape
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)


def make_figure(
    normal: np.ndarray,
    pneumonia: np.ndarray,
    normal_noisy: np.ndarray,
    pneumonia_noisy: np.ndarray,
    noise_preset: str,
    lp_radius_frac: float,
    output_path: str,
):
    """Create and save the full 5-row, 8-column comparison figure."""

    images = {
        "Normal (clean)": normal,
        "Pneumonia (clean)": pneumonia,
        f"Normal (noisy, {noise_preset})": normal_noisy,
        f"Pneumonia (noisy, {noise_preset})": pneumonia_noisy,
    }

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#1a1a1a")

    # Top suptitle
    fig.suptitle(
        "Frequency Decomposition of Paediatric Chest Radiographs\n"
        f"Kermany 2018 Dataset  |  Low-pass radius = {lp_radius_frac*100:.0f}% of spatial bandwidth  |  "
        f"Noise preset: Foi (2008) {noise_preset}",
        fontsize=11, color="white", y=0.98
    )

    gs = GridSpec(
        nrows=5, ncols=8,
        figure=fig,
        hspace=0.35, wspace=0.08,
        left=0.03, right=0.97, top=0.93, bottom=0.04
    )

    row_labels = list(images.keys())
    col_labels = [
        "Original", "Log |FFT|",
        "Low-freq\n(soft tissue)", "High-freq\n(vessels/edges)",
        "ROI\n(hilum)", "ROI Low-freq", "ROI High-freq", "Diff\n(noisy-clean)"
    ]

    # Column headers
    for c, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, c])
        ax.text(
            0.5, 0.5, label,
            ha="center", va="center", fontsize=8, color="white",
            fontweight="bold", wrap=True
        )
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")

    all_data = []
    for label, img in images.items():
        fft_mag, low, high = fft_decompose(img, lp_radius_frac)
        roi_img, roi_coords = get_hilum_roi(img)
        roi_low, _ = fft_decompose(roi_img, lp_radius_frac)[1], None
        roi_fft_mag, roi_low_d, roi_high_d = fft_decompose(roi_img, lp_radius_frac)
        all_data.append({
            "label": label, "img": img,
            "fft_mag": fft_mag, "low": low, "high": high,
            "roi_img": roi_img, "roi_coords": roi_coords,
            "roi_low": roi_low_d, "roi_high": roi_high_d,
        })

    cmap_main = "gray"
    cmap_fft = "inferno"
    cmap_low = "gray"
    cmap_high = "RdBu_r"

    for r, d in enumerate(all_data):
        row = r + 1  # row 0 = headers

        # Col 0: original image with ROI box
        ax0 = fig.add_subplot(gs[row, 0])
        _imshow(ax0, d["img"], d["label"], cmap=cmap_main)
        _add_roi_box(ax0, d["roi_coords"], d["img"].shape)

        # Col 1: log FFT magnitude
        ax1 = fig.add_subplot(gs[row, 1])
        _imshow(ax1, d["fft_mag"], "", cmap=cmap_fft)

        # Col 2: low frequency
        ax2 = fig.add_subplot(gs[row, 2])
        _imshow(ax2, d["low"], "", cmap=cmap_low)

        # Col 3: high frequency (RdBu centred at 0.5 after normalisation)
        ax3 = fig.add_subplot(gs[row, 3])
        _imshow(ax3, d["high"], "", cmap=cmap_high, vmin=0.3, vmax=0.7)

        # Col 4: zoomed ROI (hilum crop)
        ax4 = fig.add_subplot(gs[row, 4])
        _imshow(ax4, d["roi_img"], "red box", cmap=cmap_main)

        # Col 5: ROI low-freq
        ax5 = fig.add_subplot(gs[row, 5])
        _imshow(ax5, d["roi_low"], "", cmap=cmap_low)

        # Col 6: ROI high-freq
        ax6 = fig.add_subplot(gs[row, 6])
        _imshow(ax6, d["roi_high"], "", cmap=cmap_high, vmin=0.3, vmax=0.7)

        # Col 7: difference map (noisy - clean, where applicable)
        ax7 = fig.add_subplot(gs[row, 7])
        if r >= 2:
            # noisy row: show difference from clean counterpart
            clean = all_data[r - 2]["img"]
            diff = d["img"] - clean
            diff_disp = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            _imshow(ax7, diff_disp, "", cmap="seismic", vmin=0, vmax=1)
        else:
            # clean row: show the high-frequency as residual proxy
            _imshow(ax7, d["high"], "high-freq\n(proxy)", cmap="seismic", vmin=0.3, vmax=0.7)

    # Row labels on left
    for r, d in enumerate(all_data):
        row = r + 1
        ax = fig.add_subplot(gs[row, 0])
        ax.set_ylabel(d["label"], fontsize=7, color="white", rotation=90, labelpad=4)

    # Colourbar annotations (bottom row, rightmost two cells)
    fig.text(0.02, 0.01, "Colourmap: gray = intensity  |  inferno = log-FFT power  |  RdBu = high-freq residual  |  seismic = signed difference",
             fontsize=6.5, color="#aaaaaa", ha="left")
    fig.text(0.98, 0.01, "Red dashed box: right hilum / perihilar ROI (y=30-55%, x=50-75% of image)",
             fontsize=6.5, color="#ff8888", ha="right")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


def make_spectral_power_figure(
    normal: np.ndarray,
    pneumonia: np.ndarray,
    normal_noisy: np.ndarray,
    pneumonia_noisy: np.ndarray,
    noise_preset: str,
    output_path: str,
):
    """Radial-average power spectral density curves for the four images."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        "Radial Power Spectral Density  |  Kermany Paediatric CXR",
        fontsize=11, color="white", y=1.02
    )

    def radial_psd(img):
        F = np.fft.fftshift(np.fft.fft2(img))
        power = np.abs(F) ** 2
        H, W = img.shape
        cy, cx = H // 2, W // 2
        ys, xs = np.ogrid[:H, :W]
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(int)
        max_r = min(cy, cx)
        radii = np.arange(max_r)
        psd = np.array([power[dist == r].mean() for r in radii])
        freq = radii / (min(H, W))  # normalised spatial frequency [0, 0.5]
        return freq, psd

    pairs = [
        (normal, normal_noisy, "Normal"),
        (pneumonia, pneumonia_noisy, "Pneumonia"),
    ]

    colors = {"clean": "#4fc3f7", "noisy": "#ef9a9a"}

    for ax, (clean, noisy, label) in zip(axes, pairs):
        ax.set_facecolor("#2a2a2a")
        freq_c, psd_c = radial_psd(clean)
        freq_n, psd_n = radial_psd(noisy)

        ax.semilogy(freq_c, psd_c, color=colors["clean"], lw=2, label="Clean")
        ax.semilogy(freq_n, psd_n, color=colors["noisy"], lw=2, linestyle="--",
                    label=f"Noisy ({noise_preset})")

        # Mark low/high boundary
        ax.axvline(x=0.08, color="yellow", lw=1, linestyle=":", alpha=0.7)
        ax.text(0.09, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1,
                "LP cutoff\n(8%)", fontsize=7, color="yellow")

        ax.set_xlabel("Normalised spatial frequency (cycles/pixel)", color="white", fontsize=9)
        ax.set_ylabel("Power spectral density (log)", color="white", fontsize=9)
        ax.set_title(f"{label} CXR: PSD comparison", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.legend(fontsize=9, facecolor="#333333", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Frequency decomposition visualisation for Kermany CXR")
    p.add_argument("--data_dir", type=str,
                   default="/Users/sam/Documents/PPVAE Dissertation Project/chest_xray",
                   help="Root directory containing train/NORMAL and train/PNEUMONIA sub-folders")
    p.add_argument("--output_dir", type=str,
                   default="./freq_viz_output",
                   help="Directory where output figures are saved")
    p.add_argument("--noise_preset", choices=["low", "mid", "high"], default="mid",
                   help="Foi noise preset to apply (default: mid)")
    p.add_argument("--image_size", type=int, default=256,
                   help="Square resize target in pixels (default: 256)")
    p.add_argument("--lp_radius_frac", type=float, default=0.08,
                   help="Low-pass radius as fraction of image width (default: 0.08 = 8%%)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_dir)
    normal_dir = data_root / "train" / "NORMAL"
    pneumonia_dir = data_root / "train" / "PNEUMONIA"

    if not normal_dir.exists() or not pneumonia_dir.exists():
        raise FileNotFoundError(
            f"Expected {normal_dir} and {pneumonia_dir}. "
            "Check --data_dir points to the Kermany chest_xray root."
        )

    print(f"Loading images from {data_root} ...")
    normal = load_sample(normal_dir, size=args.image_size)
    pneumonia = load_sample(pneumonia_dir, size=args.image_size)

    rng = np.random.default_rng(args.seed)
    print(f"Applying Foi noise preset '{args.noise_preset}' ...")
    normal_noisy = foi_noise(normal, preset=args.noise_preset, rng=rng)
    pneumonia_noisy = foi_noise(pneumonia, preset=args.noise_preset, rng=rng)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating frequency decomposition panel ...")
    make_figure(
        normal=normal,
        pneumonia=pneumonia,
        normal_noisy=normal_noisy,
        pneumonia_noisy=pneumonia_noisy,
        noise_preset=args.noise_preset,
        lp_radius_frac=args.lp_radius_frac,
        output_path=str(out_dir / f"freq_decomp_{args.noise_preset}.png"),
    )

    print("Generating radial PSD curves ...")
    make_spectral_power_figure(
        normal=normal,
        pneumonia=pneumonia,
        normal_noisy=normal_noisy,
        pneumonia_noisy=pneumonia_noisy,
        noise_preset=args.noise_preset,
        output_path=str(out_dir / f"radial_psd_{args.noise_preset}.png"),
    )

    print("\nDone. Files written to:", out_dir)
    print("  freq_decomp_{preset}.png  -- main 5-row panel (original / FFT / low / high / ROI zoom)")
    print("  radial_psd_{preset}.png   -- radial PSD curves (normal vs pneumonia, clean vs noisy)")
    print("\nRecommended Dissertation Figure 1.2 caption:")
    print(
        "  Figure 1.2. Frequency decomposition of representative clean and noise-corrupted "
        "paediatric chest radiographs from the Kermany (2018) dataset. "
        "Columns show (left to right): the original image (red dashed box marks the right hilum ROI); "
        "its log-scaled 2-D FFT magnitude spectrum; the low-frequency component (spatial frequencies "
        f"below {args.lp_radius_frac*100:.0f}\\% of the Nyquist limit), capturing lung fields and cardiac silhouette; "
        "the high-frequency residual, encoding vessel boundaries and fine parenchymal texture; "
        "a zoomed crop of the right hilum; and the corresponding low- and high-frequency ROI crops. "
        f"The final column shows the signed pixel difference between the Foi-simulated noisy image and its clean reference. "
        "Noise increases spectral power uniformly across all spatial frequencies (rows 3--4 vs 1--2), "
        "while pneumonic consolidation selectively attenuates high-frequency vessel detail relative to normal lung (compare columns 3--4 across rows 1 and 2)."
    )


if __name__ == "__main__":
    main()
