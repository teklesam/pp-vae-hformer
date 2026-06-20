"""
Download pretrained weights for all comparison baselines and ensure the
KAIR model-code repository is present for network definitions.

Run on a machine with internet access (CSD3 login node or local).

Usage:
    python scripts/download_baseline_weights.py
    python scripts/download_baseline_weights.py --kair_only   # skip BM3D check
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAIR_ROOT = ROOT.parent / "KAIR"
WEIGHTS_DIR = ROOT / "model_zoo" / "baselines"

# (local_filename, download_url)  — URLs verified from KAIR's own download script
WEIGHTS = [
    ("dncnn_25.pth",
     "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth"),
    ("dncnn_gray_blind.pth",
     "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth"),
    ("ffdnet_gray.pth",
     "https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth"),
    ("drunet_gray.pth",
     "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"),
    ("ircnn_gray.pth",
     "https://github.com/cszn/KAIR/releases/download/v1.0/ircnn_gray.pth"),
    ("004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
     "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
     "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"),
]


def ensure_kair() -> None:
    """Clone KAIR if not present — needed for network_*.py definitions."""
    if KAIR_ROOT.exists():
        print(f"  [ok] KAIR already at {KAIR_ROOT}")
        return
    print(f"  [clone] KAIR → {KAIR_ROOT}")
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/cszn/KAIR.git",
        str(KAIR_ROOT),
    ])
    print("  [ok] KAIR cloned")


def download_weight(filename: str, url: str) -> None:
    dest = WEIGHTS_DIR / filename
    if dest.exists():
        print(f"  [skip] {filename} ({dest.stat().st_size // 1024} KB)")
        return
    print(f"  [download] {filename} ...")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [ok]   {filename} ({dest.stat().st_size // 1024} KB)")
    except Exception as exc:
        print(f"  [FAIL] {filename}: {exc}")
        if dest.exists():
            dest.unlink()


def check_bm3d() -> None:
    try:
        import bm3d  # noqa
        print("  [ok] bm3d already installed")
    except ImportError:
        print("  [installing] bm3d ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bm3d"])
        print("  [ok] bm3d installed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kair_only", action="store_true",
                        help="Skip BM3D pip install check")
    args = parser.parse_args()

    print(f"Weights dir : {WEIGHTS_DIR}")
    print(f"KAIR root   : {KAIR_ROOT}")
    print()

    print("=== KAIR source (network definitions) ===")
    ensure_kair()

    print()
    print("=== Pretrained weights ===")
    for filename, url in WEIGHTS:
        download_weight(filename, url)

    if not args.kair_only:
        print()
        print("=== BM3D (pip) ===")
        check_bm3d()

    print()
    downloaded = sorted(WEIGHTS_DIR.glob("*.pth"))
    if downloaded:
        print(f"Weights in {WEIGHTS_DIR}:")
        for f in downloaded:
            print(f"  {f.name:<62s}  {f.stat().st_size // 1024:>6d} KB")
    print("Done.")


if __name__ == "__main__":
    main()
