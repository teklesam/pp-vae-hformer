#!/usr/bin/env python3
"""
Pre-download VGG-16 ImageNet weights to ~/.cache/torch before submitting
arm_n_perc (perceptual loss).  Run this ONCE on the login node (which has
internet); compute nodes will then find the weights in the local cache.

    python scripts/prefetch_vgg.py
"""
from torchvision.models import vgg16, VGG16_Weights
import torch, os

print("Downloading VGG-16 ImageNet weights...")
vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
cache = os.path.join(torch.hub.get_dir(), "checkpoints")
print(f"Weights cached at: {cache}")
print("Done — safe to submit array jobs including arm_n_perc.")
