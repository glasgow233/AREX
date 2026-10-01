"""Shared constants and the CIFAR-10 loader."""
import os, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "cifar-10-batches-py")    # CIFAR-10 python batches (FID reference set)
CKPT_DIR = os.path.join(ROOT, "ckpt_ext")                    # pretrained checkpoints, see README
OUT = os.path.join(ROOT, "runs")                             # moments, profiles, results
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
TMIN = 1e-3                                                  # first node of every time grid


def load_cifar(train=True):
    """CIFAR-10 as (N, 3072) float32 in [-1, 1], layout C,H,W flattened."""
    xs = []
    if train:
        for i in range(1, 6):
            with open(os.path.join(DATA, f"data_batch_{i}"), "rb") as f:
                xs.append(pickle.load(f, encoding="bytes")[b"data"])
    else:
        with open(os.path.join(DATA, "test_batch"), "rb") as f:
            xs.append(pickle.load(f, encoding="bytes")[b"data"])
    return np.concatenate(xs).astype(np.float32) / 127.5 - 1.0
