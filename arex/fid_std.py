"""Standard-protocol FID for CIFAR-10, matching the pipeline used by
DPM-Solver-v3, Wang & Vastola and the wider literature.

Protocol:
  1. TF-ported InceptionV3 weights (pt_inception-2015-12-05), not torchvision's.
  2. Input in [0,1]; the network itself resizes to 299 bilinear then maps to [-1,1].
     No ImageNet mean/std normalisation.
  3. 50 000 generated samples.
  4. Reference statistics from the CIFAR-10 *training* split (50 000 images).
  5. fp32 features, and samples quantised to uint8 before feature extraction --
     published numbers are computed on 8-bit images.

Usage
  from arex.fid_std import StdFID
  fid = StdFID()                       # loads inception, caches train-set stats
  value = fid(x)                       # x: (N,3072) float in [-1,1]
"""
import os
import numpy as np
import torch
from scipy import linalg

from arex.common import load_cifar, OUT, DEV
from arex.fid_inception import InceptionV3

REF_CACHE = os.path.join(OUT, "fid_ref_train50k.npz")


def to_uint8(x):
    """(N,3072) float in [-1,1]  ->  (N,3,32,32) uint8 in [0,255].

    Inverse of load_cifar's `uint8/127.5 - 1`, so real data round-trips exactly.
    """
    x = torch.clamp(x, -1.0, 1.0)
    q = torch.round((x + 1.0) * 127.5).clamp_(0, 255).to(torch.uint8)
    return q.reshape(-1, 3, 32, 32)


@torch.no_grad()
def inception_feats_std(x, model, bs=250):
    """x: (N,3072) float in [-1,1] -> (N,2048) float64 pool3 features."""
    out = []
    for i in range(0, len(x), bs):
        q = to_uint8(x[i:i + bs]).to(DEV)
        inp = q.float() / 255.0                      # network expects [0,1]
        f = model(inp)[0]                            # block 3 -> (B,2048,1,1)
        out.append(f.squeeze(-1).squeeze(-1).double().cpu())
    return torch.cat(out).numpy()


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Verbatim from pytorch_fid.fid_score.calculate_frechet_distance."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("fid: singular product; adding %s to diagonal" % eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component %s" % np.max(np.abs(covmean.imag)))
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2 * np.trace(covmean))


def stats_of(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


class StdFID:
    def __init__(self, ref_cache=REF_CACHE, verbose=True):
        self.model = InceptionV3([3], resize_input=True, normalize_input=True,
                                 use_fid_inception=True).to(DEV).eval()
        if os.path.isfile(ref_cache):
            d = np.load(ref_cache)
            self.mu_r, self.sig_r = d["mu"], d["sigma"]
            if verbose:
                print(f"[fid_std] reference stats loaded from {ref_cache}")
        else:
            if verbose:
                print("[fid_std] computing reference stats on CIFAR-10 train (50k)...")
            real = torch.tensor(load_cifar(True))
            f = self.feats(real)
            self.mu_r, self.sig_r = stats_of(f)
            np.savez(ref_cache, mu=self.mu_r, sigma=self.sig_r)
            if verbose:
                print(f"[fid_std] saved {ref_cache}")

    def feats(self, x):
        return inception_feats_std(x, self.model)

    def __call__(self, x):
        mu, sig = stats_of(self.feats(x))
        return frechet_distance(mu, sig, self.mu_r, self.sig_r)

    def between(self, xa, xb):
        """FID between two arbitrary sample sets (for validation checks)."""
        mu1, s1 = stats_of(self.feats(xa))
        mu2, s2 = stats_of(self.feats(xb))
        return frechet_distance(mu1, s1, mu2, s2)
