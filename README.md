# AREX: Affine–Residual Exponential Integrator for few-step flow-matching sampling

Reference implementation of AREX and its conditional variant AREX-C, training-free samplers for pretrained
flow-matching models.  The learned velocity field is split into the affine field of the Gaussian with the
target's mean and covariance, which is propagated exactly in the eigenbasis of the covariance, and a residual,
which is integrated with a one-node or Adams–Bashforth quadrature that costs one network evaluation per step.
A residual magnitude profile, measured once per checkpoint, enters the quadrature weights.  AREX-C uses the
within-condition covariance and sets the affine center of each trajectory from the first network evaluation.

This repository contains the sampler and the scripts that run it on the three backbones of the paper.
Baseline samplers and ablations are not included.

## Layout

```
arex/                         core library
  common.py                   paths, device, CIFAR-10 loader
  surrogate.py                moment-matched Gaussian surrogate: eigenbasis, propagator F(t,s), quadrature weights
  cfm.py                      TorchCFM CIFAR-10 checkpoint loader and target moments (m1, Sigma_1)
  fid_std.py, fid_inception.py   standard CIFAR-10 FID (pytorch-fid Inception, training-set reference)
scripts/cifar10/arex_cifar10.py           AREX on CIFAR-10 (I-CFM), FID at 50k
scripts/imagenet/sit_moments.py           AREX-C moments for SiT-XL/2 from the ImageNet-1k training set
scripts/imagenet/arex_sit.py              AREX-C on SiT-XL/2 (ImageNet-256, w = 1.5): FID, sFID, IS, precision, recall
scripts/sana/mjhq_get.py                  MJHQ-30K download
scripts/sana/sana_selfsample.py           self-samples of SANA-0.6B (the raw material of the moments)
scripts/sana/sana_moments.py              within-prompt covariance S_bar, pooled mean m1, calibration coefficient alpha
scripts/sana/arex_sana.py                 AREX-C on SANA-0.6B (MJHQ-30K, w = 4.5): FID with SANA's toolkit
setup_ext.sh                  clones the third-party code the scripts import (TorchCFM, SiT, SANA toolkit)
```

Every script is self-contained and is run from the repository root.  The identifiers `cfs` / `cfsc` in
command-line options and result files are the legacy names of AREX / AREX-C.

## Installation

```bash
pip install -r requirements.txt     # Python 3.11, the versions used for the paper
./setup_ext.sh                      # ext/torchcfm, ext/sit, ext/Sana
```

Checkpoints and data are placed as follows (none of them is redistributed here).

| Path | Content | Source |
|---|---|---|
| `data/cifar-10-batches-py/` | CIFAR-10 python batches (FID reference set) | `torchvision.datasets.CIFAR10(download=True)` |
| `ckpt_ext/cfm_cifar10_weights_step_400000.pt` | I-CFM CIFAR-10 checkpoint | TorchCFM release |
| `ckpt_ext/SiT-XL-2-256.pt` | SiT-XL/2 checkpoint | SiT release (`ext/sit/download.py`) |
| `ckpt_ext/sd-vae-ft-mse/` | SD-VAE | `stabilityai/sd-vae-ft-mse` on Hugging Face |
| `runs/VIRTUAL_imagenet256_labeled.npz` | ADM reference batch (contains `mu`, `sigma`, `mu_s`, `sigma_s`, `arr_0`) | openai/guided-diffusion |
| `data/test/PG-eval-data/MJHQ-30K/` | MJHQ-30K prompts and reference images | `python scripts/sana/mjhq_get.py` |
| Hugging Face cache | `benjamin-paine/imagenet-1k-256x256` shards (ImageNet-1k train, 256px) | downloaded and deleted shard by shard by `sit_moments.py` |
| Hugging Face cache | `Efficient-Large-Model/Sana_600M_512px_diffusers` | downloaded on first use (`export HF_HOME=$PWD/ckpt_ext/hf`) |

The pytorch-fid Inception weights are downloaded to `~/.cache/torch/hub/checkpoints/` on first use.

## Reproducing the AREX rows of the paper

All FIDs use the protocol stated in the paper: 50k samples on CIFAR-10 and ImageNet-256, 30k on MJHQ-30K.
The node position `rho` is selected per NFE on a held-out split (seed 777 on CIFAR-10 and SiT; the first
1,000 prompts on SANA) and the reported run uses seed 1234 (SANA: seed 0).

### CIFAR-10 (I-CFM, pixel space)

```bash
python scripts/cifar10/arex_cifar10.py --schemes frozen,ab2,ab3 --nfes 2,4,8,16,32
```

The moments of the flip-augmented training set are computed on first use (`runs/cfm_stats.pt`), the residual
profile is measured at the start of the run, and `rho` is selected from `{0.05, 0.1, 0.2, 0.35, 0.5}` on 10,000
held-out samples.  Results: `runs/arex_cifar10.json`.

### ImageNet-256 (SiT-XL/2, latent space, classifier-free guidance w = 1.5)

```bash
python scripts/imagenet/sit_moments.py                             # runs/sit_data_full_stats.pt (1.28M training images + flips)
python scripts/imagenet/arex_sit.py --method cfsc_ab3 --phi        # AREX-C-AB3 with the profile
python scripts/imagenet/arex_sit.py --method cfsc_ab2 --phi        # AREX-C-AB2
python scripts/imagenet/arex_sit.py --method cfsc     --phi        # one-node rule
```

`rho` is selected from `{0.05, 0.1, 0.2, 0.35}` (AB2, AB3) or `{0.2, 0.35, 0.5}` (one node) on 2,500 held-out
samples.  Results: `runs/sit_cfg_<method>_phi.json` with FID, sFID, IS, precision and recall per NFE.  The moments
(pooled mean, within-class covariance, calibration coefficient) are computed once from the ImageNet-1k training
set at 256px (1.28M images and their horizontal flips, streamed shard by shard) in the SD-VAE latent space.

### MJHQ-30K (SANA-0.6B, 512px, classifier-free guidance w = 4.5)

```bash
export HF_HOME=$PWD/ckpt_ext/hf
python scripts/sana/mjhq_get.py                                    # data/test/PG-eval-data/MJHQ-30K
python scripts/sana/sana_selfsample.py --n 50000 --steps 20        # runs/sana512_latents/z_*.pt, prompts.json
python scripts/sana/sana_moments.py                                # runs/sana512_latents/sbar_N50000.pt, alpha.json
python scripts/sana/arex_sana.py --scheme ab2 --rho 0.35 --phi --nfes 4,5,6,7,8,9,10 --n 30000   # AREX-C-AB2
python scripts/sana/arex_sana.py --scheme ab3 --rho 0.35 --phi --nfes 4,5,6,7,8,9,10 --n 30000   # AREX-C-AB3
python scripts/sana/arex_sana.py --rho 0.5 --phi --nfes 4,5,6,7,8,9,10 --n 30000                 # one-node rule
```

On SANA, `rho` was selected on the first 1,000 prompts (`--n 1000`) among `{0.2, 0.35, 0.5}` at NFE 4 and 8.
Results are merged into `runs/sana_bench/fid_n30000.json` under keys such as `cfsc-ab2-rho0.35-phi_nfe4`.
The SANA moments are estimated from self-samples because the training set is not released; the calibration
prompts are synthetic and disjoint from MJHQ-30K.

## Notes

- Hardware: every run in the paper used one NVIDIA GH200 (120 GB); CIFAR-10 runs in fp32, SiT and SANA in bf16.
- One network evaluation counts as one NFE.  With classifier-free guidance, one double-batch forward is one NFE.
- The residual profile is measured along 50-step Euler trajectories (2,048 on CIFAR-10 and SiT, 1,024 synthetic
  prompts on SANA) and is shared across conditions; the quadrature weights are precomputed once per grid.

## License

MIT for the code in this repository; see `THIRD_PARTY.md` for the vendored FID network and the external
dependencies.
