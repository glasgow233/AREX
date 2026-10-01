# Third-party code and assets

## Vendored in this repository

- `arex/fid_inception.py`: the FID Inception network of
  [pytorch-fid](https://github.com/mseitzer/pytorch-fid) 0.3.0 (Apache-2.0, Maximilian Seitzer),
  copied verbatim except that the weights are first looked up in a local cache.

## Cloned at set-up time by `setup_ext.sh` (not redistributed)

| Directory | Project | License | Used for |
|---|---|---|---|
| `ext/torchcfm` | [TorchCFM](https://github.com/atong01/conditional-flow-matching) | MIT | UNet definition of the CIFAR-10 I-CFM checkpoint |
| `ext/sit` | [SiT](https://github.com/willisma/SiT) | MIT | SiT-XL/2 model definition (`models.py`) |
| `ext/Sana` | [SANA](https://github.com/NVlabs/Sana) | Apache-2.0 | FID toolkit `tools/metrics/pytorch-fid` for MJHQ-30K |

## Checkpoints and data (downloaded by the user, see README)

- TorchCFM CIFAR-10 checkpoint `cfm_cifar10_weights_step_400000.pt` (TorchCFM release).
- SiT-XL/2 checkpoint `SiT-XL-2-256.pt` (SiT release) and the `stabilityai/sd-vae-ft-mse` autoencoder.
- ADM reference batch `VIRTUAL_imagenet256_labeled.npz` ([guided-diffusion](https://github.com/openai/guided-diffusion), MIT).
- `Efficient-Large-Model/Sana_600M_512px_diffusers` (Hugging Face) and the MJHQ-30K dataset
  (`playgroundai/MJHQ-30K`, Hugging Face; gated, license as stated there).
- ImageNet-1k at 256px, `benjamin-paine/imagenet-1k-256x256` (Hugging Face), for the SiT-XL/2 moments.
