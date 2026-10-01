#!/usr/bin/env bash
# Third-party code that the scripts import at run time.  It is cloned here, not redistributed.
set -e
cd "$(dirname "$0")"
mkdir -p ext ckpt_ext data runs
# UNet definition of the TorchCFM CIFAR-10 checkpoints (MIT)
[ -d ext/torchcfm ] || git clone --depth 1 https://github.com/atong01/conditional-flow-matching ext/torchcfm
# SiT-XL/2 model definition, models.py (MIT)
[ -d ext/sit ] || git clone --depth 1 https://github.com/willisma/SiT ext/sit
# SANA metric toolkit, tools/metrics/pytorch-fid (Apache-2.0)
[ -d ext/Sana ] || git clone --depth 1 https://github.com/NVlabs/Sana ext/Sana
echo "done. Now place the checkpoints and data as described in README.md."
