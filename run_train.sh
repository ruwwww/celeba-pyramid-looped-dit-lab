#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-pyramid-looped-dit-lab

mkdir -p outputs/pyramid_looped_dit_50m

# Pyramid-JiT Looped-DiT ~49.1M Parameters (Pixel-Space 256x256, Zero-VAE):
# Input: RGB [3, 256, 256]
# Bottleneck patch embedding: Conv2d(3, 128, k=16, s=16) + Linear(128, 512)
# Looped core: loop_split=(2, 4, 2) at D=512, 4 loops -> Effective depth 20 layers
# Stabilized: QK-Norm + Loop-boundary RMSNorm + Bounded adaLN
# Training objective: JiT x-prediction rectified flow loss with v-weighting
# Batch size 16, 105,000 steps (~60 epochs over 28,000 CelebA-HQ images)
# Live W&B tracking to project "celeba-flow-dit"

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --output-dir outputs/pyramid_looped_dit_50m \
  --batch-size 16 \
  --epochs 60 \
  --max-steps 105000 \
  --resume outputs/pyramid_looped_dit_50m/checkpoints/latest.pt \
  --num-workers 2 \
  --learning-rate 2e-4 \
  --gradient-clip 0.5 \
  --cfg-scale 4.0 \
  --log-every 200 \
  --save-every 10000 \
  --sample-every 1000 \
  --sample-steps 50 \
  --sample-count 6 \
  --wandb-project "celeba-flow-dit" \
  --wandb-name "Pyramid-JiT-Looped-DiT-50M" \
  --amp > outputs/pyramid_looped_dit_50m/train.log 2>&1
