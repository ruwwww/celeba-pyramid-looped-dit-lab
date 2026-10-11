#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-pyramid-looped-dit-lab

mkdir -p outputs/pyramid_looped_dit_v2_perceptual

# Pyramid-JiT Looped-DiT 50.4M Parameters (V2: Pure Linear Heads + Perceptual Supervision):
# 1. Linear OutputHead + direct unpatchify (CNN decoder completely removed)
# 2. Multi-scale Pyramid Readout (intermediate head decodes 128x128 target)
# 3. Multi-objective Perceptual Supervision (PixelGen arXiv:2602.02493 & Linum AI):
#    - Flow matching MSE (final 256x256 + pyramid 128x128)
#    - LPIPS (VGG) with noise gating g(t) = (t >= 0.3) [weight 0.1]
#    - P-DINO (DINOv2 patch cosine loss) with noise gating [weight 0.01]
#    - REPA (intermediate token alignment to clean image DINOv2 tokens) [weight 0.1]
# 4. Looped-DiT Core: loop_split=(2, 4, 2) at D=512, 4 loops -> Effective depth 20 layers
# 5. Stabilized: QK-Norm + Loop-boundary RMSNorm + Bounded adaLN

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --output-dir outputs/pyramid_looped_dit_v2_perceptual \
  --batch-size 16 \
  --epochs 60 \
  --max-steps 105000 \
  --num-workers 2 \
  --learning-rate 2e-4 \
  --gradient-clip 0.5 \
  --use-pretrained-perceptual \
  --cfg-scale 4.0 \
  --log-every 200 \
  --save-every 10000 \
  --sample-every 1000 \
  --sample-steps 50 \
  --sample-count 6 \
  --wandb-project "celeba-flow-dit" \
  --wandb-name "Pyramid-JiT-Looped-DiT-V2-Perceptual" \
  --amp > outputs/pyramid_looped_dit_v2_perceptual/train.log 2>&1
