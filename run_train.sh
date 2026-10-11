#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-pyramid-looped-dit-lab

mkdir -p outputs/pixelgen_looped_dit_v3

# PixelGen-Aligned Looped-DiT 40.2M Parameters (V3):
# 1. BottleneckPatchEmbed (128 pca dim -> 512 embed dim)
# 2. Fixed 2D Sin-Cos PosEmbed + 2D RoPE (VisionRotaryEmbeddingFast) on q & k
# 3. JiTBlock: RMSNorm + QK-Norm + SwiGLUFFN (mlp_ratio=4.0) + Zero-Init adaLN
# 4. Looped-DiT Core: loop_split=(2, 4, 2) at D=512, 4 loops -> Effective depth 20 layers
# 5. FinalLayer: Zero-Init Linear + Direct Unpatchify (TANPA torch.tanh)
# 6. REPA: 3-Layer MLP Projector (512 -> 512 -> 512 -> 384) to DINOv2 Layer 11
# 7. Multi-Objective Gated Perceptual Loss: Flow Matching + Gated LPIPS (0.1) + Gated P-DINO (0.01) + REPA (0.1)
# 8. Dynamic EMA Warmup: decay = min(0.9999, (1 + step) / (10 + step))

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --output-dir outputs/pixelgen_looped_dit_v3 \
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
  --wandb-name "PixelGen-Looped-DiT-V3-RoPE" \
  --amp > outputs/pixelgen_looped_dit_v3/train.log 2>&1
