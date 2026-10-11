# Implementation Plan: PixelGen Components Porting & EMA Warmup

## Tasks Checklist
- [ ] Implement `models/pyramid_looped_dit.py`:
  - Port `BottleneckPatchEmbed` (128 pca dim -> 512)
  - Port `get_2d_sincos_pos_embed` (fixed, requires_grad=False)
  - Port `VisionRotaryEmbeddingFast` (2D RoPE for q and k)
  - Port `JiTBlock` with RMSNorm, QK-Norm, SwiGLUFFN, and zero-initialized adaLN modulation
  - Preserve looped architecture `loop_split = (2, 4, 2)` with 4 loops and loop-boundary RMSNorm
  - Port `FinalLayer` with zero-initialized linear layer and adaLN modulation, unpatchify without `torch.tanh`
  - Implement 3-layer MLP REPA projector (`512 -> 512 -> 512 -> 384`)
- [ ] Implement `models/losses.py`:
  - Flow matching loss with v-prediction from x-prediction
  - Noise-gated LPIPS (weight 0.1) and P-DINO layer 11 (weight 0.01) with threshold 0.3
  - REPA cosine loss (weight 0.1) using the 3-layer MLP projector
- [ ] Update `train.py`:
  - Update `ModelEMA` to implement dynamic warmup decay: `min(self.decay, (1.0 + step) / (10.0 + step))`
  - Update `ema.update(model, step=global_step)`
- [ ] Update `tests/test_pyramid_looped_dit.py` to test all new layers, forward pass, backward pass, and Euler sampling
- [ ] Run pytest suite to ensure 100% pass
