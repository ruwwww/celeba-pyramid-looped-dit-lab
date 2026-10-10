# Pyramid-JiT Looped-DiT Contracts & Plan

## Module Specifications

### 1. `models/pyramid_looped_dit.py`
- Arsitektur Pixel-Space Transformer untuk input RGB `[B, 3, 256, 256]`.
- **Patch Embedding:**
  - Low-rank bottleneck projection (JiT-style): `Conv2d(3, 128, kernel_size=16, stride=16)` diikuti linear ke `hidden_size=512`. Grid token: $16 \times 16 = 256$ token.
- **Looped Transformer Backbone:**
  - `pre_blocks`: 2 blok adaLN.
  - `core_blocks`: 4 blok adaLN dengan QK-Norm dan per-head Exclusive Self-Attention (XSA). Di-loop sebanyak $N=4$ kali dengan `loop_boundary_norm` (RMSNorm).
  - `post_blocks`: 2 blok adaLN.
- **Pyramid Detail Refinement / Unpatchify:**
  - Output dari transformer ($16 \times 16$ token di $D=512$) di-unpatchify atau diproyeksikan melalui residual pyramid decoder block ke resolusi $256 \times 256 \times 3$.
  - Menghasilkan prediksi langsung $\hat{x}_0 \in [-1, 1]$.
- **Conditioning:** Timestep $t$ (sinusoidal + MLP) dan class label $y \in \{0, \dots, 64\}$ via AdaLN.

### 2. `models/diffusion.py`
- Rectified Flow dengan formulasi JiT $x$-prediction:
  - Interpolasi: $x_t = t \cdot x_1 + (1 - t) \cdot \epsilon$ di mana $\epsilon \sim \mathcal{N}(0, I)$ dan $t \in [0, 1]$.
  - Target velocity: $v = x_1 - \epsilon$.
  - Prediksi velocity dari $\hat{x}_0$:
    $$\hat{v} = \frac{\hat{x}_0 - x_t}{\max(1 - t, 0.05)}$$
  - Loss MSE: $\|\hat{v} - v\|^2$.
- Sampler Euler ODE:
  - Bergerak dari $t=0$ ke $t=1$.
  - Dukungan CFG: $\hat{v} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$.

### 3. `tests/test_pyramid_looped_dit.py`
- Test 1: Shape forward `[2, 3, 256, 256]` -> `[2, 3, 256, 256]`.
- Test 2: Backward pass menghasilkan gradien valid dan finite untuk seluruh parameter.
- Test 3: 5-step Euler sampling menghasilkan citra finite dalam rentang valid.

### 4. `train.py` & `sample.py`
- Memanfaatkan `tracker.py` (W&B live tracking + offline jsonl).
- Flag screening run (`--screening-steps 100`) untuk verifikasi cepat sebelum commit run penuh.
