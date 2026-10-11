# Architecture & Training Plan: Pure Linear Pyramid-JiT + Looped-DiT with Perceptual Supervision

## Contracts & Requirements

### 1. `models/pyramid_looped_dit.py`
- Arsitektur transformer DiT untuk citra RGB $256 \times 256 \times 3$.
- Patch size: $16 \times 16$ (256 tokens).
- Bottleneck patch embedding: `Conv2d(3, 128, kernel_size=16, stride=16)` + `Linear(128, 512)` + learned positional embeddings.
- **Looped Blocks:**
  - `pre_blocks`: 2 blok transformer adaLN ($D=512$, heads=8, mlp_ratio=6.0 SwiGLU).
  - `core_blocks`: 4 blok transformer dengan QK-Norm dan per-head Exclusive Self-Attention (XSA). Di-loop $N=4$ kali dengan `loop_boundary_norm` (RMSNorm).
  - `post_blocks`: 2 blok transformer.
- **Readout Heads (Linear + Unpatchify, NO CNN):**
  - `final_head`: `AdaLN-LayerNorm + Linear(512, 16 * 16 * 3)` -> unpatchify ke $256 \times 256 \times 3$.
  - `pyramid_head` (intermediate readout): `AdaLN-LayerNorm + Linear(512, 8 * 8 * 3)` -> unpatchify ke $128 \times 128 \times 3$.
  - `repa_proj`: `Linear(512, 384)` untuk memproyeksikan token tengah ke dimensi DINOv2 (`dinov2_vits14`).
- Output `forward(x_t, t, y=None, return_pyramid=False)`:
  - Default: mengembalikan `x_hat` (shape $[B, 3, 256, 256]$).
  - Jika `return_pyramid=True`: mengembalikan `(x_hat, x_hat_pyramid, repa_tokens)`.

### 2. `models/losses.py` (atau di `models/diffusion.py`)
- Implementasi multi-objective perceptual supervision (PixelGen & Linum AI):
  - `LPIPSLoss`: Wrapper lazy loading `lpips.LPIPS(net="vgg")`.
  - `DINOPerceptualLoss`: Memuat `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`, mengekstrak patch tokens dari $\hat{x}_0$ dan $x_0$, menghitung $1 - \cos(f_{\text{pred}}, f_{\text{target}})$.
  - `REPALoss`: Menghitung $1 - \cos(\text{repa\_tokens}, f_{\text{dino}}(x_0))$.
  - `combined_training_loss(model, x0, y, ...)`:
    - Sample $t \sim \text{logit-normal}$ atau uniform.
    - $x_t = t \cdot x_0 + (1 - t) \cdot \epsilon$, $v = x_0 - \epsilon$.
    - $\hat{v} = (\hat{x}_0 - x_t) / \max(1 - t, 0.05)$, $\mathcal{L}_{\text{final}} = \|\hat{v} - v\|^2$.
    - Target pyramid $x_{0,\text{pyramid}} = \text{interpolate}(x_0, (128, 128))$.
    - $\hat{v}_{\text{pyramid}} = (\hat{x}_{0,\text{pyramid}} - x_{t,\text{pyramid}}) / \max(1 - t, 0.05)$, $\mathcal{L}_{\text{pyramid}} = \|\hat{v}_{\text{pyramid}} - v_{\text{pyramid}}\|^2$.
    - Noise gating: $g(t) = (t \ge 0.3)$.
    - $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{final}} + 1.0 \cdot \mathcal{L}_{\text{pyramid}} + g(t) \cdot (0.1 \cdot \mathcal{L}_{\text{lpips}} + 0.01 \cdot \mathcal{L}_{\text{pdino}}) + 0.1 \cdot \mathcal{L}_{\text{repa}}$.

### 3. `models/diffusion.py`
- Sampler Euler ODE tetap clean: hanya mengevaluasi `model(samples, t, labels)` (tanpa overhead DINO/LPIPS saat inferensi) dengan CFG.

### 4. `tests/test_pyramid_looped_dit.py`
- Test 1: Shape forward pass `[2, 3, 256, 256]` -> `[2, 3, 256, 256]`.
- Test 2: Pyramid readout menghasilkan `[2, 3, 128, 128]` dan repa tokens `[2, 256, 384]`.
- Test 3: Backward pass loss total menghasilkan gradien finite untuk seluruh parameter model.
- Test 4: 5-step Euler sampling menghasilkan citra valid dalam $[-1, 1]$.
