# CelebA-HQ Pyramid-JiT Looped-DiT Lab (~50M Parameters)

Implementasi arsitektur **Pyramid-JiT + Looped-DiT** untuk generative modeling langsung di **Pixel Space** (resolusi $256 \times 256$, tanpa VAE) pada dataset CelebA-HQ.

---

## 1. Konsep Inti Arsitektur: Pyramid-JiT + Looped-DiT

Menggabungkan 3 prinsip utama:
1. **JiT (Just image Transformers - Kaiming He, arXiv:2511.13720):**
   - Beroperasi langsung pada raw pixel RGB tanpa VAE.
   - Menggunakan **$x$-prediction** (memprediksi citra bersih $x_0$) dengan **$v$-weighted loss**:
     $$\mathcal{L} = \left\| \frac{\hat{x} - x_t}{\max(1 - t, 0.05)} - v \right\|^2$$
   - Bottleneck linear patch embedding.
2. **Pyramid Multi-Scale Processing:**
   - Image input $256 \times 256 \times 3$.
   - **Stage 1 (Coarse/Semantic):** Patch size $16 \times 16 \to 16 \times 16 = 256$ token.
   - **Stage 2 (Fine/Detail):** Patch size $8 \times 8 \to 32 \times 32 = 1024$ token (atau dekonvolusi/upsampler progresif).
   - Menghilangkan konflik gradien antara struktur makro (low-frequency) dan tekstur piksel (high-frequency).
3. **Looped-DiT (OpenSenseNova, arXiv:2609.40305):**
   - Blok transformer dibagi menjadi `loop_split = (pre=2, core=4, post=2)`.
   - Core blocks di-loop sebanyak $N=4$ kali, memberikan kedalaman efektif $2 + (4 \times 4) + 2 = 20$ layer dengan memori parameter fisik hanya seukuran 8 blok (~48M–52M parameter).
   - Dilengkapi **QK-Norm** dan **Loop-Boundary RMSNorm** untuk menjamin stabilitas numerik 100%.

---

## 2. Spesifikasi Model
- **Input:** $3 \times 256 \times 256$ RGB images.
- **Backbone Width:** $D = 512$, heads $= 8$, mlp_ratio $= 6.0$ (SwiGLU).
- **Physical Parameters:** ~50M parameter.
- **Conditioning:** 64 semantic DINOv2 pseudo-classes dengan class dropout 0.15 untuk Classifier-Free Guidance (CFG).
- **Sampling:** Euler ODE $x$-prediction integrator dari $t=0$ (noise) ke $t=1$ (clean image).
- **Tracking:** Live Weights & Biases (W&B) + `metrics.jsonl` + PNG figures via `tracker.py`.

---

## 3. Struktur Modul
- `models/pyramid_looped_dit.py`: Arsitektur model Pyramid-JiT Looped-DiT.
- `models/diffusion.py`: Flow-matching $x$-prediction loss, $v$-weighted objective, dan Euler ODE sampler dengan CFG.
- `models/__init__.py`: Export modul.
- `tests/test_pyramid_looped_dit.py`: Unit test forward pass shape, loss calculation, backward gradient, dan sampler.
- `train.py`: Training CLI terstandarisasi dengan W&B live tracking, AMP bf16, EMA, screening test support.
- `sample.py`: CLI standalone image generation.
