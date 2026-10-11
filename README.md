# CelebA-HQ Pyramid-JiT Looped-DiT Lab (v2: Pure Linear Heads + Perceptual Supervision)

Implementasi arsitektur **Pyramid-JiT + Looped-DiT** langsung di **Pixel Space** (resolusi $256 \times 256$, tanpa VAE) dengan supervisi perseptual modern (**PixelGen arXiv:2602.02493** & **Linum AI P-JiT**).

---

## 1. Arsitektur: Fusi Linear Pyramid-JiT & Looped-DiT (~50M Params)

1. **Pembuangan Bottleneck CNN:**
   - Membuang total modul CNN upsampler `PyramidDecoder`.
   - Menggunakan **Linear OutputHead** (`AdaLN-LayerNorm + Linear(512, 16*16*3=768)`) dan direct `unpatchify` (reshape) per-token seperti arsitektur resmi Linum AI dan JiT Kaiming He. Seluruh parameter transformer mengontrol piksel secara langsung.
2. **Pyramid Multi-Scale Readout:**
   - **Intermediate Readout (Pyramid Tap):** Membaca token setelah core loop blok untuk memprediksi citra resolusi bertingkat ($128 \times 128$, patch $8 \times 8$).
   - **Final Head:** Memprediksi citra resolusi penuh ($256 \times 256$, patch $16 \times 16$).
   - Kedua head adalah modul Linear murni tanpa konvolusi.
3. **Looped-DiT Core (OpenSenseNova):**
   - `loop_split = (2, 4, 2)` dengan $D=512$, $N=4$ loops untuk 4 blok inti.
   - Kedalaman efektif: 20 blok transformer.
   - Stabilisasi numerik terpasang: **QK-Norm** per-head, **Loop-Boundary RMSNorm**, dan **Bounded adaLN**.

---

## 2. Multi-Objective Perceptual Supervision (PixelGen & Linum AI)

Formulasi loss gabungan:
$$\mathcal{L} = \mathcal{L}_{\text{FM-final}} + 1.0 \cdot \mathcal{L}_{\text{FM-pyramid}} + g(t) \cdot (0.1 \cdot \mathcal{L}_{\text{LPIPS}} + 0.01 \cdot \mathcal{L}_{\text{P-DINO}}) + 0.1 \cdot \mathcal{L}_{\text{REPA}}$$

1. **Flow Matching Loss ($\mathcal{L}_{\text{FM}}$):**
   - $x$-prediction rectified flow loss dengan $v$-weighting:
     $$\hat{v} = \frac{\hat{x}_0 - x_t}{\max(1 - t, 0.05)}, \quad \mathcal{L} = \|\hat{v} - v\|^2$$
2. **Perceptual Texture Loss ($\mathcal{L}_{\text{LPIPS}}$):**
   - LPIPS VGG pada prediksi clean image $\hat{x}_0$ versus ground truth $x_0$.
3. **Perceptual Semantic Loss ($\mathcal{L}_{\text{P-DINO}}$):**
   - Cosine distance antara patch features DINOv2 dari $\hat{x}_0$ dan $x_0$.
4. **Representation Alignment ($\mathcal{L}_{\text{REPA}}$):**
   - Proyeksi linear dari token layer tengah DiT ke DINOv2 patch tokens citra bersih $x_0$.
5. **Noise Gating $g(t)$ (PixelGen):**
   - $g(t) = \mathbb{I}(t \ge 0.3)$ (di mana $t=1$ clean, $t=0$ noise). Menonaktifkan loss perseptual pada fase awal high-noise agar tidak merusak diversitas dan recall.
