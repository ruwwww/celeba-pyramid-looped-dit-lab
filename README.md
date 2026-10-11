# PixelGen-Aligned Looped-DiT Architecture & Implementation Plan

Implementasi arsitektur **PixelGen-aligned Looped-DiT** langsung di **Pixel Space** (resolusi $256 \times 256$, tanpa VAE) dengan memporting komponen resmi dari **[Zehong-Ma/PixelGen](https://github.com/Zehong-Ma/PixelGen)** (*arXiv:2602.02493*):

---

## 1. Spesifikasi Modul `models/pyramid_looped_dit.py` (Porting PixelGen)

1. **BottleneckPatchEmbed (PixelGen):**
   - Mengganti patch embed konvensional dengan implementasi persis PixelGen:
     `proj1 = nn.Conv2d(in_chans, pca_dim=128, kernel_size=16, stride=16, bias=False)`
     `proj2 = nn.Conv2d(pca_dim=128, embed_dim=hidden_size=512, kernel_size=1, stride=1, bias=True)`
   - Inisialisasi: `xavier_uniform_` pada view 2D bobot `proj1` dan `proj2`, `proj2.bias = 0`.
2. **Fixed 2D Sin-Cos Positional Embedding (PixelGen):**
   - Menghitung `get_2d_sincos_pos_embed(embed_dim=512, grid_size=16)`.
   - Disimpan sebagai parameter `requires_grad=False`.
3. **2D Rotary Position Embedding (2D RoPE - VisionRotaryEmbeddingFast):**
   - Porting `VisionRotaryEmbeddingFast` dan `rotate_half` dari PixelGen.
   - Dipasang pada query ($q$) dan key ($k$) di setiap layer self-attention.
4. **Looped Transformer Core dengan JiTBlock (PixelGen style):**
   - Setiap block terdiri dari:
     - `norm1 = RMSNorm(hidden_size, eps=1e-6)`
     - `attn = Attention(dim=512, num_heads=8, qkv_bias=True, qk_norm=True)` dengan 2D RoPE.
     - `norm2 = RMSNorm(hidden_size, eps=1e-6)`
     - `mlp = SwiGLUFFN(dim=512, hidden_dim=512 * 4)`
     - `adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(512, 6 * 512))` dengan **zero-init pada bobot dan bias terakhir**.
   - Arsitektur Looped: `loop_split = (2, 4, 2)` (2 pre-blocks, 4 core blocks di-loop 4 kali dengan loop-boundary RMSNorm, 2 post-blocks).
5. **FinalLayer (PixelGen Zero-Init, TANPA TANH):**
   - Menggantikan output head lama dengan `FinalLayer` resmi PixelGen:
     - `norm_final = RMSNorm(hidden_size)`
     - `adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))`
     - `linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels = 16 * 16 * 3 = 768)`
   - **Inisialisasi Penting:** `nn.init.zeros_(linear.weight)`, `nn.init.zeros_(linear.bias)`, `nn.init.zeros_(adaLN_modulation[1].weight)`, `nn.init.zeros_(adaLN_modulation[1].bias)`.
   - **Unpatchify:** Menggunakan fungsi `unpatchify` resmi PixelGen (`torch.einsum('nhwpqc->nchpwq', x)`), **TANPA `torch.tanh`**.
6. **REPA Projector (PixelGen 3-Layer MLP):**
   - `proj = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 384))` (atau 768 jika DINOv2-B).
   - Ditap dari token setelah iterasi loop ke-4.

---

## 2. Spesifikasi Supervisi `models/losses.py` (PixelGen Gated Loss)

1. **Perceptual Gating & Loss:**
   - Flow matching loss pada clean prediction $\hat{x}_0$: $v$-weighting $(1 - t)$ clamp 0.05.
   - Noise Gating: $g(t) = \mathbb{I}(t \ge 0.3)$.
   - $\mathcal{L}_{\text{LPIPS}}$ (bobot 0.1) dengan VGG.
   - $\mathcal{L}_{\text{P-DINO}}$ (bobot 0.01) pada DINOv2 layer 11.
   - $\mathcal{L}_{\text{REPA}}$ (bobot 0.1) pada DINOv2 clean tokens.
   - Tidak ada lagi pyramid flow loss tambahan yang memecah gradien.

---

## 3. Spesifikasi EMA Warmup di `train.py`

- Implementasi dynamic decay warmup:
  `current_decay = min(self.decay, (1.0 + step) / (10.0 + step))`
- Update parameter EMA menggunakan `current_decay`.
