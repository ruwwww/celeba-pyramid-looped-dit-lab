"""PixelGen-aligned pixel-space Looped-DiT model."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by PixelGen's JiT blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent feature pairs by 90 degrees."""

    if x.shape[-1] % 2:
        raise ValueError("rotary features must have an even dimension")
    original_shape = x.shape
    x = x.reshape(*original_shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(original_shape)


class VisionRotaryEmbeddingFast(nn.Module):
    """The flattened 2D vision RoPE used for image tokens in PixelGen."""

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 16,
        ft_seq_len: int | None = None,
        custom_freqs: Tensor | None = None,
        freqs_for: str = "lang",
        theta: float = 10_000.0,
        max_freq: float = 10.0,
        num_freqs: int = 1,
        num_cls_token: int = 0,
    ) -> None:
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("dim must be a positive even number")
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        if pt_seq_len < 1 or ft_seq_len < 1:
            raise ValueError("sequence lengths must be positive")

        if custom_freqs is not None:
            freqs = torch.as_tensor(custom_freqs, dtype=torch.float32)
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2.0, dim // 2) * math.pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs, dtype=torch.float32)
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        if freqs.numel() != dim // 2:
            raise ValueError("custom frequencies must contain dim / 2 values")

        positions = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len
        one_axis = torch.einsum("n,f->nf", positions, freqs)
        one_axis = one_axis.repeat_interleave(2, dim=-1)
        height = one_axis[:, None, :].expand(-1, ft_seq_len, -1)
        width = one_axis[None, :, :].expand(ft_seq_len, -1, -1)
        frequencies = torch.cat((height, width), dim=-1).reshape(-1, 2 * dim)

        cos = frequencies.cos()
        sin = frequencies.sin()
        if num_cls_token > 0:
            cos = torch.cat((torch.ones(num_cls_token, cos.shape[-1]), cos), dim=0)
            sin = torch.cat((torch.zeros(num_cls_token, sin.shape[-1]), sin), dim=0)
        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.freqs_cos.shape[-1]:
            raise ValueError(
                f"expected rotary dimension {self.freqs_cos.shape[-1]}, got {x.shape[-1]}"
            )
        cos = self.freqs_cos.to(device=x.device, dtype=x.dtype)
        sin = self.freqs_sin.to(device=x.device, dtype=x.dtype)
        return x * cos + rotate_half(x) * sin


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Create a 1D sine/cosine embedding from a position grid."""

    if embed_dim % 2:
        raise ValueError("embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10_000**omega
    positions = pos.reshape(-1)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate((np.sin(out), np.cos(out)), axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("embedding dimension must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate((emb_h, emb_w), axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    """Return PixelGen's fixed [grid_size ** 2, embed_dim] position table."""

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate((np.zeros((extra_tokens, embed_dim)), pos_embed), axis=0)
    return pos_embed


class BottleneckPatchEmbed(nn.Module):
    """PixelGen's two-stage bottleneck patch embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pca_dim: int = 128,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj1 = nn.Conv2d(
            in_chans,
            pca_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if (height, width) != self.img_size:
            raise ValueError(
                f"input image size ({height}*{width}) does not match model "
                f"({self.img_size[0]}*{self.img_size[1]})"
            )
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    """PixelGen sinusoidal timestep embedding and conditioning MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = t[:, None].float() * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        features = self.timestep_embedding(t.reshape(-1), self.frequency_embedding_size)
        return self.mlp(features)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: Tensor) -> Tensor:
        return self.embedding_table(labels)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """QK-normalized multi-head attention with 2D rotary embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, rope: VisionRotaryEmbeddingFast) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = rope(self.q_norm(query))
        key = rope(self.k_norm(key))
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class SwiGLUFFN(nn.Module):
    """PixelGen's gated feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class JiTBlock(nn.Module):
    """AdaLN-conditioned PixelGen transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: Tensor, condition: Tensor, rope: VisionRotaryEmbeddingFast) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Zero-initialized PixelGen output layer."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class PyramidLoopedDiT(nn.Module):
    """Pixel-space Looped-DiT with a shared four-block transformer core."""

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        hidden_size: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        bottleneck_channels: int = 128,
        patch_size: int = 16,
        loop_split: Sequence[int] = (2, 4, 2),
        loop_repeats: int = 4,
        num_classes: int = 64,
        class_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        if patch_size < 2 or patch_size % 2:
            raise ValueError("patch_size must be an even integer of at least 2")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if len(loop_split) != 3 or any(depth < 0 for depth in loop_split):
            raise ValueError("loop_split must contain three non-negative depths")
        if loop_repeats < 1:
            raise ValueError("loop_repeats must be positive")
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob must be in [0, 1]")

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.null_class = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.loop_split = tuple(int(depth) for depth in loop_split)
        self.loop_repeats = loop_repeats

        self.x_embedder = BottleneckPatchEmbed(
            image_size,
            patch_size,
            in_channels,
            bottleneck_channels,
            hidden_size,
            bias=True,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, hidden_size), requires_grad=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=hidden_size // num_heads // 2,
            pt_seq_len=self.grid_size,
        )

        pre_depth, core_depth, post_depth = self.loop_split
        self.pre_blocks = nn.ModuleList(
            [JiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(pre_depth)]
        )
        self.core_blocks = nn.ModuleList(
            [JiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(core_depth)]
        )
        self.post_blocks = nn.ModuleList(
            [JiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(post_depth)]
        )
        self.loop_boundary_norm = RMSNorm(hidden_size)
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self.repa_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 384),
        )
        self.initialize_weights()

    @property
    def patch_embed(self) -> BottleneckPatchEmbed:
        return self.x_embedder

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
        with torch.no_grad():
            self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            weight = self.x_embedder.proj1.weight
            nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
            weight = self.x_embedder.proj2.weight
            nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
            if self.x_embedder.proj2.bias is not None:
                nn.init.zeros_(self.x_embedder.proj2.bias)

            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            for block in (*self.pre_blocks, *self.core_blocks, *self.post_blocks):
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
            nn.init.zeros_(self.final_layer.linear.weight)
            nn.init.zeros_(self.final_layer.linear.bias)

    def _class_labels(self, y: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if y is None:
            labels = torch.full(
                (batch_size,), self.null_class, device=device, dtype=torch.long
            )
        else:
            labels = torch.as_tensor(y, device=device, dtype=torch.long).reshape(-1)
            if labels.numel() == 1 and batch_size > 1:
                labels = labels.expand(batch_size)
            if labels.numel() != batch_size:
                raise ValueError("class labels must have one value per image")
            if self.training and self.class_dropout_prob > 0:
                drop = torch.rand(batch_size, device=device) < self.class_dropout_prob
                labels = torch.where(drop, torch.full_like(labels, self.null_class), labels)
        if torch.any(labels < 0) or torch.any(labels > self.null_class):
            raise ValueError(f"class labels must be in [0, {self.null_class}]")
        return labels

    def unpatchify(self, patches: Tensor) -> Tensor:
        batch_size, token_count, patch_values = patches.shape
        expected_values = self.patch_size * self.patch_size * self.out_channels
        if token_count != self.num_tokens or patch_values != expected_values:
            raise ValueError("readout shape does not match the configured token grid")
        patches = patches.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        return torch.einsum("nhwpqc->nchpwq", patches).reshape(
            batch_size,
            self.out_channels,
            self.grid_size * self.patch_size,
            self.grid_size * self.patch_size,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor | None = None,
        return_repa: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected [B, {self.in_channels}, H, W] input")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}x{self.image_size} input")
        batch_size = x.shape[0]
        timesteps = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1)
        if timesteps.numel() == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.numel() != batch_size:
            raise ValueError("timesteps must have one value per image")

        labels = self._class_labels(y, batch_size, x.device)
        condition = self.t_embedder(timesteps) + self.y_embedder(labels)
        tokens = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)
        for block in self.pre_blocks:
            tokens = block(tokens, condition, self.feat_rope)
        for _ in range(self.loop_repeats):
            for block in self.core_blocks:
                tokens = block(tokens, condition, self.feat_rope)
            tokens = self.loop_boundary_norm(tokens)
        repa_tokens = self.repa_proj(tokens) if return_repa else None
        for block in self.post_blocks:
            tokens = block(tokens, condition, self.feat_rope)

        x_hat = self.unpatchify(self.final_layer(tokens, condition))
        if return_repa:
            assert repa_tokens is not None
            return x_hat, repa_tokens
        return x_hat

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "Attention",
    "BottleneckPatchEmbed",
    "FinalLayer",
    "JiTBlock",
    "PyramidLoopedDiT",
    "RMSNorm",
    "SwiGLUFFN",
    "VisionRotaryEmbeddingFast",
    "get_2d_sincos_pos_embed",
    "rotate_half",
]
