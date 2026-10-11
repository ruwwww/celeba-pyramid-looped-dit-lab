"""Pixel-space Pyramid-JiT Looped-DiT model."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization that is stable under bf16 autocast."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep features followed by a small conditioning MLP."""

    def __init__(self, hidden_size: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        half = hidden_size // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.reshape(-1).float()
        angles = timesteps[:, None] * self.frequencies[None, :].to(device=timesteps.device)
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if embedding.shape[-1] < self.hidden_size:
            embedding = F.pad(embedding, (0, self.hidden_size - embedding.shape[-1]))
        return self.mlp(embedding)


class QKNormSelfAttention(nn.Module):
    """Self-attention with independent per-head RMS normalization of Q and K."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch, tokens, hidden)
        return self.proj(output)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer used by the physical transformer blocks."""

    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _ada_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaLNBlock(nn.Module):
    """One transformer block conditioned through AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = QKNormSelfAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, mlp_ratio)
        self.ada_bias = nn.Parameter(torch.zeros(6, hidden_size))

    def forward(self, x: Tensor, modulation: Tensor) -> Tensor:
        modulation = modulation + self.ada_bias.reshape(1, -1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        attended = self.attn(_ada_modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + torch.tanh(gate_attn[:, None, :]) * attended
        feed_forward = self.mlp(_ada_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + torch.tanh(gate_mlp[:, None, :]) * feed_forward


class LinearOutputHead(nn.Module):
    """Conditioned token-wise projection used by JiT/Linum-AI readouts."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada_norm = nn.Linear(hidden_size, hidden_size * 2)
        self.proj = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
        )

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.ada_norm(condition).chunk(2, dim=-1)
        normalized = _ada_modulate(self.norm(tokens), shift, scale)
        return self.proj(normalized)


class PyramidLoopedDiT(nn.Module):
    """~50M-parameter pixel-space DiT with a physically shared core loop.

    The default configuration consumes RGB images at 256x256, embeds them into
    256 bottleneck tokens, and predicts a clean RGB image in [-1, 1]. The four
    core blocks are reused for four loop iterations, so the effective depth is
    2 + 4 * 4 + 2 while the parameter count is that of eight blocks.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        hidden_size: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 6.0,
        bottleneck_channels: int = 128,
        patch_size: int = 16,
        loop_split: Sequence[int] = (2, 4, 2),
        loop_repeats: int = 4,
        num_classes: int = 64,
        class_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        if patch_size < 2 or patch_size % 2 != 0:
            raise ValueError("patch_size must be an even integer of at least 2")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if len(loop_split) != 3 or any(depth < 0 for depth in loop_split):
            raise ValueError("loop_split must contain three non-negative depths")
        if loop_repeats < 1:
            raise ValueError("loop_repeats must be positive")
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob must be in [0, 1]")

        self.image_size = image_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.null_class = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.pyramid_patch_size = patch_size // 2
        self.loop_split = tuple(int(depth) for depth in loop_split)
        self.loop_repeats = loop_repeats

        self.patch_embed = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_projection = nn.Linear(bottleneck_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))

        self.time_embed = TimestepEmbedding(hidden_size)
        self.class_embed = nn.Embedding(num_classes + 1, hidden_size)
        self.ada_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

        pre_depth, core_depth, post_depth = self.loop_split
        self.pre_blocks = nn.ModuleList(
            [AdaLNBlock(hidden_size, num_heads, mlp_ratio) for _ in range(pre_depth)]
        )
        self.core_blocks = nn.ModuleList(
            [AdaLNBlock(hidden_size, num_heads, mlp_ratio) for _ in range(core_depth)]
        )
        self.post_blocks = nn.ModuleList(
            [AdaLNBlock(hidden_size, num_heads, mlp_ratio) for _ in range(post_depth)]
        )
        self.loop_boundary_norm = RMSNorm(hidden_size)
        self.final_head = LinearOutputHead(hidden_size, patch_size, in_channels)
        self.pyramid_head = LinearOutputHead(
            hidden_size,
            self.pyramid_patch_size,
            in_channels,
        )
        self.repa_proj = nn.Linear(hidden_size, 384)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in (*self.pre_blocks, *self.core_blocks, *self.post_blocks):
            nn.init.zeros_(block.ada_bias)

    def _class_labels(self, y: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if y is None:
            labels = torch.full((batch_size,), self.null_class, device=device, dtype=torch.long)
        else:
            labels = torch.as_tensor(y, device=device, dtype=torch.long).reshape(-1)
            if labels.numel() == 1 and batch_size > 1:
                labels = labels.expand(batch_size)
            if labels.numel() != batch_size:
                raise ValueError("class labels must have one value per image")
            if self.training and self.class_dropout_prob > 0:
                drop = torch.rand(batch_size, device=device) < self.class_dropout_prob
                labels = torch.where(
                    drop,
                    torch.full_like(labels, self.null_class),
                    labels,
                )
        if torch.any(labels < 0) or torch.any(labels > self.null_class):
            raise ValueError(f"class labels must be in [0, {self.null_class}]")
        return labels

    def _unpatchify(self, patches: Tensor, patch_size: int) -> Tensor:
        batch_size, token_count, patch_values = patches.shape
        expected_values = patch_size * patch_size * self.in_channels
        if token_count != self.num_tokens or patch_values != expected_values:
            raise ValueError("readout shape does not match the configured token grid")
        image = patches.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
            patch_size,
            patch_size,
            self.in_channels,
        )
        return image.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            self.in_channels,
            self.grid_size * patch_size,
            self.grid_size * patch_size,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor | None = None,
        return_pyramid: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
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
        condition = self.time_embed(timesteps) + self.class_embed(labels)
        modulation = self.ada_projection(condition)

        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = self.patch_projection(tokens) + self.pos_embed.to(dtype=tokens.dtype)
        for block in self.pre_blocks:
            tokens = block(tokens, modulation)
        for _ in range(self.loop_repeats):
            for block in self.core_blocks:
                tokens = block(tokens, modulation)
            tokens = self.loop_boundary_norm(tokens)
        pyramid_tokens = tokens
        for block in self.post_blocks:
            tokens = block(tokens, modulation)

        final_patches = self.final_head(tokens, condition)
        x_hat = torch.tanh(self._unpatchify(final_patches, self.patch_size))
        if not return_pyramid:
            return x_hat

        pyramid_patches = self.pyramid_head(pyramid_tokens, condition)
        x_hat_pyramid = torch.tanh(self._unpatchify(pyramid_patches, self.pyramid_patch_size))
        repa_tokens = self.repa_proj(pyramid_tokens)
        return x_hat, x_hat_pyramid, repa_tokens

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
