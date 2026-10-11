"""Flow-matching and perceptual objectives for Pyramid-JiT training."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def noise_gate(t: Tensor, threshold: float = 0.3) -> Tensor:
    """Return PixelGen's binary gate, active only on sufficiently clean inputs."""

    return (torch.as_tensor(t) >= threshold).to(dtype=torch.float32)


def _batch_timesteps(
    t: Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if t is None:
        return torch.rand(batch_size, device=device, dtype=dtype)
    timesteps = torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
    if timesteps.numel() == 1 and batch_size > 1:
        timesteps = timesteps.expand(batch_size)
    if timesteps.numel() != batch_size:
        raise ValueError("t must have one value per image")
    return timesteps.clamp(0.0, 1.0)


def _validate_images(prediction: Tensor, target: Tensor) -> None:
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("perceptual losses expect images with shape [B, C, H, W]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[1] != target.shape[1]:
        raise ValueError("prediction and target must have matching batch and channel dimensions")


def _fallback_patch_features(images: Tensor, feature_dim: int = 384) -> Tensor:
    """Create deterministic patch features when optional pretrained weights are unavailable."""

    pooled = F.adaptive_avg_pool2d(images.float(), (16, 16))
    patches = pooled.flatten(2).transpose(1, 2)
    features = torch.cat((patches, patches.square(), patches.sin(), patches.cos()), dim=-1)
    repeats = math.ceil(feature_dim / features.shape[-1])
    return features.repeat(1, 1, repeats)[..., :feature_dim]


class LPIPSLoss(nn.Module):
    """Lazy LPIPS(VGG) wrapper with a differentiable offline fallback."""

    def __init__(self, net: str = "vgg", use_pretrained: bool = False) -> None:
        super().__init__()
        self.net = net
        self.use_pretrained = use_pretrained
        self._lpips: nn.Module | None = None
        self._load_failed = False

    def _load(self, device: torch.device) -> nn.Module | None:
        if not self.use_pretrained or self._load_failed:
            return None
        if self._lpips is not None:
            return self._lpips.to(device)
        try:
            import lpips

            self._lpips = lpips.LPIPS(net=self.net).eval().requires_grad_(False).to(device)
        except Exception as error:  # pragma: no cover - depends on local cache/network
            self._load_failed = True
            warnings.warn(
                f"LPIPS could not be loaded ({error!r}); using local patch loss instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return self._lpips

    def per_sample(self, prediction: Tensor, target: Tensor) -> Tensor:
        _validate_images(prediction, target)
        model = self._load(prediction.device)
        if model is None:
            difference = (prediction.float() - target.float()).abs()
            values = difference.flatten(1).mean(dim=1)
            if min(prediction.shape[-2:]) >= 4:
                pooled_prediction = F.avg_pool2d(prediction.float(), kernel_size=4, stride=4)
                pooled_target = F.avg_pool2d(target.float(), kernel_size=4, stride=4)
                values = 0.5 * values + 0.5 * (pooled_prediction - pooled_target).abs().flatten(1).mean(dim=1)
            return values
        values = model(prediction.float(), target.float())
        return values.reshape(values.shape[0], -1).mean(dim=1)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.per_sample(prediction, target).mean()


class DINOPerceptualLoss(nn.Module):
    """DINOv2 patch-token cosine loss with lazy loading and offline fallback."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        feature_dim: int = 384,
        layer: int = 11,
        use_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.layer = layer
        self.use_pretrained = use_pretrained
        self._dino: nn.Module | None = None
        self._load_failed = False

    def _load(self, device: torch.device) -> nn.Module | None:
        if not self.use_pretrained or self._load_failed:
            return None
        if self._dino is not None:
            return self._dino.to(device)
        try:
            self._dino = (
                torch.hub.load("facebookresearch/dinov2", self.model_name)
                .eval()
                .requires_grad_(False)
                .to(device)
            )
        except Exception as error:  # pragma: no cover - depends on local cache/network
            self._load_failed = True
            warnings.warn(
                f"DINOv2 could not be loaded ({error!r}); using local patch features instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return self._dino

    @staticmethod
    def _prepare(images: Tensor) -> Tensor:
        images = images.float()
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = images.add(1.0).div(2.0)
        mean = images.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
        return (images - mean) / std

    @staticmethod
    def _patch_tokens(output: Any) -> Tensor | None:
        if isinstance(output, (tuple, list)):
            for value in reversed(output):
                tokens = DINOPerceptualLoss._patch_tokens(value)
                if tokens is not None:
                    return tokens
            return None
        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "x_prenorm", "patch_tokens"):
                value = output.get(key)
                if isinstance(value, Tensor) and value.ndim == 3:
                    return value
            return None
        if isinstance(output, Tensor) and output.ndim == 3:
            return output
        return None

    def features(self, images: Tensor) -> Tensor:
        model = self._load(images.device)
        if model is None:
            return _fallback_patch_features(images, self.feature_dim)
        prepared = self._prepare(images)
        if hasattr(model, "get_intermediate_layers"):
            try:
                output = model.get_intermediate_layers(
                    prepared,
                    n=[self.layer],
                    reshape=False,
                    return_class_token=False,
                )
            except (TypeError, ValueError):
                output = model.get_intermediate_layers(
                    prepared,
                    n=self.layer + 1,
                    reshape=False,
                    return_class_token=False,
                )
        else:
            output = model.forward_features(prepared)
        patch_tokens = self._patch_tokens(output)
        if patch_tokens is None or patch_tokens.shape[-1] != self.feature_dim:
            raise RuntimeError("DINOv2 did not return the expected patch-token features")
        return patch_tokens

    @staticmethod
    def cosine_distance(features_a: Tensor, features_b: Tensor) -> Tensor:
        if features_a.shape != features_b.shape:
            raise ValueError("DINO feature tensors must have matching shapes")
        return 1.0 - F.cosine_similarity(features_a.float(), features_b.float(), dim=-1).mean(dim=-1)

    def per_sample(self, prediction: Tensor, target: Tensor) -> Tensor:
        _validate_images(prediction, target)
        prediction_features = self.features(prediction)
        with torch.no_grad():
            target_features = self.features(target)
        return self.cosine_distance(prediction_features, target_features)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.per_sample(prediction, target).mean()


class REPALoss(nn.Module):
    """Cosine alignment between intermediate DiT tokens and clean-image tokens."""

    def per_sample(self, repa_tokens: Tensor, target_tokens: Tensor) -> Tensor:
        if repa_tokens.ndim != 3 or target_tokens.ndim != 3:
            raise ValueError("REPA inputs must have shape [B, N, D]")
        if repa_tokens.shape != target_tokens.shape:
            raise ValueError("REPA and DINO tokens must have matching shapes")
        return 1.0 - F.cosine_similarity(
            repa_tokens.float(), target_tokens.float(), dim=-1
        ).mean(dim=-1)

    def forward(self, repa_tokens: Tensor, target_tokens: Tensor) -> Tensor:
        return self.per_sample(repa_tokens, target_tokens).mean()


class PerceptualLosses(nn.Module):
    """Container for LPIPS, P-DINO, REPA, and PixelGen noise gating."""

    def __init__(
        self,
        enabled: bool = True,
        use_pretrained: bool = False,
        lpips_net: str = "vgg",
        dino_model_name: str = "dinov2_vits14",
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.lpips = LPIPSLoss(net=lpips_net, use_pretrained=use_pretrained)
        self.p_dino = DINOPerceptualLoss(
            model_name=dino_model_name,
            use_pretrained=use_pretrained,
        )
        self.repa = REPALoss()

    def per_sample(
        self,
        prediction: Tensor,
        target: Tensor,
        repa_tokens: Tensor | None,
        t: Tensor,
    ) -> dict[str, Tensor]:
        zeros = prediction.new_zeros(prediction.shape[0])
        gate = noise_gate(t).to(device=prediction.device, dtype=prediction.dtype)
        if not self.enabled:
            return {"lpips": zeros, "p_dino": zeros, "repa": zeros, "gate": gate}

        lpips_values = self.lpips.per_sample(prediction, target).to(dtype=prediction.dtype)
        prediction_tokens = self.p_dino.features(prediction)
        with torch.no_grad():
            target_tokens = self.p_dino.features(target)
        p_dino_values = self.p_dino.cosine_distance(prediction_tokens, target_tokens).to(
            dtype=prediction.dtype
        )
        if repa_tokens is None:
            repa_values = zeros
        else:
            repa_values = self.repa.per_sample(repa_tokens, target_tokens).to(
                dtype=prediction.dtype
            )
        return {
            "lpips": lpips_values * gate,
            "p_dino": p_dino_values * gate,
            "repa": repa_values * gate,
            "gate": gate,
        }

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        repa_tokens: Tensor | None = None,
        t: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if t is None:
            t = torch.ones(prediction.shape[0], device=prediction.device, dtype=prediction.dtype)
        values = self.per_sample(prediction, target, repa_tokens, t)
        weighted = 0.1 * values["lpips"] + 0.01 * values["p_dino"] + 0.1 * values["repa"]
        return weighted.mean(), values


def _model_outputs(
    model: nn.Module,
    x_t: Tensor,
    timesteps: Tensor,
    y: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    try:
        outputs = model(x_t, timesteps, y, return_repa=True)
    except TypeError as error:
        if "return_repa" not in str(error):
            raise
        outputs = model(x_t, timesteps, y)
    if isinstance(outputs, (tuple, list)):
        if len(outputs) != 2:
            raise ValueError("model return_repa output must contain two tensors")
        return outputs[0], outputs[1]
    return outputs, None


def training_loss(
    model: nn.Module,
    x1: Tensor,
    y: Tensor | None = None,
    t: Tensor | None = None,
    eps: Tensor | None = None,
    perceptual_losses: PerceptualLosses | None = None,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Train x-prediction with flow matching and gated perceptual supervision."""

    if x1.ndim != 4:
        raise ValueError("x1 must have shape [B, C, H, W]")
    batch_size = x1.shape[0]
    timesteps = _batch_timesteps(t, batch_size, x1.device, x1.dtype)
    noise = torch.randn_like(x1) if eps is None else eps.to(device=x1.device, dtype=x1.dtype)
    if noise.shape != x1.shape:
        raise ValueError("eps must have the same shape as x1")
    time_view = timesteps.reshape(batch_size, 1, 1, 1)
    x_t = time_view * x1 + (1.0 - time_view) * noise
    x_hat, repa_tokens = _model_outputs(model, x_t, timesteps, y)
    if x_hat.shape != x1.shape:
        raise ValueError("model final prediction must have the same shape as x1")

    denominator = (1.0 - time_view).clamp_min(0.05)
    velocity_pred = (x_hat - x_t) / denominator
    velocity_target = x1 - noise
    flow = torch.mean((velocity_pred - velocity_target) ** 2)

    perceptual_losses = perceptual_losses or PerceptualLosses(enabled=True, use_pretrained=False)
    perceptual_values = perceptual_losses.per_sample(x_hat, x1, repa_tokens, timesteps)
    lpips_term = perceptual_values["lpips"].mean()
    p_dino_term = perceptual_values["p_dino"].mean()
    repa_term = perceptual_values["repa"].mean()
    total = flow + 0.1 * lpips_term + 0.01 * p_dino_term + 0.1 * repa_term

    components = {
        "flow": flow.detach(),
        "lpips": lpips_term.detach(),
        "p_dino": p_dino_term.detach(),
        "repa": repa_term.detach(),
        "perceptual_gate": perceptual_values["gate"].mean().detach(),
        "total": total.detach(),
    }
    if return_components:
        return total, components
    return total


def combined_training_loss(*args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Named alias for callers that use the multi-objective loss terminology."""

    return training_loss(*args, **kwargs)


noise_gating = noise_gate


__all__ = [
    "DINOPerceptualLoss",
    "LPIPSLoss",
    "PerceptualLosses",
    "REPALoss",
    "combined_training_loss",
    "noise_gate",
    "noise_gating",
    "training_loss",
]
