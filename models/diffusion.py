"""Rectified-flow objectives and Euler sampling for x-predicting DiTs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .losses import combined_training_loss, training_loss


def _x_prediction_to_velocity(x_hat: Tensor, x_t: Tensor, timestep: Tensor) -> Tensor:
    denominator = (1.0 - timestep).clamp_min(0.05).reshape(-1, 1, 1, 1)
    return (x_hat - x_t) / denominator


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    num_samples: int,
    steps: int = 50,
    cfg_scale: float = 1.0,
    y: Tensor | None = None,
    device: torch.device | str | None = None,
    image_size: int = 256,
    channels: int = 3,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Integrate the x-prediction vector field from noise to clean pixels."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model_device = next(model.parameters()).device
    if model_device != device:
        model.to(device)

    labels: Tensor | None
    if y is None:
        labels = None
    else:
        labels = torch.as_tensor(y, device=device, dtype=torch.long).reshape(-1)
        if labels.numel() == 1 and num_samples > 1:
            labels = labels.expand(num_samples)
        if labels.numel() != num_samples:
            raise ValueError("class labels must have one value per sample")

    was_training = model.training
    model.eval()
    try:
        samples = torch.randn(
            num_samples,
            channels,
            image_size,
            image_size,
            device=device,
            generator=generator,
        )
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=samples.dtype)
        null_class = getattr(model, "null_class", None)
        null_labels = None
        if null_class is not None:
            null_labels = torch.full(
                (num_samples,), int(null_class), device=device, dtype=torch.long
            )

        for index in range(steps):
            timestep = timesteps[index].expand(num_samples)
            x_hat_cond = model(samples, timestep, labels)
            if cfg_scale != 1.0:
                x_hat_uncond = model(samples, timestep, null_labels)
                velocity_cond = _x_prediction_to_velocity(x_hat_cond, samples, timestep)
                velocity_uncond = _x_prediction_to_velocity(x_hat_uncond, samples, timestep)
                velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            else:
                velocity = _x_prediction_to_velocity(x_hat_cond, samples, timestep)
            samples = samples + (timesteps[index + 1] - timesteps[index]) * velocity
        return samples.clamp(-1.0, 1.0)
    finally:
        model.train(was_training)


__all__ = ["combined_training_loss", "euler_sample", "training_loss"]
