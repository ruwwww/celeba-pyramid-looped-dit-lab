"""Behavioral tests for the Pyramid-JiT Looped-DiT model and flow helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from train import ModelEMA
from models import PerceptualLosses, PyramidLoopedDiT, euler_sample, noise_gate, training_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model() -> PyramidLoopedDiT:
    torch.manual_seed(0)
    return PyramidLoopedDiT().to(DEVICE)


def test_forward_preserves_pixel_shape(model: PyramidLoopedDiT) -> None:
    model.eval()
    x = torch.randn(2, 3, 256, 256, device=DEVICE)
    t = torch.full((2,), 0.5, device=DEVICE)
    y = torch.tensor([0, 1], device=DEVICE)

    with torch.no_grad():
        prediction = model(x, t, y)

    assert prediction.shape == x.shape
    assert torch.isfinite(prediction).all()


def test_forward_returns_repa_tokens(model: PyramidLoopedDiT) -> None:
    model.eval()
    x = torch.randn(2, 3, 256, 256, device=DEVICE)
    t = torch.full((2,), 0.5, device=DEVICE)
    y = torch.tensor([0, 1], device=DEVICE)

    with torch.no_grad():
        prediction, repa_tokens = model(x, t, y, return_repa=True)

    assert prediction.shape == (2, 3, 256, 256)
    assert repa_tokens.shape == (2, 256, 384)
    assert torch.isfinite(repa_tokens).all()


def test_noise_gate_activates_at_threshold() -> None:
    timesteps = torch.tensor([0.0, 0.299, 0.3, 1.0])
    assert torch.equal(noise_gate(timesteps), torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_perceptual_loss_gates_all_terms_before_threshold() -> None:
    prediction = torch.zeros(1, 3, 16, 16)
    target = torch.ones_like(prediction)
    repa_tokens = torch.ones(1, 256, 384)
    losses = PerceptualLosses(enabled=True, use_pretrained=False)

    gated = losses.per_sample(prediction, target, repa_tokens, torch.tensor([0.299]))
    active = losses.per_sample(prediction, target, repa_tokens, torch.tensor([0.3]))

    assert all(
        torch.equal(gated[name], torch.zeros_like(gated[name]))
        for name in ("lpips", "p_dino", "repa")
    )
    assert sum(active[name].item() for name in ("lpips", "p_dino", "repa")) > 0.0


def test_training_loss_matches_jit_x_prediction_formula() -> None:
    class ZeroPredictor(nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
            return torch.zeros_like(x)

    clean = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noise = torch.tensor([[[[0.0]]], [[[1.0]]]])
    timesteps = torch.tensor([0.0, 0.95])
    x_t = timesteps.reshape(2, 1, 1, 1) * clean + (1.0 - timesteps.reshape(2, 1, 1, 1)) * noise
    denominator = (1.0 - timesteps).clamp_min(0.05).reshape(2, 1, 1, 1)
    expected = torch.mean(((torch.zeros_like(clean) - x_t) / denominator - (clean - noise)) ** 2)

    actual = training_loss(
        ZeroPredictor(),
        clean,
        t=timesteps,
        eps=noise,
        perceptual_losses=PerceptualLosses(enabled=False),
    )

    assert torch.equal(actual, expected)


def test_training_loss_backward_is_finite(model: PyramidLoopedDiT) -> None:
    model.train()
    x1 = torch.randn(1, 3, 256, 256, device=DEVICE)
    y = torch.tensor([2], device=DEVICE)

    loss, components = training_loss(
        model,
        x1,
        y=y,
        t=torch.tensor([0.8], device=DEVICE),
        perceptual_losses=PerceptualLosses(enabled=True, use_pretrained=False),
        return_components=True,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert {"flow", "lpips", "p_dino", "repa"} <= set(components)
    assert all(torch.isfinite(value) for value in components.values())

    model.zero_grad(set_to_none=True)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_ema_uses_warmup_decay_by_step() -> None:
    source = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(1.0)
    ema = ModelEMA(source, decay=0.9)

    with torch.no_grad():
        source.weight.fill_(2.0)
    ema.update(source, step=0)
    assert torch.allclose(ema.module.weight, torch.tensor([[1.9]]))

    with torch.no_grad():
        source.weight.fill_(3.0)
    ema.update(source, step=10)
    expected = 0.55 * 1.9 + 0.45 * 3.0
    assert torch.allclose(ema.module.weight, torch.tensor([[expected]]))


def test_five_step_euler_sampling_with_cfg_is_clamped(model: PyramidLoopedDiT) -> None:
    model.eval()
    labels = torch.tensor([3], device=DEVICE)

    samples = euler_sample(
        model,
        num_samples=1,
        steps=5,
        cfg_scale=1.5,
        y=labels,
        device=DEVICE,
    )

    assert samples.shape == (1, 3, 256, 256)
    assert torch.isfinite(samples).all()
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
