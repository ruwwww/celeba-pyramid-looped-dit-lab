"""Behavioral tests for the Pyramid-JiT Looped-DiT model and flow helpers."""

from __future__ import annotations

import pytest
import torch

from models import PyramidLoopedDiT, euler_sample, training_loss


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
    assert prediction.min() >= -1.0
    assert prediction.max() <= 1.0


def test_training_loss_backward_is_finite(model: PyramidLoopedDiT) -> None:
    model.train()
    x1 = torch.randn(1, 3, 256, 256, device=DEVICE)
    y = torch.tensor([2], device=DEVICE)

    loss = training_loss(model, x1, y=y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    model.zero_grad(set_to_none=True)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


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
