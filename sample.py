"""Generate pixel-space RGB PNGs from a PyramidLoopedDiT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image

from models import PyramidLoopedDiT, euler_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("samples"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use-raw-model",
        action="store_true",
        help="Use raw training weights instead of EMA weights when both are present",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_parts(checkpoint: Any, use_raw_model: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return {}, checkpoint
    config = checkpoint.get("model_config", {})
    if use_raw_model:
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint.get("ema", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    return dict(config), state


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
    use_raw_model: bool = False,
) -> PyramidLoopedDiT:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config, state = _checkpoint_parts(checkpoint, use_raw_model)
    saved_config.setdefault("num_classes", num_classes)
    model = PyramidLoopedDiT(**saved_config).to(device)
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dict")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def generate(args: argparse.Namespace) -> list[Path]:
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("num-samples and batch-size must be positive")
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device, args.num_classes, args.use_raw_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    saved_paths: list[Path] = []

    for start in range(0, args.num_samples, args.batch_size):
        count = min(args.batch_size, args.num_samples - start)
        labels = None
        if args.class_label is not None:
            labels = torch.full((count,), args.class_label, device=device, dtype=torch.long)
        images = euler_sample(
            model,
            num_samples=count,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            y=labels,
            device=device,
            image_size=model.image_size,
            channels=model.in_channels,
            generator=generator,
        )
        for offset, image in enumerate(images):
            path = args.output_dir / f"sample_{start + offset:04d}.png"
            save_image(image.add(1.0).div(2.0).clamp(0.0, 1.0), path)
            saved_paths.append(path)
    return saved_paths


def main() -> None:
    paths = generate(parse_args())
    print(f"Saved {len(paths)} images to {paths[0].parent}")


if __name__ == "__main__":
    main()

