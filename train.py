"""Training CLI for the pixel-space Pyramid-JiT Looped-DiT."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from data.dataset import CelebAHQPixelDataset
from models import PyramidLoopedDiT, euler_sample, training_loss
from tracker import ExperimentTracker


class ModelEMA:
    """Exponential moving average of model weights used for sample previews."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            value = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(value.to(dtype=ema_value.dtype), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=str, default="/mnt/data/celeba_hq/data")
    parser.add_argument(
        "--label-cache-path",
        type=str,
        default="/mnt/data/Coding3/celeba-dit-lab/cached_data/celeba_train_latents_clustered.pt",
    )
    parser.add_argument("--output-dir", type=str, default="runs/pyramid-looped-dit")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means train for all requested epochs")
    parser.add_argument(
        "--screening-steps",
        type=int,
        default=0,
        help="Run only this many real data steps, then save a screening checkpoint and exit",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--num-classes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="celeba-flow-dit")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable live W&B while keeping JSONL logs")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Enable bfloat16 autocast")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable autocast")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "image_size": 256,
        "in_channels": 3,
        "hidden_size": 512,
        "num_heads": 8,
        "mlp_ratio": 6.0,
        "bottleneck_channels": 128,
        "patch_size": 16,
        "loop_split": (2, 4, 2),
        "loop_repeats": 4,
        "num_classes": args.num_classes,
        "class_dropout_prob": 0.15,
    }


def save_checkpoint(
    path: Path,
    model: PyramidLoopedDiT,
    ema: ModelEMA,
    optimizer: AdamW,
    step: int,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config(args),
            "train_config": vars(args),
        },
        path,
    )


def save_sample_grid(samples: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = samples.shape[0]
    nrow = max(1, min(4, int(count**0.5)))
    grid = make_grid(samples.clamp(-1.0, 1.0).add(1.0).div(2.0), nrow=nrow, padding=2)
    save_image(grid, path)


def autocast_context(device: torch.device, enabled: bool):
    can_bfloat16 = device.type in {"cuda", "cpu"}
    if not enabled or not can_bfloat16:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
    )


def train(args: argparse.Namespace) -> Path:
    if args.epochs < 1 or args.batch_size < 1 or args.num_classes < 1:
        raise ValueError("epochs, batch-size, and num-classes must be positive")
    if args.screening_steps < 0 or args.max_steps < 0:
        raise ValueError("step limits cannot be negative")
    if args.sample_count < 1 or args.sample_steps < 1:
        raise ValueError("sample-count and sample-steps must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device)
    config = {**vars(args), "device_resolved": str(device)}
    tracker = ExperimentTracker(
        args.output_dir,
        config=config,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
    )

    model = PyramidLoopedDiT(**model_config(args)).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = CelebAHQPixelDataset(
        image_dir=args.image_dir,
        label_cache_path=args.label_cache_path,
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    limit = args.screening_steps or args.max_steps
    global_step = 0
    last_checkpoint = tracker.checkpoints_dir / "latest.pt"
    try:
        model.train()
        for epoch in range(args.epochs):
            for images, labels in loader:
                if limit and global_step >= limit:
                    break
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp):
                    loss = training_loss(model, images, y=labels)
                loss.backward()
                if args.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                ema.update(model)
                global_step += 1

                if global_step % max(1, args.log_every) == 0 or global_step == 1:
                    tracker.log_metrics(
                        step=global_step,
                        epoch=epoch,
                        metrics={"loss": loss.detach().float().item(), "lr": optimizer.param_groups[0]["lr"]},
                    )

                if args.sample_every > 0 and global_step % args.sample_every == 0:
                    sample_labels = torch.arange(args.sample_count, device=device) % args.num_classes
                    samples = euler_sample(
                        ema.module,
                        num_samples=args.sample_count,
                        steps=args.sample_steps,
                        cfg_scale=args.cfg_scale,
                        y=sample_labels,
                        device=device,
                    )
                    sample_path = tracker.figures_dir / f"samples_{global_step:07d}.png"
                    save_sample_grid(samples.cpu(), sample_path)
                    tracker.log_image(global_step, "samples", sample_path)

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        tracker.checkpoints_dir / f"step_{global_step:07d}.pt",
                        model,
                        ema,
                        optimizer,
                        global_step,
                        epoch,
                        args,
                    )

            if limit and global_step >= limit:
                break

        save_checkpoint(last_checkpoint, model, ema, optimizer, global_step, epoch, args)
        tracker.plot_loss_curve()
        if args.screening_steps:
            print(f"Screening completed: {global_step} steps; checkpoint={last_checkpoint}")
        return last_checkpoint
    finally:
        tracker.finish()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
