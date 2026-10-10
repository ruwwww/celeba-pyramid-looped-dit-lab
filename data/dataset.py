"""CelebA-HQ RGB dataset loader with DINOv2 cluster pseudo-labels."""

from __future__ import annotations

import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CelebAHQPixelDataset(Dataset):
    """Loads 256x256 RGB images and matches them with clustered DINOv2 pseudo-labels."""

    def __init__(
        self,
        image_dir: str | Path = "/mnt/data/celeba_hq/data",
        label_cache_path: str | Path = "/mnt/data/Coding3/celeba-dit-lab/cached_data/celeba_train_latents_clustered.pt",
        split: str = "train",
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No JPG images found in {image_dir}")

        # Split 28000 train, 2000 val
        if split == "train":
            self.image_paths = self.image_paths[:28000]
        else:
            self.image_paths = self.image_paths[28000:30000]

        # Load clustered labels if available
        label_cache = Path(label_cache_path)
        if label_cache.exists():
            data = torch.load(label_cache, weights_only=True)
            self.labels = data["labels"].long()
            if split == "val" and len(self.labels) > 28000:
                self.labels = self.labels[28000:30000]
            elif split == "train":
                self.labels = self.labels[: len(self.image_paths)]
        else:
            self.labels = torch.zeros(len(self.image_paths), dtype=torch.long)

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Map to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(img)
        label = self.labels[idx] if idx < len(self.labels) else torch.tensor(0, dtype=torch.long)
        return tensor, label
