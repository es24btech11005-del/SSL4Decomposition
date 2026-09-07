from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SuperpositionPatchDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Return a summed patch and its two constituent patches.

    Each sample chooses one image from each requested channel, crops a random
    square from each image at an independent location, applies independent
    flips and 90-degree rotations, and sums the resulting patches.
    """

    def __init__(
        self,
        split_root: str | Path,
        channel_names: tuple[str, str] = ("T1", "T2"),
        crop_size: int = 256,
        samples_per_epoch: int = 1000,
        augment: bool = True,
        seed: int | None = None,
    ) -> None:
        if len(channel_names) != 2:
            raise ValueError("channel_names must contain exactly two channel names")
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")

        self.split_root = Path(split_root)
        self.channel_names = channel_names
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.seed = seed
        self._paths = tuple(self._find_images(channel) for channel in channel_names)
        if not all(self._paths):
            raise FileNotFoundError(
                f"Could not find image TIFFs for {channel_names} under {self.split_root}"
            )

        self._frames = tuple(self._load_frame(path) for path in self._paths[0] + self._paths[1])
        too_small = [path for path, frame in zip(self._paths[0] + self._paths[1], self._frames) if min(frame.shape) < crop_size]
        if too_small:
            raise ValueError(f"crop_size={crop_size} is too large for: {too_small}")

    def _find_images(self, channel: str) -> tuple[Path, ...]:
        paths = tuple(sorted(self.split_root.joinpath(channel).glob("*_image.tif")))
        if not paths:
            paths = tuple(sorted(self.split_root.glob(f"*/{channel}/*_image.tif")))
        return paths

    @staticmethod
    def _load_frame(path: Path) -> np.ndarray:
        frame = np.asarray(tifffile.imread(path), dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError(f"Expected a 2D TIFF frame, got {frame.shape} from {path}")
        minimum = float(frame.min())
        maximum = float(frame.max())
        if maximum > minimum:
            frame = (frame - minimum) / (maximum - minimum)
        else:
            frame.fill(0.0)
        return frame

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, index: int) -> random.Random:
        if self.seed is None:
            return random.Random()
        return random.Random(self.seed + index)

    def _make_patch(self, frame: np.ndarray, rng: random.Random) -> Tensor:
        height, width = frame.shape
        top = rng.randint(0, height - self.crop_size)
        left = rng.randint(0, width - self.crop_size)
        patch = torch.from_numpy(frame[top : top + self.crop_size, left : left + self.crop_size].copy())

        if self.augment:
            if rng.random() < 0.5:
                patch = torch.flip(patch, dims=(0,))
            if rng.random() < 0.5:
                patch = torch.flip(patch, dims=(1,))
            patch = torch.rot90(patch, k=rng.randrange(4), dims=(0, 1))
        return patch

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rng = self._rng(index)
        first_frame = self._frames[rng.randrange(len(self._paths[0]))]
        second_frame = self._frames[len(self._paths[0]) + rng.randrange(len(self._paths[1]))]
        first_patch = self._make_patch(first_frame, rng)
        second_patch = self._make_patch(second_frame, rng)
        superimposed = (first_patch + second_patch).unsqueeze(0)
        targets = torch.stack((first_patch, second_patch), dim=0)
        return superimposed, targets[0:1], targets[1:2]