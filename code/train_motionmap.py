#!/usr/bin/env python3
"""Train and run a compact, from-scratch dual-view IR motion-map classifier.

The model consumes two clip summaries:

* cached full-frame IR at 120x160;
* cached 224x224 person IR ROI, with a full-frame fallback when ROI is absent.

Each view has five channels: temporal mean, temporal standard deviation,
directional dynamic-rank image, positive motion, and negative motion.  The
full-frame and ROI maps are encoded by separate compact 2D CNN stems and fused
with a two-bit view-availability mask for 40-way classification.  No external
or pretrained weights are used.

Subject folds always come from research/artifacts/cv_folds.json.  Training uses
ordinary shuffled sampling (the natural train prior) and a fixed epoch count.
The outer subject fold is evaluated exactly once after training; it is never
used for checkpoint selection or early stopping.

Typical commands
----------------
Quick data/model/backward check, without writing files:

    python3 code/train_motionmap.py smoke --fold 0
    python3 code/train_motionmap.py train --tag ir_motionmap --dry-run

Two-fold screen:

    python3 code/train_motionmap.py train --tag ir_motionmap --folds 0,2

Complete the other folds, then assemble canonical OOF and test probabilities:

    python3 code/train_motionmap.py train --tag ir_motionmap --folds 1,3
    python3 code/train_motionmap.py oof --tag ir_motionmap --folds 0,1,2,3
    python3 code/train_motionmap.py test --tag ir_motionmap --folds 0,1,2,3

Partial-fold artifacts receive a suffix such as ``_f02``.  Four-fold artifacts
use the repository convention ``oof_<tag>.npz`` and
``testprobs_<tag>.npz``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache"
DEFAULT_FOLD_FILE = ROOT / "research" / "artifacts" / "cv_folds.json"
DEFAULT_ARTIFACT_DIR = ROOT / "research" / "artifacts"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints"

NUM_CLASSES = 40
FULL_SIZE = (120, 160)
MAP_CHANNELS = ("mean", "std", "dynamic_rank", "motion_positive", "motion_negative")
SUMMARY_VERSION = "mean_std_rank_posneg_center_cache_v2"
CHECKPOINT_KIND = "dual_ir_motionmap"
CHECKPOINT_VERSION = 1
MAX_PARAMETERS = 5_000_000


@dataclass(frozen=True)
class DataConfig:
    """Preprocessing fields stored in every checkpoint."""

    frames: int = 16
    full_height: int = FULL_SIZE[0]
    full_width: int = FULL_SIZE[1]
    roi_height: int = 224
    roi_width: int = 224
    roi_cache_suffix: str = "_roi224"
    summary_version: str = SUMMARY_VERSION

    @property
    def full_size(self) -> tuple[int, int]:
        return self.full_height, self.full_width

    @property
    def roi_size(self) -> tuple[int, int]:
        return self.roi_height, self.roi_width


@dataclass(frozen=True)
class ModelConfig:
    """Architecture fields stored in every checkpoint."""

    in_channels: int = len(MAP_CHANNELS)
    widths: tuple[int, ...] = (32, 48, 80, 128)
    head_width: int = 256
    dropout: float = 0.25
    availability_features: int = 2
    num_classes: int = NUM_CLASSES


def read_metadata(cache_dir: Path, split: str) -> dict[str, dict[str, str]]:
    path = cache_dir / f"meta_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    with path.open(newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"metadata is empty: {path}")
    return rows


def read_fold_users(fold_file: Path) -> dict[int, set[str]]:
    if not fold_file.is_file():
        raise FileNotFoundError(f"fold definition not found: {fold_file}")
    with fold_file.open() as handle:
        payload = json.load(handle)
    result: dict[int, set[str]] = {}
    for item in payload.get("folds", []):
        fold = int(item["fold"])
        users = {str(user) for user in item["val_users"]}
        if fold in result:
            raise ValueError(f"duplicate fold {fold} in {fold_file}")
        if not users:
            raise ValueError(f"fold {fold} has no validation users")
        result[fold] = users
    if not result:
        raise ValueError(f"no folds found in {fold_file}")
    return result


def parse_folds(text: str, fold_file: Path) -> list[int]:
    available = read_fold_users(fold_file)
    try:
        folds = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid --folds value {text!r}") from exc
    if not folds:
        raise ValueError("--folds must contain at least one fold")
    if len(folds) != len(set(folds)):
        raise ValueError(f"--folds contains duplicates: {text}")
    unknown = sorted(set(folds) - set(available))
    if unknown:
        raise ValueError(
            f"unknown fold(s) {unknown}; available folds are {sorted(available)}"
        )
    return folds


def validate_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(
            "--tag must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return tag


def fold_suffix(folds: Sequence[int], fold_file: Path) -> str:
    all_folds = sorted(read_fold_users(fold_file))
    return "" if sorted(folds) == all_folds else "_f" + "".join(map(str, folds))


def _uniform_positions(frames: int, training: bool, rng: np.random.Generator) -> np.ndarray:
    """Return shared normalized temporal positions for the full and ROI clips."""

    edges = np.linspace(0.0, 1.0, frames + 1, dtype=np.float32)
    if training:
        positions = edges[:-1] + rng.random(frames).astype(np.float32) * np.diff(edges)
        # Mild common temporal crop.  This simulates trimming while preserving
        # full/ROI temporal alignment.
        if rng.random() < 0.5:
            span = float(rng.uniform(0.65, 1.0))
            start = float(rng.uniform(0.0, 1.0 - span))
            positions = start + span * positions
    else:
        positions = (edges[:-1] + edges[1:]) * 0.5
    return np.clip(positions, 0.0, np.nextafter(np.float32(1.0), np.float32(0.0)))


def _sample_clip(frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Sample T,H,W uint8 frames using normalized positions."""

    if frames.ndim != 3:
        raise ValueError(f"expected IR array shaped (T,H,W), got {frames.shape}")
    if len(frames) == 0:
        return np.zeros((len(positions), *frames.shape[1:]), dtype=np.float32)
    indices = np.minimum(
        (positions * len(frames)).astype(np.int64), len(frames) - 1
    )
    return frames[indices].astype(np.float32) / 255.0


def make_motion_map(clip: np.ndarray) -> torch.Tensor:
    """Convert sampled T,H,W IR frames into five direction-aware 2D maps."""

    if clip.ndim != 3 or len(clip) == 0:
        raise ValueError(f"expected non-empty (T,H,W) clip, got {clip.shape}")
    count = len(clip)
    mean = clip.mean(axis=0)
    std = clip.std(axis=0)

    # Approximate rank pooling.  Early frames receive negative weight and late
    # frames positive weight, so reversing a clip reverses this channel.
    weights = np.arange(count, dtype=np.float32) - (count - 1.0) * 0.5
    denom = float(np.abs(weights).sum())
    if denom > 0.0:
        dynamic_rank = np.tensordot(weights / denom, clip, axes=(0, 0))
    else:
        dynamic_rank = np.zeros_like(mean)

    if count > 1:
        delta = np.diff(clip, axis=0)
        motion_positive = np.maximum(delta, 0.0).mean(axis=0)
        motion_negative = np.maximum(-delta, 0.0).mean(axis=0)
    else:
        motion_positive = np.zeros_like(mean)
        motion_negative = np.zeros_like(mean)

    # Fixed, label-free scaling keeps all channels in [0,1].  GroupNorm in the
    # network handles the remaining per-domain scale difference.
    maps = np.stack(
        (
            mean,
            np.clip(std * 4.0, 0.0, 1.0),
            np.clip(0.5 + 0.5 * dynamic_rank, 0.0, 1.0),
            np.clip(motion_positive * 4.0, 0.0, 1.0),
            np.clip(motion_negative * 4.0, 0.0, 1.0),
        )
    ).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(maps))


def _resize_maps(maps: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(maps.shape[-2:]) == tuple(size):
        return maps.contiguous()
    return F.interpolate(
        maps.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def _random_resized_crop(
    maps: torch.Tensor, rng: np.random.Generator, min_scale: float = 0.85
) -> torch.Tensor:
    if rng.random() >= 0.5:
        return maps
    height, width = maps.shape[-2:]
    scale = float(rng.uniform(min_scale, 1.0))
    crop_h = max(2, int(round(height * scale)))
    crop_w = max(2, int(round(width * scale)))
    top = int(rng.integers(0, height - crop_h + 1))
    left = int(rng.integers(0, width - crop_w + 1))
    crop = maps[:, top : top + crop_h, left : left + crop_w]
    return F.interpolate(
        crop.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def _random_erase(maps: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    if rng.random() >= 0.2:
        return maps
    height, width = maps.shape[-2:]
    erase_h = max(1, int(height * float(rng.uniform(0.10, 0.25))))
    erase_w = max(1, int(width * float(rng.uniform(0.10, 0.25))))
    top = int(rng.integers(0, height - erase_h + 1))
    left = int(rng.integers(0, width - erase_w + 1))
    maps = maps.clone()
    maps[:, top : top + erase_h, left : left + erase_w] = 0.0
    # Zero means strong negative direction in the rank channel; 0.5 is its
    # no-motion baseline and therefore the correct erased value.
    rank_channel = MAP_CHANNELS.index("dynamic_rank")
    maps[rank_channel, top : top + erase_h, left : left + erase_w] = 0.5
    return maps


class MotionMapDataset(Dataset):
    """Read cached IR clips and produce aligned full/ROI clip summaries."""

    def __init__(
        self,
        cache_dir: Path,
        fold_file: Path,
        split: str,
        data_config: DataConfig,
        fold: int | None = None,
        part: str = "train",
        augment: bool = False,
        seed: int = 0,
        ids: Sequence[str] | None = None,
        write_summary_cache: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be train or test, got {split!r}")
        if part not in {"train", "val", "all"}:
            raise ValueError(f"part must be train, val or all, got {part!r}")
        if data_config.summary_version != SUMMARY_VERSION:
            raise ValueError(
                f"unsupported summary version {data_config.summary_version!r}; "
                f"this code supports {SUMMARY_VERSION!r}"
            )
        if data_config.frames < 2:
            raise ValueError("at least two sampled frames are required for motion maps")

        self.cache_dir = Path(cache_dir)
        self.fold_file = Path(fold_file)
        self.split = split
        self.data_config = data_config
        self.augment = bool(augment)
        self.write_summary_cache = bool(write_summary_cache)
        self.seed = int(seed)
        self.epoch = 0
        self.meta = read_metadata(self.cache_dir, split)

        if ids is not None:
            missing = sorted(set(ids) - set(self.meta))
            if missing:
                raise KeyError(f"{len(missing)} requested IDs are absent from metadata")
            selected = list(ids)
        elif split == "test" or part == "all":
            selected = sorted(self.meta)
        else:
            if fold is None:
                raise ValueError("a fold is required for train/validation partitions")
            fold_users = read_fold_users(self.fold_file)
            if fold not in fold_users:
                raise ValueError(f"unknown fold {fold}; available: {sorted(fold_users)}")
            validation_users = fold_users[fold]
            if part == "val":
                selected = sorted(
                    sid
                    for sid, row in self.meta.items()
                    if row["user"] in validation_users
                )
            else:
                selected = sorted(
                    sid
                    for sid, row in self.meta.items()
                    if row["user"] not in validation_users
                )

        self.ids = selected
        if not self.ids:
            raise ValueError(
                f"empty dataset for split={split}, fold={fold}, part={part}"
            )
        self.labels = (
            {sid: int(self.meta[sid]["class_id"]) for sid in self.ids}
            if split == "train"
            else None
        )

    def __len__(self) -> int:
        return len(self.ids)

    def _rng(self, sid: str) -> np.random.Generator:
        value = (
            zlib.crc32(sid.encode("utf-8"))
            + self.seed * 1_000_003
            + self.epoch * 97_409
        ) % (1 << 32)
        return np.random.default_rng(value)

    def _load_full(self, sid: str) -> np.ndarray | None:
        path = self.cache_dir / self.split / f"{sid}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"full-frame cache file missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if "ir" not in archive.files or len(archive["ir"]) == 0:
                return None
            clip = archive["ir"]
        if clip.ndim != 3:
            raise ValueError(f"invalid full IR shape in {path}: {clip.shape}")
        return clip

    def _load_roi(self, sid: str) -> np.ndarray | None:
        directory = self.split + self.data_config.roi_cache_suffix
        path = self.cache_dir / directory / f"{sid}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            if "ir_roi" not in archive.files or len(archive["ir_roi"]) == 0:
                return None
            clip = archive["ir_roi"]
        if clip.ndim != 3:
            raise ValueError(f"invalid ROI IR shape in {path}: {clip.shape}")
        return clip

    def _summary_cache_path(self, sid: str) -> Path:
        key = (
            f"{self.split}_motionmap_{self.data_config.summary_version}_"
            f"f{self.data_config.frames}_r{self.data_config.roi_height}_"
            f"{self.data_config.roi_cache_suffix.removeprefix('_')}"
        )
        return self.cache_dir / key / f"{sid}.npz"

    def _load_or_build_summary(
        self, sid: str
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        path = self._summary_cache_path(sid)
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                full_maps = torch.from_numpy(archive["full"].astype(np.float32))
                roi_maps = torch.from_numpy(archive["roi"].astype(np.float32))
                available = archive["available"].astype(np.float32)
            return full_maps, roi_maps, float(available[0]), float(available[1])

        # The expensive temporal summaries are deterministic center views and
        # are cached once. Map-space augmentation below remains epoch-varying.
        positions = _uniform_positions(
            self.data_config.frames, False, np.random.default_rng(0)
        )
        roi_raw = self._load_roi(sid)
        full_raw = self._load_full(sid)
        full_available = float(full_raw is not None)
        roi_available = float(roi_raw is not None)
        full_clip = (
            _sample_clip(full_raw, positions) if full_raw is not None else None
        )
        roi_clip = _sample_clip(roi_raw, positions) if roi_raw is not None else None
        if full_clip is None and roi_clip is None:
            full_clip = np.zeros(
                (self.data_config.frames, *self.data_config.full_size),
                dtype=np.float32,
            )
            roi_clip = np.zeros(
                (self.data_config.frames, *self.data_config.roi_size),
                dtype=np.float32,
            )
        elif roi_clip is None:
            roi_clip = full_clip
        elif full_clip is None:
            full_clip = roi_clip
        assert full_clip is not None and roi_clip is not None
        full_maps = _resize_maps(
            make_motion_map(full_clip), self.data_config.full_size
        )
        roi_maps = _resize_maps(make_motion_map(roi_clip), self.data_config.roi_size)

        if self.write_summary_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(
                temporary,
                full=full_maps.numpy().astype(np.float16),
                roi=roi_maps.numpy().astype(np.float16),
                available=np.asarray(
                    (full_available, roi_available), dtype=np.uint8
                ),
            )
            os.replace(temporary, path)
        return full_maps, roi_maps, full_available, roi_available

    @staticmethod
    def _photometric_maps(
        maps: torch.Tensor, gain: float, bias: float
    ) -> torch.Tensor:
        maps = maps.clone()
        maps[0] = (maps[0] * gain + bias).clamp(0.0, 1.0)
        maps[1] = (maps[1] * gain).clamp(0.0, 1.0)
        maps[2] = ((maps[2] - 0.5) * gain + 0.5).clamp(0.0, 1.0)
        maps[3:] = (maps[3:] * gain).clamp(0.0, 1.0)
        return maps

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
        sid = self.ids[index]
        rng = self._rng(sid)
        full_maps, roi_maps, full_available, roi_available = (
            self._load_or_build_summary(sid)
        )

        if self.augment and (full_available or roi_available):
            # The same affine brightness perturbation is applied analytically
            # to both cached summaries; dynamic-rank bias cancels by design.
            gain = float(rng.uniform(0.90, 1.10))
            bias = float(rng.uniform(-0.04, 0.04))
            full_maps = self._photometric_maps(full_maps, gain, bias)
            roi_maps = self._photometric_maps(roi_maps, gain, bias)
            full_maps = _random_resized_crop(full_maps, rng)
            roi_maps = _random_resized_crop(roi_maps, rng)
            if rng.random() < 0.5:
                full_maps = full_maps.flip(-1)
                roi_maps = roi_maps.flip(-1)
            full_maps = _random_erase(full_maps, rng)
            roi_maps = _random_erase(roi_maps, rng)

        label = self.labels[sid] if self.labels is not None else -1
        return (
            full_maps,
            roi_maps,
            torch.tensor(
                (full_available, roi_available), dtype=torch.float32
            ),
            label,
            sid,
        )


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.norm2(self.conv2(self.conv1(inputs))), inplace=True)


class MapStem(nn.Module):
    """Compact 2D encoder for one five-channel clip summary."""

    def __init__(self, in_channels: int, widths: Sequence[int]) -> None:
        super().__init__()
        if len(widths) != 4:
            raise ValueError("MapStem expects exactly four widths")
        w0, w1, w2, w3 = map(int, widths)
        self.features = nn.Sequential(
            ConvNormAct(in_channels, w0, kernel_size=5, stride=2),
            ConvNormAct(w0, w1, stride=2),
            ResidualBlock(w1),
            ConvNormAct(w1, w2, stride=2),
            ResidualBlock(w2),
            ConvNormAct(w2, w3, stride=2),
            ResidualBlock(w3),
        )
        self.out_channels = w3 * 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        mean_pool = F.adaptive_avg_pool2d(features, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(features, 1).flatten(1)
        return torch.cat((mean_pool, max_pool), dim=1)


class DualMotionMapCNN(nn.Module):
    """Separate full-frame and person-ROI stems followed by compact late fusion."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.full_stem = MapStem(config.in_channels, config.widths)
        self.roi_stem = MapStem(config.in_channels, config.widths)
        fused_width = (
            self.full_stem.out_channels
            + self.roi_stem.out_channels
            + config.availability_features
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_width),
            nn.Linear(fused_width, config.head_width),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_width, config.num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        full_maps: torch.Tensor,
        roi_maps: torch.Tensor,
        view_available: torch.Tensor,
    ) -> torch.Tensor:
        full_features = self.full_stem(full_maps)
        roi_features = self.roi_stem(roi_maps)
        availability = view_available.reshape(
            -1, self.config.availability_features
        ).to(full_features.dtype)
        return self.classifier(
            torch.cat((full_features, roi_features, availability), dim=1)
        )


def build_model(config: ModelConfig) -> DualMotionMapCNN:
    if config.in_channels != len(MAP_CHANNELS):
        raise ValueError(
            f"model expects {config.in_channels} map channels, but "
            f"{SUMMARY_VERSION} produces {len(MAP_CHANNELS)}"
        )
    if config.availability_features != 2:
        raise ValueError("the dataset produces exactly two view-availability features")
    if config.num_classes != NUM_CLASSES:
        raise ValueError(
            f"model has {config.num_classes} classes, expected {NUM_CLASSES}"
        )
    model = DualMotionMapCNN(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= MAX_PARAMETERS:
        raise ValueError(
            f"model has {parameters:,} parameters; must remain below "
            f"{MAX_PARAMETERS:,}"
        )
    return model


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (1 << 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]]:
    full_maps, roi_maps, view_available, labels, sids = batch
    return (
        full_maps.to(device, non_blocking=True),
        roi_maps.to(device, non_blocking=True),
        view_available.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        sids,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probability_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sids: list[str] = []
    for batch in loader:
        full_maps, roi_maps, view_available, labels, batch_sids = _move_batch(
            batch, device
        )
        logits = model(full_maps, roi_maps, view_available)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits encountered during evaluation")
        batch_probabilities = torch.softmax(logits, dim=1)
        if not torch.isfinite(batch_probabilities).all():
            raise RuntimeError("non-finite probabilities encountered during evaluation")
        row_sums = batch_probabilities.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            raise RuntimeError("evaluation probabilities do not sum to one")
        probability_parts.append(batch_probabilities.cpu().numpy())
        label_parts.append(labels.cpu().numpy())
        sids.extend(batch_sids)
    probabilities = np.concatenate(probability_parts).astype(np.float32, copy=False)
    validate_probabilities(probabilities, "evaluation")
    labels = np.concatenate(label_parts).astype(np.int64, copy=False)
    result: dict[str, Any] = {
        "probs": probabilities,
        "labels": labels,
        "sids": sids,
    }
    labelled = labels >= 0
    if labelled.any():
        predictions = probabilities[labelled].argmax(axis=1)
        targets = labels[labelled]
        result["micro"] = float((predictions == targets).mean())
        present_classes = np.unique(targets)
        result["macro"] = float(
            np.mean(
                [
                    (predictions[targets == class_id] == class_id).mean()
                    for class_id in present_classes
                ]
            )
        )
    return result


def validate_probabilities(probabilities: np.ndarray, context: str) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"{context} probabilities have shape {probabilities.shape}; "
            f"expected (N,{NUM_CLASSES})"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{context} probabilities contain non-finite values")
    if (probabilities < -1e-7).any():
        raise ValueError(f"{context} probabilities contain negative values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{context} probability rows do not sum to one")


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    scale = args.width / 32.0
    widths = tuple(max(8, int(round(value * scale / 8.0)) * 8) for value in (32, 48, 80, 128))
    head_width = max(64, int(round(args.head_width / 8.0)) * 8)
    return ModelConfig(widths=widths, head_width=head_width, dropout=args.dropout)


def data_config_from_args(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        frames=args.frames,
        roi_height=args.roi_size,
        roi_width=args.roi_size,
        roi_cache_suffix="_roi" if args.roi_size <= 112 else "_roi224",
    )


def checkpoint_path(checkpoint_dir: Path, tag: str, fold: int) -> Path:
    return checkpoint_dir / f"{tag}_f{fold}.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: ModelConfig,
    data_config: DataConfig,
    fold: int,
    epoch: int,
    metrics: dict[str, float],
    train_config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    package = {
        "kind": CHECKPOINT_KIND,
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": state,
        "model_config": asdict(model_config),
        "data_config": asdict(data_config),
        "fold": int(fold),
        "epoch": int(epoch),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "train_config": train_config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(package, temporary)
    os.replace(temporary, path)


def _safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        package = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        package = torch.load(path, map_location=device)
    if not isinstance(package, dict):
        raise ValueError(f"{path} is not a packaged motion-map checkpoint")
    return package


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[DualMotionMapCNN, ModelConfig, DataConfig, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    package = _safe_torch_load(path, device)
    if package.get("kind") != CHECKPOINT_KIND:
        raise ValueError(
            f"{path} has kind {package.get('kind')!r}, expected {CHECKPOINT_KIND!r}"
        )
    if int(package.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version in {path}: "
            f"{package.get('checkpoint_version')!r}"
        )
    try:
        model_config_dict = dict(package["model_config"])
        model_config_dict["widths"] = tuple(model_config_dict["widths"])
        model_config = ModelConfig(**model_config_dict)
        data_config = DataConfig(**package["data_config"])
        state_dict = package["state_dict"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed motion-map checkpoint: {path}") from exc
    model = build_model(model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_config, data_config, package


def train_one_fold(
    args: argparse.Namespace,
    fold: int,
    data_config: DataConfig,
    model_config: ModelConfig,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(args.seed + fold * 101)
    train_dataset = MotionMapDataset(
        cache_dir=args.cache_dir,
        fold_file=args.fold_file,
        split="train",
        fold=fold,
        part="train",
        data_config=data_config,
        augment=args.augment,
        seed=args.seed + fold * 101,
    )
    train_users = sorted({train_dataset.meta[sid]["user"] for sid in train_dataset.ids})
    validation_users = sorted(read_fold_users(args.fold_file)[fold])
    validation_user_set = set(validation_users)
    validation_count = sum(
        row["user"] in validation_user_set for row in train_dataset.meta.values()
    )
    if validation_count == 0:
        raise ValueError(f"fold {fold} has no held-out samples in train metadata")
    overlap = set(train_users) & set(validation_users)
    if overlap:
        raise RuntimeError(f"subject leakage in fold {fold}: {sorted(overlap)}")

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        True,
        device,
        args.seed + fold,
    )
    model = build_model(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.10,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    path = checkpoint_path(args.checkpoint_dir, args.tag, fold)
    started = time.time()

    print(
        f"fold {fold}: {len(train_dataset)} train / "
        f"{validation_count} held-out clips; "
        f"users {train_users} -> {validation_users}",
        flush=True,
    )
    print(
        f"fold {fold}: {parameter_count:,} parameters, "
        f"natural-prior shuffle, fixed {args.epochs}-epoch protocol; "
        "outer validation runs once",
        flush=True,
    )

    for epoch in range(args.epochs):
        train_dataset.epoch = epoch
        model.train()
        running_loss = 0.0
        correct = 0
        seen = 0
        for batch in train_loader:
            full_maps, roi_maps, view_available, labels, _ = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(full_maps, roi_maps, view_available)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite training loss in fold {fold}, epoch {epoch + 1}"
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = len(labels)
            running_loss += float(loss.detach()) * batch_size
            correct += int((logits.detach().argmax(1) == labels).sum())
            seen += batch_size

        train_loss = running_loss / max(seen, 1)
        train_accuracy = correct / max(seen, 1)
        print(
            f"fold {fold} epoch {epoch + 1:03d}/{args.epochs}: "
            f"loss {train_loss:.4f}, train {train_accuracy:.4f}",
            flush=True,
        )

    # Constructing this dataset after optimization makes the protocol obvious:
    # no outer-fold IR array was loaded while fitting the model.
    validation_dataset = MotionMapDataset(
        cache_dir=args.cache_dir,
        fold_file=args.fold_file,
        split="train",
        fold=fold,
        part="val",
        data_config=data_config,
        augment=False,
        seed=args.seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        max(0, args.workers // 2),
        False,
        device,
        args.seed,
    )
    # This is deliberately the first and only outer-fold evaluation performed
    # by training.  The fixed final epoch is saved irrespective of this score.
    outer_validation = evaluate(model, validation_loader, device)
    outer_metrics = {
        "micro": float(outer_validation["micro"]),
        "macro": float(outer_validation["macro"]),
    }
    save_checkpoint(
        path,
        model,
        model_config,
        data_config,
        fold,
        args.epochs,
        outer_metrics,
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "augment": args.augment,
            "seed": args.seed,
            "selection": "fixed_epochs_outer_evaluated_once",
            "sampling": "natural_prior_shuffle",
            "train_users": train_users,
            "validation_users": validation_users,
        },
    )
    print(
        f"fold {fold} outer result (one evaluation): "
        f"micro {outer_metrics['micro']:.4f}, "
        f"macro {outer_metrics['macro']:.4f}",
        flush=True,
    )

    # Verify the package through the public loader without touching outer data
    # a second time.
    loaded_model, loaded_model_config, loaded_data_config, package = load_checkpoint(
        path, device
    )
    if loaded_model_config != model_config or loaded_data_config != data_config:
        raise RuntimeError(f"checkpoint configuration round-trip failed for {path}")
    if int(package.get("epoch", -1)) != args.epochs:
        raise RuntimeError(f"checkpoint epoch round-trip failed for {path}")
    loaded_state = loaded_model.state_dict()
    for key, value in model.state_dict().items():
        if not torch.equal(value, loaded_state[key]):
            raise RuntimeError(f"checkpoint tensor round-trip failed for {path}: {key}")
    return {
        "fold": fold,
        "micro": outer_metrics["micro"],
        "macro": outer_metrics["macro"],
        "epoch": args.epochs,
        "probs": outer_validation["probs"],
        "labels": outer_validation["labels"],
        "sids": outer_validation["sids"],
        "parameter_count": parameter_count,
        "train_users": train_users,
        "validation_users": validation_users,
        "outer_validation_evaluations": 1,
        "checkpoint": str(path),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }


def save_oof(
    artifact_dir: Path,
    tag: str,
    folds: Sequence[int],
    fold_file: Path,
    results: Sequence[dict[str, Any]],
) -> Path:
    if len(results) != len(folds):
        raise ValueError("one OOF result is required per requested fold")
    probabilities = np.concatenate([result["probs"] for result in results])
    validate_probabilities(probabilities, "OOF")
    labels = np.concatenate([result["labels"] for result in results])
    sids = np.asarray(
        [sid for result in results for sid in result["sids"]], dtype=str
    )
    fold_indices = np.concatenate(
        [
            np.full(len(result["sids"]), int(result["fold"]), dtype=np.int8)
            for result in results
        ]
    )
    if len(sids) != len(set(sids.tolist())):
        raise RuntimeError("OOF partitions overlap; refusing to write duplicate sample IDs")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"oof_{tag}{fold_suffix(folds, fold_file)}.npz"
    np.savez_compressed(
        path,
        probs=probabilities.astype(np.float32, copy=False),
        labels=labels.astype(np.int64, copy=False),
        sids=sids,
        folds=fold_indices,
    )
    return path


def save_test_probabilities(
    artifact_dir: Path,
    tag: str,
    folds: Sequence[int],
    fold_file: Path,
    probabilities: np.ndarray,
    sids: Sequence[str],
) -> Path:
    validate_probabilities(probabilities, "test")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"testprobs_{tag}{fold_suffix(folds, fold_file)}.npz"
    np.savez_compressed(
        path,
        probs=probabilities.astype(np.float32, copy=False),
        sids=np.asarray(sids, dtype=str),
        folds=np.asarray(folds, dtype=np.int8),
    )
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(convert(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def checkpoint_configs(
    args: argparse.Namespace,
    folds: Sequence[int],
    device: torch.device,
) -> tuple[ModelConfig, DataConfig, list[dict[str, Any]]]:
    reference_model: ModelConfig | None = None
    reference_data: DataConfig | None = None
    packages: list[dict[str, Any]] = []
    current_fold_users = read_fold_users(args.fold_file)
    for fold in folds:
        path = checkpoint_path(args.checkpoint_dir, args.tag, fold)
        _, model_config, data_config, package = load_checkpoint(path, device)
        if int(package.get("fold", -1)) != fold:
            raise ValueError(
                f"{path} says it is fold {package.get('fold')}, expected fold {fold}"
            )
        checkpoint_users = package.get("train_config", {}).get("validation_users")
        if checkpoint_users is not None and sorted(checkpoint_users) != sorted(
            current_fold_users[fold]
        ):
            raise ValueError(
                f"fold {fold} users differ from the checkpoint: "
                f"{sorted(current_fold_users[fold])} != {sorted(checkpoint_users)}"
            )
        if reference_model is None:
            reference_model = model_config
            reference_data = data_config
        elif model_config != reference_model or data_config != reference_data:
            raise ValueError(
                "requested checkpoints use different model/data configurations"
            )
        packages.append(package)
    assert reference_model is not None and reference_data is not None
    return reference_model, reference_data, packages


def infer_oof_fold(
    args: argparse.Namespace,
    fold: int,
    data_config: DataConfig,
    device: torch.device,
) -> dict[str, Any]:
    model, _, loaded_data, package = load_checkpoint(
        checkpoint_path(args.checkpoint_dir, args.tag, fold), device
    )
    if loaded_data != data_config:
        raise ValueError(f"data configuration mismatch for fold {fold}")
    dataset = MotionMapDataset(
        cache_dir=args.cache_dir,
        fold_file=args.fold_file,
        split="train",
        fold=fold,
        part="val",
        data_config=data_config,
        augment=False,
        seed=0,
    )
    loader = make_loader(
        dataset, args.batch_size, args.workers, False, device, args.seed
    )
    result = evaluate(model, loader, device)
    result["fold"] = fold
    print(
        f"fold {fold}: micro {result['micro']:.4f}, "
        f"macro {result['macro']:.4f}, checkpoint epoch "
        f"{package.get('epoch')}",
        flush=True,
    )
    return result


def run_smoke(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    data_config = data_config_from_args(args)
    model_config = model_config_from_args(args)
    dataset = MotionMapDataset(
        cache_dir=args.cache_dir,
        fold_file=args.fold_file,
        split="train",
        fold=args.fold,
        part="train",
        data_config=data_config,
        augment=args.augment,
        seed=args.seed,
        write_summary_cache=False,
    )
    count = min(args.samples, len(dataset))
    if count < 1:
        raise ValueError("--samples must be positive")
    items = [dataset[index] for index in range(count)]
    full_maps = torch.stack([item[0] for item in items]).to(device)
    roi_maps = torch.stack([item[1] for item in items]).to(device)
    view_available = torch.stack([item[2] for item in items]).to(device)
    labels = torch.tensor([item[3] for item in items], device=device)
    model = build_model(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.train()
    logits = model(full_maps, roi_maps, view_available)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("smoke test produced non-finite gradients")
    print(f"smoke OK on {device}")
    print(f"  samples: {[item[4] for item in items]}")
    print(f"  channels: {MAP_CHANNELS}")
    print(f"  full maps: {tuple(full_maps.shape)}")
    print(f"  ROI maps:  {tuple(roi_maps.shape)}")
    print(
        "  view available [full, ROI]: "
        f"{view_available.int().tolist()}"
    )
    print(f"  logits: {tuple(logits.shape)}, loss: {float(loss.detach()):.4f}")
    print(
        f"  parameters: {parameter_count:,} "
        f"({parameter_count / 1e6:.3f}M, limit <{MAX_PARAMETERS / 1e6:.1f}M)"
    )
    print("  no files written")


def run_train(args: argparse.Namespace) -> None:
    validate_tag(args.tag)
    folds = parse_folds(args.folds, args.fold_file)
    if args.dry_run:
        args.fold = folds[0]
        args.samples = min(args.batch_size, 4)
        run_smoke(args)
        return

    seed_everything(args.seed)
    device = choose_device(args.device)
    data_config = data_config_from_args(args)
    model_config = model_config_from_args(args)
    parameter_count = sum(
        parameter.numel() for parameter in build_model(model_config).parameters()
    )
    started = time.time()
    results = [
        train_one_fold(args, fold, data_config, model_config, device) for fold in folds
    ]
    oof_path = save_oof(
        args.artifact_dir, args.tag, folds, args.fold_file, results
    )
    mean_micro = float(np.mean([result["micro"] for result in results]))
    std_micro = float(np.std([result["micro"] for result in results]))
    manifest = {
        "kind": CHECKPOINT_KIND,
        "tag": args.tag,
        "folds": folds,
        "selection_protocol": (
            "fixed epochs; outer subject validation evaluated exactly once after training"
        ),
        "sampling_protocol": "natural-prior shuffled training samples",
        "model_config": asdict(model_config),
        "data_config": asdict(data_config),
        "parameter_count": parameter_count,
        "fold_results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"probs", "labels", "sids"}
            }
            for result in results
        ],
        "mean_micro": mean_micro,
        "std_micro": std_micro,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "oof_path": str(oof_path),
        "command_args": {
            key: value for key, value in vars(args).items() if key != "handler"
        },
    }
    manifest_path = (
        args.artifact_dir
        / f"run_motionmap_{args.tag}{fold_suffix(folds, args.fold_file)}.json"
    )
    write_json(manifest_path, manifest)
    print(
        f"RESULT {args.tag}: micro {mean_micro:.4f} ± {std_micro:.4f}; "
        f"{parameter_count / 1e6:.3f}M parameters",
        flush=True,
    )
    print(f"OOF: {oof_path}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)


def run_oof(args: argparse.Namespace) -> None:
    validate_tag(args.tag)
    folds = parse_folds(args.folds, args.fold_file)
    seed_everything(args.seed)
    device = choose_device(args.device)
    _, data_config, _ = checkpoint_configs(args, folds, device)
    results = [
        infer_oof_fold(args, fold, data_config, device) for fold in folds
    ]
    path = save_oof(args.artifact_dir, args.tag, folds, args.fold_file, results)
    predictions = np.concatenate([result["probs"] for result in results]).argmax(1)
    labels = np.concatenate([result["labels"] for result in results])
    weighted_micro = float((predictions == labels).mean())
    print(f"OOF micro {weighted_micro:.4f}; wrote {path}")


def run_test(args: argparse.Namespace) -> None:
    validate_tag(args.tag)
    folds = parse_folds(args.folds, args.fold_file)
    seed_everything(args.seed)
    device = choose_device(args.device)
    _, data_config, _ = checkpoint_configs(args, folds, device)
    dataset = MotionMapDataset(
        cache_dir=args.cache_dir,
        fold_file=args.fold_file,
        split="test",
        part="all",
        data_config=data_config,
        augment=False,
        seed=0,
    )
    loader = make_loader(
        dataset, args.batch_size, args.workers, False, device, args.seed
    )
    probability_sum = np.zeros((len(dataset), NUM_CLASSES), dtype=np.float64)
    reference_sids: list[str] | None = None
    for fold in folds:
        model, _, loaded_data, package = load_checkpoint(
            checkpoint_path(args.checkpoint_dir, args.tag, fold), device
        )
        if loaded_data != data_config:
            raise ValueError(f"data configuration mismatch for fold {fold}")
        result = evaluate(model, loader, device)
        if reference_sids is None:
            reference_sids = result["sids"]
        elif result["sids"] != reference_sids:
            raise RuntimeError("test sample order changed between fold models")
        probability_sum += result["probs"]
        print(
            f"fold {fold}: inferred {len(dataset)} test clips from checkpoint "
            f"epoch {package.get('epoch')}",
            flush=True,
        )
    assert reference_sids is not None
    probabilities = probability_sum / len(folds)
    path = save_test_probabilities(
        args.artifact_dir,
        args.tag,
        folds,
        args.fold_file,
        probabilities,
        reference_sids,
    )
    print(f"wrote {path}")


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fold-file", type=Path, default=DEFAULT_FOLD_FILE)


def add_data_options(parser: argparse.ArgumentParser) -> None:
    add_paths(parser)
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="aligned temporal samples used to form each summary (default: 16)",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=224,
        help="ROI summary input size; source is cache/*_roi224 (default: 224)",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mild temporal, photometric and spatial training augmentation",
    )


def add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="base CNN width; parameter limit is enforced (default: 32)",
    )
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)


def add_inference_options(parser: argparse.ArgumentParser) -> None:
    add_paths(parser)
    add_runtime_options(parser)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "From-scratch dual-view IR classifier over static and directional "
            "motion maps."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="one forward/backward data and model check; writes nothing"
    )
    add_data_options(smoke)
    add_model_options(smoke)
    add_runtime_options(smoke)
    smoke.add_argument("--fold", type=int, default=0)
    smoke.add_argument("--samples", type=int, default=2)
    smoke.set_defaults(handler=run_smoke)

    train = subparsers.add_parser(
        "train",
        help="train subject folds, save packaged checkpoints and an OOF artifact",
    )
    add_data_options(train)
    add_model_options(train)
    add_runtime_options(train)
    train.add_argument("--tag", required=True)
    train.add_argument(
        "--folds",
        default="0,2",
        help="comma-separated subject folds (default screen: 0,2)",
    )
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--label-smoothing", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA automatic mixed precision (ignored on CPU)",
    )
    train.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    train.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="run one in-memory forward/backward check and write nothing",
    )
    train.set_defaults(handler=run_train)

    oof = subparsers.add_parser(
        "oof",
        aliases=["predict-oof"],
        help="regenerate OOF probabilities from packaged checkpoints",
    )
    add_inference_options(oof)
    oof.set_defaults(handler=run_oof)

    test = subparsers.add_parser(
        "test",
        aliases=["predict-test"],
        help="average fold checkpoints into test probability artifacts",
    )
    add_inference_options(test)
    test.set_defaults(handler=run_test)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "workers", "seed"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if hasattr(args, "frames") and args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if hasattr(args, "roi_size") and args.roi_size < 32:
        raise ValueError("--roi-size must be at least 32")
    if hasattr(args, "width") and args.width < 8:
        raise ValueError("--width must be at least 8")
    if hasattr(args, "head_width") and args.head_width < 8:
        raise ValueError("--head-width must be at least 8")
    if hasattr(args, "dropout") and not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0,1)")
    if hasattr(args, "epochs") and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if hasattr(args, "lr") and args.lr <= 0:
        raise ValueError("--lr must be positive")
    if hasattr(args, "weight_decay") and args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative")
    if hasattr(args, "label_smoothing") and not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0,1)")
    if hasattr(args, "grad_clip") and args.grad_clip < 0:
        raise ValueError("--grad-clip must be non-negative")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        args.handler(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
