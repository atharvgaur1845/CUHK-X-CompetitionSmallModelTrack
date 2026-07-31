#!/usr/bin/env python3
"""Leakage-safe world-frame IMU experiment.

The five wearable sensors report acceleration, angular velocity, and a unit
quaternion.  The quaternion convention was verified directly on this dataset:
``R(q) @ acceleration`` maps gravity from every device to approximately
``[0, 0, 1]``.  This experiment therefore uses, per device:

* gravity-removed world acceleration (3);
* world angular velocity (3);
* acceleration and angular-velocity magnitudes (2);
* a temporal device-presence mask (1).

No absolute Euler angles, magnetometer values, raw mounting-frame vectors, or
absolute quaternion components are exposed to the classifier.

Outer validation is evaluated exactly once after a fixed epoch count.  The
script never selects epochs, hyperparameters, or checkpoints on an outer fold.
It is standalone: feature extraction, model definition, checkpoint loading,
OOF generation, test inference, and the existing-ensemble marginal audit all
live in this file.

Typical workflow:

    python3 code/train_imu_world.py smoke
    python3 code/train_imu_world.py audit --folds 0,2
    python3 code/train_imu_world.py train --folds 0,2 --amp
    python3 code/train_imu_world.py oof --folds 0,2
    python3 code/train_imu_world.py marginal --folds 0,2

If the paired screen passes:

    python3 code/train_imu_world.py train --folds 1,3 --amp
    python3 code/train_imu_world.py oof --folds 0,1,2,3
    python3 code/train_imu_world.py marginal --folds 0,1,2,3
    python3 code/train_imu_world.py test --folds 0,1,2,3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache"
DEFAULT_FOLDS = ROOT / "research" / "artifacts" / "cv_folds.json"
DEFAULT_ARTIFACTS = ROOT / "research" / "artifacts"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints"

DEVICES = ("WTC", "WTLA", "WTRA", "WTLL", "WTRL")
N_CLASSES = 40
CHANNELS_PER_DEVICE = 9
CHECKPOINT_KIND = "imu_world_tcn"
CHECKPOINT_VERSION = 1

# Existing imu_inv4 reference, recorded in results.csv.  These are printed as
# paired screening controls; the marginal command also verifies them from OOF.
IMU_INV4_FOLD_ACCURACY = {
    0: 0.2982,
    1: 0.3388,
    2: 0.2806,
    3: 0.2678,
}


@dataclass(frozen=True)
class FeatureConfig:
    steps: int = 64
    gyro_scale: float = 500.0
    gravity_z: float = 1.0

    @property
    def channels(self) -> int:
        return len(DEVICES) * CHANNELS_PER_DEVICE


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = len(DEVICES) * CHANNELS_PER_DEVICE
    width: int = 128
    depth: int = 4
    dropout: float = 0.3
    classes: int = N_CLASSES


def parse_folds(text: str) -> list[int]:
    try:
        folds = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--folds must be a comma-separated subset of 0,1,2,3") from exc
    if (
        not folds
        or len(set(folds)) != len(folds)
        or any(fold not in range(4) for fold in folds)
    ):
        raise ValueError("--folds must be a unique, non-empty subset of 0,1,2,3")
    return folds


def fold_suffix(folds: Sequence[int]) -> str:
    return "" if list(folds) == [0, 1, 2, 3] else "_f" + "".join(map(str, folds))


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rotate_world(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply the verified local-to-world rotation ``R(q)``.

    Quaternions are ordered ``(w, x, y, z)`` in the raw CSV/cache.  The formula
    is sign invariant, so ``q`` and ``-q`` produce identical vectors.
    """
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    if q.shape != v.shape[:-1] + (4,) or v.shape[-1] != 3:
        raise ValueError(f"incompatible quaternion/vector shapes: {q.shape}, {v.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    valid = norm[..., 0] > 1e-8
    safe_q = q / np.where(valid[..., None], norm, 1.0)
    scalar = safe_q[..., :1]
    axis = safe_q[..., 1:]
    rotated = (
        v
        + 2.0 * scalar * np.cross(axis, v)
        + 2.0 * np.cross(axis, np.cross(axis, v))
    )
    return np.where(valid[..., None], rotated, 0.0)


def _clean_device_rows(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 16 or len(values) != len(timestamps):
        return np.empty(0, np.float64), np.empty((0, 16), np.float64)
    finite = np.isfinite(timestamps) & np.isfinite(values[:, :16]).all(1)
    timestamps, values = timestamps[finite], values[finite, :16]
    if not len(timestamps):
        return timestamps, values
    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[order]
    timestamps, unique = np.unique(timestamps, return_index=True)
    return timestamps, values[unique]


def world_device_values(values: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Convert one sensor's raw 16 channels into eight physical channels.

    The ninth per-device channel (temporal presence) is added after resampling.
    """
    values = np.asarray(values, dtype=np.float64)
    acceleration = values[:, 0:3]
    angular_velocity = values[:, 3:6]
    quaternion = values[:, 12:16]
    world_acceleration = rotate_world(quaternion, acceleration)
    world_angular_velocity = rotate_world(quaternion, angular_velocity)

    dynamic_acceleration = world_acceleration.copy()
    dynamic_acceleration[:, 2] -= config.gravity_z
    scaled_gyro = world_angular_velocity / config.gyro_scale
    features = np.concatenate(
        (
            dynamic_acceleration,
            scaled_gyro,
            np.linalg.norm(dynamic_acceleration, axis=1, keepdims=True),
            np.linalg.norm(scaled_gyro, axis=1, keepdims=True),
        ),
        axis=1,
    )
    return np.nan_to_num(features, copy=False).astype(np.float32)


def extract_world_imu(archive, config: FeatureConfig) -> torch.Tensor:
    """Resample all devices onto a common physical-time grid."""
    cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    starts, ends = [], []
    files = set(archive.files)
    for device in DEVICES:
        time_key, value_key = f"imu_{device}_t", f"imu_{device}_x"
        if time_key not in files or value_key not in files:
            continue
        timestamps, values = _clean_device_rows(archive[time_key], archive[value_key])
        if len(timestamps) < 2:
            continue
        cleaned[device] = (timestamps, values)
        starts.append(float(timestamps[0]))
        ends.append(float(timestamps[-1]))

    output = np.zeros(
        (config.steps, len(DEVICES), CHANNELS_PER_DEVICE), dtype=np.float32
    )
    if not starts or min(starts) >= max(ends):
        return torch.from_numpy(output.reshape(config.steps, -1))
    grid = np.linspace(min(starts), max(ends), config.steps, dtype=np.float64)
    for device_index, device in enumerate(DEVICES):
        if device not in cleaned:
            continue
        timestamps, raw_values = cleaned[device]
        device_values = world_device_values(raw_values, config)
        for channel in range(device_values.shape[1]):
            output[:, device_index, channel] = np.interp(
                grid, timestamps, device_values[:, channel]
            )
        # np.interp repeats edge values. Mask those repetitions so an absent
        # device contributes only an explicit zero-valued presence channel.
        support = (
            (grid >= timestamps[0]) & (grid <= timestamps[-1])
        )
        output[:, device_index, :-1] *= support[:, None]
        output[:, device_index, -1] = support.astype(np.float32)
    return torch.from_numpy(output.reshape(config.steps, -1))


def read_metadata(cache_dir: Path, split: str) -> dict[str, dict[str, str]]:
    path = cache_dir / f"meta_{split}.csv"
    with path.open(newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"metadata is empty: {path}")
    return rows


def read_fold_users(path: Path) -> dict[int, set[str]]:
    with path.open() as handle:
        payload = json.load(handle)
    return {
        int(item["fold"]): {str(user) for user in item["val_users"]}
        for item in payload["folds"]
    }


class IMUWorldDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        fold_path: Path,
        split: str,
        feature_config: FeatureConfig,
        fold: int | None = None,
        part: str = "test",
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"unsupported split {split!r}")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.feature_config = feature_config
        self.meta = read_metadata(self.cache_dir, split)
        if split == "test":
            self.ids = sorted(self.meta)
            self.labels = None
        else:
            if fold is None or part not in {"train", "val"}:
                raise ValueError("train split requires fold and part=train|val")
            validation_users = read_fold_users(fold_path)[fold]
            if part == "val":
                self.ids = sorted(
                    sid
                    for sid, row in self.meta.items()
                    if row["user"] in validation_users
                )
            else:
                self.ids = sorted(
                    sid
                    for sid, row in self.meta.items()
                    if row["user"] not in validation_users
                )
            self.labels = {sid: int(self.meta[sid]["class_id"]) for sid in self.ids}
        if not self.ids:
            raise ValueError(f"empty {split}/{part} dataset")
        self._preloaded: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def _load_features(self, sid: str) -> torch.Tensor:
        path = self.cache_dir / self.split / f"{sid}.npz"
        with np.load(path) as archive:
            return extract_world_imu(archive, self.feature_config)

    def preload(self) -> "IMUWorldDataset":
        """Materialize deterministic features once instead of once per epoch."""
        if self._preloaded is None:
            started = time.time()
            self._preloaded = torch.stack(
                [self._load_features(sid) for sid in self.ids]
            )
            mebibytes = self._preloaded.numel() * self._preloaded.element_size() / 2**20
            print(
                f"preloaded {self.split}/{len(self.ids)} world-IMU tensors "
                f"({mebibytes:.1f} MiB) in {time.time() - started:.1f}s",
                flush=True,
            )
        return self

    def __getitem__(self, index: int):
        sid = self.ids[index]
        features = (
            self._preloaded[index]
            if self._preloaded is not None
            else self._load_features(sid)
        )
        label = -1 if self.labels is None else self.labels[sid]
        return features, label, sid


def conv_block(channels_in: int, channels_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        ),
        nn.GroupNorm(8, channels_out),
        nn.SiLU(),
    )


class IMUWorldTCN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width % 8:
            raise ValueError("model width must be divisible by 8")
        channels = [
            config.width * min(2 ** (index // 2), 2)
            for index in range(config.depth)
        ]
        body, channels_in = [], config.in_channels
        for index, channels_out in enumerate(channels):
            body.append(
                conv_block(
                    channels_in,
                    channels_out,
                    stride=2 if index in (1, 3) else 1,
                )
            )
            channels_in = channels_out
        self.config = config
        self.body = nn.Sequential(*body)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(2 * channels_in, config.classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.in_channels:
            raise ValueError(
                f"expected [batch,time,{self.config.in_channels}], "
                f"got {tuple(features.shape)}"
            )
        hidden = self.body(features.transpose(1, 2))
        pooled = torch.cat((hidden.mean(-1), hidden.amax(-1)), dim=1)
        return self.head(self.dropout(pooled))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probabilities, labels, sids = [], [], []
    with torch.inference_mode():
        for features, target, batch_sids in loader:
            logits = model(features.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(target.numpy())
            sids.extend(batch_sids)
    probs = np.concatenate(probabilities)
    targets = np.concatenate(labels)
    predictions = probs.argmax(1)
    micro = float((predictions == targets).mean())
    macro = float(
        np.mean(
            [
                (predictions[targets == label] == label).mean()
                for label in np.unique(targets)
            ]
        )
    )
    return {
        "micro": micro,
        "macro": macro,
        "probs": probs,
        "labels": targets,
        "sids": sids,
    }


def checkpoint_path(directory: Path, tag: str, fold: int) -> Path:
    return directory / f"{tag}_f{fold}.pt"


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(
    path: Path,
    model: IMUWorldTCN,
    feature_config: FeatureConfig,
    fold: int,
    args: argparse.Namespace,
    metrics: dict[str, object],
) -> None:
    package = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": asdict(model.config),
        "feature_config": asdict(feature_config),
        "fold": int(fold),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "selection_protocol": (
            "fixed epochs; outer validation evaluated once after optimization"
        ),
        "outer_metrics": {
            "micro": float(metrics["micro"]),
            "macro": float(metrics["macro"]),
        },
    }
    atomic_torch_save(package, path)


def load_checkpoint(path: Path, device: torch.device):
    package = torch.load(path, map_location=device)
    if (
        not isinstance(package, dict)
        or package.get("checkpoint_kind") != CHECKPOINT_KIND
        or package.get("checkpoint_version") != CHECKPOINT_VERSION
    ):
        raise ValueError(f"not a supported {CHECKPOINT_KIND} checkpoint: {path}")
    feature_config = FeatureConfig(**package["feature_config"])
    model_config = ModelConfig(**package["model_config"])
    if model_config.in_channels != feature_config.channels:
        raise ValueError("checkpoint feature/model channel mismatch")
    model = IMUWorldTCN(model_config).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    return model, feature_config, package


def validate_fold_partition(
    train_dataset: IMUWorldDataset,
    validation_dataset: IMUWorldDataset,
    fold: int,
) -> tuple[set[str], set[str]]:
    train_ids, validation_ids = set(train_dataset.ids), set(validation_dataset.ids)
    train_users = {
        train_dataset.meta[sid]["user"] for sid in train_dataset.ids
    }
    validation_users = {
        validation_dataset.meta[sid]["user"] for sid in validation_dataset.ids
    }
    if train_ids & validation_ids:
        raise RuntimeError(f"fold {fold}: sample leakage")
    if train_users & validation_users:
        raise RuntimeError(f"fold {fold}: user leakage")
    return train_users, validation_users


def run_training_fold(args: argparse.Namespace, fold: int) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    feature_config = FeatureConfig(
        steps=args.steps,
        gyro_scale=args.gyro_scale,
        gravity_z=args.gravity_z,
    )
    model_config = ModelConfig(
        in_channels=feature_config.channels,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
    )
    train_dataset = IMUWorldDataset(
        args.cache_dir, args.fold_file, "train", feature_config, fold, "train"
    )
    validation_dataset = IMUWorldDataset(
        args.cache_dir, args.fold_file, "train", feature_config, fold, "val"
    )
    train_users, validation_users = validate_fold_partition(
        train_dataset, validation_dataset, fold
    )
    train_dataset.preload()
    validation_dataset.preload()
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        shuffle=True,
        seed=args.seed,
        device=device,
        drop_last=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.eval_batch_size,
        max(0, args.workers // 2),
        shuffle=False,
        seed=args.seed,
        device=device,
    )
    model = IMUWorldTCN(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"fold {fold}: train={len(train_dataset)} clips/{len(train_users)} users; "
        f"outer-val={len(validation_dataset)} clips/{len(validation_users)} users; "
        f"params={parameter_count}; device={device}; amp={amp_enabled}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = examples = 0
        for features, target, _ in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = criterion(model(features), target)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach()) * len(target)
            examples += len(target)
        if (
            epoch == 0
            or (epoch + 1) % args.log_every == 0
            or epoch + 1 == args.epochs
        ):
            print(
                f"  fold{fold} epoch {epoch + 1}/{args.epochs}: "
                f"train loss {running_loss / examples:.4f} "
                "(outer validation untouched)",
                flush=True,
            )

    metrics = evaluate(model, validation_loader, device)
    delta = float(metrics["micro"]) - IMU_INV4_FOLD_ACCURACY[fold]
    print(
        f"  fold{fold} final outer evaluation: micro={metrics['micro']:.4f} "
        f"macro={metrics['macro']:.4f}; "
        f"paired vs imu_inv4={delta:+.4f}",
        flush=True,
    )
    path = checkpoint_path(args.checkpoint_dir, args.tag, fold)
    save_checkpoint(path, model, feature_config, fold, args, metrics)
    print(f"  checkpoint: {path}", flush=True)
    return {
        "fold": fold,
        "micro": float(metrics["micro"]),
        "macro": float(metrics["macro"]),
        "delta": delta,
    }


def train_command(args: argparse.Namespace) -> None:
    folds = parse_folds(args.folds)
    started = time.time()
    results = [run_training_fold(args, fold) for fold in folds]
    print(
        "SCREEN "
        + ", ".join(
            f"f{item['fold']}={item['micro']:.4f} "
            f"(Δ{item['delta']:+.4f})"
            for item in results
        )
        + f"; mean={np.mean([item['micro'] for item in results]):.4f}; "
        + f"minutes={(time.time() - started) / 60:.1f}",
        flush=True,
    )


def common_checkpoint_configs(
    args: argparse.Namespace,
    folds: Sequence[int],
    device: torch.device,
):
    loaded = [
        load_checkpoint(checkpoint_path(args.checkpoint_dir, args.tag, fold), device)
        for fold in folds
    ]
    feature_config = loaded[0][1]
    model_config = loaded[0][0].config
    for expected_fold, (model, candidate_feature, package) in zip(folds, loaded):
        if candidate_feature != feature_config or model.config != model_config:
            raise ValueError("fold checkpoints do not share feature/model configuration")
        if int(package["fold"]) != expected_fold:
            raise ValueError(
                f"checkpoint fold metadata mismatch: expected {expected_fold}, "
                f"found {package['fold']}"
            )
    return loaded, feature_config


def oof_command(args: argparse.Namespace) -> None:
    folds = parse_folds(args.folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded, feature_config = common_checkpoint_configs(args, folds, device)
    probabilities, labels, sids = [], [], []
    for fold, (model, _, _) in zip(folds, loaded):
        dataset = IMUWorldDataset(
            args.cache_dir, args.fold_file, "train", feature_config, fold, "val"
        ).preload()
        loader = make_loader(
            dataset,
            args.eval_batch_size,
            args.workers,
            shuffle=False,
            seed=0,
            device=device,
        )
        metrics = evaluate(model, loader, device)
        probabilities.append(metrics["probs"])
        labels.append(metrics["labels"])
        sids.extend(metrics["sids"])
        print(
            f"fold {fold}: micro={metrics['micro']:.4f} "
            f"macro={metrics['macro']:.4f}",
            flush=True,
        )
    probs = np.concatenate(probabilities)
    targets = np.concatenate(labels)
    path = args.artifact_dir / f"oof_{args.tag}{fold_suffix(folds)}.npz"
    atomic_npz(path, probs=probs, labels=targets, sids=np.asarray(sids, dtype=str))
    print(
        f"OOF {path}: n={len(targets)} micro={(probs.argmax(1) == targets).mean():.4f}",
        flush=True,
    )


def test_command(args: argparse.Namespace) -> None:
    folds = parse_folds(args.folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded, feature_config = common_checkpoint_configs(args, folds, device)
    dataset = IMUWorldDataset(
        args.cache_dir, args.fold_file, "test", feature_config
    ).preload()
    loader = make_loader(
        dataset,
        args.eval_batch_size,
        args.workers,
        shuffle=False,
        seed=0,
        device=device,
    )
    average = np.zeros((len(dataset), N_CLASSES), dtype=np.float64)
    reference_sids = None
    for fold, (model, _, _) in zip(folds, loaded):
        model.eval()
        fold_probs, fold_sids = [], []
        with torch.inference_mode():
            for features, _, batch_sids in loader:
                fold_probs.append(
                    torch.softmax(
                        model(features.to(device, non_blocking=True)), dim=1
                    )
                    .cpu()
                    .numpy()
                )
                fold_sids.extend(batch_sids)
        if reference_sids is None:
            reference_sids = fold_sids
        elif fold_sids != reference_sids:
            raise RuntimeError("test sample order changed between fold passes")
        average += np.concatenate(fold_probs)
        print(f"fold {fold}: test probabilities complete", flush=True)
    average /= len(folds)
    path = args.artifact_dir / f"testprobs_{args.tag}{fold_suffix(folds)}.npz"
    atomic_npz(path, probs=average, sids=np.asarray(reference_sids, dtype=str))
    print(f"TEST {path}: n={len(average)}", flush=True)


def align_rows(
    probabilities: np.ndarray,
    sids: Iterable[str],
    reference: Sequence[str],
) -> np.ndarray:
    index = {str(sid): row for row, sid in enumerate(sids)}
    missing = [sid for sid in reference if sid not in index]
    if missing:
        raise ValueError(f"{len(missing)} missing rows; first={missing[0]}")
    return probabilities[[index[sid] for sid in reference]]


def paired_delta(
    old_predictions: np.ndarray,
    new_predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[int, int, int, float]:
    old_correct = old_predictions == labels
    new_correct = new_predictions == labels
    rescues = int((~old_correct & new_correct).sum())
    harms = int((old_correct & ~new_correct).sum())
    difference = (new_correct.astype(np.float64) - old_correct).astype(np.float64)
    standard_error = float(difference.std(ddof=1) / math.sqrt(len(difference)))
    return rescues, harms, rescues - harms, standard_error


def marginal_command(args: argparse.Namespace) -> None:
    """Report paired solo, replacement, and additive current-stack deltas."""
    folds = parse_folds(args.folds)
    path = args.artifact_dir / f"oof_{args.tag}{fold_suffix(folds)}.npz"
    with np.load(path) as archive:
        world_probs = archive["probs"]
        world_labels = archive["labels"]
        world_sids = [str(sid) for sid in archive["sids"]]

    # Import lazily so smoke/audit/training remain completely standalone.
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    import assemble_candidates as candidates

    current, labels, reference = candidates.block(
        "train",
        gcn_weight=0.50,
        dynamic=None,
        multitcn_weight=0.075,
        imu_weight=0.175,
        adaptive_weight=0.20,
    )
    baseline_imu, _, baseline_sids = candidates.load_standard("imu_inv4", "train")
    baseline_imu = candidates.align(
        baseline_imu, baseline_sids, reference
    )
    subset = {sid: row for row, sid in enumerate(reference)}
    indices = np.asarray([subset[sid] for sid in world_sids])
    labels = labels[indices]
    current = current[indices]
    baseline_imu = baseline_imu[indices]
    if not np.array_equal(labels, world_labels):
        raise RuntimeError("world OOF labels do not align with current-stack labels")

    skeleton = (current - 0.175 * baseline_imu) / 0.825
    baseline_imu_predictions = baseline_imu.argmax(1)
    world_predictions = world_probs.argmax(1)
    metadata = read_metadata(args.cache_dir, "train")
    fold_users = read_fold_users(args.fold_file)
    fold_masks = {
        fold: np.asarray(
            [metadata[sid]["user"] in fold_users[fold] for sid in world_sids]
        )
        for fold in folds
    }
    print("PAIRED SOLO")
    for fold in folds:
        mask = fold_masks[fold]
        rescues, harms, net, standard_error = paired_delta(
            baseline_imu_predictions[mask], world_predictions[mask], labels[mask]
        )
        print(
            f"  fold{fold}: imu_inv4="
            f"{(baseline_imu_predictions[mask] == labels[mask]).mean():.4f} "
            f"world={(world_predictions[mask] == labels[mask]).mean():.4f} "
            f"rescues={rescues} harms={harms} net={net:+d} "
            f"paired_SE={standard_error:.4f}"
        )

    current_predictions = current.argmax(1)
    print(
        f"CURRENT exact stack: {(current_predictions == labels).mean():.4f}",
        flush=True,
    )
    print("REPLACE EXISTING IMU (same-budget control)")
    for weight in (0.05, 0.10, 0.15, 0.175, 0.20, 0.25, 0.30):
        fused = (1.0 - weight) * skeleton + weight * world_probs
        predictions = fused.argmax(1)
        rescues, harms, net, standard_error = paired_delta(
            current_predictions, predictions, labels
        )
        print(
            f"  world weight={weight:.3f}: acc="
            f"{(predictions == labels).mean():.4f} "
            f"delta={net / len(labels):+.4f} rescues={rescues} harms={harms} "
            f"paired_SE={standard_error:.4f}",
            flush=True,
        )
    additive_weights = (0.0, 0.05, 0.10, 0.15, 0.175, 0.20, 0.25, 0.30)
    additive_predictions = {
        weight: ((1.0 - weight) * current + weight * world_probs).argmax(1)
        for weight in additive_weights
    }
    print("ADDITIVE RETAINING CURRENT STACK (descriptive)")
    for weight in additive_weights[1:]:
        predictions = additive_predictions[weight]
        rescues, harms, net, standard_error = paired_delta(
            current_predictions, predictions, labels
        )
        fold_delta_values = []
        for fold, mask in fold_masks.items():
            new_accuracy = (predictions[mask] == labels[mask]).mean()
            old_accuracy = (current_predictions[mask] == labels[mask]).mean()
            fold_delta_values.append(f"f{fold}={new_accuracy - old_accuracy:+.4f}")
        fold_deltas = ", ".join(fold_delta_values)
        print(
            f"  world weight={weight:.3f}: acc="
            f"{(predictions == labels).mean():.4f} "
            f"delta={net / len(labels):+.4f} rescues={rescues} harms={harms} "
            f"paired_SE={standard_error:.4f}; {fold_deltas}",
            flush=True,
        )
    if len(folds) >= 2:
        nested_predictions = current_predictions.copy()
        print("LEAVE-ONE-FOLD-OUT ADDITIVE WEIGHT")
        for fold, outer_mask in fold_masks.items():
            inner_mask = ~outer_mask
            inner_accuracies = {
                weight: (predictions[inner_mask] == labels[inner_mask]).mean()
                for weight, predictions in additive_predictions.items()
            }
            # Conservative deterministic tie break: prefer the smaller weight.
            chosen_weight = max(
                additive_weights,
                key=lambda weight: (inner_accuracies[weight], -weight),
            )
            nested_predictions[outer_mask] = additive_predictions[chosen_weight][
                outer_mask
            ]
            outer_accuracy = (
                nested_predictions[outer_mask] == labels[outer_mask]
            ).mean()
            outer_baseline = (
                current_predictions[outer_mask] == labels[outer_mask]
            ).mean()
            print(
                f"  fold{fold}: inner weight={chosen_weight:.3f}; "
                f"outer={outer_accuracy:.4f} delta="
                f"{outer_accuracy - outer_baseline:+.4f}",
                flush=True,
            )
        rescues, harms, net, standard_error = paired_delta(
            current_predictions, nested_predictions, labels
        )
        print(
            f"  nested aggregate={(nested_predictions == labels).mean():.4f} "
            f"delta={net / len(labels):+.4f} rescues={rescues} harms={harms} "
            f"paired_SE={standard_error:.4f}",
            flush=True,
        )
    print(
        "The replacement 0.175 row is the isolated same-weight control. All "
        "same-data weight sweeps are descriptive; use the leave-one-fold-out "
        "row as the leakage-safe marginal estimate.",
        flush=True,
    )


def audit_command(args: argparse.Namespace) -> None:
    folds = parse_folds(args.folds)
    feature_config = FeatureConfig(
        steps=args.steps,
        gyro_scale=args.gyro_scale,
        gravity_z=args.gravity_z,
    )
    for fold in folds:
        train_dataset = IMUWorldDataset(
            args.cache_dir, args.fold_file, "train", feature_config, fold, "train"
        )
        validation_dataset = IMUWorldDataset(
            args.cache_dir, args.fold_file, "train", feature_config, fold, "val"
        )
        train_users, validation_users = validate_fold_partition(
            train_dataset, validation_dataset, fold
        )
        class_users: dict[int, set[str]] = {}
        for sid in train_dataset.ids:
            label = int(train_dataset.meta[sid]["class_id"])
            class_users.setdefault(label, set()).add(train_dataset.meta[sid]["user"])
        print(
            f"fold {fold}: train={len(train_dataset)}/{len(train_users)} users; "
            f"outer-val={len(validation_dataset)}/{len(validation_users)} users; "
            f"classes={len(class_users)}; "
            f"min users/class={min(map(len, class_users.values()))}",
            flush=True,
        )

    raw_medians, world_medians = [], []
    metadata = read_metadata(args.cache_dir, "train")
    ordered_ids = sorted(metadata)
    audit_count = min(args.audit_samples, len(ordered_ids))
    audit_indices = np.linspace(
        0, len(ordered_ids) - 1, audit_count, dtype=np.int64
    )
    for sid in (ordered_ids[index] for index in audit_indices):
        path = args.cache_dir / "train" / f"{sid}.npz"
        with np.load(path) as archive:
            for device in DEVICES:
                key = f"imu_{device}_x"
                if key not in archive.files or len(archive[key]) < 2:
                    continue
                values = np.asarray(archive[key], dtype=np.float64)
                raw_medians.append(np.median(values[:, 0:3], axis=0))
                world = rotate_world(values[:, 12:16], values[:, 0:3])
                world_medians.append(np.median(world, axis=0))
    raw_medians = np.asarray(raw_medians)
    world_medians = np.asarray(world_medians)
    print(
        f"R(q) audit over {len(world_medians)} device-clips: "
        f"raw median={np.median(raw_medians, axis=0).round(4).tolist()}, "
        f"world median={np.median(world_medians, axis=0).round(4).tolist()}, "
        f"median |world-g|="
        f"{np.median(np.linalg.norm(world_medians - [0, 0, 1], axis=1)):.4f}",
        flush=True,
    )
    sample = IMUWorldDataset(
        args.cache_dir, args.fold_file, "train", feature_config, folds[0], "train"
    )[0][0]
    print(
        f"feature tensor={tuple(sample.shape)} finite={bool(torch.isfinite(sample).all())}; "
        "AUDIT PASSED (no artifacts written)",
        flush=True,
    )


class _SyntheticArchive:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self._arrays = arrays
        self.files = list(arrays)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]


def smoke_command() -> None:
    seed_everything(123)
    rng = np.random.default_rng(123)
    quaternion = rng.normal(size=(32, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    gravity_world = np.tile([0.0, 0.0, 1.0], (len(quaternion), 1))
    # R(q)^T is R(conjugate(q)); synthesize sensor-frame gravity.
    conjugate = quaternion.copy()
    conjugate[:, 1:] *= -1
    gravity_sensor = rotate_world(conjugate, gravity_world)
    recovered = rotate_world(quaternion, gravity_sensor)
    if not np.allclose(recovered, gravity_world, atol=1e-8):
        raise AssertionError("R(q) failed the known gravity rotation")
    if not np.allclose(
        rotate_world(-quaternion, gravity_sensor), recovered, atol=1e-8
    ):
        raise AssertionError("quaternion sign invariance failed")

    timestamps = np.linspace(10.0, 12.0, len(quaternion))
    raw = np.zeros((len(quaternion), 16), np.float32)
    raw[:, 0:3] = gravity_sensor
    raw[:, 3:6] = rng.normal(0, 20, size=(len(quaternion), 3))
    raw[:, 12:16] = quaternion
    archive = _SyntheticArchive(
        {"imu_WTC_t": timestamps, "imu_WTC_x": raw}
    )
    feature_config = FeatureConfig(steps=64)
    features = extract_world_imu(archive, feature_config)
    if features.shape != (64, feature_config.channels):
        raise AssertionError(f"unexpected feature shape {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise AssertionError("world features are not finite")
    # Static gravity must disappear from the dynamic-acceleration channels.
    if float(features[:, 0:3].abs().max()) > 1e-5:
        raise AssertionError("gravity was not removed after world rotation")

    config = ModelConfig(in_channels=feature_config.channels, width=16)
    model = IMUWorldTCN(config)
    batch = torch.stack((features, features))
    labels = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(batch), labels)
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError("model backward produced a non-finite loss")

    package = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": {key: value.clone() for key, value in model.state_dict().items()},
        "model_config": asdict(config),
        "feature_config": asdict(feature_config),
        "fold": 0,
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "smoke.pt"
        torch.save(package, path)
        reloaded, reloaded_features, reloaded_package = load_checkpoint(
            path, torch.device("cpu")
        )
    if reloaded_features != feature_config or reloaded_package["fold"] != 0:
        raise AssertionError("self-contained checkpoint metadata did not round-trip")
    model.eval()
    reloaded.eval()
    with torch.inference_mode():
        if not torch.allclose(model(batch), reloaded(batch)):
            raise AssertionError("self-contained checkpoint predictions changed")
    print(
        "SMOKE PASSED: R(q), q-sign invariance, gravity removal, resampling, "
        f"finite backward, and strict state reload; feature shape={tuple(features.shape)}",
        flush=True,
    )


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fold-file", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--tag", default="imu_world")
    parser.add_argument("--folds", default="0,2")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=128)


def add_features(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gyro-scale", type=float, default=500.0)
    parser.add_argument("--gravity-z", type=float, default=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="synthetic math/model test; writes nothing")

    audit = subparsers.add_parser(
        "audit", help="audit folds and real cached R(q) features; writes nothing"
    )
    add_paths(audit)
    add_features(audit)
    audit.add_argument("--audit-samples", type=int, default=256)

    train = subparsers.add_parser(
        "train", help="fixed-epoch training and one final outer-fold evaluation"
    )
    add_paths(train)
    add_features(train)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--label-smoothing", type=float, default=0.1)
    train.add_argument("--width", type=int, default=128)
    train.add_argument("--depth", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.3)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--amp", action="store_true")

    oof = subparsers.add_parser("oof", help="emit standard OOF NPZ from checkpoints")
    add_paths(oof)

    test = subparsers.add_parser(
        "test", help="average fold checkpoints and emit standard test-probability NPZ"
    )
    add_paths(test)

    marginal = subparsers.add_parser(
        "marginal",
        help="paired solo and exact current-ensemble replacement report",
    )
    add_paths(marginal)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "smoke":
        return
    parse_folds(args.folds)
    if args.workers < 0 or args.eval_batch_size <= 0:
        raise ValueError("workers must be non-negative and eval batch size positive")
    if not args.tag:
        raise ValueError("--tag cannot be empty")
    if args.command == "audit":
        if args.steps <= 1 or args.gyro_scale <= 0 or args.audit_samples <= 0:
            raise ValueError("invalid feature/audit configuration")
    if args.command == "train":
        positive = (
            args.steps,
            args.gyro_scale,
            args.epochs,
            args.batch_size,
            args.lr,
            args.width,
            args.depth,
            args.log_every,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("training sizes, rates, and epoch counts must be positive")
        if args.width % 8:
            raise ValueError("--width must be divisible by 8")
        if args.weight_decay < 0 or args.grad_clip < 0:
            raise ValueError("weight decay and gradient clipping cannot be negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.command == "smoke":
        smoke_command()
    elif args.command == "audit":
        audit_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "oof":
        oof_command(args)
    elif args.command == "test":
        test_command(args)
    elif args.command == "marginal":
        marginal_command(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
