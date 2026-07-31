#!/usr/bin/env python3
"""Leakage-safe subject-canonical bone Adaptive ST-GCN screen.

The candidate keeps the accepted raw station-frame features and appends only
one new view:

    xyz (3) + frame delta (3) + raw bone (3)
        + raw bone / subject median bone length (3)

Each of the 16 non-root bone lengths is estimated without labels, exclusively
from clips assigned to the same subject.  An outer-validation subject therefore
uses all of that subject's held-out clips transductively, exactly as a test
subject does.  Test subjects are represented by the predeclared E1/E2/L1/L2
cohorts in ``research/artifacts/test_cohorts.csv``; audit fails if even one test
ID is missing, duplicated, extra, or assigned to an unknown cohort.

``--variant raw`` is the exact 9-channel feature/model control.  The default
``canonbone`` variant has 12 channels.  Both use the accepted width-64 Adaptive
ST-GCN topology.  Training is deliberately fixed: 60 epochs, AdamW 1e-3,
weight decay 0.05, label smoothing 0.1, OneCycle, and truncation-only
augmentation.  The outer fold is evaluated once, after optimization.

First-pass workflow:

    python3 code/train_canonbone.py smoke
    python3 code/train_canonbone.py audit --folds 0,2
    python3 code/train_canonbone.py train --tag canonbone_v1 \
        --variant canonbone --folds 0,2 --device cuda --amp

Run the exact raw control under a distinct tag:

    python3 code/train_canonbone.py train --tag canonbone_raw60_control \
        --variant raw --folds 0,2 --device cuda --amp

This initial screen intentionally emits checkpoints and sidecars only.  OOF,
test-probability, and replacement-fusion commands are added after the paired
screen passes, so no untested branch can silently become an upload artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "cache"
DEFAULT_FOLDS = ROOT / "research" / "artifacts" / "cv_folds.json"
DEFAULT_COHORTS = ROOT / "research" / "artifacts" / "test_cohorts.csv"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints"
DEFAULT_ARTIFACTS = ROOT / "research" / "artifacts"

N_CLASSES = 40
N_JOINTS = 17
N_TEST = 405
PARENTS = np.asarray(
    (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15),
    dtype=np.int64,
)
EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)
TEST_COHORTS = frozenset(("E1", "E2", "L1", "L2"))

FEATURE_VERSION = "raw-preserving-subject-canonical-bone-v1"
CHECKPOINT_KIND = "canonbone_adaptive_stgcn"
CHECKPOINT_VERSION = 1
RAW_CHANNELS = 9
CANONBONE_CHANNELS = 12

# The screen recipe is fixed rather than tuneable on an outer fold.
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
WIDTH = 64
DROPOUT = 0.3
GRAD_CLIP = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, ids))).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(path: Path, payload: object) -> None:
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


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_folds(value: str) -> tuple[int, ...]:
    try:
        folds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--folds must be a comma-separated subset of 0,1,2,3") from exc
    if (
        not folds
        or len(folds) != len(set(folds))
        or any(fold not in range(4) for fold in folds)
    ):
        raise ValueError("--folds must be a unique non-empty subset of 0,1,2,3")
    return tuple(sorted(folds))


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


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def read_metadata(cache_dir: Path, split: str) -> dict[str, dict[str, str]]:
    path = cache_dir / f"meta_{split}.csv"
    with path.open(newline="") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"empty metadata: {path}")
    return rows


def read_fold_users(path: Path) -> dict[int, set[str]]:
    with path.open() as handle:
        payload = json.load(handle)
    records = payload.get("folds")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError(f"{path}: expected exactly four fold records")
    result: dict[int, set[str]] = {}
    seen: set[str] = set()
    for expected, record in enumerate(records):
        fold = int(record.get("fold", -1))
        users = {str(value) for value in record.get("val_users", ())}
        if fold != expected or not users:
            raise ValueError(f"{path}: invalid fold record at index {expected}")
        if seen & users:
            raise ValueError(f"{path}: validation users repeat across folds")
        result[fold] = users
        seen.update(users)
    always = {str(value) for value in payload.get("always_train", ())}
    if always & seen:
        raise ValueError(f"{path}: always-train user appears in validation")
    return result


def validate_group_mapping(
    mapping: Mapping[str, str],
    expected_ids: Iterable[str],
    allowed_groups: set[str] | frozenset[str] | None = None,
) -> None:
    expected = set(expected_ids)
    actual = set(mapping)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        detail = (
            f"missing={len(missing)}"
            + (f" first={missing[0]}" if missing else "")
            + f"; extra={len(extra)}"
            + (f" first={extra[0]}" if extra else "")
        )
        raise ValueError(f"incomplete group mapping: {detail}")
    blank = [sid for sid, group in mapping.items() if not str(group)]
    if blank:
        raise ValueError(f"blank group for {blank[0]}")
    if allowed_groups is not None:
        unknown = sorted(set(mapping.values()) - set(allowed_groups))
        if unknown:
            raise ValueError(f"unknown group(s): {unknown}")


def read_test_cohorts(path: Path, test_ids: Iterable[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    duplicates: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sm_id", "cohort_guess"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: columns {sorted(required)} are required")
        for row in reader:
            sid, cohort = row["sm_id"].strip(), row["cohort_guess"].strip()
            if sid in rows:
                duplicates.append(sid)
            rows[sid] = cohort
    if duplicates:
        raise ValueError(f"{path}: duplicate test ID {duplicates[0]}")
    validate_group_mapping(rows, test_ids, TEST_COHORTS)
    if len(rows) != N_TEST:
        raise ValueError(f"{path}: expected {N_TEST} mappings, found {len(rows)}")
    missing_cohorts = sorted(TEST_COHORTS - set(rows.values()))
    if missing_cohorts:
        raise ValueError(f"{path}: empty required cohort(s) {missing_cohorts}")
    return rows


@dataclass(frozen=True)
class FeatureConfig:
    variant: str = "canonbone"
    steps: int = 32

    def validate(self) -> None:
        if self.variant not in ("canonbone", "raw"):
            raise ValueError("variant must be canonbone or raw")
        if self.steps < 8:
            raise ValueError("steps must be at least 8")

    @property
    def channels(self) -> int:
        return CANONBONE_CHANNELS if self.variant == "canonbone" else RAW_CHANNELS


@dataclass(frozen=True)
class TemplateBundle:
    templates: Mapping[str, np.ndarray]
    members: Mapping[str, tuple[str, ...]]
    contributing_clips: Mapping[str, int]
    digest: str


def _first_person_pose(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as archive:
        if "skel_pos" not in archive.files:
            return None
        positions = archive["skel_pos"]
        if positions.ndim != 4 or positions.shape[1] < 1:
            return None
        pose = np.asarray(positions[:, 0], dtype=np.float32)
    if pose.shape[1:] != (N_JOINTS, 3) or len(pose) == 0:
        return None
    return np.nan_to_num(pose, copy=False)


def _clip_bone_lengths(pose: np.ndarray | None) -> np.ndarray | None:
    if pose is None or len(pose) == 0:
        return None
    bones = pose - pose[:, PARENTS]
    lengths = np.linalg.norm(bones, axis=-1)
    result = np.full(N_JOINTS, np.nan, dtype=np.float64)
    result[0] = 1.0
    for joint in range(1, N_JOINTS):
        valid = lengths[:, joint]
        valid = valid[np.isfinite(valid) & (valid > 1e-5)]
        if len(valid):
            result[joint] = float(np.median(valid))
    return result


def _template_digest(
    templates: Mapping[str, np.ndarray],
    members: Mapping[str, Sequence[str]],
) -> str:
    digest = hashlib.sha256()
    for group in sorted(templates):
        digest.update(group.encode())
        digest.update(b"\0")
        digest.update("\n".join(sorted(members[group])).encode())
        digest.update(b"\0")
        digest.update(np.asarray(templates[group], dtype="<f4").tobytes())
    return digest.hexdigest()


def build_templates(
    cache_dir: Path,
    split: str,
    ids: Sequence[str],
    group_of: Mapping[str, str],
) -> TemplateBundle:
    """Build group-local templates without accepting labels as an input."""
    validate_group_mapping(
        {sid: group_of[sid] for sid in ids if sid in group_of}, ids
    )
    members: dict[str, list[str]] = {}
    values: dict[str, list[np.ndarray]] = {}
    for sid in sorted(ids):
        group = str(group_of[sid])
        members.setdefault(group, []).append(sid)
        lengths = _clip_bone_lengths(
            _first_person_pose(cache_dir / split / f"{sid}.npz")
        )
        if lengths is not None:
            values.setdefault(group, []).append(lengths)

    templates: dict[str, np.ndarray] = {}
    contributing: dict[str, int] = {}
    for group, group_members in sorted(members.items()):
        rows = values.get(group, [])
        if not rows:
            raise ValueError(f"{split}/{group}: no usable skeleton clips")
        matrix = np.stack(rows)
        with np.errstate(all="ignore"):
            template = np.nanmedian(matrix, axis=0)
        template[0] = 1.0
        if (
            not np.isfinite(template).all()
            or np.any(template[1:] <= 1e-5)
        ):
            raise ValueError(f"{split}/{group}: incomplete bone template")
        templates[group] = template.astype(np.float32)
        contributing[group] = len(rows)
        if set(group_members) != {
            sid for sid in ids if str(group_of[sid]) == group
        }:
            raise AssertionError("template membership isolation failed")

    frozen_members = {
        group: tuple(sorted(group_members))
        for group, group_members in members.items()
    }
    return TemplateBundle(
        templates=templates,
        members=frozen_members,
        contributing_clips=contributing,
        digest=_template_digest(templates, frozen_members),
    )


def uniform_center_indices(length: int, steps: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(steps, dtype=np.int64)
    edges = np.linspace(0.0, length - 1e-6, steps + 1)
    return np.clip(
        ((edges[:-1] + edges[1:]) * 0.5).astype(np.int64), 0, length - 1
    )


def make_features(
    pose: np.ndarray | None,
    template: np.ndarray,
    config: FeatureConfig,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    config.validate()
    if pose is None or len(pose) == 0:
        return torch.zeros(config.channels, config.steps, N_JOINTS)
    if rng is not None and len(pose) > 4 and rng.random() < 0.5:
        keep = max(4, int(len(pose) * (0.4 + 0.6 * rng.random())))
        pose = pose[:keep]
    sampled = pose[uniform_center_indices(len(pose), config.steps)].astype(
        np.float32, copy=False
    )
    delta = np.diff(sampled, axis=0, prepend=sampled[:1])
    raw_bone = sampled - sampled[:, PARENTS]
    raw_bone[:, 0] = 0.0
    raw = np.concatenate((sampled, delta, raw_bone), axis=-1)
    if config.variant == "canonbone":
        canonical_bone = raw_bone / template[None, :, None]
        # Root has no anatomical parent and is exactly zero by contract.
        canonical_bone[:, 0] = 0.0
        features = np.concatenate((raw, canonical_bone), axis=-1)
    else:
        features = raw
    features = np.ascontiguousarray(features.transpose(2, 0, 1))
    expected = (config.channels, config.steps, N_JOINTS)
    if features.shape != expected or not np.isfinite(features).all():
        raise ValueError(f"invalid feature tensor {features.shape}, expected {expected}")
    return torch.from_numpy(features)


class CanonBoneDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        fold_file: Path,
        split: str,
        config: FeatureConfig,
        fold: int | None = None,
        part: str = "test",
        cohort_file: Path = DEFAULT_COHORTS,
        augment: bool = False,
        seed: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.config = config
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch_seed = int(seed)
        self.meta = read_metadata(self.cache_dir, split)
        if split == "train":
            if fold not in range(4) or part not in ("train", "val"):
                raise ValueError("train split requires fold 0..3 and part train|val")
            validation_users = read_fold_users(fold_file)[int(fold)]
            want_validation = part == "val"
            self.ids = tuple(sorted(
                sid for sid, row in self.meta.items()
                if (row["user"] in validation_users) == want_validation
            ))
            self.group_of = {sid: self.meta[sid]["user"] for sid in self.ids}
            self.labels = {sid: int(self.meta[sid]["class_id"]) for sid in self.ids}
        elif split == "test":
            self.ids = tuple(sorted(self.meta))
            full_mapping = read_test_cohorts(cohort_file, self.ids)
            self.group_of = {sid: full_mapping[sid] for sid in self.ids}
            self.labels = None
        else:
            raise ValueError("split must be train or test")
        if not self.ids:
            raise ValueError(f"empty dataset: {split}/{part}")
        self.bundle = build_templates(
            self.cache_dir, self.split, self.ids, self.group_of
        )

    def __len__(self) -> int:
        return len(self.ids)

    def _rng(self, sid: str) -> np.random.Generator | None:
        if not self.augment:
            return None
        value = (
            zlib.crc32(sid.encode())
            + self.epoch_seed * 1_000_003
            + self.seed * 97
        ) % (1 << 31)
        return np.random.default_rng(value)

    def __getitem__(self, index: int):
        sid = self.ids[index]
        pose = _first_person_pose(self.cache_dir / self.split / f"{sid}.npz")
        features = make_features(
            pose,
            self.bundle.templates[self.group_of[sid]],
            self.config,
            self._rng(sid),
        )
        label = -1 if self.labels is None else self.labels[sid]
        return features, label, sid


def h36m_adjacency_partitions() -> torch.Tensor:
    adjacency = torch.zeros(3, N_JOINTS, N_JOINTS)
    adjacency[0] = torch.eye(N_JOINTS)
    for parent, child in EDGES:
        adjacency[1, child, parent] = 1.0
        adjacency[2, parent, child] = 1.0
    return adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.0)


class AdaptiveSTGCNBlock(nn.Module):
    def __init__(
        self, channels_in: int, channels_out: int, adjacency: torch.Tensor,
        stride: int = 1,
    ):
        super().__init__()
        self.register_buffer("A", adjacency)
        self.PA = nn.Parameter(torch.zeros_like(adjacency))
        self.gconv = nn.Conv2d(
            channels_in, len(adjacency) * channels_out, 1, bias=False
        )
        self.pre = nn.Sequential(nn.GroupNorm(8, channels_out), nn.SiLU())
        quarter = channels_out // 4
        self.temporal = nn.ModuleList((
            nn.Conv2d(
                channels_out, quarter, (3, 1), (stride, 1), (1, 0), bias=False
            ),
            nn.Conv2d(
                channels_out, quarter, (3, 1), (stride, 1), (2, 0),
                dilation=(2, 1), bias=False,
            ),
            nn.Sequential(
                nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                nn.Conv2d(channels_out, quarter, 1, bias=False),
            ),
            nn.Conv2d(channels_out, quarter, 1, (stride, 1), bias=False),
        ))
        self.post = nn.GroupNorm(8, channels_out)
        hidden = max(8, channels_out // 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_out, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels_out, 1),
            nn.Sigmoid(),
        )
        self.res = (
            nn.Identity()
            if channels_in == channels_out and stride == 1
            else nn.Conv2d(
                channels_in, channels_out, 1, (stride, 1), bias=False
            )
        )
        self.act = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, time_steps, vertices = inputs.shape
        hidden = self.gconv(inputs).reshape(
            batch, len(self.A), -1, time_steps, vertices
        )
        hidden = torch.einsum(
            "bkctv,kvw->bctw", hidden, self.A + self.PA
        )
        hidden = self.pre(hidden)
        hidden = torch.cat([branch(hidden) for branch in self.temporal], dim=1)
        hidden = self.post(hidden)
        hidden = hidden * self.channel_gate(hidden)
        return self.act(hidden + self.res(inputs))


class AdaptiveSTGCN(nn.Module):
    """Accepted graph model generalized only at its input channel count."""

    def __init__(
        self, in_channels: int, width: int = WIDTH,
        classes: int = N_CLASSES, dropout: float = DROPOUT,
    ):
        super().__init__()
        if in_channels % 3 or width % 8:
            raise ValueError("input channels must divide 3 groups; width divides 8")
        adjacency = h36m_adjacency_partitions()
        self.in_channels = int(in_channels)
        self.input_norm = nn.GroupNorm(3, in_channels)
        self.blocks = nn.Sequential(
            AdaptiveSTGCNBlock(in_channels, width, adjacency),
            AdaptiveSTGCNBlock(width, width, adjacency),
            AdaptiveSTGCNBlock(width, 2 * width, adjacency, stride=2),
            AdaptiveSTGCNBlock(2 * width, 2 * width, adjacency),
            AdaptiveSTGCNBlock(2 * width, 4 * width, adjacency, stride=2),
            AdaptiveSTGCNBlock(4 * width, 4 * width, adjacency),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(8 * width, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [batch,{self.in_channels},time,17], "
                f"got {tuple(inputs.shape)}"
            )
        hidden = self.blocks(self.input_norm(inputs))
        pooled = torch.cat(
            (hidden.mean((2, 3)), hidden.amax(3).amax(2)), dim=-1
        )
        return self.head(self.drop(pooled))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=shuffle,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=generator,
    )


def fold_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predictions = probabilities.argmax(1)
    present = np.unique(labels)
    macro = np.mean([
        np.mean(predictions[labels == class_id] == class_id)
        for class_id in present
    ])
    return {
        "micro": float(np.mean(predictions == labels)),
        "macro_present": float(macro),
        "correct": int(np.sum(predictions == labels)),
        "samples": int(len(labels)),
    }


def evaluate_once(
    model: nn.Module,
    dataset: CanonBoneDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[dict[str, object], list[str]]:
    probabilities, labels, sids = [], [], []
    model.eval()
    loader = make_loader(dataset, batch_size, workers, device, False, 0)
    with torch.inference_mode():
        for features, targets, batch_sids in loader:
            logits = model(features.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(targets.numpy())
            sids.extend(batch_sids)
    probs = np.concatenate(probabilities)
    targets = np.concatenate(labels).astype(np.int64)
    return fold_metrics(probs, targets), sids


def checkpoint_paths(
    directory: Path, tag: str, fold: int
) -> tuple[Path, Path, Path]:
    checkpoint = directory / f"{tag}_f{fold}.pt"
    return (
        checkpoint,
        directory / f"{tag}_f{fold}.meta.json",
        directory / f"{tag}_f{fold}.sha256",
    )


def ensure_targets_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "no-overwrite policy: target already exists: "
            + ", ".join(map(str, existing))
        )


def fold_partition_audit(
    cache_dir: Path, fold_file: Path, fold: int
) -> dict[str, object]:
    meta = read_metadata(cache_dir, "train")
    validation_users = read_fold_users(fold_file)[fold]
    validation_ids = sorted(
        sid for sid, row in meta.items() if row["user"] in validation_users
    )
    train_ids = sorted(set(meta) - set(validation_ids))
    train_users = {meta[sid]["user"] for sid in train_ids}
    actual_validation_users = {meta[sid]["user"] for sid in validation_ids}
    if (
        set(train_ids) & set(validation_ids)
        or train_users & actual_validation_users
        or actual_validation_users != validation_users
    ):
        raise RuntimeError(f"fold {fold}: subject/sample leakage")
    return {
        "fold": fold,
        "train_samples": len(train_ids),
        "validation_samples": len(validation_ids),
        "train_users": sorted(train_users),
        "validation_users": sorted(actual_validation_users),
        "train_ids_sha256": ids_sha256(train_ids),
        "validation_ids_sha256": ids_sha256(validation_ids),
        "sample_overlap": 0,
        "subject_overlap": 0,
    }


def train_fold(args: argparse.Namespace, fold: int) -> dict[str, object]:
    checkpoint, sidecar, hash_path = checkpoint_paths(
        args.checkpoint_dir, args.tag, fold
    )
    ensure_targets_absent((checkpoint, sidecar, hash_path))
    partition = fold_partition_audit(args.cache_dir, args.fold_file, fold)
    run_seed = args.seed + fold * 10_007
    seed_everything(run_seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    config = FeatureConfig(args.variant, args.steps)
    config.validate()

    # Only optimizer-side subjects are touched before optimization.
    training = CanonBoneDataset(
        args.cache_dir, args.fold_file, "train", config, fold, "train",
        args.cohort_file, augment=True, seed=run_seed,
    )
    if set(training.bundle.templates) & set(partition["validation_users"]):
        raise RuntimeError("validation template entered optimizer-side dataset")
    train_loader = make_loader(
        training, args.batch_size, args.workers, device, True, run_seed
    )
    if not len(train_loader):
        raise RuntimeError("training loader has no complete batch")

    model = AdaptiveSTGCN(config.channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    started = time.time()
    skipped_steps = 0
    final_loss = math.nan
    print(
        f"fold {fold}: variant={config.variant} train={len(training)} "
        f"clips/{len(training.bundle.templates)} subjects; params={parameters:,}; "
        f"device={device}; amp={amp_enabled}; outer validation untouched",
        flush=True,
    )
    for epoch in range(EPOCHS):
        model.train()
        training.epoch_seed = run_seed * 1_000 + epoch
        loss_sum = examples = 0
        for features, targets, _ in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                loss = criterion(model(features), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler lowers its scale exactly when optimizer.step is skipped.
            # OneCycle must advance only after a real optimizer update.
            if not amp_enabled or scaler.get_scale() >= scale_before:
                scheduler.step()
            else:
                skipped_steps += 1
            loss_sum += float(loss.detach()) * len(targets)
            examples += len(targets)
        final_loss = loss_sum / examples
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == EPOCHS:
            print(
                f"  fold{fold} epoch {epoch + 1}/{EPOCHS}: "
                f"train_loss={final_loss:.5f}; skipped_amp_steps={skipped_steps} "
                "(outer validation untouched)",
                flush=True,
            )

    # First construction/access of held-out skeleton features occurs here.
    validation = CanonBoneDataset(
        args.cache_dir, args.fold_file, "train", config, fold, "val",
        args.cohort_file, augment=False, seed=run_seed,
    )
    if set(training.bundle.templates) & set(validation.bundle.templates):
        raise RuntimeError("train and validation template subjects overlap")
    metrics, validation_sids = evaluate_once(
        model, validation, args.batch_size, max(0, args.workers // 2), device
    )
    if ids_sha256(validation_sids) != partition["validation_ids_sha256"]:
        raise RuntimeError("outer-validation row lineage changed")

    package = {
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "feature_config": asdict(config),
        "model_config": {
            "in_channels": config.channels,
            "width": WIDTH,
            "classes": N_CLASSES,
            "dropout": DROPOUT,
        },
        "fold": fold,
        "seed": run_seed,
        "recipe": {
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "max_lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "scheduler": "OneCycleLR",
            "augmentation": "prefix truncation only",
            "outer_validation_evaluations": 1,
        },
        "lineage": {
            "feature_version": FEATURE_VERSION,
            "fold_file_sha256": sha256_file(args.fold_file),
            "source_sha256": sha256_file(Path(__file__)),
            "train_ids_sha256": partition["train_ids_sha256"],
            "validation_ids_sha256": partition["validation_ids_sha256"],
            "train_template_sha256": training.bundle.digest,
            "validation_template_sha256": validation.bundle.digest,
        },
        "outer_metrics": metrics,
    }
    atomic_torch_save(checkpoint, package)
    checkpoint_hash = sha256_file(checkpoint)
    metadata = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_kind": CHECKPOINT_KIND,
        "checkpoint_version": CHECKPOINT_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_config": asdict(config),
        "fold": fold,
        "parameters": parameters,
        "partition": partition,
        "train_template_sha256": training.bundle.digest,
        "validation_template_sha256": validation.bundle.digest,
        "outer_metrics": metrics,
        "outer_validation_evaluations": 1,
        "skipped_amp_steps": skipped_steps,
        "final_train_loss": final_loss,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "source_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(sidecar, metadata)
    atomic_text(hash_path, f"{checkpoint_hash}  {checkpoint.name}\n")
    print(
        f"fold {fold} OUTER-ONCE: micro={metrics['micro']:.5f} "
        f"macro={metrics['macro_present']:.5f} "
        f"({metrics['correct']}/{metrics['samples']}); "
        f"checkpoint={checkpoint.stat().st_size / 1e6:.2f} MB "
        f"sha256={checkpoint_hash}",
        flush=True,
    )
    return metadata


def smoke_command(_: argparse.Namespace) -> None:
    rng = np.random.default_rng(7)
    pose = rng.normal(size=(19, N_JOINTS, 3)).astype(np.float32)
    pose[:, 0] = rng.normal(size=(19, 3))
    lengths = _clip_bone_lengths(pose)
    if lengths is None:
        raise AssertionError("synthetic template failed")
    raw_config = FeatureConfig("raw", 32)
    candidate_config = FeatureConfig("canonbone", 32)
    raw = make_features(pose, lengths.astype(np.float32), raw_config)
    candidate = make_features(pose, lengths.astype(np.float32), candidate_config)
    if raw.shape != (9, 32, 17) or candidate.shape != (12, 32, 17):
        raise AssertionError("feature shape contract failed")
    if not torch.equal(raw, candidate[:RAW_CHANNELS]):
        raise AssertionError("candidate did not preserve the exact raw 9 channels")
    if not torch.equal(candidate[9:, :, 0], torch.zeros_like(candidate[9:, :, 0])):
        raise AssertionError("canonical root bone is not exactly zero")
    scaled = make_features(
        pose * 2.0, lengths.astype(np.float32) * 2.0, candidate_config
    )
    if not torch.allclose(candidate[9:], scaled[9:], atol=2e-6):
        raise AssertionError("subject-normalized bone is not scale invariant")

    raw_model = AdaptiveSTGCN(RAW_CHANNELS)
    candidate_model = AdaptiveSTGCN(CANONBONE_CHANNELS)
    if sum(parameter.numel() for parameter in raw_model.parameters()) != 841_596:
        raise AssertionError("9-channel control no longer matches accepted model size")
    criterion = nn.CrossEntropyLoss()
    raw_logits = raw_model(raw.unsqueeze(0))
    candidate_logits = candidate_model(candidate.unsqueeze(0))
    loss = criterion(raw_logits, torch.tensor([3]))
    loss = loss + criterion(candidate_logits, torch.tensor([3]))
    loss.backward()
    if (
        raw_logits.shape != (1, N_CLASSES)
        or candidate_logits.shape != (1, N_CLASSES)
        or not torch.isfinite(loss)
    ):
        raise AssertionError("model finite-forward/backward contract failed")

    try:
        validate_group_mapping({"a": "E1"}, ("a", "b"), TEST_COHORTS)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete cohort mapping was not rejected")
    print(
        "SMOKE PASSED: raw-channel identity, root-zero canonical bone, "
        "scale invariance, exact 841596-param raw control, finite backward, "
        "and incomplete-cohort rejection",
        flush=True,
    )


def audit_command(args: argparse.Namespace) -> None:
    folds = parse_folds(args.folds)
    train_meta = read_metadata(args.cache_dir, "train")
    test_meta = read_metadata(args.cache_dir, "test")
    train_groups = {sid: row["user"] for sid, row in train_meta.items()}
    test_groups = read_test_cohorts(args.cohort_file, test_meta)
    train_bundle = build_templates(
        args.cache_dir, "train", tuple(sorted(train_meta)), train_groups
    )
    test_bundle = build_templates(
        args.cache_dir, "test", tuple(sorted(test_meta)), test_groups
    )
    for fold in folds:
        record = fold_partition_audit(args.cache_dir, args.fold_file, fold)
        train_users = set(record["train_users"])
        validation_users = set(record["validation_users"])
        if train_users & validation_users:
            raise RuntimeError(f"fold {fold}: template group leakage")
        print(
            f"fold {fold}: train={record['train_samples']}/"
            f"{len(train_users)} subjects; outer-val="
            f"{record['validation_samples']}/{len(validation_users)} subjects; "
            "sample_overlap=0 subject_overlap=0",
            flush=True,
        )
    cohort_counts = {
        cohort: sum(group == cohort for group in test_groups.values())
        for cohort in sorted(TEST_COHORTS)
    }
    minima = [
        float(template[1:].min())
        for template in list(train_bundle.templates.values())
        + list(test_bundle.templates.values())
    ]
    maxima = [
        float(template[1:].max())
        for template in list(train_bundle.templates.values())
        + list(test_bundle.templates.values())
    ]
    print(
        f"train templates: {len(train_bundle.templates)} subjects, "
        f"sha256={train_bundle.digest}; test templates: "
        f"{cohort_counts}, sha256={test_bundle.digest}",
        flush=True,
    )
    print(
        f"AUDIT PASSED: all {len(train_meta)} train and {len(test_meta)} test "
        f"IDs mapped exactly once; group-local median bone range="
        f"[{min(minima):.6f}, {max(maxima):.6f}]; labels were not accepted by "
        "the template builder",
        flush=True,
    )


def train_command(args: argparse.Namespace) -> None:
    folds = parse_folds(args.folds)
    config = FeatureConfig(args.variant, args.steps)
    config.validate()
    if not args.tag or any(character in args.tag for character in "/\\"):
        raise ValueError("--tag must be a filename-safe non-empty value")
    manifest = args.artifact_dir / (
        f"run_canonbone_{args.tag}_f{''.join(map(str, folds))}.json"
    )
    targets: list[Path] = [manifest]
    for fold in folds:
        targets.extend(checkpoint_paths(args.checkpoint_dir, args.tag, fold))
    ensure_targets_absent(targets)
    results = [train_fold(args, fold) for fold in folds]
    payload = {
        "experiment": "raw-preserving subject-canonical bone screen",
        "feature_version": FEATURE_VERSION,
        "tag": args.tag,
        "variant": args.variant,
        "folds": list(folds),
        "fixed_recipe": {
            "epochs": EPOCHS,
            "max_lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "augmentation": "truncation only",
            "outer_validation_evaluations_per_fold": 1,
        },
        "results": results,
        "source_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(manifest, payload)
    print(f"run manifest: {manifest}", flush=True)


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fold-file", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cohort-file", type=Path, default=DEFAULT_COHORTS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="synthetic feature/model checks")
    smoke.set_defaults(function=smoke_command)

    audit = commands.add_parser(
        "audit", help="real-data fold/template/cohort leakage audit; writes nothing"
    )
    add_paths(audit)
    audit.add_argument("--folds", default="0,2")
    audit.set_defaults(function=audit_command)

    train = commands.add_parser(
        "train", help="fixed 60-epoch screen and one outer evaluation per fold"
    )
    add_paths(train)
    train.add_argument("--tag", default="canonbone_v1")
    train.add_argument("--variant", choices=("canonbone", "raw"), default="canonbone")
    train.add_argument("--folds", default="0,2")
    train.add_argument("--steps", type=int, default=32)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--workers", type=int, default=6)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--amp", action="store_true")
    train.set_defaults(function=train_command)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if hasattr(args, "batch_size") and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if hasattr(args, "workers") and args.workers < 0:
        raise ValueError("--workers cannot be negative")
    args.function(args)


if __name__ == "__main__":
    main()
