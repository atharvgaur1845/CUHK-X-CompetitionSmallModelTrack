#!/usr/bin/env python3
"""Leakage-safe dual-frame skeleton/kinematics experiment.

This branch deliberately keeps two complementary coordinate systems:

* ``raw`` preserves the station-aligned position, velocity, and bone cues that
  already transfer in this dataset.
* ``canonical`` removes clip-level yaw and subject scale, then exposes physical
  velocity/acceleration, unit bones, joint angles, and endpoint distances.

Canonicalization is per clip.  It never estimates statistics from another
sample, a validation user, or the test split.

The default screen trains on folds 0 and 2 with a fixed epoch budget and
evaluates each outer validation fold once, after optimization is finished:

    python code/train_dualframe.py screen --audit-only
    python code/train_dualframe.py screen --folds 0,2 \
      --baseline-oof research/artifacts/oof_tjitter_v2_p3_astgcn_v1.npz \
      --baseline-key clean_probs

Complete the remaining folds, then emit artifacts understood by the existing
fusion code (``probs``, ``labels``, and ``sids``):

    python code/train_dualframe.py screen --folds 1,3
    python code/train_dualframe.py oof
    python code/train_dualframe.py test

An exact-architecture raw-only control is available with ``--variant raw``.
Use a different tag for it.  Its canonical/global inputs are masked to zero,
while the parameterization and training recipe remain unchanged.
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
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import har_data


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache")
ARTIFACTS = os.path.join(ROOT, "research", "artifacts")
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
FOLD_PATH = os.path.join(ARTIFACTS, "cv_folds.json")
FEATURE_VERSION = "dualframe-kinematics-v1"
N_CLASSES = 40
N_JOINTS = 17

# Human3.6M topology used throughout the repository.
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

# (endpoint, vertex, endpoint) triples.  Cosines avoid angle wraparound.
ANGLE_TRIPLES = (
    (0, 1, 2), (1, 2, 3),
    (0, 4, 5), (4, 5, 6),
    (0, 7, 8), (7, 8, 9), (8, 9, 10),
    (7, 8, 11), (8, 11, 12), (11, 12, 13),
    (7, 8, 14), (8, 14, 15), (14, 15, 16),
)

# Scale-normalized distances that directly describe hand, foot, and head pose.
DISTANCE_PAIRS = (
    (13, 16), (13, 10), (16, 10),
    (13, 0), (16, 0), (10, 0),
    (3, 6), (3, 0), (6, 0),
    (13, 3), (16, 6), (13, 6), (16, 3),
)

RAW_CHANNELS = 9                 # xyz + frame delta + bone
CANONICAL_CHANNELS = 12          # xyz + physical v + physical a + unit bone
GLOBAL_CHANNELS = (
    len(ANGLE_TRIPLES)
    + len(DISTANCE_PAIRS)
    + 2                           # sin/cos normalized phase
    + 1                           # log duration
    + 1                           # skeleton present
    + 1                           # multi-person level
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_digest(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str, payload: Mapping) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def atomic_torch_save(path: str, state: Mapping[str, torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path)
    )
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def atomic_npz(path: str, **arrays: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp.npz",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


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
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(value)


def parse_folds(value: str) -> tuple[int, ...]:
    try:
        folds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"invalid fold list {value!r}") from exc
    if not folds:
        raise ValueError("at least one fold is required")
    if len(folds) != len(set(folds)):
        raise ValueError(f"duplicate fold in {value!r}")
    if any(fold not in range(4) for fold in folds):
        raise ValueError("folds must be drawn from 0,1,2,3")
    return folds


def checkpoint_path(tag: str, fold: int) -> str:
    return os.path.join(CHECKPOINTS, f"{tag}_f{fold}.pt")


def checkpoint_meta_path(tag: str, fold: int) -> str:
    return os.path.join(CHECKPOINTS, f"{tag}_f{fold}.meta.json")


def load_tensor_state(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path}: expected a non-empty state-dict")
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise ValueError(f"{path}: checkpoint is not a plain tensor state-dict")
    return state


@dataclass(frozen=True)
class BranchConfig:
    tag: str
    width: int
    t_steps: int
    variant: str
    seed: int

    def validate(self) -> None:
        if not self.tag or any(char in self.tag for char in "/\\"):
            raise ValueError("--tag must be a non-empty filename-safe name")
        if self.width < 16 or self.width % 8:
            raise ValueError("--width must be >=16 and divisible by 8")
        if self.t_steps < 8:
            raise ValueError("--t-steps must be at least 8")
        if self.variant not in ("dual", "raw", "canonical"):
            raise ValueError("--variant must be dual, raw, or canonical")


def load_fold_definition() -> dict:
    with open(FOLD_PATH) as handle:
        definition = json.load(handle)
    folds = definition.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise ValueError(f"{FOLD_PATH}: expected exactly four folds")
    seen: set[str] = set()
    always = set(definition.get("always_train", ()))
    for expected, record in enumerate(folds):
        if int(record.get("fold", -1)) != expected:
            raise ValueError(f"{FOLD_PATH}: fold order/id mismatch at {expected}")
        users = set(record.get("val_users", ()))
        if not users:
            raise ValueError(f"{FOLD_PATH}: fold {expected} has no validation users")
        overlap = seen & users
        if overlap:
            raise ValueError(
                f"{FOLD_PATH}: validation users repeated across folds: {sorted(overlap)}"
            )
        if users & always:
            raise ValueError(
                f"{FOLD_PATH}: always-train user appears in validation: "
                f"{sorted(users & always)}"
            )
        seen.update(users)
    return definition


def _normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


def _safe_timestamps(raw: np.ndarray | None, n_frames: int) -> np.ndarray:
    if raw is None or len(raw) != n_frames:
        return np.arange(n_frames, dtype=np.float64) * 0.1
    timestamps = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(timestamps).all():
        return np.arange(n_frames, dtype=np.float64) * 0.1
    ordered = np.sort(timestamps, kind="stable")
    if n_frames > 1 and np.diff(ordered).max(initial=0.0) > 100.0:
        # Defensive unit fallback.  Normal cache timestamps are epoch seconds
        # spaced by about 0.1 s; a huge step indicates malformed metadata.
        return np.arange(n_frames, dtype=np.float64) * 0.1
    return timestamps


def _resample_pose(
    pose: np.ndarray,
    timestamps: np.ndarray,
    steps: int,
    temporal_jitter: bool,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Linearly resample a pose while retaining a physical-time grid."""
    if len(pose) == 0:
        return (
            np.zeros((steps, N_JOINTS, 3), dtype=np.float32),
            np.arange(steps, dtype=np.float64) * 0.1,
            0.0,
        )

    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    pose = pose[order]
    timestamps, unique_index = np.unique(timestamps, return_index=True)
    pose = pose[unique_index]
    if len(pose) == 1 or timestamps[-1] - timestamps[0] < 1e-5:
        return (
            np.repeat(pose[:1].astype(np.float32), steps, axis=0),
            np.arange(steps, dtype=np.float64) * 0.1,
            0.0,
        )

    edges = np.linspace(timestamps[0], timestamps[-1], steps + 1)
    if temporal_jitter:
        if rng is None:
            raise ValueError("temporal jitter requires an RNG")
        query = edges[:-1] + rng.random(steps) * np.diff(edges)
    else:
        query = (edges[:-1] + edges[1:]) * 0.5

    flat = pose.reshape(len(pose), -1)
    sampled = np.empty((steps, flat.shape[1]), dtype=np.float32)
    for channel in range(flat.shape[1]):
        sampled[:, channel] = np.interp(query, timestamps, flat[:, channel])
    duration = float(timestamps[-1] - timestamps[0])
    return sampled.reshape(steps, N_JOINTS, 3), query, duration


def _physical_derivatives(
    coordinates: np.ndarray,
    query: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    if duration <= 1e-5:
        zeros = np.zeros_like(coordinates, dtype=np.float32)
        return zeros, zeros.copy()
    edge_order = 2 if len(query) >= 3 else 1
    velocity = np.gradient(
        coordinates.astype(np.float64), query, axis=0, edge_order=edge_order
    )
    acceleration = np.gradient(velocity, query, axis=0, edge_order=edge_order)
    # Pose lifting can create single-frame spikes.  Fixed physical clips avoid
    # fold-derived clipping constants and make the transform split-independent.
    velocity = np.clip(velocity, -12.0, 12.0)
    acceleration = np.clip(acceleration, -60.0, 60.0)
    return velocity.astype(np.float32), acceleration.astype(np.float32)


def _body_frame(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a stable clip-level rotation and mean-bone-length scale."""
    # H36M left-minus-right hip and shoulder vectors.  Removing z makes the
    # vertical axis exactly the known floor-aligned world z.
    lateral_by_frame = 0.5 * (
        (pose[:, 4] - pose[:, 1]) + (pose[:, 11] - pose[:, 14])
    )
    lateral_by_frame[:, 2] = 0.0
    norms = np.linalg.norm(lateral_by_frame, axis=1)
    valid = norms > 1e-4
    if valid.any():
        unit = lateral_by_frame[valid] / norms[valid, None]
        lateral = unit.sum(axis=0)
        # A near-zero resultant means the person turned about 180 degrees.
        # The middle valid frame is deterministic and anatomically signed.
        if np.linalg.norm(lateral) < 0.2 * len(unit):
            lateral = unit[len(unit) // 2]
    else:
        lateral = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    lateral[2] = 0.0
    lateral = _normalize(lateral[None])[0]
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    forward = _normalize(np.cross(up, lateral)[None])[0]
    # Row-vector coordinates use pose @ rotation to obtain dot products with
    # [lateral, forward, up].
    rotation = np.stack((lateral, forward, up), axis=1).astype(np.float32)

    lengths = np.stack(
        [np.linalg.norm(pose[:, child] - pose[:, parent], axis=-1)
         for parent, child in EDGES],
        axis=1,
    )
    per_frame = lengths.mean(axis=1)
    finite = per_frame[np.isfinite(per_frame) & (per_frame > 1e-4)]
    scale = float(np.median(finite)) if len(finite) else 0.25
    return rotation, max(scale, 0.05)


def _joint_cosines(coordinates: np.ndarray) -> np.ndarray:
    values = []
    for left, vertex, right in ANGLE_TRIPLES:
        first = coordinates[:, left] - coordinates[:, vertex]
        second = coordinates[:, right] - coordinates[:, vertex]
        first = _normalize(first)
        second = _normalize(second)
        values.append(np.sum(first * second, axis=-1))
    return np.stack(values, axis=-1).astype(np.float32)


def _endpoint_distances(coordinates: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.linalg.norm(coordinates[:, first] - coordinates[:, second], axis=-1)
         for first, second in DISTANCE_PAIRS],
        axis=-1,
    ).astype(np.float32)


def make_dualframe_features(
    pose: np.ndarray,
    timestamps: np.ndarray,
    person_counts: np.ndarray | None,
    steps: int,
    temporal_jitter: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Pure per-clip transform used by both training and inference."""
    sampled, query, duration = _resample_pose(
        pose, timestamps, steps, temporal_jitter, rng
    )
    rotation, scale = _body_frame(sampled)

    raw_delta = np.diff(sampled, axis=0, prepend=sampled[:1])
    raw_bone = sampled - sampled[:, PARENTS]
    raw = np.concatenate((sampled, raw_delta, raw_bone), axis=-1)

    centered = sampled - sampled[:, :1]
    canonical = (centered @ rotation) / scale
    velocity, acceleration = _physical_derivatives(canonical, query, duration)
    canonical_bone = canonical - canonical[:, PARENTS]
    unit_bone = _normalize(canonical_bone)
    unit_bone[:, 0] = 0.0
    canonical_features = np.concatenate(
        (canonical, velocity, acceleration, unit_bone), axis=-1
    )

    angles = _joint_cosines(canonical)
    distances = _endpoint_distances(canonical)
    phase = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    phase_features = np.stack(
        (np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)), axis=-1
    )
    duration_feature = np.full(
        (steps, 1),
        np.log1p(min(max(duration, 0.0), 10.0)) / np.log(11.0),
        dtype=np.float32,
    )
    present = np.ones((steps, 1), dtype=np.float32)
    if person_counts is None or len(person_counts) == 0:
        multi_level = 0.0
    else:
        multi_level = float(np.clip(np.median(person_counts) - 1.0, 0.0, 2.0) / 2.0)
    multi = np.full((steps, 1), multi_level, dtype=np.float32)
    global_features = np.concatenate(
        (angles, distances, phase_features, duration_feature, present, multi), axis=-1
    )

    arrays = {
        "raw": np.ascontiguousarray(raw.transpose(2, 0, 1), dtype=np.float32),
        "canonical": np.ascontiguousarray(
            canonical_features.transpose(2, 0, 1), dtype=np.float32
        ),
        "global": np.ascontiguousarray(global_features.T, dtype=np.float32),
    }
    expected = {
        "raw": (RAW_CHANNELS, steps, N_JOINTS),
        "canonical": (CANONICAL_CHANNELS, steps, N_JOINTS),
        "global": (GLOBAL_CHANNELS, steps),
    }
    for key, value in arrays.items():
        if value.shape != expected[key]:
            raise RuntimeError(f"{key} shape {value.shape}, expected {expected[key]}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite {key} feature")
    return {key: torch.from_numpy(value) for key, value in arrays.items()}


def empty_dualframe_features(steps: int) -> dict[str, torch.Tensor]:
    return {
        "raw": torch.zeros(RAW_CHANNELS, steps, N_JOINTS),
        "canonical": torch.zeros(CANONICAL_CHANNELS, steps, N_JOINTS),
        "global": torch.zeros(GLOBAL_CHANNELS, steps),
    }


class DualFrameSkeletonDataset(Dataset):
    """Subject-folded dataset with deterministic, epoch-varying augmentation."""

    def __init__(
        self,
        split: str,
        fold: int | None = None,
        part: str = "train",
        t_steps: int = 32,
        aug_spec: str = "",
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if split == "train" and fold not in range(4):
            raise ValueError("training data requires fold 0..3")
        self.split = split
        self.fold = fold
        self.part = part
        self.t_steps = int(t_steps)
        self.aug_set = {part.strip() for part in aug_spec.split(",") if part.strip()}
        unsupported = self.aug_set - {"trunc", "tjit"}
        if unsupported:
            raise ValueError(
                f"unsupported augmentation(s) {sorted(unsupported)}; "
                "only trunc,tjit are representation-safe here"
            )
        self.seed = int(seed)
        self.epoch_seed = int(seed)

        # Reuse the repository's authoritative ID/fold construction, but load
        # raw poses ourselves so no baseline representation is silently mixed in.
        index = har_data.HARDataset(
            split,
            "skel",
            fold=fold,
            part=part,
            t_skel=t_steps,
            aug=False,
            person="first",
        )
        self.ids = tuple(index.ids)
        self.meta = index.meta
        self.labels = index.labels

    def __len__(self) -> int:
        return len(self.ids)

    def _rng(self, sid: str) -> np.random.Generator | None:
        if not self.aug_set:
            return None
        value = (
            zlib.crc32(sid.encode("utf-8"))
            + self.epoch_seed * 1_000_003
            + self.seed * 97
        ) % (1 << 31)
        return np.random.default_rng(value)

    @staticmethod
    def _first_person(archive) -> np.ndarray | None:
        if "skel_pos" not in archive.files or len(archive["skel_pos"]) == 0:
            return None
        return archive["skel_pos"][:, 0].astype(np.float32)

    def __getitem__(self, index: int):
        sid = self.ids[index]
        rng = self._rng(sid)
        path = os.path.join(CACHE, self.split, sid + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            pose = self._first_person(archive)
            if pose is None:
                features = empty_dualframe_features(self.t_steps)
            else:
                raw_time = archive["skel_t"] if "skel_t" in archive.files else None
                timestamps = _safe_timestamps(raw_time, len(pose))
                counts = (
                    archive["skel_np"].astype(np.float32)
                    if "skel_np" in archive.files else None
                )
                if (
                    "trunc" in self.aug_set
                    and rng is not None
                    and len(pose) > 4
                    and rng.random() < 0.5
                ):
                    keep = max(4, int(len(pose) * (0.4 + 0.6 * rng.random())))
                    pose = pose[:keep]
                    timestamps = timestamps[:keep]
                    if counts is not None:
                        counts = counts[:keep]
                features = make_dualframe_features(
                    pose,
                    timestamps,
                    counts,
                    self.t_steps,
                    temporal_jitter="tjit" in self.aug_set,
                    rng=rng,
                )
        label = self.labels[sid] if self.labels is not None else -1
        return features, label, sid


def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def h36m_adjacency() -> torch.Tensor:
    adjacency = torch.zeros(3, N_JOINTS, N_JOINTS)
    adjacency[0] = torch.eye(N_JOINTS)
    for parent, child in EDGES:
        adjacency[1, child, parent] = 1.0
        adjacency[2, parent, child] = 1.0
    return adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.0)


class AdaptiveGraphBlock(nn.Module):
    """Partitioned graph convolution plus multi-scale temporal convolution."""

    def __init__(self, channels_in: int, channels_out: int, adjacency: torch.Tensor,
                 stride: int = 1):
        super().__init__()
        if channels_out % 4:
            raise ValueError("graph block output width must be divisible by four")
        self.register_buffer("adjacency", adjacency.clone())
        self.adaptive = nn.Parameter(torch.zeros_like(adjacency))
        self.graph = nn.Conv2d(
            channels_in, len(adjacency) * channels_out, 1, bias=False
        )
        self.pre = nn.Sequential(
            nn.GroupNorm(group_count(channels_out), channels_out),
            nn.SiLU(),
        )
        quarter = channels_out // 4
        self.temporal = nn.ModuleList(
            (
                nn.Conv2d(
                    channels_out, quarter, (3, 1), (stride, 1), (1, 0), bias=False
                ),
                nn.Conv2d(
                    channels_out,
                    quarter,
                    (3, 1),
                    (stride, 1),
                    (2, 0),
                    dilation=(2, 1),
                    bias=False,
                ),
                nn.Sequential(
                    nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                    nn.Conv2d(channels_out, quarter, 1, bias=False),
                ),
                nn.Conv2d(channels_out, quarter, 1, (stride, 1), bias=False),
            )
        )
        self.post = nn.GroupNorm(group_count(channels_out), channels_out)
        hidden = max(8, channels_out // 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_out, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels_out, 1),
            nn.Sigmoid(),
        )
        self.residual = (
            nn.Identity()
            if channels_in == channels_out and stride == 1
            else nn.Conv2d(channels_in, channels_out, 1, (stride, 1), bias=False)
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, time_steps, vertices = inputs.shape
        hidden = self.graph(inputs).reshape(
            batch, len(self.adjacency), -1, time_steps, vertices
        )
        hidden = torch.einsum(
            "bkctv,kvw->bctw", hidden, self.adjacency + self.adaptive
        )
        hidden = self.pre(hidden)
        hidden = torch.cat([branch(hidden) for branch in self.temporal], dim=1)
        hidden = self.post(hidden)
        hidden = hidden * self.gate(hidden)
        return self.activation(hidden + self.residual(inputs))


class GraphEncoder(nn.Module):
    def __init__(self, channels_in: int, width: int):
        super().__init__()
        adjacency = h36m_adjacency()
        self.input_norm = nn.GroupNorm(group_count(channels_in), channels_in)
        self.blocks = nn.Sequential(
            AdaptiveGraphBlock(channels_in, width, adjacency),
            AdaptiveGraphBlock(width, width, adjacency),
            AdaptiveGraphBlock(width, 2 * width, adjacency, stride=2),
            AdaptiveGraphBlock(2 * width, 2 * width, adjacency),
            AdaptiveGraphBlock(2 * width, 4 * width, adjacency, stride=2),
            AdaptiveGraphBlock(4 * width, 4 * width, adjacency),
        )
        self.output_dim = 8 * width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.input_norm(inputs))
        return torch.cat(
            (hidden.mean(dim=(2, 3)), hidden.amax(dim=3).amax(dim=2)), dim=-1
        )


def temporal_block(channels_in: int, channels_out: int, stride: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(
            channels_in, channels_out, 5, stride=stride, padding=2, bias=False
        ),
        nn.GroupNorm(group_count(channels_out), channels_out),
        nn.SiLU(),
    )


class GlobalKinematicsEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            temporal_block(GLOBAL_CHANNELS, width),
            temporal_block(width, width, stride=2),
            temporal_block(width, 2 * width),
            temporal_block(2 * width, 2 * width, stride=2),
        )
        self.output_dim = 4 * width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.body(inputs)
        return torch.cat((hidden.mean(-1), hidden.amax(-1)), dim=-1)


class DualFrameKinematicsNet(nn.Module):
    """Raw station graph + canonical kinematics graph with compact late fusion."""

    def __init__(
        self,
        width: int = 64,
        n_classes: int = N_CLASSES,
        drop: float = 0.3,
        variant: str = "dual",
    ):
        super().__init__()
        if variant not in ("dual", "raw", "canonical"):
            raise ValueError("variant must be dual, raw, or canonical")
        self.variant = variant
        self.raw_encoder = GraphEncoder(RAW_CHANNELS, width)
        self.canonical_encoder = GraphEncoder(CANONICAL_CHANNELS, width)
        self.global_encoder = GlobalKinematicsEncoder(width)
        graph_dim = self.raw_encoder.output_dim
        fusion_in = 4 * graph_dim + self.global_encoder.output_dim
        fusion_dim = 4 * width
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
            nn.Dropout(drop),
        )
        self.head = nn.Linear(fusion_dim, n_classes)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raw = inputs["raw"]
        canonical = inputs["canonical"]
        global_features = inputs["global"]
        if self.variant == "raw":
            canonical = torch.zeros_like(canonical)
            global_features = torch.zeros_like(global_features)
        elif self.variant == "canonical":
            raw = torch.zeros_like(raw)

        raw_embedding = self.raw_encoder(raw)
        canonical_embedding = self.canonical_encoder(canonical)
        global_embedding = self.global_encoder(global_features)
        interactions = torch.cat(
            (
                raw_embedding,
                canonical_embedding,
                torch.abs(raw_embedding - canonical_embedding),
                torch.tanh(raw_embedding) * torch.tanh(canonical_embedding),
                global_embedding,
            ),
            dim=-1,
        )
        return self.head(self.fusion(interactions))


def move_features(
    features: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in features.items()
    }


def model_for(config: BranchConfig, device: torch.device) -> DualFrameKinematicsNet:
    config.validate()
    return DualFrameKinematicsNet(
        width=config.width, variant=config.variant
    ).to(device)


def audit_fold(
    fold: int,
    t_steps: int,
    sample_count: int = 8,
    exercise_validation_features: bool = False,
) -> dict:
    definition = load_fold_definition()
    expected_val_users = set(definition["folds"][fold]["val_users"])
    train = DualFrameSkeletonDataset("train", fold, "train", t_steps=t_steps)
    validation = DualFrameSkeletonDataset("train", fold, "val", t_steps=t_steps)
    train_ids, validation_ids = set(train.ids), set(validation.ids)
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(
            f"fold {fold}: train/validation sample overlap; first={sorted(overlap)[0]}"
        )
    train_users = {train.meta[sid]["user"] for sid in train.ids}
    validation_users = {validation.meta[sid]["user"] for sid in validation.ids}
    if validation_users != expected_val_users:
        raise RuntimeError(
            f"fold {fold}: validation users {sorted(validation_users)} != "
            f"definition {sorted(expected_val_users)}"
        )
    if train_users & validation_users:
        raise RuntimeError(
            f"fold {fold}: user leakage {sorted(train_users & validation_users)}"
        )
    if any(sid not in train.labels for sid in train.ids):
        raise RuntimeError(f"fold {fold}: missing training label")
    if any(sid not in validation.labels for sid in validation.ids):
        raise RuntimeError(f"fold {fold}: missing validation label")

    # Normal training screens exercise only optimizer-facing training features
    # here.  ``--audit-only`` additionally checks held-out feature loading, but
    # that command creates neither an optimizer nor an artifact.
    checked_datasets = (train, validation) if exercise_validation_features else (train,)
    for dataset in checked_datasets:
        if len(dataset):
            indices = np.linspace(
                0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int
            )
            for index in np.unique(indices):
                features, label, sid = dataset[int(index)]
                expected = {
                    "raw": (RAW_CHANNELS, t_steps, N_JOINTS),
                    "canonical": (CANONICAL_CHANNELS, t_steps, N_JOINTS),
                    "global": (GLOBAL_CHANNELS, t_steps),
                }
                for key, shape in expected.items():
                    if tuple(features[key].shape) != shape:
                        raise RuntimeError(
                            f"{sid}: {key} shape {tuple(features[key].shape)} != {shape}"
                        )
                    if not torch.isfinite(features[key]).all():
                        raise RuntimeError(f"{sid}: non-finite {key} feature")
                if not 0 <= int(label) < N_CLASSES:
                    raise RuntimeError(f"{sid}: invalid class {label}")

    return {
        "fold": fold,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_users": sorted(train_users),
        "validation_users": sorted(validation_users),
        "train_ids_sha256": ids_digest(train.ids),
        "validation_ids_sha256": ids_digest(validation.ids),
        "train_validation_overlap": 0,
        "user_overlap": 0,
        "optimizer_split": "train only",
        "outer_validation_policy": "one evaluation after fixed training budget",
        "normalization_scope": "per clip only",
    }


def loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool = False,
    seed: int = 0,
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


def predict(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    probabilities, labels, sids = [], [], []
    model.eval()
    with torch.inference_mode():
        for features, label, sid in loader(
            dataset, batch_size, workers, device, shuffle=False
        ):
            logits = model(move_features(features, device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(label.numpy())
            sids.extend(sid)
    if not probabilities:
        return (
            np.empty((0, N_CLASSES), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )
    return (
        np.concatenate(probabilities).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
        sids,
    )


def fold_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    micro = float(np.mean(predictions == labels))
    per_class = [
        float(np.mean(predictions[labels == class_id] == class_id))
        for class_id in np.unique(labels)
    ]
    return {
        "micro": micro,
        "macro_present": float(np.mean(per_class)),
        "correct": int(np.sum(predictions == labels)),
        "samples": int(len(labels)),
    }


def checkpoint_metadata(config: BranchConfig, fold: int) -> dict:
    path = checkpoint_meta_path(config.tag, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is required to prevent loading a checkpoint with the "
            "wrong feature/model configuration"
        )
    with open(path) as handle:
        metadata = json.load(handle)
    expected = {
        "feature_version": FEATURE_VERSION,
        "tag": config.tag,
        "fold": fold,
        "width": config.width,
        "t_steps": config.t_steps,
        "variant": config.variant,
        "seed": config.seed,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{path}: {key}={metadata.get(key)!r}, expected {value!r}"
            )
    checkpoint = checkpoint_path(config.tag, fold)
    actual_hash = sha256_file(checkpoint)
    if metadata.get("checkpoint_sha256") != actual_hash:
        raise ValueError(f"{checkpoint}: hash does not match metadata sidecar")
    if metadata.get("checkpoint_bytes") != os.path.getsize(checkpoint):
        raise ValueError(f"{checkpoint}: size does not match metadata sidecar")
    return metadata


def train_fold(args, config: BranchConfig, fold: int, audit: dict) -> dict:
    path = checkpoint_path(config.tag, fold)
    metadata_path = checkpoint_meta_path(config.tag, fold)
    if (os.path.exists(path) or os.path.exists(metadata_path)) and not args.force:
        raise FileExistsError(
            f"{path} or its metadata already exists; use --force to replace this fold"
        )

    seed_everything(config.seed + fold * 10_007)
    device = resolve_device(args.device)
    train = DualFrameSkeletonDataset(
        "train",
        fold,
        "train",
        t_steps=config.t_steps,
        aug_spec=args.aug_spec,
        seed=config.seed,
    )
    train_loader = loader(
        train,
        args.bs,
        args.workers,
        device,
        shuffle=True,
        seed=config.seed + fold * 10_007,
    )
    if len(train_loader) == 0:
        raise RuntimeError("training loader has no complete batch")
    model = model_for(config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
    started = time.time()
    final_loss = math.nan
    for epoch in range(args.epochs):
        model.train()
        # Workers are intentionally non-persistent, so they receive this
        # epoch-specific seed instead of replaying one frozen augmentation.
        train.epoch_seed = config.seed * 1_000 + epoch
        loss_sum, sample_sum = 0.0, 0
        for features, labels, _ in train_loader:
            labels = labels.to(device, non_blocking=True)
            logits = model(move_features(features, device))
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(labels)
            sample_sum += len(labels)
        final_loss = loss_sum / max(1, sample_sum)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(
                f"fold {fold} epoch {epoch + 1}/{args.epochs}: "
                f"train_loss={final_loss:.5f}",
                flush=True,
            )

    # Fixed-budget policy: the outer validation examples have not been loaded
    # at any point above and cannot select an epoch or hyperparameter.
    validation = DualFrameSkeletonDataset(
        "train", fold, "val", t_steps=config.t_steps
    )
    probabilities, labels, sids = predict(
        model, validation, args.bs, max(0, args.workers // 2), device
    )
    metrics = fold_metrics(probabilities, labels)
    state = {key: value.detach().cpu().clone()
             for key, value in model.state_dict().items()}
    atomic_torch_save(path, state)
    checkpoint_hash = sha256_file(path)
    metadata = {
        "feature_version": FEATURE_VERSION,
        **asdict(config),
        "fold": fold,
        "epochs": args.epochs,
        "augmentation": args.aug_spec,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "parameters": parameter_count,
        "checkpoint_bytes": os.path.getsize(path),
        "checkpoint_sha256": checkpoint_hash,
        "source_sha256": sha256_file(__file__),
        "audit": audit,
        "validation": metrics,
        "validation_ids_sha256": ids_digest(sids),
        "final_train_loss": final_loss,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "outer_validation_evaluations": 1,
    }
    atomic_json(metadata_path, metadata)
    print(
        f"fold {fold}: micro={metrics['micro']:.5f} "
        f"macro={metrics['macro_present']:.5f} "
        f"({metrics['correct']}/{metrics['samples']}); "
        f"checkpoint={os.path.getsize(path) / 1e6:.2f} MB",
        flush=True,
    )
    return metadata


def load_model_checkpoint(
    config: BranchConfig, fold: int, device: torch.device
) -> tuple[nn.Module, dict]:
    path = checkpoint_path(config.tag, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    metadata = checkpoint_metadata(config, fold)
    model = model_for(config, device)
    model.load_state_dict(load_tensor_state(path, device), strict=True)
    model.eval()
    return model, metadata


def checkpoint_signature(config: BranchConfig, folds: Sequence[int]) -> str:
    records = []
    for fold in folds:
        metadata = checkpoint_metadata(config, fold)
        records.append(
            f"{os.path.basename(checkpoint_path(config.tag, fold))}:"
            f"{metadata['checkpoint_bytes']}:{metadata['checkpoint_sha256']}"
        )
    return "|".join(records)


def oof_path(tag: str, folds: Sequence[int]) -> str:
    if set(folds) == set(range(4)) and len(folds) == 4:
        return os.path.join(ARTIFACTS, f"oof_{tag}.npz")
    suffix = "-".join(str(fold) for fold in folds)
    return os.path.join(ARTIFACTS, f"oof_{tag}_folds{suffix}.npz")


def compare_baseline(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sids: Sequence[str],
    baseline_path: str,
    baseline_key: str,
) -> dict | None:
    if not baseline_path:
        return None
    if not os.path.isabs(baseline_path):
        baseline_path = os.path.join(ROOT, baseline_path)
    with np.load(baseline_path, allow_pickle=False) as archive:
        if baseline_key not in archive.files:
            raise KeyError(
                f"{baseline_path}: key {baseline_key!r} absent; "
                f"available={archive.files}"
            )
        baseline = archive[baseline_key]
        baseline_sids = [str(value) for value in archive["sids"]]
        baseline_labels = archive["labels"] if "labels" in archive.files else None
    order = {sid: index for index, sid in enumerate(baseline_sids)}
    missing = [sid for sid in sids if sid not in order]
    if missing:
        raise ValueError(
            f"{baseline_path}: missing {len(missing)} candidate rows; first={missing[0]}"
        )
    selected = np.asarray([order[sid] for sid in sids])
    aligned = baseline[selected]
    if baseline_labels is not None and not np.array_equal(
        baseline_labels[selected].astype(np.int64), labels
    ):
        raise ValueError(f"{baseline_path}: labels disagree after SID alignment")
    candidate_hit = probabilities.argmax(1) == labels
    baseline_hit = aligned.argmax(1) == labels
    report = {
        "path": os.path.relpath(baseline_path, ROOT),
        "key": baseline_key,
        "candidate_micro": float(candidate_hit.mean()),
        "baseline_micro": float(baseline_hit.mean()),
        "delta": float(candidate_hit.mean() - baseline_hit.mean()),
        "rescued": int(np.sum(candidate_hit & ~baseline_hit)),
        "harmed": int(np.sum(~candidate_hit & baseline_hit)),
        "argmax_changes": int(
            np.sum(probabilities.argmax(1) != aligned.argmax(1))
        ),
    }
    print(
        f"paired baseline: candidate={report['candidate_micro']:.5f} "
        f"baseline={report['baseline_micro']:.5f} "
        f"delta={report['delta']:+.5f} rescued={report['rescued']} "
        f"harmed={report['harmed']} changes={report['argmax_changes']}",
        flush=True,
    )
    return report


def generate_oof(args, config: BranchConfig, folds: Sequence[int]) -> tuple[str, dict]:
    path = oof_path(config.tag, folds)
    if os.path.exists(path) and not args.force:
        raise FileExistsError(f"{path} exists; use --force to replace it")
    device = resolve_device(args.device)
    all_probabilities, all_labels, all_sids, row_folds = [], [], [], []
    per_fold = {}
    for fold in folds:
        model, _ = load_model_checkpoint(config, fold, device)
        validation = DualFrameSkeletonDataset(
            "train", fold, "val", t_steps=config.t_steps
        )
        probabilities, labels, sids = predict(
            model, validation, args.bs, args.workers, device
        )
        metrics = fold_metrics(probabilities, labels)
        per_fold[str(fold)] = metrics
        all_probabilities.append(probabilities)
        all_labels.append(labels)
        all_sids.extend(sids)
        row_folds.extend([fold] * len(sids))
        print(f"OOF fold {fold}: micro={metrics['micro']:.5f}", flush=True)
    probabilities = np.concatenate(all_probabilities).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int64)
    if len(all_sids) != len(set(all_sids)):
        raise RuntimeError("OOF contains duplicate sample IDs")
    overall = fold_metrics(probabilities, labels)
    baseline = compare_baseline(
        probabilities,
        labels,
        all_sids,
        getattr(args, "baseline_oof", ""),
        getattr(args, "baseline_key", "probs"),
    )
    signature = checkpoint_signature(config, folds)
    atomic_npz(
        path,
        probs=probabilities,
        labels=labels,
        sids=np.asarray(all_sids),
        row_folds=np.asarray(row_folds, dtype=np.int8),
        folds=np.asarray(folds, dtype=np.int8),
        checkpoint_signature=np.asarray(signature),
        feature_version=np.asarray(FEATURE_VERSION),
        variant=np.asarray(config.variant),
    )
    report = {
        "artifact": os.path.relpath(path, ROOT),
        "overall": overall,
        "per_fold": per_fold,
        "baseline": baseline,
        "checkpoint_signature": signature,
    }
    print(
        f"OOF {path}: micro={overall['micro']:.5f} "
        f"({overall['correct']}/{overall['samples']})",
        flush=True,
    )
    return path, report


def expected_test_ids() -> list[str]:
    with open(os.path.join(CACHE, "meta_test.csv")) as handle:
        return sorted(row["sample_id"] for row in csv.DictReader(handle))


def generate_test(args, config: BranchConfig) -> tuple[str, dict]:
    folds = tuple(range(4))
    # Validate every checkpoint and sidecar before loading test data.
    signature = checkpoint_signature(config, folds)
    path = os.path.join(ARTIFACTS, f"testprobs_{config.tag}.npz")
    if os.path.exists(path) and not args.force:
        raise FileExistsError(f"{path} exists; use --force to replace it")
    device = resolve_device(args.device)
    test = DualFrameSkeletonDataset("test", t_steps=config.t_steps)
    expected = expected_test_ids()
    if list(test.ids) != expected or len(test.ids) != len(set(test.ids)):
        raise RuntimeError("test IDs do not exactly match sorted meta_test.csv")
    total = np.zeros((len(test), N_CLASSES), dtype=np.float64)
    reference_sids: list[str] | None = None
    for fold in folds:
        model, _ = load_model_checkpoint(config, fold, device)
        probabilities, labels, sids = predict(
            model, test, args.bs, args.workers, device
        )
        if not np.all(labels == -1):
            raise RuntimeError("test dataset unexpectedly exposed labels")
        if reference_sids is None:
            reference_sids = sids
        elif sids != reference_sids:
            raise RuntimeError(f"test order changed on fold {fold}")
        total += probabilities
    probabilities = (total / len(folds)).astype(np.float32)
    if reference_sids != expected:
        raise RuntimeError("predicted test rows do not match meta_test.csv")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("test probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(1), 1.0, atol=2e-5):
        raise RuntimeError("test probabilities are not row-normalized")
    atomic_npz(
        path,
        probs=probabilities,
        sids=np.asarray(reference_sids),
        folds=np.asarray(folds, dtype=np.int8),
        checkpoint_signature=np.asarray(signature),
        feature_version=np.asarray(FEATURE_VERSION),
        variant=np.asarray(config.variant),
    )
    report = {
        "artifact": os.path.relpath(path, ROOT),
        "rows": len(reference_sids),
        "classes": N_CLASSES,
        "checkpoint_signature": signature,
        "four_fold_checkpoint_bytes": sum(
            os.path.getsize(checkpoint_path(config.tag, fold)) for fold in folds
        ),
    }
    print(
        f"test probabilities {path}: rows={report['rows']}; "
        f"four-fold checkpoints={report['four_fold_checkpoint_bytes'] / 1e6:.2f} MB",
        flush=True,
    )
    return path, report


def branch_config(args) -> BranchConfig:
    config = BranchConfig(
        tag=args.tag,
        width=args.width,
        t_steps=args.t_steps,
        variant=args.variant,
        seed=args.seed,
    )
    config.validate()
    return config


def run_screen(args) -> None:
    config = branch_config(args)
    folds = parse_folds(args.folds)
    audits = {}
    for fold in folds:
        audits[str(fold)] = audit_fold(
            fold,
            config.t_steps,
            sample_count=args.audit_samples,
            exercise_validation_features=args.audit_only,
        )
        record = audits[str(fold)]
        print(
            f"AUDIT fold {fold}: train={record['train_samples']} "
            f"val={record['validation_samples']} "
            f"val_users={record['validation_users']} overlap=0",
            flush=True,
        )
    if args.audit_only:
        device = resolve_device(args.device)
        dataset = DualFrameSkeletonDataset(
            "train", folds[0], "train", t_steps=config.t_steps
        )
        features, _, _ = dataset[0]
        model = model_for(config, device).eval()
        batch = {key: value.unsqueeze(0) for key, value in features.items()}
        with torch.inference_mode():
            logits = model(move_features(batch, device))
        if logits.shape != (1, N_CLASSES) or not torch.isfinite(logits).all():
            raise RuntimeError("model dry-run failed")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"AUDIT PASSED: feature/model dry-run finite; params={parameters:,}; "
            "no artifacts written",
            flush=True,
        )
        return

    runs = {}
    for fold in folds:
        runs[str(fold)] = train_fold(args, config, fold, audits[str(fold)])
    _, oof_report = generate_oof(args, config, folds)
    fold_suffix = "-".join(str(fold) for fold in folds)
    manifest_path = os.path.join(
        ARTIFACTS, f"run_dualframe_{config.tag}_{fold_suffix}.json"
    )
    manifest = {
        "feature_version": FEATURE_VERSION,
        "config": asdict(config),
        "folds": list(folds),
        "fixed_budget_outer_evaluation": True,
        "training": runs,
        "oof": oof_report,
        "source_sha256": sha256_file(__file__),
    }
    atomic_json(manifest_path, manifest)
    print(f"run manifest {manifest_path}", flush=True)


def run_oof(args) -> None:
    config = branch_config(args)
    folds = parse_folds(args.folds)
    generate_oof(args, config, folds)


def run_test(args) -> None:
    config = branch_config(args)
    generate_test(args, config)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", default="skel_dualframe_v1")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--t-steps", type=int, default=32)
    parser.add_argument(
        "--variant",
        choices=("dual", "raw", "canonical"),
        default="dual",
        help="dual branch or exact-architecture single-view control",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=48)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")


def add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-oof",
        default="",
        help="optional NPZ for SID-aligned paired accuracy/rescue comparison",
    )
    parser.add_argument(
        "--baseline-key",
        default="probs",
        help="probability key in --baseline-oof (for probes, use clean_probs)",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dual station/body-frame skeleton experiment"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    screen = commands.add_parser(
        "screen",
        help="audit, train fixed-budget folds, evaluate once, and emit partial OOF",
    )
    add_common_arguments(screen)
    add_baseline_arguments(screen)
    screen.add_argument("--folds", default="0,2")
    screen.add_argument("--epochs", type=int, default=100)
    screen.add_argument("--lr", type=float, default=1e-3)
    screen.add_argument("--weight-decay", type=float, default=0.05)
    screen.add_argument("--label-smoothing", type=float, default=0.1)
    screen.add_argument("--grad-clip", type=float, default=5.0)
    screen.add_argument(
        "--aug-spec",
        default="trunc",
        help="comma list drawn from trunc,tjit; default keeps only proven truncation",
    )
    screen.add_argument("--audit-only", action="store_true")
    screen.add_argument("--audit-samples", type=int, default=8)
    screen.set_defaults(function=run_screen)

    oof = commands.add_parser(
        "oof", help="load fold checkpoints and write standard or partial OOF NPZ"
    )
    add_common_arguments(oof)
    add_baseline_arguments(oof)
    oof.add_argument("--folds", default="0,1,2,3")
    oof.set_defaults(function=run_oof)

    test = commands.add_parser(
        "test", help="require all four checkpoints and write testprobs_<tag>.npz"
    )
    add_common_arguments(test)
    test.set_defaults(function=run_test)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.bs < 1:
        raise ValueError("--bs must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    args.function(args)


if __name__ == "__main__":
    main()
