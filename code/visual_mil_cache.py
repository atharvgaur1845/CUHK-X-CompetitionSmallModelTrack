#!/usr/bin/env python3
"""Deterministic high-resolution visual cache for the compact MIL reset.

This script performs preprocessing and validation only.  It does not download
weights, instantiate a model, or train anything.

Each output clip contains 16 normalized segment-center samples:

* ``ir``:      uint8 ``(16, 192, 256)``
* ``depth``:   uint8 ``(16, 192, 256)`` scalar depth from the supplied JET PNG
* ``thermal``: uint8 ``(16, 192, 256, 3)`` RGB false-color frames

IR is the preferred temporal anchor when its filenames contain timestamps.
Depth is matched to each selected IR timestamp by nearest-neighbour search with
an explicit tolerance.  If only Depth has timestamps, it becomes the temporal
anchor; any untimestamped stream is sampled independently rather than being
discarded or falsely aligned.  Thermal has no timestamps, so it is sampled
independently at the same normalized segment positions.  Missing frames and
modalities are zero-filled and represented by masks; canonical clips are never
filtered.

The output NPZ writer fixes ZIP metadata and array order.  Consequently two
builds from the same raw inputs have the same file SHA-256, which ``smoke`` and
``validate --compare-dir`` verify.

Examples
--------
Small deterministic train+test smoke cache:

    python3 code/visual_mil_cache.py smoke --output-dir /tmp/visual_mil_smoke

Estimate the complete cache footprint without reading image payloads:

    python3 code/visual_mil_cache.py estimate

Build the complete cache only after reviewing the estimate:

    python3 code/visual_mil_cache.py build --split both --confirm-full-build

Validate complete train and test caches:

    python3 code/visual_mil_cache.py validate --split both
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

import build_cache as legacy_cache

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = ROOT / "Small-Model-Track" / "Training" / "data" / "HAR" / "data"
DEFAULT_TEST_ROOT = (
    ROOT / "Small-Model-Track" / "Testing" / "data" / "small_model_track_test"
)
DEFAULT_METADATA_DIR = ROOT / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "cache" / "visual_mil_v1"

# Cache geometry is env-overridable so cache v2 can be built without forking this
# module.  EXP-068 measured accuracy flat down to 96x128 (sedentary +0.45) while
# 16->8 frames costs 4.1 points, so v2 spends the freed pixel budget on frames.
# Defaults reproduce v1 byte-for-byte; CACHE_VERSION carries the geometry so a v2
# build can never be mistaken for a v1 one by the manifest/SHA checks.
SEGMENTS = int(os.environ.get("CUHKX_CACHE_SEGMENTS", "16"))
HEIGHT = int(os.environ.get("CUHKX_CACHE_HEIGHT", "192"))
WIDTH = int(os.environ.get("CUHKX_CACHE_WIDTH", "256"))
CACHE_VERSION = (
    "visual_mil_highres_v1"
    if (SEGMENTS, HEIGHT, WIDTH) == (16, 192, 256)
    else f"visual_mil_s{SEGMENTS}_{HEIGHT}x{WIDTH}_v2"
)
# Thermal carries no timestamps, so it is sampled at the same normalized positions
# as the IR anchor.  That is valid only while both streams span the same window.
# Measured over 572 clips the thermal/IR frame-count ratio is median 2.43 (24.2 Hz
# against 10 Hz) but std 0.70, and 23.6% of clips fall outside 2.0-3.0 (min 0.02,
# max 7.60).  For those the streams do not span the same interval and thermal
# segment k is a different instant from IR segment k.  Rather than silently
# feeding misaligned frames, those clips get their thermal modality mask cleared;
# the network already handles an absent modality and EXP-062's modality dropout
# trains it to.  Set the band to 0/inf to restore the v1 behaviour.
THERMAL_RATIO_MIN = float(os.environ.get("CUHKX_THERMAL_RATIO_MIN", "0"))
THERMAL_RATIO_MAX = float(os.environ.get("CUHKX_THERMAL_RATIO_MAX", "inf"))
ALIGN_TOLERANCE_SECONDS = 0.075
MODALITIES = ("ir", "depth", "thermal")
EXPECTED_IDS = {"train": 2933, "test": 405}
EXPECTED_SOURCE_PRESENCE = {
    # Thermal has 2,891 raw trials, but 103 are Thermal-only records outside
    # the canonical 2,933-clip IR/Depth/Skeleton/IMU universe.  Their
    # intersection is therefore 2,788 canonical train clips.
    "train": {"ir": 2933, "depth": 2931, "thermal": 2788},
    "test": {"ir": 405, "depth": 405, "thermal": 395},
}
# Four nominal test IR streams contain only zero-filled, non-PNG payloads.
# ``source_present`` deliberately records that their files exist, while the
# decoded frame/modality masks correctly mark them unusable.
EXPECTED_DECODED_PRESENCE = {
    "train": {"ir": 2933, "depth": 2931, "thermal": 2788},
    "test": {"ir": 401, "depth": 405, "thermal": 395},
}
SEGMENT_POSITIONS = (np.arange(SEGMENTS, dtype=np.float64) + 0.5) / float(SEGMENTS)
THERMAL_INDEX_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocessing_config() -> dict[str, Any]:
    """Return every field that changes cached numerical content."""

    dependency_path = Path(legacy_cache.__file__).resolve()
    return {
        "cache_version": CACHE_VERSION,
        "segments": SEGMENTS,
        "height": HEIGHT,
        "width": WIDTH,
        "segment_sampling": "floor((i+0.5)*n/16), clipped",
        "ir_depth_anchor": (
            "IR when fully timestamped, otherwise Depth when fully timestamped"
        ),
        "ir_depth_alignment": "nearest timestamp",
        "alignment_tolerance_seconds": ALIGN_TOLERANCE_SECONDS,
        "untimestamped_visual_fallback": "independent normalized segment centers",
        "duration_seconds": "full span of preferred finite timestamp stream",
        "ir_resize": "cv2.INTER_AREA",
        "depth_decode": "build_cache.invert_jet",
        "depth_resize": "cv2.INTER_NEAREST",
        "thermal_sampling": "independent normalized segment centers",
        "thermal_ratio_band": [THERMAL_RATIO_MIN, THERMAL_RATIO_MAX],
        "thermal_resize": "cv2.INTER_AREA",
        "thermal_color_order": "RGB",
        "source_dtype": "uint8",
        "missing_policy": "zero-fill with explicit frame/modality masks",
        "build_cache_sha256": sha256_file(dependency_path),
    }


CONFIG = preprocessing_config()
CONFIG_JSON = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))
CONFIG_SHA256 = hashlib.sha256(CONFIG_JSON.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClipJob:
    sample_id: str
    split: str
    ir_dir: str
    depth_dir: str
    thermal_dir: str

    def directory(self, modality: str) -> Path | None:
        value = {
            "ir": self.ir_dir,
            "depth": self.depth_dir,
            "thermal": self.thermal_dir,
        }[modality]
        return Path(value) if value else None


def _existing_directory(path: Path) -> str:
    return str(path.resolve()) if path.is_dir() else ""


def read_metadata(metadata_dir: Path, split: str) -> list[dict[str, str]]:
    path = metadata_dir / f"meta_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"metadata not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metadata is empty: {path}")
    required = {"sample_id"}
    if split == "train":
        required.update(("class_id", "user", "trial"))
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate sample IDs in {path}")
    return sorted(rows, key=lambda row: row["sample_id"])


def _class_directories(root: Path, modality: str) -> dict[int, Path]:
    modality_root = root / modality
    if not modality_root.is_dir():
        return {}
    result: dict[int, Path] = {}
    for path in modality_root.iterdir():
        if not path.is_dir():
            continue
        try:
            class_id = int(path.name.split("_", 1)[0])
        except ValueError:
            continue
        if class_id in result:
            raise ValueError(f"duplicate class prefix {class_id} under {modality_root}")
        result[class_id] = path
    return result


def resolve_jobs(
    split: str,
    metadata_dir: Path,
    train_root: Path,
    test_root: Path,
) -> list[ClipJob]:
    """Resolve canonical metadata IDs to raw visual directories."""

    rows = read_metadata(metadata_dir, split)
    jobs: list[ClipJob] = []
    if split == "train":
        class_maps = {
            modality: _class_directories(train_root, raw_name)
            for modality, raw_name in (
                ("ir", "IR"),
                ("depth", "Depth_Color"),
                ("thermal", "Thermal"),
            )
        }
        for row in rows:
            class_id = int(row["class_id"])
            paths: dict[str, str] = {}
            for modality in MODALITIES:
                class_dir = class_maps[modality].get(class_id)
                candidate = (
                    class_dir / row["user"] / row["trial"]
                    if class_dir is not None
                    else Path("/nonexistent")
                )
                paths[modality] = _existing_directory(candidate)
            jobs.append(
                ClipJob(
                    sample_id=row["sample_id"],
                    split=split,
                    ir_dir=paths["ir"],
                    depth_dir=paths["depth"],
                    thermal_dir=paths["thermal"],
                )
            )
    elif split == "test":
        for row in rows:
            sample_id = row["sample_id"]
            sample_root = test_root / sample_id
            jobs.append(
                ClipJob(
                    sample_id=sample_id,
                    split=split,
                    ir_dir=_existing_directory(sample_root / "IR"),
                    depth_dir=_existing_directory(sample_root / "Depth_Color"),
                    thermal_dir=_existing_directory(sample_root / "Thermal"),
                )
            )
    else:
        raise ValueError(f"unknown split {split!r}")

    if len(jobs) != len(rows) or len({job.sample_id for job in jobs}) != len(jobs):
        raise RuntimeError(f"failed to resolve canonical {split} jobs")
    return jobs


def source_presence(jobs: Sequence[ClipJob]) -> dict[str, int]:
    counts = {modality: 0 for modality in MODALITIES}
    for job in jobs:
        for modality in MODALITIES:
            directory = job.directory(modality)
            files = (
                _thermal_files(directory)
                if modality == "thermal"
                else _png_sequence(directory)[0]
            )
            counts[modality] += int(bool(files))
    return counts


def _sequence_sort_key(path: Path) -> tuple[int, int | str, str]:
    match = THERMAL_INDEX_RE.search(path.name)
    if match is None:
        return (1, path.name, path.name)
    return (0, int(match.group(1)), path.name)


def _png_sequence(
    directory: Path | None,
) -> tuple[list[Path], np.ndarray]:
    """Return every PNG and its timestamp, falling back to frame-number order.

    A few supplied training directories use names such as ``IR_00000092.png``
    without wall-clock timestamps.  Those frames remain useful but cannot be
    aligned safely across modalities, so callers sample them independently and
    leave their selected timestamp/alignment masks empty.
    """

    if directory is None:
        return [], np.empty(0, dtype=np.float64)
    records: list[tuple[float | None, Path]] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        timestamp = legacy_cache.fname_ts(path.name)
        records.append((float(timestamp) if timestamp is not None else None, path))
    if records and all(timestamp is not None for timestamp, _ in records):
        records.sort(key=lambda item: (float(item[0]), item[1].name))
    else:
        records.sort(key=lambda item: _sequence_sort_key(item[1]))
    files = [path for _, path in records]
    timestamps = np.asarray(
        [
            float(timestamp) if timestamp is not None else np.nan
            for timestamp, _ in records
        ],
        dtype=np.float64,
    )
    return files, timestamps


def _thermal_sort_key(path: Path) -> tuple[int, int | str, str]:
    return _sequence_sort_key(path)


def _thermal_files(directory: Path | None) -> list[Path]:
    if directory is None:
        return []
    allowed = {".jpg", ".jpeg", ".png"}
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        ),
        key=_thermal_sort_key,
    )


def segment_indices(length: int) -> np.ndarray:
    if length <= 0:
        return np.full(SEGMENTS, -1, dtype=np.int32)
    indices = np.floor(SEGMENT_POSITIONS * length).astype(np.int64)
    return np.minimum(indices, length - 1).astype(np.int32)


def _nearest_indices(
    timestamps: np.ndarray,
    anchors: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Map finite anchors to nearest timestamps, using -1 outside tolerance."""

    result = np.full(len(anchors), -1, dtype=np.int32)
    if not len(timestamps):
        return result
    insertion = np.searchsorted(timestamps, anchors)
    for position, (anchor, right) in enumerate(zip(anchors, insertion)):
        if not np.isfinite(anchor):
            continue
        candidates = []
        if right < len(timestamps):
            candidates.append(int(right))
        if right > 0:
            candidates.append(int(right - 1))
        if not candidates:
            continue
        best = min(
            candidates, key=lambda index: (abs(timestamps[index] - anchor), index)
        )
        if abs(float(timestamps[best] - anchor)) <= tolerance:
            result[position] = best
    return result


def _as_uint8(image: np.ndarray, context: Path) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError(f"{context}: expected uint8 source, got {image.dtype}")
    return np.ascontiguousarray(image)


def _read_ir(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"{path}: unsupported IR shape {image.shape}")
    image = _as_uint8(image, path)
    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def _read_depth(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(_as_uint8(image, path), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = _as_uint8(image, path)
    else:
        raise ValueError(f"{path}: unsupported Depth shape {image.shape}")
    scalar = legacy_cache.invert_jet(image)
    return cv2.resize(scalar, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


def _read_thermal(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = _as_uint8(image, path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def _sample_images(
    files: Sequence[Path],
    indices: np.ndarray,
    reader: Any,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((SEGMENTS, *shape), dtype=np.uint8)
    mask = np.zeros(SEGMENTS, dtype=np.uint8)
    decoded: dict[int, np.ndarray | None] = {}
    for position, source_index in enumerate(indices):
        source_index = int(source_index)
        if source_index < 0:
            continue
        if source_index not in decoded:
            decoded[source_index] = reader(files[source_index])
        image = decoded[source_index]
        if image is None:
            continue
        if image.shape != shape or image.dtype != np.uint8:
            raise ValueError(
                f"decoded image has {image.shape}/{image.dtype}; "
                f"expected {shape}/uint8"
            )
        output[position] = image
        mask[position] = 1
    return output, mask


def _thermal_frame_numbers(files: Sequence[Path], indices: np.ndarray) -> np.ndarray:
    output = np.full(SEGMENTS, -1, dtype=np.int32)
    for position, source_index in enumerate(indices):
        source_index = int(source_index)
        if source_index < 0:
            continue
        match = THERMAL_INDEX_RE.search(files[source_index].name)
        output[position] = int(match.group(1)) if match is not None else source_index
    return output


def canonical_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash named numerical content independently of NPZ container metadata."""

    digest = hashlib.sha256()
    for key in sorted(arrays):
        if key == "content_sha256":
            continue
        array = np.asarray(arrays[key])
        if array.ndim:
            array = np.ascontiguousarray(array)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_clip_arrays(job: ClipJob) -> dict[str, np.ndarray]:
    ir_files, ir_times = _png_sequence(job.directory("ir"))
    depth_files, depth_times = _png_sequence(job.directory("depth"))
    thermal_files = _thermal_files(job.directory("thermal"))

    ir_timestamped = bool(len(ir_times) and np.isfinite(ir_times).all())
    depth_timestamped = bool(len(depth_times) and np.isfinite(depth_times).all())
    timestamp_alignment = False
    if ir_timestamped:
        ir_indices = segment_indices(len(ir_files))
        anchor_times = ir_times[ir_indices]
        if depth_timestamped:
            depth_indices = _nearest_indices(
                depth_times, anchor_times, ALIGN_TOLERANCE_SECONDS
            )
            alignment_mode = "timestamp_ir_anchor"
            timestamp_alignment = True
        else:
            depth_indices = segment_indices(len(depth_files))
            alignment_mode = (
                "timestamp_ir_anchor_depth_independent"
                if depth_files
                else "timestamp_ir_only"
            )
    elif depth_timestamped:
        depth_indices = segment_indices(len(depth_files))
        anchor_times = depth_times[depth_indices]
        ir_indices = segment_indices(len(ir_files))
        alignment_mode = (
            "timestamp_depth_anchor_ir_independent"
            if ir_files
            else "timestamp_depth_only"
        )
    else:
        anchor_times = np.full(SEGMENTS, np.nan, dtype=np.float64)
        ir_indices = segment_indices(len(ir_files))
        depth_indices = segment_indices(len(depth_files))
        alignment_mode = "independent_no_common_timestamps"

    ir, ir_mask = _sample_images(ir_files, ir_indices, _read_ir, (HEIGHT, WIDTH))
    depth, depth_mask = _sample_images(
        depth_files, depth_indices, _read_depth, (HEIGHT, WIDTH)
    )
    thermal_indices = segment_indices(len(thermal_files))
    thermal, thermal_mask = _sample_images(
        thermal_files,
        thermal_indices,
        _read_thermal,
        (HEIGHT, WIDTH, 3),
    )
    # Thermal has no timestamps, so normalized-position sampling is the only
    # option -- and it is correct only when thermal and IR span the same window.
    # Drop thermal for clips whose frame-count ratio says they do not (see the
    # THERMAL_RATIO_MIN/MAX note at the top of this module).
    if thermal_files and ir_files:
        thermal_ratio = len(thermal_files) / len(ir_files)
        if not (THERMAL_RATIO_MIN <= thermal_ratio <= THERMAL_RATIO_MAX):
            # The cache invariant is that masked-out frames are zero-filled, so
            # the payload must be cleared alongside the mask, not just the mask.
            thermal_mask = np.zeros_like(thermal_mask)
            thermal = np.zeros_like(thermal)

    ir_selected_times = np.full(SEGMENTS, np.nan, dtype=np.float64)
    depth_selected_times = np.full(SEGMENTS, np.nan, dtype=np.float64)
    valid_ir_indices = ir_indices >= 0
    valid_depth_indices = depth_indices >= 0
    if valid_ir_indices.any():
        ir_selected_times[valid_ir_indices] = ir_times[ir_indices[valid_ir_indices]]
    if valid_depth_indices.any():
        depth_selected_times[valid_depth_indices] = depth_times[
            depth_indices[valid_depth_indices]
        ]
    ir_selected_times[ir_mask == 0] = np.nan
    depth_selected_times[depth_mask == 0] = np.nan
    ir_depth_aligned_mask = np.zeros(SEGMENTS, dtype=np.uint8)
    if timestamp_alignment:
        paired = (ir_mask == 1) & (depth_mask == 1)
        differences = np.abs(ir_selected_times - depth_selected_times)
        ir_depth_aligned_mask[
            paired
            & np.isfinite(differences)
            & (differences <= ALIGN_TOLERANCE_SECONDS + 1e-9)
        ] = 1

    modality_mask = np.asarray(
        [ir_mask.any(), depth_mask.any(), thermal_mask.any()], dtype=np.uint8
    )
    source_counts = np.asarray(
        [len(ir_files), len(depth_files), len(thermal_files)], dtype=np.int32
    )
    timestamped_source_counts = np.asarray(
        [
            int(np.isfinite(ir_times).sum()),
            int(np.isfinite(depth_times).sum()),
        ],
        dtype=np.int32,
    )
    source_present = (source_counts > 0).astype(np.uint8)
    duration_reference = (
        ir_times
        if ir_timestamped
        else depth_times if depth_timestamped else np.empty(0, dtype=np.float64)
    )
    finite_duration_reference = duration_reference[np.isfinite(duration_reference)]
    duration_seconds = np.asarray(
        (
            float(finite_duration_reference[-1] - finite_duration_reference[0])
            if len(finite_duration_reference) > 1
            else 0.0
        ),
        dtype=np.float32,
    )

    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray(job.sample_id),
        "split": np.asarray(job.split),
        "alignment_mode": np.asarray(alignment_mode),
        "cache_version": np.asarray(CACHE_VERSION),
        "config_json": np.asarray(CONFIG_JSON),
        "config_sha256": np.asarray(CONFIG_SHA256),
        "normalized_positions": SEGMENT_POSITIONS.astype(np.float32),
        "anchor_t": anchor_times.astype(np.float64),
        "duration_seconds": duration_seconds,
        "ir": ir,
        "depth": depth,
        "thermal": thermal,
        "ir_frame_mask": ir_mask,
        "depth_frame_mask": depth_mask,
        "thermal_frame_mask": thermal_mask,
        "ir_depth_aligned_mask": ir_depth_aligned_mask,
        "modality_mask": modality_mask,
        "source_present": source_present,
        "source_counts": source_counts,
        "timestamped_source_counts": timestamped_source_counts,
        "ir_t": ir_selected_times,
        "depth_t": depth_selected_times,
        "ir_source_index": ir_indices.astype(np.int32),
        "depth_source_index": depth_indices.astype(np.int32),
        "thermal_source_index": thermal_indices.astype(np.int32),
        "thermal_frame_number": _thermal_frame_numbers(thermal_files, thermal_indices),
    }
    arrays["content_sha256"] = np.asarray(canonical_content_sha256(arrays))
    return arrays


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    value = np.asarray(array)
    if value.ndim:
        value = np.ascontiguousarray(value)
    np.lib.format.write_array(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomically write a byte-deterministic compressed NPZ."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for key in sorted(arrays):
                info = zipfile.ZipInfo(
                    filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    _npy_bytes(arrays[key]),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _scalar_text(array: np.ndarray, key: str) -> str:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar, got {value.shape}")
    return str(value.item())


def validate_clip(
    path: Path,
    expected_id: str | None = None,
    expected_split: str | None = None,
) -> dict[str, Any]:
    arrays = load_npz_arrays(path)
    required = {
        "sample_id",
        "split",
        "alignment_mode",
        "cache_version",
        "config_json",
        "config_sha256",
        "content_sha256",
        "normalized_positions",
        "anchor_t",
        "duration_seconds",
        "ir",
        "depth",
        "thermal",
        "ir_frame_mask",
        "depth_frame_mask",
        "thermal_frame_mask",
        "ir_depth_aligned_mask",
        "modality_mask",
        "source_present",
        "source_counts",
        "timestamped_source_counts",
        "ir_t",
        "depth_t",
        "ir_source_index",
        "depth_source_index",
        "thermal_source_index",
        "thermal_frame_number",
    }
    missing = required.difference(arrays)
    extras = set(arrays).difference(required)
    if missing or extras:
        raise ValueError(
            f"{path}: key mismatch; missing={sorted(missing)}, extras={sorted(extras)}"
        )

    sample_id = _scalar_text(arrays["sample_id"], "sample_id")
    if expected_id is not None and sample_id != expected_id:
        raise ValueError(f"{path}: sample_id {sample_id!r} != {expected_id!r}")
    if path.stem != sample_id:
        raise ValueError(f"{path}: filename and sample_id differ")
    split = _scalar_text(arrays["split"], "split")
    if split not in EXPECTED_IDS:
        raise ValueError(f"{path}: invalid split {split!r}")
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"{path}: split {split!r} != expected {expected_split!r}"
        )
    alignment_mode = _scalar_text(arrays["alignment_mode"], "alignment_mode")
    allowed_alignment_modes = {
        "timestamp_ir_anchor",
        "timestamp_ir_anchor_depth_independent",
        "timestamp_ir_only",
        "timestamp_depth_anchor_ir_independent",
        "timestamp_depth_only",
        "independent_no_common_timestamps",
    }
    if alignment_mode not in allowed_alignment_modes:
        raise ValueError(f"{path}: invalid alignment mode {alignment_mode!r}")
    if _scalar_text(arrays["cache_version"], "cache_version") != CACHE_VERSION:
        raise ValueError(f"{path}: cache version mismatch")
    if _scalar_text(arrays["config_json"], "config_json") != CONFIG_JSON:
        raise ValueError(f"{path}: preprocessing config mismatch")
    if _scalar_text(arrays["config_sha256"], "config_sha256") != CONFIG_SHA256:
        raise ValueError(f"{path}: preprocessing fingerprint mismatch")
    expected_content_hash = canonical_content_sha256(arrays)
    stored_content_hash = _scalar_text(arrays["content_sha256"], "content_sha256")
    if stored_content_hash != expected_content_hash:
        raise ValueError(f"{path}: content SHA-256 mismatch")

    specifications = {
        "ir": ((SEGMENTS, HEIGHT, WIDTH), np.dtype(np.uint8)),
        "depth": ((SEGMENTS, HEIGHT, WIDTH), np.dtype(np.uint8)),
        "thermal": ((SEGMENTS, HEIGHT, WIDTH, 3), np.dtype(np.uint8)),
        "normalized_positions": ((SEGMENTS,), np.dtype(np.float32)),
        "anchor_t": ((SEGMENTS,), np.dtype(np.float64)),
        "ir_t": ((SEGMENTS,), np.dtype(np.float64)),
        "depth_t": ((SEGMENTS,), np.dtype(np.float64)),
        "ir_frame_mask": ((SEGMENTS,), np.dtype(np.uint8)),
        "depth_frame_mask": ((SEGMENTS,), np.dtype(np.uint8)),
        "thermal_frame_mask": ((SEGMENTS,), np.dtype(np.uint8)),
        "ir_depth_aligned_mask": ((SEGMENTS,), np.dtype(np.uint8)),
        "modality_mask": ((3,), np.dtype(np.uint8)),
        "source_present": ((3,), np.dtype(np.uint8)),
        "source_counts": ((3,), np.dtype(np.int32)),
        "timestamped_source_counts": ((2,), np.dtype(np.int32)),
        "ir_source_index": ((SEGMENTS,), np.dtype(np.int32)),
        "depth_source_index": ((SEGMENTS,), np.dtype(np.int32)),
        "thermal_source_index": ((SEGMENTS,), np.dtype(np.int32)),
        "thermal_frame_number": ((SEGMENTS,), np.dtype(np.int32)),
    }
    for key, (shape, dtype) in specifications.items():
        array = arrays[key]
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"{path}: {key} has {array.shape}/{array.dtype}; "
                f"expected {shape}/{dtype}"
            )
    if (
        arrays["duration_seconds"].shape != ()
        or arrays["duration_seconds"].dtype != np.float32
    ):
        raise ValueError(f"{path}: invalid duration_seconds scalar")
    duration_seconds = float(arrays["duration_seconds"])
    if not np.isfinite(duration_seconds) or duration_seconds < 0.0:
        raise ValueError(f"{path}: duration_seconds must be finite and non-negative")
    if not np.allclose(arrays["normalized_positions"], SEGMENT_POSITIONS, atol=1e-7):
        raise ValueError(f"{path}: normalized positions changed")

    masks = {
        "ir": arrays["ir_frame_mask"],
        "depth": arrays["depth_frame_mask"],
        "thermal": arrays["thermal_frame_mask"],
    }
    for modality, mask in masks.items():
        if not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{path}: {modality} frame mask is non-binary")
        if np.any(arrays[modality][mask == 0]):
            raise ValueError(f"{path}: {modality} missing frames are not zero")
    modality_mask = arrays["modality_mask"]
    source_present = arrays["source_present"]
    aligned_mask = arrays["ir_depth_aligned_mask"]
    if not np.isin(modality_mask, (0, 1)).all():
        raise ValueError(f"{path}: modality_mask is non-binary")
    if not np.isin(source_present, (0, 1)).all():
        raise ValueError(f"{path}: source_present is non-binary")
    if not np.isin(aligned_mask, (0, 1)).all():
        raise ValueError(f"{path}: ir_depth_aligned_mask is non-binary")
    expected_modality_mask = np.asarray(
        [masks[modality].any() for modality in MODALITIES], dtype=np.uint8
    )
    if not np.array_equal(modality_mask, expected_modality_mask):
        raise ValueError(f"{path}: modality mask disagrees with frame masks")
    if not np.array_equal(
        source_present, (arrays["source_counts"] > 0).astype(np.uint8)
    ):
        raise ValueError(f"{path}: source presence disagrees with source counts")

    source_counts = arrays["source_counts"]
    timestamped_counts = arrays["timestamped_source_counts"]
    if np.any(source_counts < 0) or np.any(timestamped_counts < 0):
        raise ValueError(f"{path}: source counts must be non-negative")
    if np.any(timestamped_counts > source_counts[:2]):
        raise ValueError(f"{path}: timestamped counts exceed source counts")

    indices_by_modality = {
        "ir": arrays["ir_source_index"],
        "depth": arrays["depth_source_index"],
        "thermal": arrays["thermal_source_index"],
    }
    for position, modality in enumerate(MODALITIES):
        indices = indices_by_modality[modality]
        count = int(source_counts[position])
        if np.any(indices < -1) or np.any(indices >= count):
            raise ValueError(f"{path}: {modality} source index is out of range")
        if np.any((masks[modality] == 1) & (indices < 0)):
            raise ValueError(f"{path}: {modality} present frame lacks a source index")
    if np.any(
        (arrays["thermal_source_index"] < 0) != (arrays["thermal_frame_number"] < 0)
    ):
        raise ValueError(f"{path}: Thermal frame numbers disagree with source indices")

    ir_fully_timestamped = (
        source_counts[0] > 0 and timestamped_counts[0] == source_counts[0]
    )
    depth_fully_timestamped = (
        source_counts[1] > 0 and timestamped_counts[1] == source_counts[1]
    )
    if ir_fully_timestamped:
        expected_alignment_mode = (
            "timestamp_ir_anchor"
            if depth_fully_timestamped
            else (
                "timestamp_ir_anchor_depth_independent"
                if source_counts[1] > 0
                else "timestamp_ir_only"
            )
        )
    elif depth_fully_timestamped:
        expected_alignment_mode = (
            "timestamp_depth_anchor_ir_independent"
            if source_counts[0] > 0
            else "timestamp_depth_only"
        )
    else:
        expected_alignment_mode = "independent_no_common_timestamps"
    if alignment_mode != expected_alignment_mode:
        raise ValueError(
            f"{path}: alignment mode {alignment_mode!r} disagrees with "
            f"timestamp/source counts (expected {expected_alignment_mode!r})"
        )
    if alignment_mode == "independent_no_common_timestamps":
        if np.isfinite(arrays["anchor_t"]).any():
            raise ValueError(
                f"{path}: independent sampling must not claim anchor timestamps"
            )
    elif not np.isfinite(arrays["anchor_t"]).all():
        raise ValueError(f"{path}: timestamp anchor contains non-finite values")

    aligned = aligned_mask == 1
    if np.any(aligned & ((masks["ir"] == 0) | (masks["depth"] == 0))):
        raise ValueError(f"{path}: aligned mask includes a missing frame")
    expected_aligned = np.zeros(SEGMENTS, dtype=np.uint8)
    if alignment_mode == "timestamp_ir_anchor":
        differences = np.abs(arrays["ir_t"] - arrays["depth_t"])
        expected_aligned[
            (masks["ir"] == 1)
            & (masks["depth"] == 1)
            & np.isfinite(differences)
            & (differences <= ALIGN_TOLERANCE_SECONDS + 1e-9)
        ] = 1
    if not np.array_equal(aligned_mask, expected_aligned):
        raise ValueError(f"{path}: IR/Depth aligned mask is inconsistent")
    if aligned.any():
        differences = np.abs(arrays["ir_t"][aligned] - arrays["depth_t"][aligned])
        if not np.isfinite(differences).all():
            raise ValueError(f"{path}: aligned timestamps are non-finite")
        if float(differences.max()) > ALIGN_TOLERANCE_SECONDS + 1e-9:
            raise ValueError(f"{path}: IR/Depth alignment exceeds tolerance")
    if np.isfinite(arrays["ir_t"][masks["ir"] == 0]).any():
        raise ValueError(f"{path}: missing IR timestamps must be NaN")
    if np.isfinite(arrays["depth_t"][masks["depth"] == 0]).any():
        raise ValueError(f"{path}: missing Depth timestamps must be NaN")

    return {
        "sample_id": sample_id,
        "split": split,
        "content_sha256": stored_content_hash,
        "file_sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "source_present": source_present.astype(int).tolist(),
        "frame_counts": [int(masks[modality].sum()) for modality in MODALITIES],
    }


def _build_worker(payload: tuple[ClipJob, str, bool]) -> dict[str, Any]:
    job, output_name, overwrite = payload
    output_path = Path(output_name)
    if output_path.is_file() and not overwrite:
        try:
            result = validate_clip(output_path, job.sample_id, job.split)
            result["status"] = "valid-existing"
            return result
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(
                f"{output_path} exists but is invalid; use --overwrite: {exc}"
            ) from exc
    arrays = build_clip_arrays(job)
    write_npz_deterministic(output_path, arrays)
    result = validate_clip(output_path, job.sample_id, job.split)
    result["status"] = "written"
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def aggregate_file_sha256(results: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for result in sorted(results, key=lambda item: item["sample_id"]):
        digest.update(result["sample_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(result["file_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _select_jobs(
    jobs: Sequence[ClipJob],
    ids: Sequence[str] | None,
    limit: int | None,
) -> list[ClipJob]:
    selected = list(jobs)
    if ids:
        requested = set(ids)
        known = {job.sample_id for job in jobs}
        missing = sorted(requested - known)
        if missing:
            raise KeyError(f"unknown sample IDs: {missing[:5]}")
        selected = [job for job in selected if job.sample_id in requested]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no clips selected")
    return selected


def parse_ids(text: str) -> list[str] | None:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError("--ids contains duplicate sample IDs")
    return values or None


def partition_ids(
    jobs_by_split: dict[str, Sequence[ClipJob]],
    ids: Sequence[str] | None,
) -> dict[str, list[str] | None]:
    """Assign requested IDs to their canonical split.

    This lets ``--split both --ids train_id,test_id`` work without requiring
    each ID to exist in both splits.
    """

    if ids is None:
        return {split: None for split in jobs_by_split}
    known_by_split = {
        split: {job.sample_id for job in jobs} for split, jobs in jobs_by_split.items()
    }
    requested = set(ids)
    unknown = requested.difference(set().union(*known_by_split.values()))
    if unknown:
        raise KeyError(f"unknown sample IDs: {sorted(unknown)[:5]}")
    return {
        split: [sample_id for sample_id in ids if sample_id in known]
        for split, known in known_by_split.items()
    }


def build_jobs(
    jobs: Sequence[ClipJob],
    output_dir: Path,
    workers: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    split = jobs[0].split
    split_dir = output_dir / split
    payloads = [
        (job, str(split_dir / f"{job.sample_id}.npz"), overwrite) for job in jobs
    ]
    started = time.time()
    results: list[dict[str, Any]] = []
    if workers == 1:
        iterator: Iterable[dict[str, Any]] = map(_build_worker, payloads)
        pool = None
    else:
        pool = Pool(workers)
        iterator = pool.imap_unordered(_build_worker, payloads, chunksize=2)
    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index == 1 or index % 100 == 0 or index == len(payloads):
                print(
                    f"{split}: {index}/{len(payloads)} "
                    f"({(time.time() - started) / 60.0:.1f} min)",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    manifest = {
        "kind": "visual_mil_highres_cache_manifest",
        "cache_version": CACHE_VERSION,
        "config": CONFIG,
        "config_sha256": CONFIG_SHA256,
        "split": split,
        "clips": len(results),
        "canonical_full_split": len(results) == EXPECTED_IDS[split],
        "sample_ids": sorted(result["sample_id"] for result in results),
        "source_presence": {
            modality: sum(result["source_present"][index] for result in results)
            for index, modality in enumerate(MODALITIES)
        },
        "decoded_presence": {
            modality: sum(
                int(result["frame_counts"][index] > 0) for result in results
            )
            for index, modality in enumerate(MODALITIES)
        },
        "bytes": sum(result["bytes"] for result in results),
        "aggregate_file_sha256": aggregate_file_sha256(results),
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    _atomic_json(output_dir / f"manifest_{split}.json", manifest)
    return results


def resolve_all(args: argparse.Namespace, split: str) -> list[ClipJob]:
    return resolve_jobs(
        split=split,
        metadata_dir=args.metadata_dir,
        train_root=args.train_root,
        test_root=args.test_root,
    )


def split_values(value: str) -> list[str]:
    if value == "both":
        return ["train", "test"]
    if value in EXPECTED_IDS:
        return [value]
    raise ValueError(f"invalid split {value!r}")


def _format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def storage_estimate(
    output_dir: Path,
    counts: dict[str, int],
) -> dict[str, Any]:
    per_clip_payload = (
        SEGMENTS * HEIGHT * WIDTH  # IR
        + SEGMENTS * HEIGHT * WIDTH  # Depth
        + SEGMENTS * HEIGHT * WIDTH * 3  # Thermal RGB
        + 3 * SEGMENTS  # frame masks
        + SEGMENTS  # IR/Depth aligned mask
        + 3  # modality mask
        + 3  # source present
        + 3 * np.dtype(np.int32).itemsize  # source counts
        + 2 * np.dtype(np.int32).itemsize  # timestamped source counts
        + 3 * SEGMENTS * np.dtype(np.int32).itemsize  # source indices
        + SEGMENTS * np.dtype(np.int32).itemsize  # thermal frame numbers
        + 3 * SEGMENTS * np.dtype(np.float64).itemsize  # timestamps
        + SEGMENTS * np.dtype(np.float32).itemsize  # normalized positions
    )
    total_clips = sum(counts.values())
    uncompressed = per_clip_payload * total_clips
    existing = list(output_dir.glob("train/*.npz")) + list(
        output_dir.glob("test/*.npz")
    )
    projected_compressed: int | None = None
    mean_existing: float | None = None
    if existing:
        mean_existing = float(np.mean([path.stat().st_size for path in existing]))
        projected_compressed = int(round(mean_existing * total_clips))
    usage = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else ROOT)
    return {
        "clips": counts,
        "total_clips": total_clips,
        "uncompressed_array_upper_bound_bytes": uncompressed,
        "existing_npz_samples": len(existing),
        "mean_existing_npz_bytes": mean_existing,
        "projected_compressed_bytes": projected_compressed,
        "disk_free_bytes": usage.free,
    }


def print_storage_estimate(estimate: dict[str, Any]) -> None:
    print(
        "storage estimate: "
        f"{estimate['total_clips']} clips, "
        f"raw uint8 arrays about "
        f"{_format_bytes(estimate['uncompressed_array_upper_bound_bytes'])}",
        flush=True,
    )
    if estimate["projected_compressed_bytes"] is not None:
        print(
            f"  extrapolated from {estimate['existing_npz_samples']} cache files: "
            f"{_format_bytes(estimate['projected_compressed_bytes'])}",
            flush=True,
        )
    else:
        print(
            "  compressed size unavailable until at least one cache file exists",
            flush=True,
        )
    print(
        f"  free space: {_format_bytes(estimate['disk_free_bytes'])}",
        flush=True,
    )


def validate_directory(
    split: str,
    output_dir: Path,
    canonical_jobs: Sequence[ClipJob],
    expected_ids: Sequence[str] | None,
    allow_partial: bool,
    compare_dir: Path | None = None,
) -> dict[str, Any]:
    split_dir = output_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"cache split directory not found: {split_dir}")
    canonical_ids = {job.sample_id for job in canonical_jobs}
    actual_paths = sorted(split_dir.glob("*.npz"))
    actual_ids = {path.stem for path in actual_paths}
    if expected_ids is not None:
        required_ids = set(expected_ids)
    elif allow_partial:
        required_ids = actual_ids
    else:
        required_ids = canonical_ids
    unknown = actual_ids - canonical_ids
    missing = required_ids - actual_ids
    # With --allow-partial, --ids selects the files to validate rather than
    # requiring the cache directory to contain only those files.  This makes
    # targeted validation useful on a complete cache as well as on a smoke
    # cache.  Without --allow-partial the previous exact-set/full-split
    # contract remains in force.
    extras = (
        actual_ids - required_ids
        if expected_ids is not None and not allow_partial
        else set()
    )
    if unknown or missing or extras:
        raise ValueError(
            f"{split}: ID mismatch unknown={len(unknown)}, "
            f"missing={len(missing)}, extras={len(extras)}"
        )
    if not allow_partial and len(actual_ids) != EXPECTED_IDS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_IDS[split]} cache files, "
            f"found {len(actual_ids)}"
        )

    results = [
        validate_clip(
            split_dir / f"{sample_id}.npz",
            sample_id,
            split,
        )
        for sample_id in sorted(required_ids)
    ]
    if not results:
        raise ValueError(f"{split}: no cache files validated")

    source_counts = {
        modality: sum(result["source_present"][index] for result in results)
        for index, modality in enumerate(MODALITIES)
    }
    decoded_counts = {
        modality: sum(
            int(result["frame_counts"][index] > 0) for result in results
        )
        for index, modality in enumerate(MODALITIES)
    }
    if not allow_partial and source_counts != EXPECTED_SOURCE_PRESENCE[split]:
        raise ValueError(
            f"{split}: modality presence {source_counts} != "
            f"{EXPECTED_SOURCE_PRESENCE[split]}"
        )
    if (
        not allow_partial
        and decoded_counts != EXPECTED_DECODED_PRESENCE[split]
    ):
        raise ValueError(
            f"{split}: decoded modality presence {decoded_counts} != "
            f"{EXPECTED_DECODED_PRESENCE[split]}"
        )

    repeat_mismatches = 0
    if compare_dir is not None:
        for result in results:
            other = compare_dir / split / f"{result['sample_id']}.npz"
            if not other.is_file():
                raise FileNotFoundError(f"comparison cache missing: {other}")
            if result["file_sha256"] != sha256_file(other):
                repeat_mismatches += 1
        if repeat_mismatches:
            raise ValueError(
                f"{split}: {repeat_mismatches} deterministic file-hash mismatches"
            )

    summary = {
        "split": split,
        "clips": len(results),
        "source_presence": source_counts,
        "decoded_presence": decoded_counts,
        "bytes": sum(result["bytes"] for result in results),
        "aggregate_file_sha256": aggregate_file_sha256(results),
        "repeat_mismatches": repeat_mismatches,
    }
    print(
        f"VALID {split}: clips={summary['clips']} "
        f"source={source_counts} decoded={decoded_counts} "
        f"bytes={_format_bytes(summary['bytes'])} "
        f"aggregate_sha256={summary['aggregate_file_sha256']}",
        flush=True,
    )
    return summary


def run_estimate(args: argparse.Namespace) -> None:
    counts = {
        split: len(resolve_all(args, split)) for split in split_values(args.split)
    }
    print_storage_estimate(storage_estimate(args.output_dir, counts))


def run_build(args: argparse.Namespace) -> None:
    ids = parse_ids(args.ids)
    splits = split_values(args.split)
    all_jobs = {split: resolve_all(args, split) for split in splits}
    for split, jobs in all_jobs.items():
        if len(jobs) != EXPECTED_IDS[split]:
            raise ValueError(
                f"canonical {split} metadata has {len(jobs)} IDs; "
                f"expected {EXPECTED_IDS[split]}"
            )
        presence = source_presence(jobs)
        if presence != EXPECTED_SOURCE_PRESENCE[split]:
            raise ValueError(
                f"raw {split} modality presence {presence} != "
                f"{EXPECTED_SOURCE_PRESENCE[split]}"
            )
        print(
            f"resolved {split}: {len(jobs)} canonical IDs; presence={presence}",
            flush=True,
        )

    ids_by_split = partition_ids(all_jobs, ids)
    selected = {
        split: _select_jobs(jobs, ids_by_split[split], args.limit)
        for split, jobs in all_jobs.items()
        if ids_by_split[split] is None or ids_by_split[split]
    }
    if not selected:
        raise ValueError("no clips selected")
    is_full = all(
        len(selected[split]) == EXPECTED_IDS[split] for split in selected
    ) and len(selected) == len(all_jobs)
    estimate = storage_estimate(
        args.output_dir, {split: len(jobs) for split, jobs in all_jobs.items()}
    )
    print_storage_estimate(estimate)
    if is_full and not args.confirm_full_build:
        raise ValueError(
            "refusing the full multi-gigabyte build without " "--confirm-full-build"
        )

    for split in selected:
        results = build_jobs(
            selected[split],
            output_dir=args.output_dir,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        print(
            f"BUILT {split}: {len(results)} clips, "
            f"{_format_bytes(sum(result['bytes'] for result in results))}, "
            f"aggregate_sha256={aggregate_file_sha256(results)}",
            flush=True,
        )


def run_validate(args: argparse.Namespace) -> None:
    ids = parse_ids(args.ids)
    summaries = []
    splits = split_values(args.split)
    all_jobs = {split: resolve_all(args, split) for split in splits}
    ids_by_split = partition_ids(all_jobs, ids)
    for split, jobs in all_jobs.items():
        if ids_by_split[split] == []:
            continue
        summaries.append(
            validate_directory(
                split=split,
                output_dir=args.output_dir,
                canonical_jobs=jobs,
                expected_ids=ids_by_split[split],
                allow_partial=args.allow_partial,
                compare_dir=args.compare_dir,
            )
        )
    if len(summaries) > 1:
        digest = hashlib.sha256()
        for summary in summaries:
            digest.update(summary["aggregate_file_sha256"].encode("ascii"))
        print(f"VALID combined_sha256={digest.hexdigest()}", flush=True)


def _choose_smoke_job(jobs: Sequence[ClipJob], requested: str) -> ClipJob:
    if requested:
        matches = [job for job in jobs if job.sample_id == requested]
        if not matches:
            raise KeyError(f"smoke sample not found: {requested}")
        return matches[0]
    all_present = [
        job for job in jobs if all(job.directory(modality) for modality in MODALITIES)
    ]
    return (all_present or list(jobs))[0]


def run_smoke(args: argparse.Namespace) -> None:
    train_jobs = resolve_all(args, "train")
    test_jobs = resolve_all(args, "test")
    chosen = [
        _choose_smoke_job(train_jobs, args.train_id),
        _choose_smoke_job(test_jobs, args.test_id),
    ]
    first_root = args.output_dir / "repeat_a"
    second_root = args.output_dir / "repeat_b"
    print(
        f"SMOKE config_sha256={CONFIG_SHA256} "
        f"train={chosen[0].sample_id} test={chosen[1].sample_id}",
        flush=True,
    )

    first_results = []
    second_results = []
    for job in chosen:
        first_results.append(
            _build_worker(
                (
                    job,
                    str(first_root / job.split / f"{job.sample_id}.npz"),
                    True,
                )
            )
        )
        second_results.append(
            _build_worker(
                (
                    job,
                    str(second_root / job.split / f"{job.sample_id}.npz"),
                    True,
                )
            )
        )

    for first, second in zip(first_results, second_results):
        if first["content_sha256"] != second["content_sha256"]:
            raise RuntimeError(f"{first['sample_id']}: repeated content hashes differ")
        if first["file_sha256"] != second["file_sha256"]:
            raise RuntimeError(f"{first['sample_id']}: repeated NPZ hashes differ")
        path = first_root / first["split"] / f"{first['sample_id']}.npz"
        arrays = load_npz_arrays(path)
        print(
            f"  {first['split']} {first['sample_id']}: "
            f"ir={arrays['ir'].shape}/{arrays['ir'].dtype} "
            f"depth={arrays['depth'].shape}/{arrays['depth'].dtype} "
            f"thermal={arrays['thermal'].shape}/{arrays['thermal'].dtype} "
            f"masks={first['frame_counts']} "
            f"bytes={_format_bytes(first['bytes'])} "
            f"sha256={first['file_sha256']}",
            flush=True,
        )

    validate_directory(
        "train",
        first_root,
        train_jobs,
        [chosen[0].sample_id],
        allow_partial=True,
        compare_dir=second_root,
    )
    validate_directory(
        "test",
        first_root,
        test_jobs,
        [chosen[1].sample_id],
        allow_partial=True,
        compare_dir=second_root,
    )
    print_storage_estimate(
        storage_estimate(
            first_root,
            {"train": EXPECTED_IDS["train"], "test": EXPECTED_IDS["test"]},
        )
    )
    print("SMOKE PASS: repeated content and NPZ SHA-256 values match", flush=True)


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate a deterministic 16-frame, 192x256 "
            "IR/Depth/Thermal cache. No model or external weights are used."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="build one train and one test clip twice and compare exact hashes",
    )
    add_paths(smoke)
    smoke.add_argument("--train-id", default="")
    smoke.add_argument("--test-id", default="")
    smoke.set_defaults(handler=run_smoke)

    estimate = subparsers.add_parser(
        "estimate", help="estimate full storage without decoding images"
    )
    add_paths(estimate)
    estimate.add_argument("--split", choices=("train", "test", "both"), default="both")
    estimate.set_defaults(handler=run_estimate)

    build = subparsers.add_parser(
        "build", help="build selected clips atomically; full builds need confirmation"
    )
    add_paths(build)
    build.add_argument("--split", choices=("train", "test", "both"), default="both")
    build.add_argument("--ids", default="", help="comma-separated sample IDs")
    build.add_argument("--limit", type=int)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--confirm-full-build", action="store_true")
    build.set_defaults(handler=run_build)

    validate = subparsers.add_parser(
        "validate", help="validate shapes, masks, IDs, hashes, and alignment"
    )
    add_paths(validate)
    validate.add_argument("--split", choices=("train", "test", "both"), default="both")
    validate.add_argument("--ids", default="", help="comma-separated expected IDs")
    validate.add_argument("--allow-partial", action="store_true")
    validate.add_argument(
        "--compare-dir",
        type=Path,
        help="second cache root whose per-file SHA-256 must match",
    )
    validate.set_defaults(handler=run_validate)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for path_name in ("train_root", "test_root", "metadata_dir"):
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f"{path_name.replace('_', '-')} not found: {path}")
    if hasattr(args, "workers") and args.workers < 1:
        raise ValueError("--workers must be positive")
    if hasattr(args, "limit") and args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if getattr(args, "ids", "") and getattr(args, "limit", None) is not None:
        raise ValueError("use either --ids or --limit, not both")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        args.handler(args)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
