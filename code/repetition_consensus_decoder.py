#!/usr/bin/env python3
"""Repetition consensus followed by leakage-safe transition decoding.

CUHK-X contains repeated recordings of short, ordered action sequences.  This
tool aligns only *consecutive, equal-length* recordings that look like repeats,
averages their per-position class probabilities, and optionally applies the
first-order transition decoder validated by the repository audit.

The grouping rules are deliberately explicit:

* train recording: ``(user, trial)``, clips ordered by ``meta_train.t0``;
* test recording: ``test_cohorts.radar_rec_ts``, clips ordered by
  ``meta_test.t0`` (a missing Radar key is a singleton);
* possible train repeats must share ``user``;
* possible test repeats must share ``test_cohorts.slot_id``;
* adjacent recordings link only when their lengths match, their end-to-start
  gap is at most ``--gap-seconds``, and their mean aligned probability cosine
  is at least ``--cosine-threshold``.

A recording with a missing ``t0``/``t1`` or two ``t0`` values within 1e-6 is
order-ambiguous. It is never used for transition fitting or repetition
clustering and is never decoded; its probabilities and base argmax are left
unchanged. In particular, sample ID is never used to resolve a timestamp tie.

Consensus is transductive but label-free.  With weight ``w``, each probability
row becomes ``(1-w) * original + w * aligned_cluster_mean``.  Weight zero is an
exact no-op; weight one is full per-position consensus.

The optional transition stage uses a 40x40 row-conditional transition matrix
with uniform add-one smoothing and no start-class term.  In OOF mode, the
transition matrix used for fold f excludes every validation user in fold f.
Grid selection is strictly nested: for outer fold f and inner validation fold
j, the transition matrix used to score candidates excludes users in
``f UNION j``. Outer-fold labels are neither fitted nor scored.

Examples
--------

Leakage-safe OOF evaluation using the audit defaults (dry run):

  python3 code/repetition_consensus_decoder.py oof \
      --probs research/artifacts/oof_astgcn_world25.npz

Leave-one-fold-out selection over all decoder parameters:

  python3 code/repetition_consensus_decoder.py oof \
      --probs research/artifacts/oof_astgcn_world25.npz \
      --gap-grid 120,300,600 --cosine-grid .5,.6,.7 \
      --consensus-weight-grid .5,1 --lambda-grid .25,.5,.75 \
      --output research/artifacts/oof_world25_repeat_transition.npz

Test dry run in the exact official sample-submission order:

  python3 code/repetition_consensus_decoder.py test \
      --probs research/artifacts/testprobs_astgcn_world25_int8.npz

Write an upload CSV plus adjusted-probability NPZ and provenance JSON:

  python3 code/repetition_consensus_decoder.py test \
      --probs research/artifacts/testprobs_astgcn_world25_int8.npz \
      --output submissions/sub_world25_repeat_transition.csv

Run dependency-free synthetic regression tests:

  python3 code/repetition_consensus_decoder.py self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


VERSION = "1.1"
N_CLASSES = 40
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
ARTIFACTS = ROOT / "research" / "artifacts"
DEFAULT_FOLDS = ARTIFACTS / "cv_folds.json"
DEFAULT_TEST_COHORTS = ARTIFACTS / "test_cohorts.csv"
DEFAULT_SAMPLE_SUBMISSION = (
    ROOT / "Small-Model-Track" / "Testing" / "sample_submission.csv"
)
DEFAULT_META_TRAIN = CACHE / "meta_train.csv"
DEFAULT_META_TEST = CACHE / "meta_test.csv"
TEST_SID_RE = re.compile(r"(SM_test_\d+)")


@dataclass(frozen=True)
class Clip:
    """One probability row plus metadata needed for recording alignment."""

    sid: str
    owner: str
    recording_key: str
    t0: float | None
    t1: float | None
    user: str | None
    label: int | None


@dataclass(frozen=True)
class Recording:
    """A metadata-native recording.

    ``clips`` is chronological only when ``ambiguous`` is false. Ambiguous
    recordings retain metadata input order solely for auditability and are
    never fitted, aligned, or decoded.
    """

    key: str
    owner: str
    clips: tuple[Clip, ...]
    start: float
    end: float
    complete: bool
    ambiguous: bool


@dataclass(frozen=True)
class Candidate:
    """One repetition/transition hyperparameter combination."""

    gap_seconds: float
    cosine_threshold: float
    consensus_weight: float
    transition_weight: float

    def label(self) -> str:
        return (
            f"gap={self.gap_seconds:g},cos={self.cosine_threshold:g},"
            f"w={self.consensus_weight:g},lambda={self.transition_weight:g}"
        )


@dataclass
class ConsensusStats:
    owners: int = 0
    recordings: int = 0
    complete_recordings: int = 0
    incomplete_recordings: int = 0
    ambiguous_recordings: int = 0
    adjacent_pairs: int = 0
    equal_length_pairs: int = 0
    gap_eligible_pairs: int = 0
    cosine_evaluated_pairs: int = 0
    linked_pairs: int = 0
    consensus_clusters: int = 0
    consensus_recordings: int = 0
    consensus_clips: int = 0
    max_cluster_recordings: int = 0
    mean_link_gap_seconds: float | None = None
    mean_link_cosine: float | None = None
    argmax_changes: int = 0
    mean_probability_l1: float = 0.0


@dataclass(frozen=True)
class TransitionStats:
    source_recordings: int
    source_clips: int
    source_edges: int
    source_ambiguous_recordings: int = 0
    target_recordings: int = 0
    target_ambiguous_recordings: int = 0
    target_ambiguous_clips: int = 0
    target_multi_recordings: int = 0
    target_multi_clips: int = 0
    argmax_changes: int = 0


@dataclass
class CandidateEvaluation:
    candidate: Candidate
    predictions: np.ndarray
    consensus_predictions: np.ndarray
    consensus_stats: dict[int, ConsensusStats]
    transition_stats: dict[int, TransitionStats]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def _unique_rows(
    path: Path,
    key: str,
) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value:
            raise ValueError(f"{path} contains an empty {key!r}")
        if value in result:
            raise ValueError(f"{path} contains duplicate {key}={value!r}")
        result[value] = row
    return result


def _optional_time(raw: str | None, *, field: str, sid: str) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} for {sid}: {raw!r}")
    return value


def _recording_bounds(clips: Sequence[Clip]) -> tuple[float, float]:
    starts = [clip.t0 for clip in clips if clip.t0 is not None]
    ends = [
        clip.t1 if clip.t1 is not None else clip.t0
        for clip in clips
        if clip.t1 is not None or clip.t0 is not None
    ]
    start = min(starts) if starts else math.inf
    end = max(float(value) for value in ends if value is not None) if ends else -math.inf
    return float(start), float(end)


def _recording_order_ambiguous(
    clips: Sequence[Clip],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Return whether timestamps cannot establish a unique safe clip order."""

    if any(clip.t0 is None or clip.t1 is None for clip in clips):
        return True
    starts = sorted(float(clip.t0) for clip in clips if clip.t0 is not None)
    return any(
        following - previous <= tolerance
        for previous, following in zip(starts, starts[1:])
    )


def load_train_clips(meta_path: Path) -> dict[str, Clip]:
    """Load all train clips; recording identity is exactly ``(user, trial)``."""

    rows = _unique_rows(meta_path, "sample_id")
    result: dict[str, Clip] = {}
    for sid, row in rows.items():
        user = row.get("user", "")
        trial = row.get("trial", "")
        if not user or not trial:
            raise ValueError(f"missing user/trial metadata for {sid}")
        label = int(row["class_id"])
        if not 0 <= label < N_CLASSES:
            raise ValueError(f"class_id out of range for {sid}: {label}")
        result[sid] = Clip(
            sid=sid,
            owner=user,
            recording_key=f"train:{user}:{trial}",
            t0=_optional_time(row.get("t0"), field="t0", sid=sid),
            t1=_optional_time(row.get("t1"), field="t1", sid=sid),
            user=user,
            label=label,
        )
    return result


def load_test_clips(
    meta_path: Path,
    cohort_path: Path,
) -> dict[str, Clip]:
    """Join test timing with cohort slot and Radar recording identities."""

    meta = _unique_rows(meta_path, "sample_id")
    cohorts = _unique_rows(cohort_path, "sm_id")
    if set(meta) != set(cohorts):
        missing = sorted(set(meta) - set(cohorts))
        extra = sorted(set(cohorts) - set(meta))
        raise ValueError(
            "meta_test/test_cohorts ID mismatch: "
            f"missing_cohorts={missing[:3]} extra_cohorts={extra[:3]}"
        )

    result: dict[str, Clip] = {}
    for sid, row in meta.items():
        cohort = cohorts[sid]
        slot = cohort.get("slot_id", "")
        if not slot:
            raise ValueError(f"missing slot_id for {sid}")
        radar_timestamp = cohort.get("radar_rec_ts", "").strip()
        recording = (
            f"test:radar:{radar_timestamp}"
            if radar_timestamp
            else f"test:singleton:{sid}"
        )
        result[sid] = Clip(
            sid=sid,
            owner=f"slot:{slot}",
            recording_key=recording,
            t0=_optional_time(row.get("t0"), field="t0", sid=sid),
            t1=_optional_time(row.get("t1"), field="t1", sid=sid),
            user=None,
            label=None,
        )
    return result


def build_recordings(
    clips: Mapping[str, Clip],
    allowed_sids: set[str] | None = None,
) -> list[Recording]:
    """Build ordered recordings, retaining whether selected rows are complete."""

    grouped: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips.values():
        grouped[clip.recording_key].append(clip)

    recordings: list[Recording] = []
    for key, full_clips in grouped.items():
        selected = [
            clip
            for clip in full_clips
            if allowed_sids is None or clip.sid in allowed_sids
        ]
        if not selected:
            continue
        owners = {clip.owner for clip in full_clips}
        if len(owners) != 1:
            raise ValueError(f"recording {key!r} spans owners: {sorted(owners)}")
        ambiguous = _recording_order_ambiguous(full_clips)
        # Never use SID as an ordering fallback: train IDs begin with class and
        # would leak the answer into an otherwise unresolved temporal tie.
        if not ambiguous:
            selected.sort(key=lambda clip: float(clip.t0))
        start, end = _recording_bounds(selected)
        recordings.append(
            Recording(
                key=key,
                owner=selected[0].owner,
                clips=tuple(selected),
                start=start,
                end=end,
                complete=len(selected) == len(full_clips),
                ambiguous=ambiguous,
            )
        )
    recordings.sort(key=lambda rec: (rec.owner, rec.start, rec.key))
    return recordings


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
        raise ValueError(
            f"probabilities must have shape (N,{N_CLASSES}), got "
            f"{probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if (probabilities < 0).any():
        raise ValueError("probabilities contain negative values")
    totals = probabilities.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("at least one probability row sums to zero")
    return probabilities / totals


def load_probability_file(
    path: Path,
    key: str,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files or "sids" not in payload.files:
            raise KeyError(f"{path} must contain {key!r} and 'sids'")
        probabilities = validate_probabilities(payload[key])
        sids = [str(value) for value in payload["sids"]]
        labels = (
            np.asarray(payload["labels"], dtype=np.int64)
            if "labels" in payload.files
            else None
        )
    if len(sids) != len(probabilities):
        raise ValueError("probability and sample-ID lengths differ")
    if len(sids) != len(set(sids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    if labels is not None:
        if len(labels) != len(sids):
            raise ValueError("label and sample-ID lengths differ")
        if ((labels < 0) | (labels >= N_CLASSES)).any():
            raise ValueError("labels fall outside the 40-class range")
    return probabilities, labels, sids


def mean_aligned_cosine(
    left: Recording,
    right: Recording,
    probabilities: np.ndarray,
    sid_to_row: Mapping[str, int],
) -> float:
    """Mean cosine between probability rows at aligned clip positions."""

    if len(left.clips) != len(right.clips):
        raise ValueError("aligned cosine requires equal-length recordings")
    left_rows = np.asarray([sid_to_row[clip.sid] for clip in left.clips])
    right_rows = np.asarray([sid_to_row[clip.sid] for clip in right.clips])
    a = probabilities[left_rows]
    b = probabilities[right_rows]
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if (denominator <= 0).any():
        raise ValueError("zero-norm probability vector in cosine calculation")
    return float(np.mean(np.sum(a * b, axis=1) / denominator))


def cluster_repetitions(
    recordings: Sequence[Recording],
    probabilities: np.ndarray,
    sids: Sequence[str],
    *,
    gap_seconds: float,
    cosine_threshold: float,
) -> tuple[list[tuple[Recording, ...]], ConsensusStats]:
    """Form maximal runs linked by eligible consecutive recording pairs."""

    if gap_seconds < 0:
        raise ValueError("gap threshold must be non-negative")
    if not -1.0 <= cosine_threshold <= 1.0:
        raise ValueError("cosine threshold must be in [-1,1]")
    sid_to_row = {sid: row for row, sid in enumerate(sids)}
    if len(sid_to_row) != len(sids):
        raise ValueError("duplicate sample IDs")
    recording_sids = {clip.sid for rec in recordings for clip in rec.clips}
    if not recording_sids.issubset(sid_to_row):
        missing = sorted(recording_sids - set(sid_to_row))
        raise ValueError(f"recording rows absent from probabilities: {missing[:3]}")

    by_owner: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        by_owner[recording.owner].append(recording)
    for owner_recordings in by_owner.values():
        owner_recordings.sort(key=lambda rec: (rec.start, rec.key))

    stats = ConsensusStats(
        owners=len(by_owner),
        recordings=len(recordings),
        complete_recordings=sum(recording.complete for recording in recordings),
        incomplete_recordings=sum(not recording.complete for recording in recordings),
        ambiguous_recordings=sum(recording.ambiguous for recording in recordings),
    )
    clusters: list[tuple[Recording, ...]] = []
    linked_gaps: list[float] = []
    linked_cosines: list[float] = []

    for owner in sorted(by_owner):
        ordered = by_owner[owner]
        if not ordered:
            continue
        current = [ordered[0]]
        for previous, following in zip(ordered, ordered[1:]):
            stats.adjacent_pairs += 1
            linked = False
            if (
                previous.complete
                and following.complete
                and not previous.ambiguous
                and not following.ambiguous
                and len(previous.clips) == len(following.clips)
            ):
                stats.equal_length_pairs += 1
                gap = following.start - previous.end
                if math.isfinite(gap) and gap <= gap_seconds:
                    stats.gap_eligible_pairs += 1
                    cosine = mean_aligned_cosine(
                        previous, following, probabilities, sid_to_row
                    )
                    stats.cosine_evaluated_pairs += 1
                    if cosine >= cosine_threshold:
                        linked = True
                        stats.linked_pairs += 1
                        linked_gaps.append(float(gap))
                        linked_cosines.append(float(cosine))
            if linked:
                current.append(following)
            else:
                if len(current) > 1:
                    clusters.append(tuple(current))
                current = [following]
        if len(current) > 1:
            clusters.append(tuple(current))

    stats.consensus_clusters = len(clusters)
    stats.consensus_recordings = sum(len(cluster) for cluster in clusters)
    stats.consensus_clips = sum(
        len(recording.clips)
        for cluster in clusters
        for recording in cluster
    )
    stats.max_cluster_recordings = max(
        (len(cluster) for cluster in clusters), default=0
    )
    stats.mean_link_gap_seconds = (
        float(np.mean(linked_gaps)) if linked_gaps else None
    )
    stats.mean_link_cosine = (
        float(np.mean(linked_cosines)) if linked_cosines else None
    )
    return clusters, stats


def apply_repetition_consensus(
    probabilities: np.ndarray,
    sids: Sequence[str],
    recordings: Sequence[Recording],
    *,
    gap_seconds: float,
    cosine_threshold: float,
    weight: float,
) -> tuple[np.ndarray, ConsensusStats]:
    """Blend each linked repetition cluster with its aligned position mean."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("consensus weight must be in [0,1]")
    probabilities = validate_probabilities(probabilities)
    if len(probabilities) != len(sids):
        raise ValueError("probability and sample-ID lengths differ")
    clusters, stats = cluster_repetitions(
        recordings,
        probabilities,
        sids,
        gap_seconds=gap_seconds,
        cosine_threshold=cosine_threshold,
    )
    sid_to_row = {sid: row for row, sid in enumerate(sids)}
    adjusted = probabilities.copy()
    for cluster in clusters:
        aligned_rows = np.asarray(
            [
                [sid_to_row[clip.sid] for clip in recording.clips]
                for recording in cluster
            ],
            dtype=np.int64,
        )
        aligned = probabilities[aligned_rows]
        position_mean = aligned.mean(axis=0)
        adjusted[aligned_rows] = (
            (1.0 - weight) * aligned + weight * position_mean[None, :, :]
        )

    adjusted = validate_probabilities(adjusted)
    stats.argmax_changes = int(
        np.sum(adjusted.argmax(axis=1) != probabilities.argmax(axis=1))
    )
    stats.mean_probability_l1 = float(
        np.mean(np.abs(adjusted - probabilities).sum(axis=1))
    )
    return adjusted, stats


def fit_transition_matrix(
    recordings: Sequence[Recording],
) -> tuple[np.ndarray, TransitionStats]:
    """Fit uniform add-one, row-conditional transitions with no start term."""

    counts = np.ones((N_CLASSES, N_CLASSES), dtype=np.float64)
    source_clips = 0
    source_edges = 0
    source_recordings = 0
    source_ambiguous_recordings = 0
    for recording in recordings:
        if recording.ambiguous:
            source_ambiguous_recordings += 1
            continue
        labels = [clip.label for clip in recording.clips]
        if any(label is None for label in labels):
            raise ValueError("transition source recording contains unlabeled clips")
        row = [int(label) for label in labels if label is not None]
        if not row:
            continue
        source_recordings += 1
        source_clips += len(row)
        for left, right in zip(row, row[1:]):
            counts[left, right] += 1.0
            source_edges += 1
    if source_clips == 0:
        raise ValueError("no labeled source clips available for transitions")
    conditional = counts / counts.sum(axis=1, keepdims=True)
    return np.log(conditional), TransitionStats(
        source_recordings=source_recordings,
        source_clips=source_clips,
        source_edges=source_edges,
        source_ambiguous_recordings=source_ambiguous_recordings,
    )


def recordings_excluding_owners(
    recordings: Sequence[Recording],
    excluded_owners: set[str],
) -> list[Recording]:
    """Return a transition source with an explicit owner-disjointness guard."""

    source = [
        recording
        for recording in recordings
        if recording.owner not in excluded_owners
    ]
    overlap = {recording.owner for recording in source} & excluded_owners
    if overlap:
        raise AssertionError(
            f"transition source contains excluded owners: {sorted(overlap)}"
        )
    return source


def viterbi_decode(
    probabilities: np.ndarray,
    log_transition: np.ndarray,
    transition_weight: float,
) -> np.ndarray:
    """Exact first-order Viterbi decode without a start-class score."""

    probabilities = validate_probabilities(probabilities)
    if log_transition.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(
            "transition matrix must have shape "
            f"({N_CLASSES},{N_CLASSES}), got {log_transition.shape}"
        )
    if not np.isfinite(log_transition).all():
        raise ValueError("transition matrix contains NaN or infinity")
    if transition_weight < 0:
        raise ValueError("transition weight must be non-negative")
    if len(probabilities) == 0:
        return np.empty(0, dtype=np.int64)
    if transition_weight == 0:
        return probabilities.argmax(axis=1).astype(np.int64)

    emissions = np.log(np.clip(probabilities, 1e-12, 1.0))
    dynamic = emissions[0].copy()
    backpointers = np.zeros((len(probabilities), N_CLASSES), dtype=np.int16)
    for step in range(1, len(probabilities)):
        scores = dynamic[:, None] + transition_weight * log_transition
        backpointers[step] = scores.argmax(axis=0)
        dynamic = emissions[step] + scores.max(axis=0)
    path = np.empty(len(probabilities), dtype=np.int64)
    path[-1] = int(dynamic.argmax())
    for step in range(len(probabilities) - 1, 0, -1):
        path[step - 1] = backpointers[step, path[step]]
    return path


def decode_recordings(
    probabilities: np.ndarray,
    sids: Sequence[str],
    recordings: Sequence[Recording],
    log_transition: np.ndarray,
    *,
    transition_weight: float,
    source_stats: TransitionStats,
) -> tuple[np.ndarray, TransitionStats]:
    """Decode each metadata-native recording independently."""

    probabilities = validate_probabilities(probabilities)
    sid_to_row = {sid: row for row, sid in enumerate(sids)}
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    multi_recordings = 0
    multi_clips = 0
    ambiguous_recordings = 0
    ambiguous_clips = 0
    for recording in recordings:
        rows = np.asarray([sid_to_row[clip.sid] for clip in recording.clips])
        if recording.ambiguous:
            ambiguous_recordings += 1
            ambiguous_clips += len(rows)
            continue
        decoded = viterbi_decode(
            probabilities[rows], log_transition, transition_weight
        )
        predictions[rows] = decoded
        if len(rows) > 1:
            multi_recordings += 1
            multi_clips += len(rows)
    base = probabilities.argmax(axis=1)
    stats = TransitionStats(
        source_recordings=source_stats.source_recordings,
        source_clips=source_stats.source_clips,
        source_edges=source_stats.source_edges,
        source_ambiguous_recordings=source_stats.source_ambiguous_recordings,
        target_recordings=len(recordings),
        target_ambiguous_recordings=ambiguous_recordings,
        target_ambiguous_clips=ambiguous_clips,
        target_multi_recordings=multi_recordings,
        target_multi_clips=multi_clips,
        argmax_changes=int(np.sum(predictions != base)),
    )
    return predictions, stats


def _load_fold_users(path: Path) -> list[set[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        payload = json.load(handle)
    if "folds" not in payload or not payload["folds"]:
        raise ValueError(f"{path} does not contain a non-empty 'folds' list")
    folds: list[set[str]] = []
    seen: set[str] = set()
    for index, fold in enumerate(payload["folds"]):
        users = set(fold.get("val_users", ()))
        if not users:
            raise ValueError(f"fold {index} has no val_users")
        overlap = seen & users
        if overlap:
            raise ValueError(f"users appear in multiple folds: {sorted(overlap)}")
        seen |= users
        folds.append(users)
    return folds


def _parse_grid(
    raw: str,
    fallback: float,
    *,
    name: str,
    lower: float,
    upper: float | None = None,
) -> list[float]:
    values: list[float] = []
    text = raw.strip()
    tokens = text.split(",") if text else [str(fallback)]
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value")
        if value < lower or (upper is not None and value > upper):
            interval = f"[{lower},{upper}]" if upper is not None else f">={lower}"
            raise ValueError(f"{name} value {value} is outside {interval}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{name} did not contain any values")
    return sorted(values)


def _candidate_grid(args: argparse.Namespace) -> list[Candidate]:
    gaps = _parse_grid(
        args.gap_grid,
        args.gap_seconds,
        name="gap grid",
        lower=0.0,
    )
    cosines = _parse_grid(
        args.cosine_grid,
        args.cosine_threshold,
        name="cosine grid",
        lower=-1.0,
        upper=1.0,
    )
    weights = _parse_grid(
        args.consensus_weight_grid,
        args.consensus_weight,
        name="consensus-weight grid",
        lower=0.0,
        upper=1.0,
    )
    lambdas = _parse_grid(
        args.lambda_grid,
        args.transition_weight,
        name="lambda grid",
        lower=0.0,
    )
    candidates = [
        Candidate(*values)
        for values in itertools.product(gaps, cosines, weights, lambdas)
    ]
    if len(candidates) > args.max_grid_candidates:
        raise ValueError(
            f"grid has {len(candidates)} candidates; limit is "
            f"{args.max_grid_candidates} (raise --max-grid-candidates explicitly)"
        )
    return candidates


def _candidate_tie_key(candidate: Candidate, index: int) -> tuple[float, ...]:
    """Prefer the least intervention when validation correct counts tie."""

    return (
        -candidate.consensus_weight,
        -candidate.transition_weight,
        -candidate.gap_seconds,
        candidate.cosine_threshold,
        -float(index),
    )


def select_candidates_strict_nested(
    evaluations: Sequence[CandidateEvaluation],
    nested_correct: np.ndarray,
) -> dict[int, int]:
    """Select candidates from scores produced by strict outer/inner fits.

    ``nested_correct[outer, candidate]`` must already aggregate inner-fold
    predictions whose transition sources excluded ``outer UNION inner`` users.
    Keeping fitting outside this pure selector makes that exclusion auditable.
    """

    if not evaluations:
        raise ValueError("no candidate evaluations")
    if (
        nested_correct.ndim != 2
        or nested_correct.shape[1] != len(evaluations)
    ):
        raise ValueError(
            "nested score matrix must have one column per candidate"
        )
    selected: dict[int, int] = {}
    for outer_fold in range(nested_correct.shape[0]):
        ranked = [
            (
                int(nested_correct[outer_fold, index]),
                *_candidate_tie_key(evaluation.candidate, index),
                index,
            )
            for index, evaluation in enumerate(evaluations)
        ]
        selected[outer_fold] = int(max(ranked)[-1])
    return selected


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> str:
    correct = int(np.sum(predictions == labels))
    return f"{correct / max(len(labels), 1):.5f} ({correct}/{len(labels)})"


def _stats_dict(stats: ConsensusStats | TransitionStats) -> dict[str, object]:
    return asdict(stats)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _common_provenance(
    *,
    mode: str,
    args: argparse.Namespace,
    inputs: Sequence[Path],
) -> dict[str, object]:
    existing = [path.resolve() for path in inputs if path.is_file()]
    return {
        "schema": "cuhkx.repetition_consensus_decoder.provenance.v2",
        "decoder_version": VERSION,
        "mode": mode,
        "command": [str(value) for value in sys.argv],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _hash_file(Path(__file__).resolve()),
        "inputs": {
            str(path): {"sha256": _hash_file(path), "bytes": path.stat().st_size}
            for path in existing
        },
        "mechanics": {
            "train_recording": "(user,trial), ordered by meta_train.t0",
            "test_recording": "test_cohorts.radar_rec_ts, ordered by meta_test.t0",
            "repeat_owner_train": "user",
            "repeat_owner_test": "test_cohorts.slot_id",
            "repeat_link": (
                "consecutive complete, order-unambiguous, equal-length recordings; "
                "end-to-start gap <= threshold; mean aligned cosine >= threshold"
            ),
            "order_ambiguity": (
                "any missing t0/t1 or within-recording t0 separation <=1e-6; "
                "ambiguous recordings are not fitted, aligned, or decoded"
            ),
            "consensus_blend": "(1-weight)*original + weight*aligned_cluster_mean",
            "transition": (
                "40x40 uniform add-one row-conditional first-order Viterbi; "
                "no start-class score"
            ),
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _official_test_order(path: Path) -> tuple[list[str], dict[str, str]]:
    rows = _read_csv(path)
    sids: list[str] = []
    paths: dict[str, str] = {}
    for row in rows:
        raw_path = row.get("path", "")
        match = TEST_SID_RE.search(raw_path)
        if not match:
            raise ValueError(f"cannot parse test ID from path={raw_path!r}")
        sid = match.group(1)
        if sid in paths:
            raise ValueError(f"duplicate {sid} in {path}")
        sids.append(sid)
        paths[sid] = raw_path
    return sids, paths


def _atomic_submission(
    path: Path,
    official_sids: Sequence[str],
    official_paths: Mapping[str, str],
    predictions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(("path", "prediction"))
            for sid, prediction in zip(official_sids, predictions):
                writer.writerow((official_paths[sid], int(prediction)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _provenance_path(
    explicit: Path | None,
    primary_output: Path | None,
) -> Path | None:
    if explicit is not None:
        return explicit
    if primary_output is None:
        return None
    return primary_output.with_name(primary_output.name + ".provenance.json")


def _histogram(values: np.ndarray) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in sorted(Counter(values.tolist()).items())
    }


def run_oof(args: argparse.Namespace) -> int:
    probabilities, labels, sids = load_probability_file(args.probs, args.probs_key)
    if labels is None:
        raise ValueError("OOF mode requires a 'labels' array")
    train_clips = load_train_clips(args.meta_train)
    missing = sorted(set(sids) - set(train_clips))
    if missing:
        raise ValueError(f"OOF IDs absent from meta_train: {missing[:3]}")
    metadata_labels = np.asarray([train_clips[sid].label for sid in sids])
    if not np.array_equal(labels, metadata_labels):
        bad = np.flatnonzero(labels != metadata_labels)
        raise ValueError(
            f"OOF labels disagree with metadata for {len(bad)} rows; "
            f"first={sids[int(bad[0])]}"
        )

    fold_users = _load_fold_users(args.folds)
    user_to_fold: dict[str, int] = {}
    for fold, users in enumerate(fold_users):
        for user in users:
            user_to_fold[user] = fold
    row_folds = np.asarray(
        [user_to_fold.get(str(train_clips[sid].user), -1) for sid in sids],
        dtype=np.int64,
    )
    if (row_folds < 0).any():
        missing_users = sorted(
            {
                str(train_clips[sids[index]].user)
                for index in np.flatnonzero(row_folds < 0)
            }
        )
        raise ValueError(
            "OOF users absent from validation folds: " + ",".join(missing_users)
        )

    all_train_recordings = build_recordings(train_clips)
    target_recordings: dict[int, list[Recording]] = {}
    target_indices: dict[int, np.ndarray] = {}
    transition_models: dict[int, tuple[np.ndarray, TransitionStats]] = {}
    for fold, validation_users in enumerate(fold_users):
        indices = np.flatnonzero(row_folds == fold)
        if not len(indices):
            raise ValueError(f"fold {fold} has no OOF rows")
        target_indices[fold] = indices
        fold_sids = {sids[index] for index in indices}
        recordings = build_recordings(train_clips, fold_sids)
        target_recordings[fold] = recordings
        source = recordings_excluding_owners(
            all_train_recordings, validation_users
        )
        transition_models[fold] = fit_transition_matrix(source)

    candidates = _candidate_grid(args)
    base = probabilities.argmax(axis=1).astype(np.int64)
    consensus_cache: dict[
        tuple[int, float, float, float], tuple[np.ndarray, ConsensusStats]
    ] = {}
    evaluations: list[CandidateEvaluation] = []
    print(
        f"OOF input: n={len(sids)} base={_accuracy(base, labels)} "
        f"candidates={len(candidates)}"
    )

    for candidate in candidates:
        final_predictions = base.copy()
        consensus_predictions = base.copy()
        consensus_stats: dict[int, ConsensusStats] = {}
        transition_stats: dict[int, TransitionStats] = {}
        for fold in range(len(fold_users)):
            indices = target_indices[fold]
            local_sids = [sids[index] for index in indices]
            cache_key = (
                fold,
                candidate.gap_seconds,
                candidate.cosine_threshold,
                candidate.consensus_weight,
            )
            cached = consensus_cache.get(cache_key)
            if cached is None:
                adjusted, stats = apply_repetition_consensus(
                    probabilities[indices],
                    local_sids,
                    target_recordings[fold],
                    gap_seconds=candidate.gap_seconds,
                    cosine_threshold=candidate.cosine_threshold,
                    weight=candidate.consensus_weight,
                )
                consensus_cache[cache_key] = (adjusted, stats)
            else:
                adjusted, stats = cached
            log_transition, source_stats = transition_models[fold]
            decoded, decode_stats = decode_recordings(
                adjusted,
                local_sids,
                target_recordings[fold],
                log_transition,
                transition_weight=candidate.transition_weight,
                source_stats=source_stats,
            )
            consensus_predictions[indices] = adjusted.argmax(axis=1)
            final_predictions[indices] = decoded
            consensus_stats[fold] = stats
            transition_stats[fold] = decode_stats
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                predictions=final_predictions,
                consensus_predictions=consensus_predictions,
                consensus_stats=consensus_stats,
                transition_stats=transition_stats,
            )
        )

    nested_correct: np.ndarray | None = None
    nested_totals: np.ndarray | None = None
    nested_source_payload: dict[str, dict[str, object]] = {}
    if len(evaluations) == 1:
        selected_indices = {fold: 0 for fold in range(len(fold_users))}
        selection = "fixed"
    else:
        print("descriptive outer-fold candidate grid (not used for selection):")
        for index, evaluation in enumerate(evaluations):
            print(
                f"  [{index:03d}] {evaluation.candidate.label()} "
                f"consensus={_accuracy(evaluation.consensus_predictions, labels)} "
                f"final={_accuracy(evaluation.predictions, labels)}"
            )
        n_folds = len(fold_users)
        nested_correct = np.zeros(
            (n_folds, len(evaluations)), dtype=np.int64
        )
        nested_totals = np.zeros(n_folds, dtype=np.int64)
        for outer_fold in range(n_folds):
            outer_payload: dict[str, object] = {}
            for inner_fold in range(n_folds):
                if inner_fold == outer_fold:
                    continue
                excluded_users = fold_users[outer_fold] | fold_users[inner_fold]
                source = recordings_excluding_owners(
                    all_train_recordings, excluded_users
                )
                nested_transition, nested_source_stats = fit_transition_matrix(
                    source
                )
                indices = target_indices[inner_fold]
                local_sids = [sids[index] for index in indices]
                nested_totals[outer_fold] += len(indices)
                for candidate_index, evaluation in enumerate(evaluations):
                    candidate = evaluation.candidate
                    adjusted, _ = consensus_cache[
                        (
                            inner_fold,
                            candidate.gap_seconds,
                            candidate.cosine_threshold,
                            candidate.consensus_weight,
                        )
                    ]
                    nested_predictions, _ = decode_recordings(
                        adjusted,
                        local_sids,
                        target_recordings[inner_fold],
                        nested_transition,
                        transition_weight=candidate.transition_weight,
                        source_stats=nested_source_stats,
                    )
                    nested_correct[outer_fold, candidate_index] += int(
                        np.sum(nested_predictions == labels[indices])
                    )
                outer_payload[str(inner_fold)] = {
                    "excluded_users": sorted(excluded_users),
                    "source_stats": _stats_dict(nested_source_stats),
                    "inner_rows": int(len(indices)),
                }
            nested_source_payload[str(outer_fold)] = outer_payload
        selected_indices = select_candidates_strict_nested(
            evaluations, nested_correct
        )
        selection = "strict-nested"
        print("strict nested selection (inner source excludes outer UNION inner):")
        for outer_fold in range(n_folds):
            selected = selected_indices[outer_fold]
            print(
                f"  outer {outer_fold}: selected=[{selected:03d}] "
                f"{evaluations[selected].candidate.label()} "
                f"inner={nested_correct[outer_fold, selected]}/"
                f"{nested_totals[outer_fold]}"
            )

    selected_probabilities = probabilities.copy()
    selected_consensus_predictions = base.copy()
    selected_predictions = base.copy()
    for fold in range(len(fold_users)):
        evaluation = evaluations[selected_indices[fold]]
        candidate = evaluation.candidate
        indices = target_indices[fold]
        adjusted, _ = consensus_cache[
            (
                fold,
                candidate.gap_seconds,
                candidate.cosine_threshold,
                candidate.consensus_weight,
            )
        ]
        selected_probabilities[indices] = adjusted
        selected_consensus_predictions[indices] = evaluation.consensus_predictions[
            indices
        ]
        selected_predictions[indices] = evaluation.predictions[indices]

        fold_labels = labels[indices]
        cstats = evaluation.consensus_stats[fold]
        tstats = evaluation.transition_stats[fold]
        print(
            f"fold {fold}: val_users={sorted(fold_users[fold])} "
            f"selected=[{selected_indices[fold]:03d}] {candidate.label()} "
            f"base={_accuracy(base[indices], fold_labels)} "
            f"consensus={_accuracy(selected_consensus_predictions[indices], fold_labels)} "
            f"final={_accuracy(selected_predictions[indices], fold_labels)}; "
            f"repeat_clusters={cstats.consensus_clusters} "
            f"repeat_clips={cstats.consensus_clips} "
            f"transition_source={tstats.source_clips}clips/{tstats.source_edges}edges "
            f"ambiguous=target:{tstats.target_ambiguous_recordings}"
            f"/{tstats.target_ambiguous_clips}clips,"
            f"source:{tstats.source_ambiguous_recordings}recs "
            f"changes={tstats.argmax_changes}"
        )

    base_correct = base == labels
    final_correct = selected_predictions == labels
    print(
        f"OOF consensus={_accuracy(selected_consensus_predictions, labels)} "
        f"final={_accuracy(selected_predictions, labels)} "
        f"delta={final_correct.mean() - base_correct.mean():+.5f} "
        f"changed={int(np.sum(selected_predictions != base))} "
        f"rescues={int(np.sum(~base_correct & final_correct))} "
        f"harms={int(np.sum(base_correct & ~final_correct))} "
        f"selection={selection}"
    )

    if (
        all(candidate.consensus_weight == 0 for candidate in candidates)
        and all(candidate.transition_weight == 0 for candidate in candidates)
        and not np.array_equal(selected_predictions, base)
    ):
        raise AssertionError("zero-weight consensus/transition must reproduce argmax")

    selected_payload = {
        str(fold): {
            "candidate_index": selected_indices[fold],
            "candidate": asdict(evaluations[selected_indices[fold]].candidate),
            "validation_users": sorted(fold_users[fold]),
            "consensus_stats": _stats_dict(
                evaluations[selected_indices[fold]].consensus_stats[fold]
            ),
            "transition_stats": _stats_dict(
                evaluations[selected_indices[fold]].transition_stats[fold]
            ),
            "selection_evidence": (
                None
                if nested_correct is None or nested_totals is None
                else {
                    "strict_nested_correct": int(
                        nested_correct[fold, selected_indices[fold]]
                    ),
                    "strict_nested_rows": int(nested_totals[fold]),
                    "inner_folds": [
                        inner
                        for inner in range(len(fold_users))
                        if inner != fold
                    ],
                }
            ),
            "base_accuracy": float(
                np.mean(base[target_indices[fold]] == labels[target_indices[fold]])
            ),
            "consensus_accuracy": float(
                np.mean(
                    selected_consensus_predictions[target_indices[fold]]
                    == labels[target_indices[fold]]
                )
            ),
            "final_accuracy": float(
                np.mean(
                    selected_predictions[target_indices[fold]]
                    == labels[target_indices[fold]]
                )
            ),
        }
        for fold in range(len(fold_users))
    }
    provenance = _common_provenance(
        mode="oof",
        args=args,
        inputs=(args.probs, args.meta_train, args.folds),
    )
    provenance.update(
        {
            "probability_key": args.probs_key,
            "selection": selection,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "index": index,
                    "parameters": asdict(evaluation.candidate),
                    "consensus_accuracy": float(
                        np.mean(evaluation.consensus_predictions == labels)
                    ),
                    "final_accuracy": float(
                        np.mean(evaluation.predictions == labels)
                    ),
                    "strict_nested_correct_by_outer": (
                        None
                        if nested_correct is None
                        else [
                            int(value)
                            for value in nested_correct[:, index]
                        ]
                    ),
                }
                for index, evaluation in enumerate(evaluations)
            ],
            "selected_by_fold": selected_payload,
            "strict_nested_sources": nested_source_payload,
            "summary": {
                "rows": len(sids),
                "base_accuracy": float(np.mean(base_correct)),
                "consensus_accuracy": float(
                    np.mean(selected_consensus_predictions == labels)
                ),
                "final_accuracy": float(np.mean(final_correct)),
                "prediction_changes": int(np.sum(selected_predictions != base)),
                "rescues": int(np.sum(~base_correct & final_correct)),
                "harms": int(np.sum(base_correct & ~final_correct)),
            },
            "leakage_guard": (
                "transition source for target fold excludes all target-fold users; "
                "for grid selection of outer fold f, every inner-fold j transition "
                "source excludes users in f UNION j, and no f labels are scored"
            ),
        }
    )

    if args.output is None and args.provenance_output is None:
        print("dry run: no artifact written (pass --output to write OOF NPZ)")
        return 0

    if args.output is not None:
        selected_matrix = np.asarray(
            [
                [
                    evaluations[selected_indices[fold]].candidate.gap_seconds,
                    evaluations[selected_indices[fold]].candidate.cosine_threshold,
                    evaluations[selected_indices[fold]].candidate.consensus_weight,
                    evaluations[selected_indices[fold]].candidate.transition_weight,
                ]
                for fold in range(len(fold_users))
            ],
            dtype=np.float64,
        )
        _atomic_npz(
            args.output,
            probs=selected_probabilities.astype(np.float32),
            original_probs=probabilities.astype(np.float32),
            preds=selected_predictions,
            consensus_preds=selected_consensus_predictions,
            base_preds=base,
            labels=labels,
            sids=np.asarray(sids),
            folds=row_folds,
            selected_parameters=selected_matrix,
            selected_candidate_indices=np.asarray(
                [selected_indices[fold] for fold in range(len(fold_users))],
                dtype=np.int64,
            ),
            parameter_columns=np.asarray(
                [
                    "gap_seconds",
                    "cosine_threshold",
                    "consensus_weight",
                    "transition_weight",
                ]
            ),
            provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
        print(f"wrote OOF artifact: {args.output}")
    provenance_path = _provenance_path(args.provenance_output, args.output)
    if provenance_path is not None:
        _atomic_json(provenance_path, provenance)
        print(f"wrote provenance: {provenance_path}")
    return 0


def run_test(args: argparse.Namespace) -> int:
    probabilities, labels, input_sids = load_probability_file(
        args.probs, args.probs_key
    )
    if labels is not None:
        raise ValueError("test probability artifact unexpectedly contains labels")
    official_sids, official_paths = _official_test_order(args.sample_submission)
    if set(input_sids) != set(official_sids):
        missing = sorted(set(official_sids) - set(input_sids))
        extra = sorted(set(input_sids) - set(official_sids))
        raise ValueError(
            "probability/sample-submission ID mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    input_index = {sid: row for row, sid in enumerate(input_sids)}
    probabilities = probabilities[
        np.asarray([input_index[sid] for sid in official_sids])
    ]

    train_clips = load_train_clips(args.meta_train)
    test_clips = load_test_clips(args.meta_test, args.test_cohorts)
    if set(official_sids) != set(test_clips):
        missing = sorted(set(official_sids) - set(test_clips))
        extra = sorted(set(test_clips) - set(official_sids))
        raise ValueError(
            "sample-submission/test-metadata ID mismatch: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    train_recordings = build_recordings(train_clips)
    test_recordings = build_recordings(test_clips, set(official_sids))

    candidate = Candidate(
        gap_seconds=args.gap_seconds,
        cosine_threshold=args.cosine_threshold,
        consensus_weight=args.consensus_weight,
        transition_weight=args.transition_weight,
    )
    adjusted, consensus_stats = apply_repetition_consensus(
        probabilities,
        official_sids,
        test_recordings,
        gap_seconds=candidate.gap_seconds,
        cosine_threshold=candidate.cosine_threshold,
        weight=candidate.consensus_weight,
    )
    log_transition, source_stats = fit_transition_matrix(train_recordings)
    predictions, transition_stats = decode_recordings(
        adjusted,
        official_sids,
        test_recordings,
        log_transition,
        transition_weight=candidate.transition_weight,
        source_stats=source_stats,
    )
    base = probabilities.argmax(axis=1).astype(np.int64)
    consensus_predictions = adjusted.argmax(axis=1).astype(np.int64)
    print(
        f"test n={len(official_sids)} {candidate.label()}; "
        f"recordings={len(test_recordings)} "
        f"repeat_clusters={consensus_stats.consensus_clusters} "
        f"repeat_recordings={consensus_stats.consensus_recordings} "
        f"repeat_clips={consensus_stats.consensus_clips}; "
        f"ambiguous=target:{transition_stats.target_ambiguous_recordings}"
        f"/{transition_stats.target_ambiguous_clips}clips,"
        f"source:{transition_stats.source_ambiguous_recordings}recs; "
        f"transition_source={transition_stats.source_clips}clips/"
        f"{transition_stats.source_edges}edges; "
        f"consensus_changes={int(np.sum(consensus_predictions != base))} "
        f"transition_changes={transition_stats.argmax_changes} "
        f"total_changes={int(np.sum(predictions != base))}"
    )
    print(f"base histogram: {_histogram(base)}")
    print(f"consensus histogram: {_histogram(consensus_predictions)}")
    print(f"final histogram: {_histogram(predictions)}")

    if (
        candidate.consensus_weight == 0
        and candidate.transition_weight == 0
        and not np.array_equal(predictions, base)
    ):
        raise AssertionError("zero-weight consensus/transition must reproduce argmax")

    provenance = _common_provenance(
        mode="test",
        args=args,
        inputs=(
            args.probs,
            args.meta_train,
            args.meta_test,
            args.test_cohorts,
            args.sample_submission,
        ),
    )
    provenance.update(
        {
            "probability_key": args.probs_key,
            "parameters": asdict(candidate),
            "consensus_stats": _stats_dict(consensus_stats),
            "transition_stats": _stats_dict(transition_stats),
            "summary": {
                "rows": len(official_sids),
                "base_to_consensus_changes": int(
                    np.sum(consensus_predictions != base)
                ),
                "consensus_to_final_changes": transition_stats.argmax_changes,
                "base_to_final_changes": int(np.sum(predictions != base)),
                "base_histogram": _histogram(base),
                "consensus_histogram": _histogram(consensus_predictions),
                "final_histogram": _histogram(predictions),
            },
            "row_order": {
                "source": str(args.sample_submission.resolve()),
                "exact_order_verified": True,
            },
        }
    )

    any_output = any(
        value is not None
        for value in (args.output, args.prob_output, args.provenance_output)
    )
    if not any_output:
        print(
            "dry run: no files written "
            "(pass --output to write CSV, probability NPZ, and provenance)"
        )
        return 0

    if args.output is not None:
        _atomic_submission(
            args.output, official_sids, official_paths, predictions
        )
        print(f"wrote submission: {args.output}")
    probability_output = args.prob_output
    if probability_output is None and args.output is not None:
        probability_output = args.output.with_name(args.output.stem + "_decode.npz")
    if probability_output is not None:
        _atomic_npz(
            probability_output,
            probs=adjusted.astype(np.float32),
            original_probs=probabilities.astype(np.float32),
            preds=predictions,
            consensus_preds=consensus_predictions,
            base_preds=base,
            sids=np.asarray(official_sids),
            parameters=np.asarray(
                [
                    candidate.gap_seconds,
                    candidate.cosine_threshold,
                    candidate.consensus_weight,
                    candidate.transition_weight,
                ],
                dtype=np.float64,
            ),
            parameter_columns=np.asarray(
                [
                    "gap_seconds",
                    "cosine_threshold",
                    "consensus_weight",
                    "transition_weight",
                ]
            ),
            provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
        print(f"wrote adjusted probabilities: {probability_output}")
    primary_output = args.output if args.output is not None else probability_output
    provenance_path = _provenance_path(args.provenance_output, primary_output)
    if provenance_path is not None:
        _atomic_json(provenance_path, provenance)
        print(f"wrote provenance: {provenance_path}")
    return 0


def _synthetic_recording(
    key: str,
    owner: str,
    starts: Sequence[float | None],
    sids: Sequence[str],
    labels: Sequence[int] | None = None,
    *,
    complete: bool = True,
    ends: Sequence[float | None] | None = None,
) -> Recording:
    if labels is None:
        labels = [0] * len(sids)
    if ends is None:
        ends = [
            None if start is None else float(start) + 1.0
            for start in starts
        ]
    clips = tuple(
        Clip(
            sid=sid,
            owner=owner,
            recording_key=key,
            t0=None if start is None else float(start),
            t1=None if end is None else float(end),
            user=owner,
            label=int(label),
        )
        for sid, start, end, label in zip(sids, starts, ends, labels)
    )
    start, end = _recording_bounds(clips)
    return Recording(
        key,
        owner,
        clips,
        start,
        end,
        complete,
        _recording_order_ambiguous(clips),
    )


def run_self_test(_: argparse.Namespace) -> int:
    """Run focused synthetic tests without reading competition data."""

    def distribution(primary: int, secondary: int | None = None) -> np.ndarray:
        row = np.full(N_CLASSES, 1e-4, dtype=np.float64)
        row[primary] = 0.9
        if secondary is not None:
            row[secondary] = 0.08
        return row / row.sum()

    r1 = _synthetic_recording("r1", "u1", [0, 2], ["a0", "a1"])
    r2 = _synthetic_recording("r2", "u1", [10, 12], ["b0", "b1"])
    r3 = _synthetic_recording("r3", "u1", [20, 22], ["c0", "c1"])
    r4 = _synthetic_recording("r4", "u1", [30], ["d0"])
    sids = ["a0", "a1", "b0", "b1", "c0", "c1", "d0"]
    probabilities = np.stack(
        [
            distribution(0, 1),
            distribution(1, 0),
            distribution(0, 2),
            distribution(1, 2),
            distribution(8),
            distribution(9),
            distribution(0),
        ]
    )
    adjusted, stats = apply_repetition_consensus(
        probabilities,
        sids,
        [r1, r2, r3, r4],
        gap_seconds=20.0,
        cosine_threshold=0.6,
        weight=1.0,
    )
    assert stats.consensus_clusters == 1
    assert stats.consensus_recordings == 2
    assert stats.consensus_clips == 4
    assert np.allclose(adjusted[0], adjusted[2])
    assert np.allclose(adjusted[1], adjusted[3])
    assert not np.allclose(adjusted[2], adjusted[4])
    assert np.allclose(adjusted[6], probabilities[6])

    unchanged, _ = apply_repetition_consensus(
        probabilities,
        sids,
        [r1, r2, r3, r4],
        gap_seconds=20.0,
        cosine_threshold=0.6,
        weight=0.0,
    )
    assert np.allclose(unchanged, validate_probabilities(probabilities))

    ambiguous_left = _synthetic_recording(
        "ambiguous-left",
        "u2",
        [40.0, 40.0 + 5e-7],
        ["amb-a0", "amb-a1"],
    )
    ambiguous_right = _synthetic_recording(
        "ambiguous-right",
        "u2",
        [50.0, 50.0 + 5e-7],
        ["amb-b0", "amb-b1"],
    )
    ambiguous_sids = ["amb-a0", "amb-a1", "amb-b0", "amb-b1"]
    ambiguous_probabilities = np.stack(
        [distribution(0), distribution(1), distribution(0), distribution(1)]
    )
    ambiguous_adjusted, ambiguous_stats = apply_repetition_consensus(
        ambiguous_probabilities,
        ambiguous_sids,
        [ambiguous_left, ambiguous_right],
        gap_seconds=20.0,
        cosine_threshold=0.6,
        weight=1.0,
    )
    assert ambiguous_left.ambiguous and ambiguous_right.ambiguous
    assert ambiguous_stats.ambiguous_recordings == 2
    assert ambiguous_stats.consensus_clusters == 0
    assert np.allclose(ambiguous_adjusted, ambiguous_probabilities)

    class_last = Clip(
        sid="39_class_last",
        owner="tie-owner",
        recording_key="tie-recording",
        t0=100.0,
        t1=101.0,
        user="tie-owner",
        label=39,
    )
    class_first = Clip(
        sid="00_class_first",
        owner="tie-owner",
        recording_key="tie-recording",
        t0=100.0 + 5e-7,
        t1=101.0,
        user="tie-owner",
        label=0,
    )
    tied = build_recordings(
        {class_last.sid: class_last, class_first.sid: class_first}
    )[0]
    assert tied.ambiguous
    assert [clip.sid for clip in tied.clips] == [
        "39_class_last",
        "00_class_first",
    ]
    missing_time = _synthetic_recording(
        "missing-time",
        "u3",
        [0.0, None],
        ["missing-a", "missing-b"],
    )
    assert missing_time.ambiguous

    transition_sources = [
        _synthetic_recording(
            f"source-{index}",
            "source",
            [0, 1],
            [f"s{index}a", f"s{index}b"],
            [0, 1],
        )
        for index in range(10)
    ]
    ambiguous_source = _synthetic_recording(
        "ambiguous-source",
        "source",
        [2.0, 2.0],
        ["amb-source-a", "amb-source-b"],
        [0, 39],
    )
    log_transition, source_stats = fit_transition_matrix(
        transition_sources + [ambiguous_source]
    )
    assert log_transition[0, 1] > log_transition[0, 2]
    emissions = np.stack(
        [
            distribution(0),
            distribution(2, 1),
        ]
    )
    assert np.array_equal(viterbi_decode(emissions, log_transition, 0), [0, 2])
    assert np.array_equal(viterbi_decode(emissions, log_transition, 2), [0, 1])
    assert source_stats.source_edges == 10
    assert source_stats.source_ambiguous_recordings == 1
    ambiguous_decoded, ambiguous_decode_stats = decode_recordings(
        emissions,
        ["amb-source-a", "amb-source-b"],
        [ambiguous_source],
        log_transition,
        transition_weight=2.0,
        source_stats=source_stats,
    )
    assert np.array_equal(ambiguous_decoded, [0, 2])
    assert ambiguous_decode_stats.target_ambiguous_recordings == 1
    assert ambiguous_decode_stats.target_ambiguous_clips == 2

    strict_sources = [
        _synthetic_recording("outer", "outer-user", [0], ["outer-row"]),
        _synthetic_recording("inner", "inner-user", [0], ["inner-row"]),
        _synthetic_recording("allowed", "allowed-user", [0], ["allowed-row"]),
    ]
    strict_source = recordings_excluding_owners(
        strict_sources, {"outer-user", "inner-user"}
    )
    assert [recording.owner for recording in strict_source] == ["allowed-user"]

    candidate_a = Candidate(10, 0.6, 0, 0)
    candidate_b = Candidate(10, 0.6, 1, 1)
    evaluation_a = CandidateEvaluation(
        candidate_a,
        predictions=np.asarray([0, 0, 0, 0]),
        consensus_predictions=np.zeros(4, dtype=np.int64),
        consensus_stats={},
        transition_stats={},
    )
    evaluation_b = CandidateEvaluation(
        candidate_b,
        predictions=np.asarray([1, 1, 1, 1]),
        consensus_predictions=np.ones(4, dtype=np.int64),
        consensus_stats={},
        transition_stats={},
    )
    selected = select_candidates_strict_nested(
        [evaluation_a, evaluation_b],
        np.asarray([[0, 2], [2, 0]], dtype=np.int64),
    )
    assert selected == {0: 1, 1: 0}

    provenance_a = _common_provenance(
        mode="self-test", args=argparse.Namespace(), inputs=()
    )
    provenance_b = _common_provenance(
        mode="self-test", args=argparse.Namespace(), inputs=()
    )
    assert provenance_a == provenance_b
    assert not any("time" in key.lower() for key in provenance_a)

    with tempfile.TemporaryDirectory() as directory:
        submission = Path(directory) / "submission.csv"
        official_sids = ["SM_test_0002", "SM_test_0001"]
        official_paths = {
            "SM_test_0001": "x/SM_test_0001",
            "SM_test_0002": "x/SM_test_0002",
        }
        _atomic_submission(
            submission,
            official_sids,
            official_paths,
            np.asarray([3, 4]),
        )
        rows = _read_csv(submission)
        assert [row["path"] for row in rows] == [
            "x/SM_test_0002",
            "x/SM_test_0001",
        ]
        assert [row["prediction"] for row in rows] == ["3", "4"]

    print(
        "self-test passed: safe clustering, ambiguity no-ops, add-one transition, "
        "no-start Viterbi, strict nested selection, deterministic provenance, "
        "and official row order"
    )
    return 0


def _add_decoder_parameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=300.0,
        help="maximum consecutive recording end-to-start gap (default: 300)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.6,
        help="minimum mean aligned probability cosine (default: .6)",
    )
    parser.add_argument(
        "--consensus-weight",
        type=float,
        default=1.0,
        help="aligned-cluster mean blend weight in [0,1] (default: 1)",
    )
    parser.add_argument(
        "--lambda",
        "--transition-weight",
        dest="transition_weight",
        type=float,
        default=0.5,
        help="conditional log-transition weight; 0 disables Viterbi (default: .5)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align likely repeated CUHK-X recordings, blend per-position "
            "probabilities, and optionally apply leakage-safe transitions."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_probability_input(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--probs",
            type=Path,
            required=True,
            help="NPZ containing a probability matrix and sids",
        )
        subparser.add_argument(
            "--probs-key",
            default="probs",
            help="probability key in the NPZ (for example probs or clean_probs)",
        )
        subparser.add_argument(
            "--meta-train", type=Path, default=DEFAULT_META_TRAIN
        )
        _add_decoder_parameters(subparser)

    oof = subparsers.add_parser(
        "oof",
        help="fold-safe OOF evaluation and optional adjusted-probability artifact",
    )
    add_probability_input(oof)
    oof.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    oof.add_argument(
        "--gap-grid",
        default="",
        help="comma-separated gap candidates; empty uses --gap-seconds",
    )
    oof.add_argument(
        "--cosine-grid",
        default="",
        help="comma-separated cosine candidates; empty uses --cosine-threshold",
    )
    oof.add_argument(
        "--consensus-weight-grid",
        default="",
        help="comma-separated consensus weights; empty uses fixed weight",
    )
    oof.add_argument(
        "--lambda-grid",
        default="",
        help="comma-separated transition weights; empty uses fixed lambda",
    )
    oof.add_argument(
        "--max-grid-candidates",
        type=int,
        default=256,
        help="safety limit for the Cartesian parameter grid",
    )
    oof.add_argument(
        "--output",
        type=Path,
        help="optional output NPZ; omitted means report-only dry run",
    )
    oof.add_argument(
        "--provenance-output",
        type=Path,
        help="optional JSON path; defaults beside --output",
    )
    oof.set_defaults(func=run_oof)

    test = subparsers.add_parser(
        "test",
        help="test decode in exact sample-submission order; dry-run by default",
    )
    add_probability_input(test)
    test.add_argument("--meta-test", type=Path, default=DEFAULT_META_TEST)
    test.add_argument(
        "--test-cohorts", type=Path, default=DEFAULT_TEST_COHORTS
    )
    test.add_argument(
        "--sample-submission",
        type=Path,
        default=DEFAULT_SAMPLE_SUBMISSION,
    )
    test.add_argument(
        "--output",
        type=Path,
        help=(
            "submission CSV; also writes <stem>_decode.npz and provenance; "
            "omitted means dry run"
        ),
    )
    test.add_argument(
        "--prob-output",
        type=Path,
        help="optional adjusted-probability NPZ path",
    )
    test.add_argument(
        "--provenance-output",
        type=Path,
        help="optional JSON path; defaults beside the primary output",
    )
    test.set_defaults(func=run_test)

    self_test = subparsers.add_parser(
        "self-test", help="run synthetic dependency-free regression tests"
    )
    self_test.set_defaults(func=run_self_test)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "max_grid_candidates", 1) < 1:
        raise ValueError("--max-grid-candidates must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
