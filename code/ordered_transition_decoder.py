#!/usr/bin/env python3
"""Leakage-safe ordered-recording transition decoding.

The CUHK-X recordings contain short, ordered sequences of different activities.
Radar filenames identify the continuous recording pass, while camera timestamps
give the within-pass clip order.  This script learns a class-transition model
from *allowed source users only* and combines it with per-clip model
probabilities using beam search.

Two operating modes are provided:

* ``oof``: for each subject fold, fit transitions without that fold's users and
  decode only that fold. Validation labels are used only after decoding to
  report accuracy.
* ``test``: fit transitions from all allowed training users, decode test
  recording groups, and optionally write a CSV in the exact
  ``sample_submission.csv`` order.

Examples:

  # A no-op regression test: decoded predictions must equal per-clip argmax.
  python3 code/ordered_transition_decoder.py oof \
      --probs research/artifacts/oof_tjitter_v2_p3_astgcn_v1.npz \
      --lambda 0 --distinctness none

  # Evaluate an order-aware, all-distinct decoder without writing a submission.
  python3 code/ordered_transition_decoder.py oof \
      --probs research/artifacts/oof_tjitter_v2_p3_astgcn_v1.npz \
      --lambda 0.2 --beam-size 256 --distinctness hard

  # Test-side dry run. Add --output submissions/name.csv only after OOF gating.
  python3 code/ordered_transition_decoder.py test \
      --probs research/artifacts/testprobs_tjitter_v2_p3_astgcn_v1.npz \
      --lambda 0.2 --beam-size 256 --distinctness hard

Negative controls deliberately destroy source order, target order, or the label
assignment of the learned transition matrix. They are intended to establish
that any OOF gain is genuinely order-specific.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "research" / "artifacts"
CACHE = ROOT / "cache"
TRAIN_RADAR = (
    ROOT / "Small-Model-Track" / "Training" / "data" / "HAR" / "data" / "Radar"
)
TEST_ROOT = (
    ROOT
    / "Small-Model-Track"
    / "Testing"
    / "data"
    / "small_model_track_test"
)
DEFAULT_SAMPLE_SUBMISSION = (
    ROOT / "Small-Model-Track" / "Testing" / "sample_submission.csv"
)
DEFAULT_FOLDS = ARTIFACTS / "cv_folds.json"
N_CLASSES = 40
RADAR_RE = re.compile(r"radar_output_T(.+)\.csv$")
TEST_SID_RE = re.compile(r"(SM_test_\d+)")


@dataclass(frozen=True)
class Clip:
    """Metadata needed for recording grouping and temporal ordering."""

    sid: str
    group: str
    t0: float
    user: str | None
    label: int | None


@dataclass(frozen=True)
class TransitionModel:
    """Log transition scores consumed by the decoder."""

    transition: np.ndarray
    start: np.ndarray
    # Retaining the exact leakage-safe source templates makes the model object
    # an extension point for a future template-alignment decoder. The current
    # ``markov`` decoder deliberately uses only ``start`` and ``transition``.
    source_templates: tuple[tuple[int, ...], ...]
    source_clips: int
    source_groups: int
    source_edges: int


@dataclass(frozen=True)
class DecodeStats:
    groups: int
    multi_groups: int
    multi_clips: int
    ambiguous_groups_skipped: int
    ambiguous_clips_skipped: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_meta(split: str) -> dict[str, dict[str, str]]:
    rows = _read_csv(CACHE / f"meta_{split}.csv")
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate sample_id entries in meta_{split}.csv")
    return result


def _radar_key(directory: Path) -> str | None:
    """Return the unique Radar recording timestamp encoded in a clip folder."""

    if not directory.is_dir():
        return None
    keys: set[str] = set()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = RADAR_RE.search(path.name)
        if match:
            keys.add(match.group(1))
    if len(keys) > 1:
        raise ValueError(f"multiple Radar recording keys in {directory}: {sorted(keys)}")
    return next(iter(keys), None)


def _train_radar_keys() -> dict[str, tuple[str, str]]:
    """Map train sample ID to (user, recording timestamp)."""

    result: dict[str, tuple[str, str]] = {}
    if not TRAIN_RADAR.is_dir():
        raise FileNotFoundError(TRAIN_RADAR)
    for class_dir in sorted(TRAIN_RADAR.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name.split("_", 1)[0])
        except ValueError as error:
            raise ValueError(f"invalid Radar class directory: {class_dir}") from error
        for user_dir in sorted(class_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            for trial_dir in sorted(user_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                key = _radar_key(trial_dir)
                if key is None:
                    continue
                sid = f"{class_id:02d}_{user_dir.name}_{trial_dir.name}"
                result[sid] = (user_dir.name, key)
    return result


def build_train_clips(grouping: str = "trial") -> dict[str, Clip]:
    """Build train clips using the requested recording identity.

    ``trial`` is the complete metadata-native recording key and covers all
    2,933 clips. ``radar`` is retained as a stricter negative/control grouping,
    but clips without a Radar filename necessarily become singletons.
    """

    meta = _load_meta("train")
    radar = _train_radar_keys()
    clips: dict[str, Clip] = {}
    for sid, row in meta.items():
        user = row["user"]
        radar_entry = radar.get(sid)
        if radar_entry is not None and radar_entry[0] != user:
            raise ValueError(f"Radar/meta user mismatch for {sid}")
        if grouping == "trial":
            group = f"trial:{user}:{row['trial']}"
        elif grouping == "radar":
            group = (
                f"radar:{user}:{radar_entry[1]}"
                if radar_entry is not None
                else f"singleton:{sid}"
            )
        else:
            raise ValueError(f"unsupported train grouping: {grouping!r}")
        clips[sid] = Clip(
            sid=sid,
            group=group,
            t0=float(row["t0"]) if row["t0"] else math.inf,
            user=user,
            label=int(row["class_id"]),
        )
    return clips


def build_test_clips() -> dict[str, Clip]:
    """Build test clips using Radar pass identity and camera/IMU start time."""

    meta = _load_meta("test")
    clips: dict[str, Clip] = {}
    for sid, row in meta.items():
        key = _radar_key(TEST_ROOT / sid / "Radar")
        group = f"radar:{key}" if key is not None else f"singleton:{sid}"
        clips[sid] = Clip(
            sid=sid,
            group=group,
            t0=float(row["t0"]) if row["t0"] else math.inf,
            user=None,
            label=None,
        )
    return clips


def group_sequences(
    clips: Iterable[Clip],
    allowed_sids: set[str] | None = None,
) -> list[list[Clip]]:
    """Group clips and sort each pass by timestamp, then stable sample ID."""

    groups: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        if allowed_sids is None or clip.sid in allowed_sids:
            groups[clip.group].append(clip)
    sequences = [
        sorted(group, key=lambda clip: (clip.t0, clip.sid))
        for group in groups.values()
    ]
    sequences.sort(key=lambda seq: (seq[0].t0, seq[0].group, seq[0].sid))
    return sequences


def ambiguous_sequence_order(sequence: Sequence[Clip]) -> bool:
    """Return whether chronological order is missing or tied.

    Train sample IDs begin with the class label while test IDs do not, so using
    a stable sample-ID tie break would create train-only label information.
    Ambiguous groups are therefore skipped by default rather than assigned an
    arbitrary order.
    """

    timestamps = [clip.t0 for clip in sequence]
    if any(not math.isfinite(value) for value in timestamps):
        return True
    ordered = sorted(timestamps)
    return any(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
        for left, right in zip(ordered, ordered[1:])
    )


def _controlled_source_sequences(
    sequences: Sequence[Sequence[Clip]],
    control: str,
    seed: int,
    ambiguous_order: str,
) -> list[list[int]]:
    labels: list[list[int]] = []
    rng = np.random.default_rng(seed)
    for sequence in sequences:
        if (
            ambiguous_order == "skip"
            and ambiguous_sequence_order(sequence)
        ):
            continue
        row = [int(clip.label) for clip in sequence if clip.label is not None]
        if not row:
            continue
        if control == "shuffle-source" and len(row) > 1:
            row = list(np.asarray(row)[rng.permutation(len(row))])
        elif control == "reverse-source":
            row = row[::-1]
        labels.append(row)
    return labels


def fit_transition_model(
    sequences: Sequence[Sequence[Clip]],
    *,
    alpha: float,
    backoff: str,
    score_mode: str,
    clip_score: float,
    control: str,
    seed: int,
    ambiguous_order: str,
) -> TransitionModel:
    """Fit smoothed start/transition scores from source sequences."""

    if alpha <= 0:
        raise ValueError("--alpha must be positive")
    labels = _controlled_source_sequences(
        sequences, control, seed, ambiguous_order
    )
    unigram = np.zeros(N_CLASSES, dtype=np.float64)
    start = np.zeros(N_CLASSES, dtype=np.float64)
    transition = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    edges = 0
    for row in labels:
        unigram += np.bincount(row, minlength=N_CLASSES)
        start[row[0]] += 1
        for left, right in zip(row, row[1:]):
            transition[left, right] += 1
            edges += 1
    source_clips = int(unigram.sum())
    if source_clips == 0:
        raise ValueError("no source clips remain after user exclusion")

    if backoff == "unigram":
        prior = (unigram + 1.0) / (unigram.sum() + N_CLASSES)
    elif backoff == "uniform":
        prior = np.full(N_CLASSES, 1.0 / N_CLASSES, dtype=np.float64)
    else:
        raise ValueError(backoff)

    # Alpha is a per-class pseudo-count. Scaling prior by N_CLASSES makes
    # alpha=1 comparable to conventional add-one smoothing.
    pseudo = alpha * N_CLASSES * prior
    conditional = transition + pseudo[None, :]
    conditional /= conditional.sum(1, keepdims=True)
    start_prob = start + pseudo
    start_prob /= start_prob.sum()

    if score_mode == "conditional":
        transition_score = np.log(conditional)
        start_score = np.log(start_prob)
    elif score_mode == "pmi":
        # Remove the next-class marginal so the decoder measures order rather
        # than double-counting the classifier's learned class prior.
        transition_score = np.log(conditional) - np.log(prior[None, :])
        start_score = np.log(start_prob) - np.log(prior)
    else:
        raise ValueError(score_mode)

    if clip_score > 0:
        transition_score = np.clip(transition_score, -clip_score, clip_score)
        start_score = np.clip(start_score, -clip_score, clip_score)

    if control == "uniform":
        transition_score.fill(0.0)
        start_score.fill(0.0)
    elif control == "label-permute":
        permutation = np.random.default_rng(seed).permutation(N_CLASSES)
        transition_score = transition_score[permutation][:, permutation]
        start_score = start_score[permutation]

    return TransitionModel(
        transition=transition_score,
        start=start_score,
        source_templates=tuple(tuple(row) for row in labels),
        source_clips=source_clips,
        source_groups=len(labels),
        source_edges=edges,
    )


def _target_order(
    length: int,
    group_key: str,
    control: str,
    seed: int,
) -> np.ndarray:
    order = np.arange(length)
    if control == "reverse-target":
        return order[::-1]
    if control == "shuffle-target" and length > 1:
        digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
        stable_seed = int.from_bytes(digest[:8], "little", signed=False)
        return np.random.default_rng(stable_seed).permutation(length)
    return order


def beam_decode(
    probabilities: np.ndarray,
    model: TransitionModel,
    *,
    transition_weight: float,
    start_weight: float,
    beam_size: int,
    candidate_topk: int,
    distinctness: str,
    distinctness_penalty: float,
) -> np.ndarray:
    """Decode one ordered recording sequence using deterministic beam search."""

    if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
        raise ValueError(f"expected probabilities shaped (T,{N_CLASSES})")
    if beam_size < 1:
        raise ValueError("--beam-size must be >= 1")
    if transition_weight < 0:
        raise ValueError("--lambda/--transition-weight must be non-negative")
    if start_weight < 0:
        raise ValueError("--start-weight must be non-negative")
    if not 1 <= candidate_topk <= N_CLASSES:
        raise ValueError(f"--candidate-topk must be in [1,{N_CLASSES}]")
    if distinctness_penalty < 0:
        raise ValueError("--distinctness-penalty must be non-negative")
    if len(probabilities) == 0:
        return np.empty(0, dtype=np.int64)

    emissions = np.log(np.clip(probabilities, 1e-12, 1.0))
    candidate_rows = [
        np.argsort(-row, kind="stable")[:candidate_topk] for row in probabilities
    ]
    if distinctness == "none":
        # A first-order Markov model has an exact O(T*C^2) Viterbi solution.
        # Using it avoids beam-width approximation in the primary decoder.
        allowed = np.zeros_like(emissions, dtype=bool)
        for step, candidates in enumerate(candidate_rows):
            allowed[step, candidates] = True
        emissions = np.where(allowed, emissions, -np.inf)
        dynamic = emissions[0] + start_weight * model.start
        backpointers = np.zeros((len(probabilities), N_CLASSES), dtype=np.int16)
        for step in range(1, len(probabilities)):
            scores = dynamic[:, None] + transition_weight * model.transition
            backpointers[step] = scores.argmax(0)
            dynamic = emissions[step] + scores.max(0)
        path = np.empty(len(probabilities), dtype=np.int64)
        path[-1] = int(dynamic.argmax())
        for step in range(len(probabilities) - 1, 0, -1):
            path[step - 1] = backpointers[step, path[step]]
        return path

    # Entries are (score, label tuple, used-label bit mask). Tuple ordering is
    # included in the sort key below, making tied runs reproducible.
    beams: list[tuple[float, tuple[int, ...], int]] = []
    for label_value in candidate_rows[0]:
        label = int(label_value)
        score = float(emissions[0, label])
        score += start_weight * float(model.start[label])
        beams.append((score, (label,), 1 << label))
    beams.sort(key=lambda item: (-item[0], item[1]))
    beams = beams[:beam_size]

    for step in range(1, len(probabilities)):
        expanded: list[tuple[float, tuple[int, ...], int]] = []
        for score, path, used in beams:
            previous = path[-1]
            for label_value in candidate_rows[step]:
                label = int(label_value)
                repeated = bool(used & (1 << label))
                if distinctness == "hard" and repeated:
                    continue
                penalty = (
                    distinctness_penalty
                    if distinctness == "penalty" and repeated
                    else 0.0
                )
                new_score = score + float(emissions[step, label])
                new_score += (
                    transition_weight * float(model.transition[previous, label])
                )
                new_score -= penalty
                expanded.append(
                    (new_score, path + (label,), used | (1 << label))
                )
        if not expanded:
            raise RuntimeError(
                "beam became empty; increase --candidate-topk or disable hard "
                "distinctness"
            )
        expanded.sort(key=lambda item: (-item[0], item[1]))
        beams = expanded[:beam_size]
    return np.asarray(beams[0][1], dtype=np.int64)


def decode_group(
    probabilities: np.ndarray,
    sequence: Sequence[Clip],
    model: TransitionModel,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one metadata-ordered group.

    This is the intended scoring extension point. A future full-template
    aligner can use ``model.source_templates`` and the ordered ``sequence``
    without changing fold exclusion, metadata grouping, test-order alignment,
    controls, or output code.

    Returns ``(row_indices_within_group, decoded_labels)``.
    """

    if args.decoder != "markov":
        raise ValueError(f"unsupported decoder: {args.decoder}")
    target_order = _target_order(
        len(sequence), sequence[0].group, args.control, args.seed
    )
    decoded = beam_decode(
        probabilities[target_order],
        model,
        transition_weight=args.transition_weight,
        start_weight=args.start_weight,
        beam_size=args.beam_size,
        candidate_topk=args.candidate_topk,
        distinctness=args.distinctness,
        distinctness_penalty=args.distinctness_penalty,
    )
    return target_order, decoded


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
        raise ValueError(
            f"probability array must have shape (N,{N_CLASSES}), got "
            f"{probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if (probabilities < 0).any():
        raise ValueError("probabilities contain negative values")
    totals = probabilities.sum(1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("at least one probability row sums to zero")
    return probabilities / totals


def load_probability_file(
    path: Path,
    key: str,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
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
    if len(sids) != len(set(sids)):
        raise ValueError(f"duplicate sample IDs in {path}")
    if len(sids) != len(probabilities):
        raise ValueError("probability and sample-ID lengths differ")
    if labels is not None and len(labels) != len(sids):
        raise ValueError("label and sample-ID lengths differ")
    return probabilities, labels, sids


def _decode_sequences(
    probabilities: np.ndarray,
    sids: Sequence[str],
    clips: dict[str, Clip],
    transition_model: TransitionModel,
    args: argparse.Namespace,
) -> tuple[np.ndarray, DecodeStats]:
    sid_to_index = {sid: index for index, sid in enumerate(sids)}
    sequences = group_sequences(
        (clips[sid] for sid in sids),
        allowed_sids=set(sids),
    )
    predictions = probabilities.argmax(1).astype(np.int64)
    multi_groups = 0
    multi_clips = 0
    ambiguous_groups_skipped = 0
    ambiguous_clips_skipped = 0
    for sequence in sequences:
        if len(sequence) > 1:
            multi_groups += 1
            multi_clips += len(sequence)
        if (
            args.ambiguous_order == "skip"
            and ambiguous_sequence_order(sequence)
        ):
            ambiguous_groups_skipped += 1
            ambiguous_clips_skipped += len(sequence)
            continue
        indices = np.asarray([sid_to_index[clip.sid] for clip in sequence])
        within_group_order, decoded = decode_group(
            probabilities[indices],
            sequence,
            transition_model,
            args,
        )
        ordered_indices = indices[within_group_order]
        predictions[ordered_indices] = decoded
    return predictions, DecodeStats(
        groups=len(sequences),
        multi_groups=multi_groups,
        multi_clips=multi_clips,
        ambiguous_groups_skipped=ambiguous_groups_skipped,
        ambiguous_clips_skipped=ambiguous_clips_skipped,
    )


def _fold_spec(path: Path) -> list[set[str]]:
    with path.open() as handle:
        payload = json.load(handle)
    return [set(fold["val_users"]) for fold in payload["folds"]]


def _format_accuracy(hit: int, total: int) -> str:
    return f"{hit / max(total, 1):.5f} ({hit}/{total})"


def _histogram(values: np.ndarray) -> dict[int, int]:
    return {
        int(label): int(count)
        for label, count in sorted(Counter(values.tolist()).items())
    }


def _lambda_values(args: argparse.Namespace) -> list[float]:
    if not args.lambda_grid:
        return [float(args.transition_weight)]
    values: list[float] = []
    for raw in args.lambda_grid.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if value < 0:
            raise ValueError("--lambda-grid values must be non-negative")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--lambda-grid did not contain any values")
    return sorted(values)


def run_oof(args: argparse.Namespace) -> int:
    probabilities, labels, sids = load_probability_file(args.probs, args.probs_key)
    if labels is None:
        raise ValueError("OOF mode requires a 'labels' array in the probability NPZ")
    train_clips = build_train_clips(args.train_grouping)
    missing = [sid for sid in sids if sid not in train_clips]
    if missing:
        raise ValueError(f"{len(missing)} OOF IDs missing from train metadata")
    meta_labels = np.asarray([train_clips[sid].label for sid in sids])
    if not np.array_equal(labels, meta_labels):
        bad = np.flatnonzero(labels != meta_labels)
        raise ValueError(
            f"OOF labels disagree with metadata for {len(bad)} rows; "
            f"first={sids[int(bad[0])]}"
        )

    fold_users = _fold_spec(args.folds)
    user_to_fold: dict[str, int] = {}
    for fold, users in enumerate(fold_users):
        for user in users:
            if user in user_to_fold:
                raise ValueError(f"user {user} appears in multiple validation folds")
            user_to_fold[user] = fold
    row_folds = np.asarray(
        [user_to_fold.get(str(train_clips[sid].user), -1) for sid in sids]
    )
    if (row_folds < 0).any():
        missing_users = sorted(
            {str(train_clips[sids[i]].user) for i in np.flatnonzero(row_folds < 0)}
        )
        raise ValueError(
            "OOF rows include users absent from cv_folds.json: "
            + ",".join(missing_users)
        )

    base = probabilities.argmax(1)
    lambda_values = _lambda_values(args)
    decoded_by_lambda = {
        value: base.copy() for value in lambda_values
    }
    fold_stats: dict[int, DecodeStats] = {}
    fold_source_stats: dict[int, tuple[int, int, int]] = {}
    print(
        f"OOF input: n={len(sids)} base={_format_accuracy(int((base == labels).sum()), len(labels))}"
    )
    for fold, validation_users in enumerate(fold_users):
        target_indices = np.flatnonzero(row_folds == fold)
        target_sids = [sids[index] for index in target_indices]
        allowed_source = [
            clip
            for clip in train_clips.values()
            if clip.user not in validation_users
        ]
        source_sequences = group_sequences(allowed_source)
        transition_model = fit_transition_model(
            source_sequences,
            alpha=args.alpha,
            backoff=args.backoff,
            score_mode=args.transition_score,
            clip_score=args.clip_transition_score,
            control=args.control,
            seed=args.seed + fold * 1009,
            ambiguous_order=args.ambiguous_order,
        )
        for value in lambda_values:
            local_args = copy.copy(args)
            local_args.transition_weight = value
            fold_predictions, stats = _decode_sequences(
                probabilities[target_indices],
                target_sids,
                train_clips,
                transition_model,
                local_args,
            )
            decoded_by_lambda[value][target_indices] = fold_predictions
            fold_stats[fold] = stats
        fold_source_stats[fold] = (
            transition_model.source_clips,
            transition_model.source_groups,
            transition_model.source_edges,
        )

    if args.lambda_grid:
        # Nested transition-label selection. For outer fold f, each inner
        # selection fold j is decoded with a transition model that excludes
        # both f and j. Thus outer-f labels cannot enter either transition
        # counts or lambda scoring. The saved base OOF emissions remain the
        # historical four-fold models; fully nested base-model refits are a
        # separate, explicitly documented limitation.
        selected_by_fold: dict[int, float] = {}
        decoded = base.copy()
        print("descriptive lambda grid:")
        for value in lambda_values:
            candidate = decoded_by_lambda[value]
            print(
                f"  lambda={value:g}: "
                f"{_format_accuracy(int((candidate == labels).sum()), len(labels))} "
                f"delta={(candidate == labels).mean() - (base == labels).mean():+.5f}"
            )
        for fold in range(len(fold_users)):
            selection_hits = {value: 0 for value in lambda_values}
            for inner_fold, inner_users in enumerate(fold_users):
                if inner_fold == fold:
                    continue
                excluded_users = fold_users[fold] | inner_users
                allowed_source = [
                    clip
                    for clip in train_clips.values()
                    if clip.user not in excluded_users
                ]
                inner_model = fit_transition_model(
                    group_sequences(allowed_source),
                    alpha=args.alpha,
                    backoff=args.backoff,
                    score_mode=args.transition_score,
                    clip_score=args.clip_transition_score,
                    control=args.control,
                    seed=args.seed + fold * 100_003 + inner_fold * 1009,
                    ambiguous_order=args.ambiguous_order,
                )
                inner_indices = np.flatnonzero(row_folds == inner_fold)
                inner_sids = [sids[index] for index in inner_indices]
                for value in lambda_values:
                    local_args = copy.copy(args)
                    local_args.transition_weight = value
                    inner_predictions, _ = _decode_sequences(
                        probabilities[inner_indices],
                        inner_sids,
                        train_clips,
                        inner_model,
                        local_args,
                    )
                    selection_hits[value] += int(
                        (inner_predictions == labels[inner_indices]).sum()
                    )
            scored = [
                (selection_hits[value], -value, value)
                for value in lambda_values
            ]
            selected = max(scored)[2]
            selected_by_fold[fold] = selected
            target_indices = np.flatnonzero(row_folds == fold)
            decoded[target_indices] = decoded_by_lambda[selected][target_indices]
    else:
        selected_by_fold = {
            fold: float(args.transition_weight) for fold in range(len(fold_users))
        }
        decoded = decoded_by_lambda[float(args.transition_weight)]

    for fold, validation_users in enumerate(fold_users):
        target_indices = np.flatnonzero(row_folds == fold)
        fold_base = base[target_indices]
        fold_labels = labels[target_indices]
        fold_predictions = decoded[target_indices]
        base_hit = int((fold_base == fold_labels).sum())
        decoded_hit = int((fold_predictions == fold_labels).sum())
        source_clips, source_groups, source_edges = fold_source_stats[fold]
        stats = fold_stats[fold]
        print(
            f"fold {fold}: val_users={sorted(validation_users)} "
            f"source={source_clips} clips/{source_groups} groups/"
            f"{source_edges} edges; "
            f"target={len(target_indices)} clips/{stats.groups} groups "
            f"({stats.multi_clips} in {stats.multi_groups} multi); "
            f"ambiguous-skip={stats.ambiguous_clips_skipped} clips/"
            f"{stats.ambiguous_groups_skipped} groups; "
            f"lambda={selected_by_fold[fold]:g} "
            f"base={_format_accuracy(base_hit, len(target_indices))} "
            f"decoded={_format_accuracy(decoded_hit, len(target_indices))} "
            f"delta={(decoded_hit - base_hit) / len(target_indices):+.5f}"
        )

    base_correct = base == labels
    decoded_correct = decoded == labels
    rescues = int((~base_correct & decoded_correct).sum())
    harms = int((base_correct & ~decoded_correct).sum())
    changed = int((base != decoded).sum())
    print(
        f"OOF decoded={_format_accuracy(int(decoded_correct.sum()), len(labels))} "
        f"delta={(decoded_correct.mean() - base_correct.mean()):+.5f} "
        f"changed={changed} rescues={rescues} harms={harms} "
        f"control={args.control} "
        f"selection={'nested-transition-source' if args.lambda_grid else 'fixed'}"
    )

    if (
        not args.lambda_grid
        and args.transition_weight == 0
        and args.start_weight == 0
        and args.distinctness == "none"
        and not np.array_equal(decoded, base)
    ):
        raise AssertionError("lambda=0, distinctness=none must reproduce argmax")
    if (
        not args.lambda_grid
        and args.control == "uniform"
        and args.distinctness == "none"
        and not np.array_equal(decoded, base)
    ):
        raise AssertionError("uniform control without distinctness must reproduce argmax")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            preds=decoded,
            base_preds=base,
            labels=labels,
            sids=np.asarray(sids),
            folds=row_folds,
            transition_weight=np.asarray(
                [selected_by_fold[fold] for fold in range(len(fold_users))]
            ),
            lambda_grid=np.asarray(lambda_values),
            control=np.asarray(args.control),
            alpha=np.asarray(args.alpha),
            backoff=np.asarray(args.backoff),
            transition_score=np.asarray(args.transition_score),
            clip_transition_score=np.asarray(args.clip_transition_score),
            start_weight=np.asarray(args.start_weight),
            train_grouping=np.asarray(args.train_grouping),
            ambiguous_order=np.asarray(args.ambiguous_order),
            probability_input_sha256=np.asarray(sha256_file(args.probs)),
            folds_sha256=np.asarray(sha256_file(args.folds)),
            train_metadata_sha256=np.asarray(
                sha256_file(CACHE / "meta_train.csv")
            ),
        )
        print(f"wrote OOF decode artifact: {args.output}")
    return 0


def _official_test_order(path: Path) -> tuple[list[str], dict[str, str]]:
    rows = _read_csv(path)
    sids: list[str] = []
    paths: dict[str, str] = {}
    for row in rows:
        match = TEST_SID_RE.search(row["path"])
        if not match:
            raise ValueError(f"cannot parse test ID from {row['path']!r}")
        sid = match.group(1)
        if sid in paths:
            raise ValueError(f"duplicate {sid} in {path}")
        sids.append(sid)
        paths[sid] = row["path"]
    return sids, paths


def _write_submission(
    path: Path,
    official_sids: Sequence[str],
    official_paths: dict[str, str],
    predictions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "prediction"))
        for sid, prediction in zip(official_sids, predictions):
            writer.writerow((official_paths[sid], int(prediction)))


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
            f"probability/test-order ID mismatch: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )
    order = {sid: index for index, sid in enumerate(input_sids)}
    probabilities = probabilities[[order[sid] for sid in official_sids]]

    train_clips = build_train_clips(args.train_grouping)
    excluded = set(args.exclude_users)
    known_users = {str(clip.user) for clip in train_clips.values()}
    unknown_exclusions = excluded - known_users
    if unknown_exclusions:
        raise ValueError(f"unknown --exclude-users: {sorted(unknown_exclusions)}")
    source_clips = [
        clip for clip in train_clips.values() if str(clip.user) not in excluded
    ]
    source_sequences = group_sequences(source_clips)
    transition_model = fit_transition_model(
        source_sequences,
        alpha=args.alpha,
        backoff=args.backoff,
        score_mode=args.transition_score,
        clip_score=args.clip_transition_score,
        control=args.control,
        seed=args.seed,
        ambiguous_order=args.ambiguous_order,
    )
    test_clips = build_test_clips()
    missing_meta = [sid for sid in official_sids if sid not in test_clips]
    if missing_meta:
        raise ValueError(f"{len(missing_meta)} official test IDs lack metadata")
    base = probabilities.argmax(1)
    decoded, stats = _decode_sequences(
        probabilities,
        official_sids,
        test_clips,
        transition_model,
        args,
    )
    base_collisions = 0
    decoded_collisions = 0
    official_index = {sid: index for index, sid in enumerate(official_sids)}
    for sequence in group_sequences(test_clips.values()):
        if len(sequence) < 2:
            continue
        idx = np.asarray([official_index[clip.sid] for clip in sequence])
        base_collisions += len(idx) - len(set(base[idx]))
        decoded_collisions += len(idx) - len(set(decoded[idx]))
    print(
        f"test source={transition_model.source_clips} clips/"
        f"{transition_model.source_groups} groups/"
        f"{transition_model.source_edges} edges; "
        f"target={len(official_sids)} clips/{stats.groups} groups "
        f"({stats.multi_clips} in {stats.multi_groups} multi); "
        f"ambiguous-skip={stats.ambiguous_clips_skipped} clips/"
        f"{stats.ambiguous_groups_skipped} groups; "
        f"changes={int((base != decoded).sum())}; "
        f"repeat-collisions={base_collisions}->{decoded_collisions}; "
        f"control={args.control}"
    )
    print(f"base histogram: {_histogram(base)}")
    print(f"decoded histogram: {_histogram(decoded)}")

    if (
        args.transition_weight == 0
        and args.start_weight == 0
        and args.distinctness == "none"
        and not np.array_equal(decoded, base)
    ):
        raise AssertionError("lambda=0, distinctness=none must reproduce argmax")
    if (
        args.control == "uniform"
        and args.distinctness == "none"
        and not np.array_equal(decoded, base)
    ):
        raise AssertionError("uniform control without distinctness must reproduce argmax")

    if args.output is None:
        print("dry run: no submission written (pass --output to write one)")
    else:
        _write_submission(
            args.output, official_sids, official_paths, decoded
        )
        output_digest = sha256_file(args.output)
        probability_provenance: dict[str, str] = {}
        with np.load(args.probs, allow_pickle=False) as payload:
            for key in ("ensemble", "artifact_sha256", "profile"):
                if key in payload.files:
                    probability_provenance[key] = str(payload[key].item())
        manifest = {
            "schema": "cuhkx.ordered-transition-submission.v1",
            "output": str(args.output.resolve()),
            "output_sha256": output_digest,
            "rows": len(official_sids),
            "changes_from_probability_argmax": int((base != decoded).sum()),
            "probability_input": str(args.probs.resolve()),
            "probability_input_sha256": sha256_file(args.probs),
            "probability_provenance": probability_provenance,
            "sample_submission_sha256": sha256_file(args.sample_submission),
            "train_metadata_sha256": sha256_file(CACHE / "meta_train.csv"),
            "test_metadata_sha256": sha256_file(CACHE / "meta_test.csv"),
            "configuration": {
                "transition_weight": args.transition_weight,
                "start_weight": args.start_weight,
                "alpha": args.alpha,
                "backoff": args.backoff,
                "transition_score": args.transition_score,
                "clip_transition_score": args.clip_transition_score,
                "train_grouping": args.train_grouping,
                "ambiguous_order": args.ambiguous_order,
                "distinctness": args.distinctness,
                "distinctness_penalty": args.distinctness_penalty,
                "candidate_topk": args.candidate_topk,
                "control": args.control,
                "seed": args.seed,
                "exclude_users": sorted(args.exclude_users),
            },
            "source": {
                "clips": transition_model.source_clips,
                "groups": transition_model.source_groups,
                "edges": transition_model.source_edges,
            },
            "target": {
                "groups": stats.groups,
                "multi_groups": stats.multi_groups,
                "multi_clips": stats.multi_clips,
                "ambiguous_groups_skipped": stats.ambiguous_groups_skipped,
                "ambiguous_clips_skipped": stats.ambiguous_clips_skipped,
            },
        }
        manifest_path = Path(str(args.output) + ".manifest.json")
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"wrote submission: {args.output} sha256={output_digest}\n"
            f"wrote manifest: {manifest_path}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe Markov/beam decoder over ordered recording groups. "
            "Transition estimates for OOF fold f exclude all fold-f users."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--probs", type=Path, required=True, help="NPZ containing probabilities and sids"
        )
        subparser.add_argument(
            "--probs-key",
            default="probs",
            help="probability array key in the NPZ (for example probs or clean_probs)",
        )
        subparser.add_argument(
            "--decoder",
            choices=("markov",),
            default="markov",
            help=(
                "group scoring backend; markov is implemented now, while the "
                "model retains leakage-safe source templates for a future "
                "template-alignment backend"
            ),
        )
        subparser.add_argument(
            "--lambda",
            "--transition-weight",
            dest="transition_weight",
            type=float,
            default=0.0,
            help="transition log-score weight; 0 is an exact per-clip control",
        )
        subparser.add_argument(
            "--start-weight",
            type=float,
            default=0.0,
            help=(
                "start-class log-score weight; kept separate because the "
                "fold-safe gate selected zero"
            ),
        )
        subparser.add_argument(
            "--beam-size", type=int, default=256, help="beam hypotheses retained per step"
        )
        subparser.add_argument(
            "--candidate-topk",
            type=int,
            default=N_CLASSES,
            help="per-clip emission candidates considered by the beam",
        )
        subparser.add_argument(
            "--distinctness",
            choices=("none", "hard", "penalty"),
            default="none",
            help="within-recording repeated-label constraint",
        )
        subparser.add_argument(
            "--distinctness-penalty",
            type=float,
            default=2.0,
            help="log-score penalty per repeated label when distinctness=penalty",
        )
        subparser.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            help="Dirichlet smoothing pseudo-count per destination class",
        )
        subparser.add_argument(
            "--backoff",
            choices=("uniform", "unigram"),
            default="uniform",
            help="transition smoothing distribution",
        )
        subparser.add_argument(
            "--transition-score",
            choices=("pmi", "conditional"),
            default="conditional",
            help="PMI isolates order; conditional also applies the learned class prior",
        )
        subparser.add_argument(
            "--clip-transition-score",
            type=float,
            default=0.0,
            help="symmetric clipping for learned transition/start log scores; <=0 disables",
        )
        subparser.add_argument(
            "--train-grouping",
            choices=("trial", "radar"),
            default="trial",
            help=(
                "source/OOF recording key; trial is complete and is the "
                "evidence-backed default, radar is a stricter control"
            ),
        )
        subparser.add_argument(
            "--ambiguous-order",
            choices=("skip", "stable"),
            default="skip",
            help=(
                "skip groups with tied/missing timestamps (default) or use "
                "the deterministic timestamp/sample-ID order"
            ),
        )
        subparser.add_argument(
            "--control",
            choices=(
                "none",
                "uniform",
                "shuffle-source",
                "reverse-source",
                "label-permute",
                "reverse-target",
                "shuffle-target",
            ),
            default="none",
            help="negative-control transformation",
        )
        subparser.add_argument("--seed", type=int, default=20260730)
        subparser.add_argument(
            "--output",
            type=Path,
            help="optional OOF NPZ or test CSV; omitted means report-only dry run",
        )

    oof = subparsers.add_parser("oof", help="fold-safe OOF evaluation")
    add_common(oof)
    oof.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    oof.add_argument(
        "--lambda-grid",
        default="",
        help=(
            "optional comma-separated candidates; selects lambda for each fold "
            "using only the other three folds' labels"
        ),
    )
    oof.set_defaults(func=run_oof)

    test = subparsers.add_parser("test", help="test decoding in official row order")
    add_common(test)
    test.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    test.add_argument(
        "--exclude-users",
        nargs="*",
        default=(),
        help="optional source users to exclude when fitting test transitions",
    )
    test.set_defaults(func=run_test)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
