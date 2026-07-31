#!/usr/bin/env python3
"""Leakage-safe exact recording-template decoder.

This experiment asks whether complete activity scripts contain useful
higher-order information beyond the pairwise ordered-transition decoder.

Source templates are label tuples from timestamp-complete, timestamp-unique
``(user, trial)`` training recordings.  For outer OOF fold ``f``, every source
recording from a fold-f validation user is excluded.  A target recording is
compared only with deduplicated source templates of exactly the same length.
The candidate template maximizes

    sum_t log p_t(template_t) + frequency_weight * log(source_frequency).

``frequency_weight=0`` is the explicitly deduplicated model.  Positive values
use source-template frequency as a prior without physically duplicating
candidates.

Hard template replacement is unsafe because many target scripts are unseen.
The decoder therefore accepts a template only when:

* its mean emission loss versus independent per-clip argmax is below a bound;
* its mean score margin over the second template exceeds a bound.

The fallback is either exact per-clip argmax or the tie-safe Markov decoder.
Nested OOF selection is strict: while selecting a gate for outer fold ``f``,
inner-fold predictions are produced from source templates/transitions that
exclude both ``f`` and the inner validation fold.  Outer-fold labels therefore
cannot affect source models or gate selection.

No CSV writer is provided.  Test mode is a dry run unless ``--output`` is
given, and even then writes an NPZ plus provenance JSON rather than a Kaggle
submission.

Examples:

    python3 code/recording_template_decoder.py oof \
      --probs research/artifacts/oof_astgcn_world25.npz

    python3 code/recording_template_decoder.py oof \
      --probs research/artifacts/oof_astgcn_world25.npz \
      --selection fixed --template-control reverse

    python3 code/recording_template_decoder.py test \
      --probs research/artifacts/testprobs_astgcn_world25_int8.npz \
      --frequency-weight 0.25 --loss-threshold 0.75 \
      --margin-threshold 0.025
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np

import ordered_transition_decoder as ordered


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDS = ROOT / "research" / "artifacts" / "cv_folds.json"
N_CLASSES = 40
SCHEMA = "cuhkx.recording-template-decode.v1"


@dataclass(frozen=True)
class GateConfig:
    frequency_weight: float
    loss_threshold: float
    margin_threshold: float

    def validate(self) -> None:
        if self.frequency_weight < 0:
            raise ValueError("frequency weight must be non-negative")
        if self.loss_threshold < 0:
            raise ValueError("loss threshold must be non-negative")
        if self.margin_threshold < 0:
            raise ValueError("margin threshold must be non-negative")


@dataclass(frozen=True)
class Proposal:
    indices: np.ndarray
    labels: np.ndarray
    emission_loss: float
    score_margin: float
    source_frequency: int
    true_template_seen: bool | None


@dataclass(frozen=True)
class CatalogStats:
    source_groups: int
    source_clips: int
    unique_templates: int
    repeated_groups: int
    lengths: Mapping[int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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


def parse_float_grid(value: str) -> tuple[float, ...]:
    values: list[float] = []
    for raw in value.split(","):
        raw = raw.strip().lower()
        if not raw:
            continue
        number = math.inf if raw in ("inf", "infinity") else float(raw)
        if number < 0 or math.isnan(number):
            raise ValueError("gate grids must contain non-negative values")
        if number not in values:
            values.append(number)
    if not values:
        raise ValueError("empty gate grid")
    return tuple(sorted(values))


def valid_sequences(
    clips: Iterable[ordered.Clip],
) -> tuple[list[list[ordered.Clip]], list[list[ordered.Clip]]]:
    valid, ambiguous = [], []
    for sequence in ordered.group_sequences(clips):
        if ordered.ambiguous_sequence_order(sequence):
            ambiguous.append(sequence)
        else:
            valid.append(sequence)
    return valid, ambiguous


def _stable_shuffle(
    labels: tuple[int, ...], group: str, seed: int
) -> tuple[int, ...]:
    if len(labels) < 2:
        return labels
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    permutation = np.random.default_rng(
        int.from_bytes(digest[:8], "little")
    ).permutation(len(labels))
    values = np.asarray(labels, dtype=np.int64)
    return tuple(int(value) for value in values[permutation])


def build_catalog(
    sequences: Sequence[Sequence[ordered.Clip]],
    excluded_users: set[str],
    control: str,
    seed: int,
) -> tuple[Counter[tuple[int, ...]], CatalogStats]:
    """Build a deduplicated template catalog from permitted source users."""
    counter: Counter[tuple[int, ...]] = Counter()
    source_groups = source_clips = 0
    for sequence in sequences:
        users = {str(clip.user) for clip in sequence}
        if len(users) != 1:
            raise RuntimeError("source recording is not user-pure")
        if users & excluded_users:
            continue
        labels = tuple(int(clip.label) for clip in sequence)
        if control == "reverse":
            labels = tuple(reversed(labels))
        elif control == "shuffle":
            labels = _stable_shuffle(labels, sequence[0].group, seed)
        elif control != "none":
            raise ValueError(control)
        counter[labels] += 1
        source_groups += 1
        source_clips += len(labels)
    if not counter:
        raise ValueError("no source templates remain after user exclusion")
    lengths = Counter(len(template) for template in counter)
    return counter, CatalogStats(
        source_groups=source_groups,
        source_clips=source_clips,
        unique_templates=len(counter),
        repeated_groups=sum(
            count for count in counter.values() if count > 1
        ),
        lengths=dict(sorted(lengths.items())),
    )


def target_sequences_for_users(
    sequences: Sequence[Sequence[ordered.Clip]],
    users: set[str],
) -> list[list[ordered.Clip]]:
    result = [
        list(sequence)
        for sequence in sequences
        if str(sequence[0].user) in users
    ]
    if any(
        len({str(clip.user) for clip in sequence}) != 1
        for sequence in result
    ):
        raise RuntimeError("target recording is not user-pure")
    return result


def build_proposals(
    probabilities: np.ndarray,
    sids: Sequence[str],
    sequences: Sequence[Sequence[ordered.Clip]],
    catalog: Counter[tuple[int, ...]],
    frequency_weight: float,
    min_length: int,
) -> list[Proposal]:
    """Choose one exact-length source template for each ordered target."""
    sid_to_index = {sid: index for index, sid in enumerate(sids)}
    templates_by_length: dict[
        int, list[tuple[tuple[int, ...], int]]
    ] = defaultdict(list)
    for template, count in sorted(catalog.items()):
        templates_by_length[len(template)].append((template, count))

    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    base = probabilities.argmax(1)
    proposals: list[Proposal] = []
    for sequence in sequences:
        if len(sequence) < min_length:
            continue
        candidates = templates_by_length.get(len(sequence), ())
        if not candidates:
            continue
        missing = [clip.sid for clip in sequence if clip.sid not in sid_to_index]
        if missing:
            # A partial recording is not an exact same-length target.
            continue
        indices = np.asarray(
            [sid_to_index[clip.sid] for clip in sequence], dtype=np.int64
        )
        candidate_labels = np.asarray(
            [template for template, _ in candidates], dtype=np.int64
        )
        frequencies = np.asarray(
            [count for _, count in candidates], dtype=np.float64
        )
        steps = np.arange(len(sequence))[:, None]
        emission_scores = log_probabilities[
            indices[steps], candidate_labels.T
        ].sum(0)
        scores = emission_scores + frequency_weight * np.log(frequencies)
        order = np.argsort(-scores, kind="stable")
        best = int(order[0])
        second = float(scores[order[1]]) if len(order) > 1 else -math.inf
        independent_score = float(
            log_probabilities[indices, base[indices]].sum()
        )
        mean_loss = (
            independent_score - float(emission_scores[best])
        ) / len(sequence)
        mean_margin = (float(scores[best]) - second) / len(sequence)
        target_labels = tuple(
            int(clip.label) for clip in sequence
            if clip.label is not None
        )
        seen = (
            target_labels in catalog
            if len(target_labels) == len(sequence)
            else None
        )
        proposals.append(
            Proposal(
                indices=indices,
                labels=candidate_labels[best],
                emission_loss=max(0.0, mean_loss),
                score_margin=mean_margin,
                source_frequency=int(frequencies[best]),
                true_template_seen=seen,
            )
        )
    return proposals


def apply_gate(
    fallback: np.ndarray,
    proposals: Sequence[Proposal],
    config: GateConfig,
) -> tuple[np.ndarray, int, int]:
    config.validate()
    predictions = fallback.copy()
    accepted_groups = accepted_clips = 0
    for proposal in proposals:
        if (
            proposal.emission_loss <= config.loss_threshold
            and proposal.score_margin >= config.margin_threshold
        ):
            predictions[proposal.indices] = proposal.labels
            accepted_groups += 1
            accepted_clips += len(proposal.indices)
    return predictions, accepted_groups, accepted_clips


def _markov_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        decoder="markov",
        control="none",
        seed=args.seed,
        transition_weight=args.markov_weight,
        start_weight=0.0,
        beam_size=256,
        candidate_topk=N_CLASSES,
        distinctness="none",
        distinctness_penalty=0.0,
        ambiguous_order="skip",
    )


def fallback_predictions(
    probabilities: np.ndarray,
    sids: Sequence[str],
    target_clips: dict[str, ordered.Clip],
    source_clips: dict[str, ordered.Clip],
    excluded_users: set[str],
    target_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    predictions = probabilities.argmax(1).astype(np.int64)
    if args.fallback == "argmax":
        return predictions
    source_sequences = ordered.group_sequences(
        clip
        for clip in source_clips.values()
        if str(clip.user) not in excluded_users
    )
    model = ordered.fit_transition_model(
        source_sequences,
        alpha=args.markov_alpha,
        backoff="uniform",
        score_mode="conditional",
        clip_score=0.0,
        control="none",
        seed=seed,
        ambiguous_order="skip",
    )
    local_args = _markov_args(args)
    local_args.seed = seed
    local, _ = ordered._decode_sequences(
        probabilities[target_indices],
        [sids[index] for index in target_indices],
        target_clips,
        model,
        local_args,
    )
    predictions[target_indices] = local
    return predictions


def _fold_rows(
    sids: Sequence[str],
    clips: dict[str, ordered.Clip],
    fold_users: Sequence[set[str]],
) -> np.ndarray:
    user_to_fold: dict[str, int] = {}
    for fold, users in enumerate(fold_users):
        for user in users:
            if user in user_to_fold:
                raise ValueError(f"user {user} appears in multiple folds")
            user_to_fold[user] = fold
    folds = np.asarray([
        user_to_fold.get(str(clips[sid].user), -1) for sid in sids
    ])
    if (folds < 0).any():
        raise ValueError("OOF probabilities include users outside validation folds")
    return folds


def _metric(predictions: np.ndarray, labels: np.ndarray) -> tuple[int, float]:
    correct = int((predictions == labels).sum())
    return correct, correct / len(labels)


def _config_grid(args: argparse.Namespace) -> tuple[GateConfig, ...]:
    frequencies = parse_float_grid(args.frequency_grid)
    losses = parse_float_grid(args.loss_grid)
    margins = parse_float_grid(args.margin_grid)
    configs = tuple(
        GateConfig(frequency, loss, margin)
        for frequency in frequencies
        for loss in losses
        for margin in margins
    )
    if len(configs) > 10_000:
        raise ValueError("gate grid is unexpectedly large")
    return configs


def _fixed_config(args: argparse.Namespace) -> GateConfig:
    config = GateConfig(
        args.frequency_weight,
        args.loss_threshold,
        args.margin_threshold,
    )
    config.validate()
    return config


def _decode_fold(
    fold: int,
    excluded_users: set[str],
    probabilities: np.ndarray,
    sids: Sequence[str],
    clips: dict[str, ordered.Clip],
    valid_train_sequences: Sequence[Sequence[ordered.Clip]],
    fold_users: Sequence[set[str]],
    row_folds: np.ndarray,
    configs: Sequence[GateConfig],
    args: argparse.Namespace,
    seed: int,
) -> tuple[
    dict[GateConfig, np.ndarray],
    dict[GateConfig, tuple[int, int]],
    CatalogStats,
    dict[float, list[Proposal]],
]:
    target_indices = np.flatnonzero(row_folds == fold)
    fallback = fallback_predictions(
        probabilities,
        sids,
        clips,
        clips,
        excluded_users,
        target_indices,
        args,
        seed,
    )
    catalog, stats = build_catalog(
        valid_train_sequences,
        excluded_users,
        args.template_control,
        seed,
    )
    target_sequences = target_sequences_for_users(
        valid_train_sequences, fold_users[fold]
    )
    proposals_by_frequency = {
        frequency: build_proposals(
            probabilities,
            sids,
            target_sequences,
            catalog,
            frequency,
            args.min_length,
        )
        for frequency in sorted({
            config.frequency_weight for config in configs
        })
    }
    predictions: dict[GateConfig, np.ndarray] = {}
    acceptance: dict[GateConfig, tuple[int, int]] = {}
    for config in configs:
        decoded, groups, accepted_clips = apply_gate(
            fallback,
            proposals_by_frequency[config.frequency_weight],
            config,
        )
        predictions[config] = decoded
        acceptance[config] = (groups, accepted_clips)
    return predictions, acceptance, stats, proposals_by_frequency


def _conservative_choice(
    configs: Sequence[GateConfig],
    correct: Mapping[GateConfig, int],
    changes: Mapping[GateConfig, int],
) -> GateConfig:
    """Maximize correct, then prefer the less invasive deterministic gate."""
    return max(
        configs,
        key=lambda config: (
            correct[config],
            -changes[config],
            -config.frequency_weight,
            -config.loss_threshold,
            config.margin_threshold,
        ),
    )


def run_oof(args: argparse.Namespace) -> int:
    probabilities, labels, sids = ordered.load_probability_file(
        args.probs, args.probs_key
    )
    if labels is None:
        raise ValueError("OOF mode requires labels in the probability artifact")
    clips = ordered.build_train_clips("trial")
    if any(sid not in clips for sid in sids):
        raise ValueError("OOF sample missing from train metadata")
    metadata_labels = np.asarray([clips[sid].label for sid in sids])
    if not np.array_equal(labels, metadata_labels):
        raise ValueError("OOF labels disagree with train metadata")
    fold_users = ordered._fold_spec(args.folds)
    row_folds = _fold_rows(sids, clips, fold_users)
    valid_train, ambiguous_train = valid_sequences(clips.values())
    base = probabilities.argmax(1)
    configs = (
        _config_grid(args)
        if args.selection == "nested"
        else (_fixed_config(args),)
    )

    chosen: dict[int, GateConfig] = {}
    if args.selection == "nested":
        for outer in range(len(fold_users)):
            correct = {config: 0 for config in configs}
            changes = {config: 0 for config in configs}
            for inner in range(len(fold_users)):
                if inner == outer:
                    continue
                excluded = fold_users[outer] | fold_users[inner]
                decoded, _, _, _ = _decode_fold(
                    inner,
                    excluded,
                    probabilities,
                    sids,
                    clips,
                    valid_train,
                    fold_users,
                    row_folds,
                    configs,
                    args,
                    args.seed + outer * 10_007 + inner * 1_009,
                )
                mask = row_folds == inner
                for config in configs:
                    correct[config] += int(
                        (decoded[config][mask] == labels[mask]).sum()
                    )
                    changes[config] += int(
                        (decoded[config][mask] != base[mask]).sum()
                    )
            chosen[outer] = _conservative_choice(
                configs, correct, changes
            )
    else:
        chosen = {
            fold: configs[0] for fold in range(len(fold_users))
        }

    final = base.copy()
    fallback_all = base.copy()
    fold_records: dict[str, object] = {}
    for fold in range(len(fold_users)):
        config = chosen[fold]
        decoded, acceptance, catalog_stats, proposals = _decode_fold(
            fold,
            fold_users[fold],
            probabilities,
            sids,
            clips,
            valid_train,
            fold_users,
            row_folds,
            (config,),
            args,
            args.seed + fold * 1_009,
        )
        target = row_folds == fold
        selected = decoded[config]
        # Reconstruct the fallback for reporting from rows untouched by no gate.
        fallback = fallback_predictions(
            probabilities,
            sids,
            clips,
            clips,
            fold_users[fold],
            np.flatnonzero(target),
            args,
            args.seed + fold * 1_009,
        )
        final[target] = selected[target]
        fallback_all[target] = fallback[target]
        base_correct, base_accuracy = _metric(base[target], labels[target])
        fallback_correct, fallback_accuracy = _metric(
            fallback[target], labels[target]
        )
        final_correct, final_accuracy = _metric(
            final[target], labels[target]
        )
        source_seen = [
            proposal.true_template_seen
            for proposal in proposals[config.frequency_weight]
        ]
        accepted_groups, accepted_clips = acceptance[config]
        fold_records[str(fold)] = {
            "config": asdict(config),
            "base_correct": base_correct,
            "fallback_correct": fallback_correct,
            "decoded_correct": final_correct,
            "samples": int(target.sum()),
            "base_accuracy": base_accuracy,
            "fallback_accuracy": fallback_accuracy,
            "decoded_accuracy": final_accuracy,
            "accepted_groups": accepted_groups,
            "accepted_clips": accepted_clips,
            "true_template_seen_groups": int(sum(value is True for value in source_seen)),
            "target_template_groups": len(source_seen),
            "catalog": asdict(catalog_stats),
        }
        print(
            f"fold {fold}: config={asdict(config)} "
            f"base={base_accuracy:.5f} fallback={fallback_accuracy:.5f} "
            f"template={final_accuracy:.5f} "
            f"accepted={accepted_groups} groups/{accepted_clips} clips",
            flush=True,
        )

    base_correct, base_accuracy = _metric(base, labels)
    fallback_correct, fallback_accuracy = _metric(fallback_all, labels)
    final_correct, final_accuracy = _metric(final, labels)
    print(
        f"OOF base={base_accuracy:.5f} ({base_correct}/{len(labels)}) "
        f"fallback={fallback_accuracy:.5f} ({fallback_correct}/{len(labels)}) "
        f"template={final_accuracy:.5f} ({final_correct}/{len(labels)}); "
        f"delta_base={final_accuracy - base_accuracy:+.5f} "
        f"delta_fallback={final_accuracy - fallback_accuracy:+.5f}; "
        f"changes={int((final != base).sum())} "
        f"rescues={int(((final == labels) & (base != labels)).sum())} "
        f"harms={int(((final != labels) & (base == labels)).sum())}; "
        f"ambiguous_source/target_policy=skip "
        f"({len(ambiguous_train)} train groups)",
        flush=True,
    )

    manifest = {
        "schema": SCHEMA,
        "mode": "oof",
        "selection": args.selection,
        "fallback": args.fallback,
        "markov_weight": args.markov_weight,
        "template_control": args.template_control,
        "min_length": args.min_length,
        "chosen": {
            str(fold): asdict(config) for fold, config in chosen.items()
        },
        "folds": fold_records,
        "summary": {
            "samples": len(labels),
            "base_correct": base_correct,
            "fallback_correct": fallback_correct,
            "decoded_correct": final_correct,
            "base_accuracy": base_accuracy,
            "fallback_accuracy": fallback_accuracy,
            "decoded_accuracy": final_accuracy,
        },
        "provenance": provenance(args),
    }
    if args.output:
        if args.output.exists() or Path(str(args.output) + ".manifest.json").exists():
            raise FileExistsError("refusing to overwrite OOF output or manifest")
        atomic_npz(
            args.output,
            preds=final,
            fallback_preds=fallback_all,
            base_preds=base,
            labels=labels,
            sids=np.asarray(sids),
            folds=row_folds,
            chosen=np.asarray(json.dumps(manifest["chosen"], sort_keys=True)),
        )
        atomic_json(Path(str(args.output) + ".manifest.json"), manifest)
        print(f"wrote {args.output} and provenance manifest", flush=True)
    return 0


def provenance(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "probabilities": args.probs,
        "source": Path(__file__),
        "ordered_backend": Path(ordered.__file__),
        "train_metadata": ROOT / "cache" / "meta_train.csv",
    }
    if hasattr(args, "folds"):
        paths["folds"] = args.folds
    if args.mode == "test":
        paths["test_metadata"] = ROOT / "cache" / "meta_test.csv"
        paths["sample_submission"] = args.sample_submission
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def run_test(args: argparse.Namespace) -> int:
    probabilities, labels, input_sids = ordered.load_probability_file(
        args.probs, args.probs_key
    )
    if labels is not None:
        raise ValueError("test probability artifact unexpectedly contains labels")
    official_sids, _ = ordered._official_test_order(args.sample_submission)
    if set(input_sids) != set(official_sids):
        raise ValueError("probability IDs do not match official test IDs")
    source_order = {sid: index for index, sid in enumerate(input_sids)}
    probabilities = probabilities[
        [source_order[sid] for sid in official_sids]
    ]

    train_clips = ordered.build_train_clips("trial")
    valid_train, ambiguous_train = valid_sequences(train_clips.values())
    test_clips = ordered.build_test_clips()
    valid_test, ambiguous_test = valid_sequences(test_clips.values())
    config = _fixed_config(args)
    fallback = fallback_predictions(
        probabilities,
        official_sids,
        test_clips,
        train_clips,
        set(),
        np.arange(len(official_sids)),
        args,
        args.seed,
    )
    catalog, catalog_stats = build_catalog(
        valid_train, set(), args.template_control, args.seed
    )
    proposals = build_proposals(
        probabilities,
        official_sids,
        valid_test,
        catalog,
        config.frequency_weight,
        args.min_length,
    )
    decoded, accepted_groups, accepted_clips = apply_gate(
        fallback, proposals, config
    )
    base = probabilities.argmax(1)
    print(
        f"TEST DRY RUN: rows={len(official_sids)} config={asdict(config)} "
        f"fallback={args.fallback} markov_weight={args.markov_weight}; "
        f"changes_vs_base={int((decoded != base).sum())} "
        f"changes_vs_fallback={int((decoded != fallback).sum())}; "
        f"accepted={accepted_groups} groups/{accepted_clips} clips; "
        f"valid_target={len(valid_test)} groups, "
        f"ambiguous_skipped={len(ambiguous_test)} groups/"
        f"{sum(map(len, ambiguous_test))} clips; "
        f"catalog={catalog_stats.unique_templates} unique templates/"
        f"{catalog_stats.source_groups} source groups",
        flush=True,
    )
    manifest = {
        "schema": SCHEMA,
        "mode": "test-dry-run",
        "configuration": {
            **asdict(config),
            "fallback": args.fallback,
            "markov_weight": args.markov_weight,
            "markov_alpha": args.markov_alpha,
            "template_control": args.template_control,
            "min_length": args.min_length,
            "ambiguous_order": "skip",
            "seed": args.seed,
        },
        "source": {
            **asdict(catalog_stats),
            "ambiguous_groups_skipped": len(ambiguous_train),
        },
        "target": {
            "rows": len(official_sids),
            "valid_groups": len(valid_test),
            "ambiguous_groups_skipped": len(ambiguous_test),
            "ambiguous_clips_skipped": sum(map(len, ambiguous_test)),
            "accepted_groups": accepted_groups,
            "accepted_clips": accepted_clips,
            "changes_from_base": int((decoded != base).sum()),
            "changes_from_fallback": int((decoded != fallback).sum()),
        },
        "provenance": provenance(args),
    }
    print(
        "provenance="
        + json.dumps(manifest["provenance"], sort_keys=True),
        flush=True,
    )
    if args.output:
        manifest_path = Path(str(args.output) + ".manifest.json")
        if args.output.exists() or manifest_path.exists():
            raise FileExistsError("refusing to overwrite test output or manifest")
        atomic_npz(
            args.output,
            preds=decoded,
            fallback_preds=fallback,
            base_preds=base,
            sids=np.asarray(official_sids),
            configuration=np.asarray(
                json.dumps(manifest["configuration"], sort_keys=True)
            ),
        )
        atomic_json(manifest_path, manifest)
        print(
            f"wrote prediction NPZ (not a submission): {args.output}\n"
            f"wrote provenance manifest: {manifest_path}",
            flush=True,
        )
    else:
        print("no output written", flush=True)
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--probs-key", default="probs")
    parser.add_argument(
        "--fallback", choices=("argmax", "markov"), default="markov"
    )
    parser.add_argument("--markov-weight", type=float, default=0.5)
    parser.add_argument("--markov-alpha", type=float, default=1.0)
    parser.add_argument("--frequency-weight", type=float, default=0.25)
    parser.add_argument("--loss-threshold", type=float, default=0.75)
    parser.add_argument("--margin-threshold", type=float, default=0.025)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument(
        "--template-control",
        choices=("none", "reverse", "shuffle"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional NPZ prediction artifact; this script never writes a CSV",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    oof = commands.add_parser("oof", help="fold-safe OOF evaluation")
    add_common(oof)
    oof.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    oof.add_argument(
        "--selection", choices=("nested", "fixed"), default="nested"
    )
    oof.add_argument(
        "--frequency-grid", default="0,0.25,0.5,1"
    )
    oof.add_argument(
        "--loss-grid",
        default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.75,1,1.25,1.5,2,inf",
    )
    oof.add_argument(
        "--margin-grid", default="0,0.01,0.025,0.05,0.1,0.2,0.3,0.5"
    )
    oof.set_defaults(function=run_oof)

    test = commands.add_parser(
        "test", help="test dry-run; optional output is NPZ, never CSV"
    )
    add_common(test)
    test.add_argument(
        "--sample-submission",
        type=Path,
        default=ordered.DEFAULT_SAMPLE_SUBMISSION,
    )
    test.set_defaults(function=run_test)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.markov_weight < 0 or args.markov_alpha <= 0:
        raise ValueError("invalid Markov configuration")
    if args.min_length < 2:
        raise ValueError("--min-length must be at least 2")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
