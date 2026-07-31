#!/usr/bin/env python3
"""Generate a complete, deterministic subject-grouped outer-CV protocol.

This protocol is intentionally separate from ``cv_protocol.py`` and its
historical artifact.  It fixes the historical always-train-user omission:

* all 18 training users are assigned to one validation fold;
* every cached training sample is validated exactly once;
* no model result is used to construct the folds;
* each outer fold is intended to be evaluated once, after a fixed training
  recipe (including its epoch budget) has been selected without that fold.

The four folds retain the competition's two user-ID blocks while distributing
all users as evenly as possible.  With the current 9+9 users this gives fold
sizes 5, 5, 4, and 4.  Candidate assignments are chosen deterministically by
class coverage first, then sample-count and class-distribution balance.

Default output:
    research/artifacts/cv_folds_all18.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "cache" / "meta_train.csv"
DEFAULT_OUTPUT = ROOT / "research" / "artifacts" / "cv_folds_all18.json"
N_FOLDS = 4
DEFAULT_SEED = 20260730
SEARCH_RESTARTS = 24


@dataclass(frozen=True)
class UserStats:
    name: str
    numeric_id: int
    samples: int
    class_counts: tuple[int, ...]

    @property
    def classes_covered(self) -> int:
        return sum(count > 0 for count in self.class_counts)


@dataclass(frozen=True)
class CandidateTable:
    """All labeled partitions of one user block at fixed fold capacities."""

    assignments: np.ndarray  # uint8 [candidate, user]
    sample_counts: np.ndarray  # int64 [candidate, fold]
    class_counts: np.ndarray  # int64 [candidate, fold, class]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_user_id(user: str) -> int:
    match = re.search(r"(\d+)$", user)
    if match is None:
        raise ValueError(f"user name has no numeric suffix: {user!r}")
    return int(match.group(1))


def load_metadata(
    path: Path,
) -> tuple[list[dict[str, str]], list[UserStats], list[int]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "user", "class_id"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} contains no samples")

    sample_ids = [row["sample_id"] for row in rows]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count != 1
    )
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"sample_id values are not unique ({preview})")

    labels = sorted({int(row["class_id"]) for row in rows})
    expected_labels = list(range(labels[-1] + 1))
    if labels != expected_labels:
        raise ValueError(
            "class IDs must be contiguous from zero; "
            f"found {labels}, expected {expected_labels}"
        )

    counts_by_user: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        counts_by_user[row["user"]][int(row["class_id"])] += 1

    users = []
    ordered_users = sorted(
        counts_by_user, key=lambda value: (numeric_user_id(value), value)
    )
    for user in ordered_users:
        class_counts = tuple(counts_by_user[user][label] for label in labels)
        users.append(
            UserStats(
                name=user,
                numeric_id=numeric_user_id(user),
                samples=sum(class_counts),
                class_counts=class_counts,
            )
        )
    return rows, users, labels


def split_user_blocks(users: Sequence[UserStats]) -> list[list[UserStats]]:
    """Split users at the largest numeric-ID gap.

    The current data split is user1--user9 and user16--user24.  Inferring the
    boundary from IDs keeps the rule explicit and deterministic without
    embedding individual users in the protocol.
    """

    if len(users) < 2:
        raise ValueError("at least two users are required")
    gaps = [
        users[index + 1].numeric_id - users[index].numeric_id
        for index in range(len(users) - 1)
    ]
    largest_gap = max(gaps)
    boundary = gaps.index(largest_gap) + 1
    if largest_gap <= 1:
        raise ValueError(
            "could not infer the expected early/late user blocks: "
            "no numeric-ID gap is present"
        )
    blocks = [list(users[:boundary]), list(users[boundary:])]
    if any(len(block) < N_FOLDS for block in blocks):
        raise ValueError(
            f"each user block must contain at least {N_FOLDS} users; "
            f"found {[len(block) for block in blocks]}"
        )
    return blocks


def block_capacities(block_size: int, block_index: int) -> tuple[int, ...]:
    """Evenly distribute one block, rotating which fold receives remainders."""

    quotient, remainder = divmod(block_size, N_FOLDS)
    capacities = [quotient] * N_FOLDS
    for offset in range(remainder):
        capacities[(block_index + offset) % N_FOLDS] += 1
    assert sum(capacities) == block_size
    return tuple(capacities)


def labeled_partitions(
    item_indices: tuple[int, ...], capacities: tuple[int, ...]
) -> Iterable[tuple[int, ...]]:
    """Yield every assignment vector satisfying labeled fold capacities."""

    assignment = [-1] * len(item_indices)

    def visit(fold: int, remaining: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
        if fold == len(capacities) - 1:
            if len(remaining) != capacities[fold]:
                return
            for item in remaining:
                assignment[item] = fold
            yield tuple(assignment)
            for item in remaining:
                assignment[item] = -1
            return

        from itertools import combinations

        for selected in combinations(remaining, capacities[fold]):
            selected_set = set(selected)
            for item in selected:
                assignment[item] = fold
            next_remaining = tuple(
                item for item in remaining if item not in selected_set
            )
            yield from visit(fold + 1, next_remaining)
            for item in selected:
                assignment[item] = -1

    yield from visit(0, item_indices)


def build_candidate_table(
    users: Sequence[UserStats], capacities: tuple[int, ...]
) -> CandidateTable:
    assignments = np.asarray(
        list(labeled_partitions(tuple(range(len(users))), capacities)),
        dtype=np.uint8,
    )
    if assignments.ndim != 2 or assignments.shape[1] != len(users):
        raise RuntimeError("failed to enumerate block assignments")

    membership = assignments[:, :, None] == np.arange(N_FOLDS, dtype=np.uint8)
    user_samples = np.asarray([user.samples for user in users], dtype=np.int64)
    user_classes = np.asarray(
        [user.class_counts for user in users], dtype=np.int64
    )
    sample_counts = np.einsum(
        "nuf,u->nf", membership, user_samples, optimize=True
    )
    class_counts = np.einsum(
        "nuf,uc->nfc", membership, user_classes, optimize=True
    )
    return CandidateTable(assignments, sample_counts, class_counts)


def objective_arrays(
    class_counts: np.ndarray,
    sample_counts: np.ndarray,
    fold_user_counts: np.ndarray,
    global_class_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lexicographic objective terms for one or many assignments."""

    if class_counts.ndim == 2:
        class_counts = class_counts[None, ...]
        sample_counts = sample_counts[None, ...]

    # Sorting makes the worst fold the primary class-coverage criterion, then
    # the second-worst, and so on.
    sorted_coverage = np.sort(np.count_nonzero(class_counts, axis=2), axis=1)
    total_samples = int(global_class_counts.sum())
    total_users = int(fold_user_counts.sum())

    # A 5-user fold should contain roughly 5/18 of all samples, while a 4-user
    # fold should contain roughly 4/18.  Integer cross-products avoid unstable
    # floating-point tie breaking.
    sample_deviation = np.abs(
        sample_counts * total_users
        - fold_user_counts[None, :] * total_samples
    ).sum(axis=1)

    # Once coverage is maximized, prefer similar label distributions after
    # accounting for each fold's actual sample count.
    class_deviation = np.abs(
        class_counts * total_samples
        - sample_counts[:, :, None] * global_class_counts[None, None, :]
    ).sum(axis=(1, 2))
    return sorted_coverage, sample_deviation, class_deviation


def best_row(
    class_counts: np.ndarray,
    sample_counts: np.ndarray,
    fold_user_counts: np.ndarray,
    global_class_counts: np.ndarray,
) -> int:
    coverage, sample_deviation, class_deviation = objective_arrays(
        class_counts, sample_counts, fold_user_counts, global_class_counts
    )
    row_index = np.arange(len(sample_counts), dtype=np.int64)
    # np.lexsort uses its last key as primary.  The negative row index makes
    # the first enumerated assignment win a complete tie.
    keys = [
        -row_index,
        -class_deviation,
        -sample_deviation,
        *[coverage[:, column] for column in reversed(range(N_FOLDS))],
    ]
    return int(np.lexsort(tuple(keys))[-1])


def assignment_key(
    early_index: int,
    late_index: int,
    early: CandidateTable,
    late: CandidateTable,
    fold_user_counts: np.ndarray,
    global_class_counts: np.ndarray,
) -> tuple[int, ...]:
    class_counts = early.class_counts[early_index] + late.class_counts[late_index]
    sample_counts = (
        early.sample_counts[early_index] + late.sample_counts[late_index]
    )
    coverage, sample_deviation, class_deviation = objective_arrays(
        class_counts, sample_counts, fold_user_counts, global_class_counts
    )
    canonical = tuple(
        int(value)
        for value in np.concatenate(
            (
                early.assignments[early_index],
                late.assignments[late_index],
            )
        )
    )
    return (
        *(int(value) for value in coverage[0]),
        -int(sample_deviation[0]),
        -int(class_deviation[0]),
        *(3 - value for value in canonical),
    )


def optimize_assignment(
    blocks: Sequence[Sequence[UserStats]],
    capacities: Sequence[tuple[int, ...]],
    global_class_counts: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic coordinate search over all partitions of each block."""

    if len(blocks) != 2:
        raise ValueError(f"expected two user blocks, found {len(blocks)}")
    early = build_candidate_table(blocks[0], capacities[0])
    late = build_candidate_table(blocks[1], capacities[1])
    fold_user_counts = np.asarray(capacities[0], dtype=np.int64) + np.asarray(
        capacities[1], dtype=np.int64
    )

    rng = random.Random(seed)
    starts = [(0, 0), (len(early.assignments) - 1, len(late.assignments) - 1)]
    while len(starts) < SEARCH_RESTARTS:
        pair = (
            rng.randrange(len(early.assignments)),
            rng.randrange(len(late.assignments)),
        )
        if pair not in starts:
            starts.append(pair)

    best_pair: tuple[int, int] | None = None
    best_key: tuple[int, ...] | None = None
    for early_index, late_index in starts:
        seen_pairs: set[tuple[int, int]] = set()
        for _ in range(20):
            pair = (early_index, late_index)
            if pair in seen_pairs:
                break
            seen_pairs.add(pair)

            combined_classes = (
                early.class_counts + late.class_counts[late_index][None, ...]
            )
            combined_samples = (
                early.sample_counts + late.sample_counts[late_index][None, ...]
            )
            next_early = best_row(
                combined_classes,
                combined_samples,
                fold_user_counts,
                global_class_counts,
            )

            combined_classes = (
                late.class_counts + early.class_counts[next_early][None, ...]
            )
            combined_samples = (
                late.sample_counts + early.sample_counts[next_early][None, ...]
            )
            next_late = best_row(
                combined_classes,
                combined_samples,
                fold_user_counts,
                global_class_counts,
            )
            if (next_early, next_late) == (early_index, late_index):
                break
            early_index, late_index = next_early, next_late

        key = assignment_key(
            early_index,
            late_index,
            early,
            late,
            fold_user_counts,
            global_class_counts,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_pair = (early_index, late_index)

    if best_pair is None:
        raise RuntimeError("assignment search produced no result")
    return (
        early.assignments[best_pair[0]].astype(np.int64),
        late.assignments[best_pair[1]].astype(np.int64),
    )


def class_count_dict(labels: Sequence[int], counts: Sequence[int]) -> dict[str, int]:
    return {str(label): int(count) for label, count in zip(labels, counts)}


def build_protocol(
    metadata_path: Path,
    rows: Sequence[dict[str, str]],
    users: Sequence[UserStats],
    labels: Sequence[int],
    blocks: Sequence[Sequence[UserStats]],
    capacities: Sequence[tuple[int, ...]],
    assignments: Sequence[np.ndarray],
    seed: int,
) -> dict[str, object]:
    all_user_names = {user.name for user in users}
    user_to_fold: dict[str, int] = {}
    user_to_block: dict[str, str] = {}
    block_names = ("early", "late")
    for block_name, block, assignment in zip(block_names, blocks, assignments):
        for user, fold in zip(block, assignment):
            if user.name in user_to_fold:
                raise RuntimeError(f"{user.name} was assigned more than once")
            user_to_fold[user.name] = int(fold)
            user_to_block[user.name] = block_name

    if set(user_to_fold) != all_user_names:
        missing = sorted(all_user_names.difference(user_to_fold))
        raise RuntimeError(f"users omitted from validation: {missing}")

    rows_by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_fold[user_to_fold[row["user"]]].append(row)
    if sum(len(fold_rows) for fold_rows in rows_by_fold.values()) != len(rows):
        raise RuntimeError("not every metadata row was assigned to validation")

    global_class_counts = Counter(int(row["class_id"]) for row in rows)
    class_user_support = {
        label: sum(user.class_counts[label] > 0 for user in users)
        for label in labels
    }
    coverage_limited_classes = {
        str(label): class_user_support[label]
        for label in labels
        if class_user_support[label] < N_FOLDS
    }
    folds = []
    validation_occurrences = Counter()
    aggregate_validation_counts = Counter()
    for fold in range(N_FOLDS):
        validation_users = sorted(
            (user for user, assigned in user_to_fold.items() if assigned == fold),
            key=lambda value: (numeric_user_id(value), value),
        )
        training_users = sorted(
            all_user_names.difference(validation_users),
            key=lambda value: (numeric_user_id(value), value),
        )
        validation_occurrences.update(validation_users)
        fold_rows = rows_by_fold[fold]
        validation_class_counts = Counter(
            int(row["class_id"]) for row in fold_rows
        )
        aggregate_validation_counts.update(validation_class_counts)
        training_class_counts = Counter(global_class_counts)
        training_class_counts.subtract(validation_class_counts)
        missing_validation = [
            label for label in labels if validation_class_counts[label] == 0
        ]
        missing_training = [
            label for label in labels if training_class_counts[label] == 0
        ]

        folds.append(
            {
                "fold": fold,
                "repeat": 0,
                "val_users": validation_users,
                "train_users": training_users,
                "val_users_by_block": {
                    block_name: sum(
                        user_to_block[user] == block_name
                        for user in validation_users
                    )
                    for block_name in block_names
                },
                "val_samples": len(fold_rows),
                "train_samples": len(rows) - len(fold_rows),
                "val_classes_covered": len(labels) - len(missing_validation),
                "val_missing_classes": missing_validation,
                "train_classes_covered": len(labels) - len(missing_training),
                "train_missing_classes": missing_training,
                "val_class_counts": class_count_dict(
                    labels, [validation_class_counts[label] for label in labels]
                ),
                "train_class_counts": class_count_dict(
                    labels, [training_class_counts[label] for label in labels]
                ),
            }
        )

    per_user = []
    for user in users:
        missing_classes = [
            label
            for label, count in zip(labels, user.class_counts)
            if count == 0
        ]
        per_user.append(
            {
                "user": user.name,
                "numeric_id": user.numeric_id,
                "block": user_to_block[user.name],
                "validation_fold": user_to_fold[user.name],
                "samples": user.samples,
                "classes_covered": user.classes_covered,
                "missing_classes": missing_classes,
                "class_counts": class_count_dict(labels, user.class_counts),
            }
        )

    occurrence_dict = {
        user.name: int(validation_occurrences[user.name]) for user in users
    }
    aggregate_matches = all(
        aggregate_validation_counts[label] == global_class_counts[label]
        for label in labels
    )
    all_users_once = all(count == 1 for count in occurrence_dict.values())
    all_samples_once = (
        sum(fold["val_samples"] for fold in folds) == len(rows)
        and aggregate_matches
    )
    if not all_users_once or not all_samples_once:
        raise RuntimeError("complete validation coverage invariant failed")

    block_description = []
    for name, block, block_capacity in zip(block_names, blocks, capacities):
        block_description.append(
            {
                "name": name,
                "users": [user.name for user in block],
                "fold_capacities": list(block_capacity),
            }
        )

    metadata_display = (
        str(metadata_path.relative_to(ROOT))
        if metadata_path.is_relative_to(ROOT)
        else str(metadata_path)
    )
    return {
        "schema_version": 1,
        "protocol_id": "all18_subject_grouped_outer_once_v1",
        "description": (
            "Deterministic 4-fold subject-grouped CV over all 18 train users. "
            "Every user and cached sample is held out exactly once; there are "
            "no always-train users."
        ),
        "outer_evaluation_policy": {
            "training": (
                "Fix architecture, hyperparameters, seed policy, and epoch "
                "budget without using the current outer fold."
            ),
            "evaluation": (
                "Evaluate each trained model on its outer fold exactly once "
                "after training; never select checkpoints or variants on that fold."
            ),
            "aggregation": (
                "Concatenate the four disjoint outer predictions. Each cached "
                "training sample contributes exactly one OOF prediction."
            ),
        },
        "source": {
            "metadata": metadata_display,
            "metadata_sha256": sha256_file(metadata_path),
            "sample_unit": "one unique sample_id row in cache metadata",
        },
        "determinism": {
            "seed": seed,
            "folds": N_FOLDS,
            "repeats": 1,
            "search_restarts": SEARCH_RESTARTS,
            "assignment_objective": [
                "maximize the worst-to-best sorted validation class coverage",
                "minimize sample-count deviation from fold user capacity",
                "minimize validation class-distribution deviation",
                "canonical user/fold assignment tie-break",
            ],
            "model_results_used_for_split": False,
        },
        "user_blocks": block_description,
        "always_train": [],
        "global_diagnostics": {
            "samples": len(rows),
            "users": len(users),
            "classes": len(labels),
            "class_ids": list(labels),
            "class_counts": class_count_dict(
                labels, [global_class_counts[label] for label in labels]
            ),
            "class_user_support": class_count_dict(
                labels, [class_user_support[label] for label in labels]
            ),
            "classes_present_in_fewer_users_than_folds": coverage_limited_classes,
            "all_folds_can_cover_every_class": not coverage_limited_classes,
        },
        "coverage_checks": {
            "all_users_held_out_exactly_once": all_users_once,
            "all_samples_held_out_exactly_once": all_samples_once,
            "aggregate_validation_class_counts_match_global": aggregate_matches,
            "validation_user_occurrences": occurrence_dict,
            "validation_samples_total": sum(
                fold["val_samples"] for fold in folds
            ),
            "fold_user_counts": [len(fold["val_users"]) for fold in folds],
            "fold_sample_counts": [fold["val_samples"] for fold in folds],
            "fold_class_coverage": [
                fold["val_classes_covered"] for fold in folds
            ],
            "minimum_fold_class_coverage": min(
                fold["val_classes_covered"] for fold in folds
            ),
        },
        "users": per_user,
        "folds": folds,
    }


def validate_protocol(protocol: dict[str, object]) -> None:
    diagnostics = protocol["global_diagnostics"]
    checks = protocol["coverage_checks"]
    folds = protocol["folds"]
    if diagnostics["users"] != 18:
        raise RuntimeError(
            f"expected 18 users for this protocol, found {diagnostics['users']}"
        )
    if diagnostics["classes"] != 40:
        raise RuntimeError(
            f"expected 40 classes for this protocol, found {diagnostics['classes']}"
        )
    if len(folds) != N_FOLDS:
        raise RuntimeError(f"expected {N_FOLDS} folds, found {len(folds)}")
    required_true = (
        "all_users_held_out_exactly_once",
        "all_samples_held_out_exactly_once",
        "aggregate_validation_class_counts_match_global",
    )
    failed = [name for name in required_true if not checks[name]]
    if failed:
        raise RuntimeError(f"protocol coverage checks failed: {failed}")
    if protocol["always_train"]:
        raise RuntimeError("the all-user protocol cannot have always-train users")
    for fold in folds:
        overlap = set(fold["val_users"]).intersection(fold["train_users"])
        if overlap:
            raise RuntimeError(
                f"fold {fold['fold']} has train/validation overlap: {sorted(overlap)}"
            )
        if fold["train_classes_covered"] != diagnostics["classes"]:
            raise RuntimeError(
                f"fold {fold['fold']} training partition omits classes: "
                f"{fold['train_missing_classes']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"cache metadata CSV (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"protocol JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="fixed deterministic coordinate-search seed",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="build and validate in memory without writing the JSON artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    rows, users, labels = load_metadata(metadata_path)
    blocks = split_user_blocks(users)
    capacities = [
        block_capacities(len(block), block_index)
        for block_index, block in enumerate(blocks)
    ]
    global_class_counts = np.asarray(
        [
            sum(user.class_counts[index] for user in users)
            for index in range(len(labels))
        ],
        dtype=np.int64,
    )
    assignments = optimize_assignment(
        blocks, capacities, global_class_counts, args.seed
    )
    protocol = build_protocol(
        metadata_path,
        rows,
        users,
        labels,
        blocks,
        capacities,
        assignments,
        args.seed,
    )
    validate_protocol(protocol)

    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(protocol, handle, indent=2)
            handle.write("\n")

    folds = protocol["folds"]
    print(
        f"validated {protocol['protocol_id']}: "
        f"{protocol['global_diagnostics']['users']} users, "
        f"{protocol['global_diagnostics']['samples']} samples, "
        f"{protocol['global_diagnostics']['classes']} classes"
    )
    for fold in folds:
        print(
            f"fold {fold['fold']}: {len(fold['val_users'])} users, "
            f"{fold['val_samples']} samples, "
            f"{fold['val_classes_covered']}/"
            f"{protocol['global_diagnostics']['classes']} classes; "
            f"val={','.join(fold['val_users'])}"
        )
    print(
        "coverage: every user once="
        f"{protocol['coverage_checks']['all_users_held_out_exactly_once']}, "
        "every sample once="
        f"{protocol['coverage_checks']['all_samples_held_out_exactly_once']}"
    )
    if not args.check_only:
        print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
