#!/usr/bin/env python3
"""Run a packaged int8 ensemble and write a validated Kaggle submission.

The default streamed loader reconstructs, runs, and releases one ensemble
member at a time.  This preserves the original member order and float64
probability accumulation while avoiding the fp32 expansion of all members
being resident simultaneously.  ``--member-loading eager`` retains the legacy
all-model loader for diagnostics.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

import har_data
from package_ensemble import (
    iter_ensemble_members,
    load_ensemble,
    member_memory_profile,
    package_storage_profile,
    read_manifest,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def dataset_key(member):
    role = member.get("role")
    if role in ("gcn_soup", "adaptive_graph"):
        return "graph"
    if role == "imu":
        return "imu"
    if role == "world_imu":
        return "world_imu"
    if role in ("tcn_soup", "multi_stream_tcn"):
        return "tcn"
    raise ValueError(f"unsupported packaged member role: {role!r}")


def load_test_inputs(batch_size, workers, members):
    specs = {
        "tcn": ("skel", {"feat": "jvb"}),
        "graph": ("skelg", {}),
        "imu": ("imu", {"imu_feat": "inv"}),
    }
    needed = {dataset_key(member) for member in members}
    loaded = {}
    reference_ids = None
    for key, (modality, kwargs) in specs.items():
        if key not in needed:
            continue
        dataset = har_data.HARDataset("test", modality, aug=False, **kwargs)
        tensors, sample_ids = [], []
        for x, _, sids in DataLoader(
            dataset, batch_size, num_workers=workers, shuffle=False
        ):
            tensors.append(x)
            sample_ids.extend(sids)
        if reference_ids is None:
            reference_ids = sample_ids
        elif sample_ids != reference_ids:
            raise RuntimeError(f"test sample order differs for {key}")
        loaded[key] = torch.cat(tensors)
    if "world_imu" in needed:
        import train_imu_world

        feature_payloads = [
            member.get("feature_config")
            for member in members
            if dataset_key(member) == "world_imu"
        ]
        if (
            not feature_payloads
            or any(payload is None for payload in feature_payloads)
            or any(payload != feature_payloads[0] for payload in feature_payloads)
        ):
            raise RuntimeError(
                "packaged world-IMU members need one shared feature_config"
            )
        feature_config = train_imu_world.FeatureConfig(**feature_payloads[0])
        dataset = train_imu_world.IMUWorldDataset(
            Path(ROOT) / "cache",
            Path(ROOT) / "research" / "artifacts" / "cv_folds.json",
            "test",
            feature_config,
        ).preload()
        tensors, sample_ids = [], []
        for x, _, sids in DataLoader(
            dataset, batch_size, num_workers=workers, shuffle=False
        ):
            tensors.append(x)
            sample_ids.extend(sids)
        if reference_ids is None:
            reference_ids = sample_ids
        elif sample_ids != reference_ids:
            raise RuntimeError("test sample order differs for world_imu")
        loaded["world_imu"] = torch.cat(tensors)
    return loaded, reference_ids


def write_submission(path, sample_ids, prediction):
    if len(sample_ids) != 405 or len(set(sample_ids)) != 405:
        raise RuntimeError("expected exactly 405 unique test sample IDs")
    if prediction.shape != (405,) or prediction.min() < 0 or prediction.max() > 39:
        raise RuntimeError("invalid prediction vector")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "prediction"))
        for sid, label in zip(sample_ids, prediction):
            writer.writerow((f"small_model_track_test/{sid}/", int(label)))
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eager_members_with_memory(members):
    """Adapt the compatibility loader to the streamed iterator contract."""
    for member, model in members:
        yield member, model, member_memory_profile(member, model)


def fuse_members(
    member_iterator,
    inputs,
    sample_count,
    batch_size,
    device,
    loading_mode,
    per_member=None,
):
    """Fuse members in manifest order and report logical resident byte bounds."""
    probabilities = np.zeros((sample_count, 40), np.float64)
    profiles: list[tuple[str, Mapping[str, int]]] = []
    with torch.no_grad():
        for index, (member, model, memory) in enumerate(member_iterator):
            x = inputs[dataset_key(member)]
            member_probs = []
            for start in range(0, len(x), batch_size):
                logits = model(x[start:start + batch_size].to(device))
                member_probs.append(torch.softmax(logits, 1).cpu().numpy())
            fused_member = np.concatenate(member_probs)
            if per_member is not None:
                per_member.append((str(member["id"]), fused_member.astype(np.float32)))
            probabilities += float(member["weight"]) * fused_member
            profiles.append((str(member["id"]), memory))
            print(
                f"[{index + 1:02d}] {member['id']}: "
                f"parameter_bytes={memory['parameter_bytes']:,} "
                f"packed_bytes={memory['packed_tensor_bytes']:,}",
                flush=True,
            )
            # In stream mode this drops the caller's references before the
            # iterator resumes, releases its own reference, and builds the
            # next member.  Eager mode intentionally retains models in its
            # compatibility list.
            del fused_member, member_probs, logits, model

    if not profiles:
        raise RuntimeError("packaged ensemble contains no members")
    total_parameter_bytes = sum(
        profile["parameter_bytes"] for _, profile in profiles
    )
    total_fp32_parameter_bytes = sum(
        profile["fp32_parameter_bytes"] for _, profile in profiles
    )
    total_model_bytes = sum(profile["model_bytes"] for _, profile in profiles)
    peak_id, peak_profile = max(
        profiles, key=lambda item: item[1]["parameter_bytes"]
    )
    stream_peak_id, stream_peak_profile = max(
        profiles,
        key=lambda item: item[1]["stream_peak_live_fp32_parameter_bytes"],
    )
    if loading_mode == "stream":
        max_live_fp32_parameter_bytes = stream_peak_profile[
            "stream_peak_live_fp32_parameter_bytes"
        ]
    else:
        # The eager compatibility path retains every completed model.  Add one
        # largest decoded parameter as a conservative reconstruction staging
        # bound while the final resident set is assembled.
        max_live_fp32_parameter_bytes = total_fp32_parameter_bytes + max(
            profile["largest_decoded_fp32_parameter_bytes"]
            for _, profile in profiles
        )
    runtime_profile = {
        "members": len(profiles),
        "eager_all_parameter_bytes": int(total_parameter_bytes),
        "eager_all_fp32_parameter_bytes": int(total_fp32_parameter_bytes),
        "eager_all_model_bytes": int(total_model_bytes),
        "peak_member_id": peak_id,
        "peak_member_parameter_bytes": int(
            peak_profile["parameter_bytes"]
        ),
        "peak_member_model_bytes": int(
            max(profile["model_bytes"] for _, profile in profiles)
        ),
        "peak_member_restored_state_bytes": int(
            max(
                profile["restored_state_bytes"]
                for _, profile in profiles
            )
        ),
        "peak_member_model_plus_state_bytes": int(
            max(
                profile["model_bytes"] + profile["restored_state_bytes"]
                for _, profile in profiles
            )
        ),
        "stream_peak_member_id": stream_peak_id,
        "stream_peak_live_fp32_parameter_bytes": int(
            stream_peak_profile["stream_peak_live_fp32_parameter_bytes"]
        ),
        "max_live_fp32_parameter_bytes": int(
            max_live_fp32_parameter_bytes
        ),
        "peak_member_packed_bytes": int(
            max(profile["packed_tensor_bytes"] for _, profile in profiles)
        ),
        "peak_tensor_payload_bytes": int(
            max(profile["largest_payload_bytes"] for _, profile in profiles)
        ),
        "persistent_model_parameter_bytes": int(
            peak_profile["parameter_bytes"]
            if loading_mode == "stream"
            else total_parameter_bytes
        ),
    }
    return probabilities, runtime_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("output")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--member-loading",
        choices=("stream", "eager"),
        default="stream",
        help=(
            "stream (default) reconstructs and releases one model at a time; "
            "eager retains all models for legacy diagnostics"
        ),
    )
    parser.add_argument("--reference", help="optional CSV for top-1 parity reporting")
    parser.add_argument(
        "--probabilities-output",
        help=(
            "optional compressed NPZ containing the exact packaged-ensemble "
            "probabilities and sample IDs"
        ),
    )
    parser.add_argument(
        "--per-member-output",
        help=(
            "optional NPZ of every member's own probabilities, keyed by member id, "
            "so pruning subsets can be evaluated offline without re-running inference"
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_manifest(args.artifact)
    member_specs = manifest["members"]
    inputs, sample_ids = load_test_inputs(
        args.batch_size, args.workers, member_specs
    )
    if args.member_loading == "stream":
        member_iterator = iter_ensemble_members(
            args.artifact, device=device, verify_payloads=True
        )
    else:
        eager_manifest, eager_members = load_ensemble(
            args.artifact, device=device, verify_payloads=True
        )
        if eager_manifest != manifest:
            raise RuntimeError("artifact manifest changed while loading")
        member_iterator = eager_members_with_memory(eager_members)
    per_member = [] if args.per_member_output else None
    probabilities, runtime_profile = fuse_members(
        member_iterator,
        inputs,
        len(sample_ids),
        args.batch_size,
        device,
        args.member_loading,
        per_member,
    )
    if per_member is not None:
        np.savez_compressed(
            args.per_member_output,
            sids=np.array([str(s) for s in sample_ids]),
            weights=np.array([float(m["weight"]) for m in member_specs], np.float64),
            ids=np.array([str(m["id"]) for m in member_specs]),
            **{mid: p for mid, p in per_member},
        )
        print(f"wrote per-member probabilities: {args.per_member_output}")
    storage_profile = package_storage_profile(args.artifact, manifest)
    print(
        f"member_loading={args.member_loading} "
        f"archive_loader=zip_per_tensor torch_load_used=no "
        f"all_int8_state_materialized_by_torch_load=no "
        f"full_int8_package_materialized=no "
        f"decoded_state_tensors_at_once=1 "
        f"package_file_bytes={storage_profile['package_bytes']:,} "
        f"package_tensor_payload_bytes="
        f"{storage_profile['package_tensor_payload_bytes']:,} "
        f"largest_archive_tensor_payload_bytes="
        f"{storage_profile['largest_tensor_payload_bytes']:,} "
        f"eager_all_parameter_bytes="
        f"{runtime_profile['eager_all_parameter_bytes']:,} "
        f"eager_all_model_bytes="
        f"{runtime_profile['eager_all_model_bytes']:,} "
        f"peak_member={runtime_profile['peak_member_id']} "
        f"peak_member_parameter_bytes="
        f"{runtime_profile['peak_member_parameter_bytes']:,} "
        f"peak_member_restored_state_bytes="
        f"{runtime_profile['peak_member_restored_state_bytes']:,} "
        f"peak_member_model_plus_state_bytes="
        f"{runtime_profile['peak_member_model_plus_state_bytes']:,} "
        f"stream_peak_member="
        f"{runtime_profile['stream_peak_member_id']} "
        f"stream_peak_live_fp32_parameter_bytes="
        f"{runtime_profile['stream_peak_live_fp32_parameter_bytes']:,} "
        f"max_live_fp32_parameter_bytes="
        f"{runtime_profile['max_live_fp32_parameter_bytes']:,} "
        f"peak_member_packed_bytes="
        f"{runtime_profile['peak_member_packed_bytes']:,} "
        f"peak_package_payload_resident_bytes="
        f"{runtime_profile['peak_tensor_payload_bytes']:,} "
        f"persistent_model_parameter_bytes="
        f"{runtime_profile['persistent_model_parameter_bytes']:,}",
        flush=True,
    )

    if not np.isfinite(probabilities).all():
        raise RuntimeError("packaged inference produced non-finite probabilities")
    if not np.allclose(probabilities.sum(1), 1.0, atol=1e-5):
        raise RuntimeError("fused probabilities do not sum to one")
    prediction = probabilities.argmax(1)
    digest = write_submission(args.output, sample_ids, prediction)

    if args.probabilities_output:
        probability_path = Path(args.probabilities_output).resolve()
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probability_path,
            # Preserve the float64 accumulation exactly. Structured decoders
            # compare whole-path log scores, where an unnecessary float32 cast
            # could change a near-tied path even when top-1 argmax is stable.
            probs=probabilities,
            sids=np.asarray(sample_ids),
            ensemble=np.asarray(manifest["ensemble"]["name"]),
            artifact_sha256=np.asarray(sha256_file(args.artifact)),
        )
        print(f"probabilities={probability_path}", flush=True)

    print(
        f"ensemble={manifest['ensemble']['name']} rows={len(sample_ids)} "
        f"output={os.path.abspath(args.output)} sha256={digest}",
        flush=True,
    )
    if args.reference:
        with open(args.reference, newline="") as handle:
            rows = list(csv.DictReader(handle))
        reference = {
            row["path"].strip("/").split("/")[-1]: int(row["prediction"])
            for row in rows
        }
        expected = np.asarray([reference[sid] for sid in sample_ids])
        changed = np.flatnonzero(prediction != expected)
        print(f"changes_vs_reference={len(changed)}", flush=True)
        for index in changed:
            print(
                f"  {sample_ids[index]}: {expected[index]} -> {prediction[index]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
