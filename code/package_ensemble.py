#!/usr/bin/env python3
"""Build and load deterministic <=100 MB int8 HAR ensemble artifacts.

The artifact is a regular ZIP file (the extension is unrestricted) containing:

* ``manifest.json``: model construction, fusion weights, source hashes, tensor
  metadata, and build-time numerical verification results.
* ``tensors/...``: one symmetric-int8 payload per floating-point state tensor.
  Non-floating tensors are stored losslessly.

The module is deliberately standalone so the same file can be imported by the
competition inference entry point.  ``iter_ensemble_members`` is the
memory-bounded deployment API: it opens the ZIP once and reconstructs exactly
one member at a time.  ``load_ensemble`` remains available for compatibility,
but deliberately retains every reconstructed model:

    from package_ensemble import iter_ensemble_members, read_manifest
    manifest = read_manifest("model.pth")
    for member, model, memory in iter_ensemble_members(
        "model.pth", device="cuda"
    ):
        ...

Each item in ``members`` is ``(member_manifest, model)``.  The current presets
use probability-space fusion, so inference should accumulate
``member["weight"] * softmax(model(x), dim=1)``.

Build the verified block8 ensemble:

    python3 code/package_ensemble.py build \
        --preset block8 --output artifacts/model_block8_int8.pth

Build an arbitrary selection using the JSON schema printed by ``example-spec``:

    python3 code/package_ensemble.py example-spec > /tmp/ensemble_spec.json
    python3 code/package_ensemble.py build \
        --spec /tmp/ensemble_spec.json --output artifacts/model_int8.pth

Inspect and cryptographically verify an artifact:

    python3 code/package_ensemble.py inspect artifacts/model_int8.pth --verify
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
import zipfile
from collections import OrderedDict
from typing import Any, Iterable, Iterator, Mapping

import torch


FORMAT_NAME = "cuhkx.har-ensemble.int8"
FORMAT_VERSION = 1
RULE_LIMIT_BYTES = 100_000_000  # Competition's strict decimal 100 MB limit.
DEFAULT_VERIFY_BATCH = 8
MANIFEST_ENTRY = "manifest.json"
SIDECAR_SUFFIX = ".sha256"


TCN5 = (
    "skel_jvb_big",
    "skel_jvb_big_s1",
    "skel_jvb_big_s2",
    "skel_jvb_big_s3",
    "skel_jvb_big_s4",
)
GCN3 = ("stgcn_v1", "stgcn_s1", "stgcn_w96")


TAG_MODELS: dict[str, dict[str, Any]] = {
    **{
        tag: {
            "modality": "skel",
            "in_ch": 153,
            "kwargs": {"width": 256},
            "input": {"shape": [DEFAULT_VERIFY_BATCH, 32, 153], "kind": "normal", "scale": 0.2},
        }
        for tag in (
            *TCN5,
            "skel_mildaug",
            "skel_dynaug",
            "skel_ls0",
            "skel_pmotion",
        )
    },
    "skel_w192": {
        "modality": "skel",
        "in_ch": 153,
        "kwargs": {"width": 192},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 32, 153], "kind": "normal", "scale": 0.2},
    },
    **{
        tag: {
            "modality": "skelg",
            "in_ch": None,
            "kwargs": {},
            "input": {"shape": [DEFAULT_VERIFY_BATCH, 9, 32, 17], "kind": "normal", "scale": 0.2},
        }
        for tag in ("stgcn_v1", "stgcn_s1")
    },
    **{
        tag: {
            "modality": "skelg",
            "in_ch": None,
            "kwargs": {"width": 96},
            "input": {"shape": [DEFAULT_VERIFY_BATCH, 9, 32, 17], "kind": "normal", "scale": 0.2},
        }
        for tag in ("stgcn_w96", "stgcn_w96_s1")
    },
    "astgcn_v1": {
        "modality": "skelg",
        "in_ch": None,
        "kwargs": {"arch": "adaptive", "width": 64},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 9, 32, 17], "kind": "normal", "scale": 0.2},
    },
    "skel_multitcn": {
        "modality": "skel",
        "in_ch": 153,
        "kwargs": {"arch": "multitcn", "width": 128},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 32, 153], "kind": "normal", "scale": 0.2},
    },
    "skel_multitcn_dann": {
        "modality": "skel",
        "in_ch": 153,
        "kwargs": {"arch": "multitcn_dann", "width": 128},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 32, 153], "kind": "normal", "scale": 0.2},
    },
    "skel_pad64": {
        "modality": "skel",
        "in_ch": 154,
        "kwargs": {"arch": "masktcn", "width": 256},
        "input": {
            "shape": [DEFAULT_VERIFY_BATCH, 64, 154],
            "kind": "masked_skeleton",
            "scale": 0.2,
        },
    },
    "skel_bigru": {
        "modality": "skel",
        "in_ch": 153,
        "kwargs": {"arch": "bigru", "width": 128},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 32, 153], "kind": "normal", "scale": 0.2},
    },
    "imu_inv4": {
        "modality": "imu",
        "in_ch": 65,
        "kwargs": {},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 65, 64], "kind": "normal", "scale": 0.2},
    },
    "imu_aug": {
        "modality": "imu",
        "in_ch": 85,
        "kwargs": {},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 85, 64], "kind": "normal", "scale": 0.2},
    },
    "imu_lstm": {
        "modality": "imu",
        "in_ch": 65,
        "kwargs": {"arch": "lstm"},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 65, 64], "kind": "normal", "scale": 0.2},
    },
    "imu_world": {
        "modality": "imu_world",
        "in_ch": 45,
        "kwargs": {"width": 128, "depth": 4, "dropout": 0.3, "classes": 40},
        "input": {"shape": [DEFAULT_VERIFY_BATCH, 64, 45], "kind": "normal", "scale": 0.2},
    },
    "droi_big": {
        "modality": "droi",
        "in_ch": None,
        "kwargs": {"width": 64},
        "input": {
            "shape": [DEFAULT_VERIFY_BATCH, 12, 112, 112],
            "kind": "uniform",
            "scale": 1.0,
        },
    },
}


DTYPES: dict[str, torch.dtype] = {
    name.removeprefix("torch."): value
    for name, value in vars(torch).items()
    if isinstance(value, torch.dtype)
}


class PackageError(RuntimeError):
    """A malformed spec, checkpoint, or artifact."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_text(path: pathlib.Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def deterministic_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def load_checkpoint(path: pathlib.Path) -> OrderedDict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility.
        loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    elif isinstance(loaded, Mapping) and "model" in loaded and isinstance(loaded["model"], Mapping):
        loaded = loaded["model"]
    if not isinstance(loaded, Mapping):
        raise PackageError(f"{path}: checkpoint is not a state_dict mapping")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in loaded.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise PackageError(f"{path}: state_dict entry {name!r} is not a tensor")
        if tensor.layout != torch.strided:
            raise PackageError(f"{path}: sparse/non-strided tensor {name!r} is unsupported")
        if tensor.is_complex():
            raise PackageError(f"{path}: complex tensor {name!r} is unsupported")
        state[name] = tensor.detach().cpu().contiguous()
    if not state:
        raise PackageError(f"{path}: state_dict is empty")
    return state


def resolve_dtype(name: str) -> torch.dtype:
    short = name.removeprefix("torch.")
    if short not in DTYPES:
        raise PackageError(f"unsupported torch dtype {name!r}")
    return DTYPES[short]


def raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.numel() == 0:
        return b""
    return tensor.contiguous().view(torch.uint8).numpy().tobytes(order="C")


def quantize_tensor(tensor: torch.Tensor) -> tuple[bytes, dict[str, Any], torch.Tensor]:
    """Return payload bytes, metadata, and a dequantized verification tensor."""
    shape = list(tensor.shape)
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all():
            raise PackageError("cannot quantize a tensor containing NaN or infinity")
        max_abs = float(tensor.float().abs().max()) if tensor.numel() else 0.0
        scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
        quantized = (
            torch.round(tensor.float() / scale)
            .clamp_(-127, 127)
            .to(torch.int8)
            .contiguous()
        )
        payload = raw_tensor_bytes(quantized)
        reconstructed = (quantized.float() * scale).to(tensor.dtype)
        meta = {
            "encoding": "symmetric_int8_per_tensor",
            "dtype": dtype_name,
            "shape": shape,
            "scale": scale,
            "zero_point": 0,
            "nbytes": len(payload),
        }
        return payload, meta, reconstructed

    payload = raw_tensor_bytes(tensor)
    meta = {
        "encoding": "raw",
        "dtype": dtype_name,
        "shape": shape,
        "nbytes": len(payload),
    }
    return payload, meta, tensor.clone()


def tensor_from_payload(payload: bytes, meta: Mapping[str, Any]) -> torch.Tensor:
    shape = tuple(int(x) for x in meta["shape"])
    count = math.prod(shape)
    encoding = meta["encoding"]
    dtype = resolve_dtype(str(meta["dtype"]))
    if len(payload) != int(meta["nbytes"]):
        raise PackageError(
            f"tensor payload length mismatch: got {len(payload)}, expected {meta['nbytes']}"
        )
    if count == 0:
        return torch.empty(shape, dtype=dtype)
    # bytearray provides a writable buffer and avoids PyTorch's read-only warning.
    backing = bytearray(payload)
    if encoding == "symmetric_int8_per_tensor":
        expected = count
        if len(payload) != expected:
            raise PackageError(f"int8 payload has {len(payload)} bytes for {count} elements")
        quantized = torch.frombuffer(backing, dtype=torch.int8).clone().reshape(shape)
        # Use one float32 staging tensor.  The non-inplace spelling
        # ``quantized.float() * scale`` can briefly retain both the conversion
        # and multiplication result, doubling the largest decoded tensor's
        # fp32 workspace.
        restored = quantized.float()
        restored.mul_(float(meta["scale"]))
        return restored.to(dtype)
    if encoding == "raw":
        itemsize = torch.empty((), dtype=dtype).element_size()
        expected = count * itemsize
        if len(payload) != expected:
            raise PackageError(f"raw payload has {len(payload)} bytes; expected {expected}")
        raw = torch.frombuffer(backing, dtype=torch.uint8).clone()
        return raw.view(dtype).reshape(shape)
    raise PackageError(f"unknown tensor encoding {encoding!r}")


def import_model_module():
    code_dir = pathlib.Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import har_models  # Local import keeps ``inspect`` usable without model code.

    return har_models


def build_model(model_spec: Mapping[str, Any]) -> torch.nn.Module:
    modality = str(model_spec["modality"])
    kwargs = copy.deepcopy(dict(model_spec.get("kwargs", {})))
    in_ch = model_spec.get("in_ch")
    if modality == "imu_world":
        code_dir = pathlib.Path(__file__).resolve().parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        import train_imu_world

        if in_ch is not None:
            kwargs["in_channels"] = int(in_ch)
        return train_imu_world.IMUWorldTCN(train_imu_world.ModelConfig(**kwargs))
    har_models = import_model_module()
    if in_ch is None:
        return har_models.build(modality, **kwargs)
    return har_models.build(modality, in_ch=int(in_ch), **kwargs)


def synthetic_input(input_spec: Mapping[str, Any], seed: int) -> torch.Tensor:
    shape = [int(x) for x in input_spec["shape"]]
    if not shape or any(x <= 0 for x in shape):
        raise PackageError(f"invalid synthetic input shape {shape!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kind = str(input_spec.get("kind", "normal"))
    scale = float(input_spec.get("scale", 1.0))
    if kind == "normal":
        return torch.randn(shape, generator=generator) * scale
    if kind == "uniform":
        return torch.rand(shape, generator=generator) * scale
    if kind == "masked_skeleton":
        x = torch.randn(shape, generator=generator) * scale
        if shape[-1] < 2:
            raise PackageError("masked_skeleton input requires a final mask channel")
        x[..., -1] = 1.0
        # Exercise padded pooling deterministically in half of each sequence.
        x[: max(1, shape[0] // 2), shape[1] // 2 :, -1] = 0.0
        x[..., :-1] *= x[..., -1:]
        return x
    raise PackageError(f"unknown synthetic input kind {kind!r}")


def verification_metrics(
    original_state: Mapping[str, torch.Tensor],
    reconstructed_state: Mapping[str, torch.Tensor],
    model_spec: Mapping[str, Any],
    input_spec: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    original = build_model(model_spec)
    reconstructed = build_model(model_spec)
    original.load_state_dict(original_state, strict=True)
    reconstructed.load_state_dict(reconstructed_state, strict=True)
    original.eval()
    reconstructed.eval()
    x = synthetic_input(input_spec, seed)
    with torch.inference_mode():
        logits_a = original(x).float()
        logits_b = reconstructed(x).float()
    if logits_a.shape != logits_b.shape or logits_a.ndim != 2:
        raise PackageError(
            f"model verification expected 2-D equal logits, got "
            f"{tuple(logits_a.shape)} and {tuple(logits_b.shape)}"
        )
    difference = (logits_a - logits_b).abs()
    cosine = torch.nn.functional.cosine_similarity(logits_a, logits_b, dim=1)
    metrics = {
        "argmax_match": round(
            float((logits_a.argmax(1) == logits_b.argmax(1)).float().mean()), 10
        ),
        "logit_mean_abs_error": round(float(difference.mean()), 10),
        "logit_max_abs_error": round(float(difference.max()), 10),
        "mean_cosine_similarity": round(float(cosine.mean()), 10),
        "batch_size": int(logits_a.shape[0]),
        "seed": int(seed),
    }
    del original, reconstructed, x, logits_a, logits_b
    return metrics


def normalized_model_spec(member: Mapping[str, Any]) -> dict[str, Any]:
    if "model" not in member:
        raise PackageError(f"member {member.get('id')!r} has no model construction spec")
    model = copy.deepcopy(dict(member["model"]))
    if "modality" not in model:
        raise PackageError(f"member {member.get('id')!r} model has no modality")
    model.setdefault("in_ch", None)
    model.setdefault("kwargs", {})
    if not isinstance(model["kwargs"], Mapping):
        raise PackageError(f"member {member.get('id')!r} model kwargs must be an object")
    return model


def inferred_input_spec(model: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    modality = str(model["modality"])
    kwargs = dict(model.get("kwargs", {}))
    in_ch = model.get("in_ch")
    arch = str(kwargs.get("arch", ""))
    if modality == "skelg":
        return {"shape": [batch_size, 9, 32, 17], "kind": "normal", "scale": 0.2}
    if modality == "skel":
        channels = int(in_ch or (154 if arch == "masktcn" else 153))
        steps = 64 if arch == "masktcn" else 32
        kind = "masked_skeleton" if arch == "masktcn" else "normal"
        return {"shape": [batch_size, steps, channels], "kind": kind, "scale": 0.2}
    if modality == "imu":
        return {
            "shape": [batch_size, int(in_ch or 85), 64],
            "kind": "normal",
            "scale": 0.2,
        }
    if modality == "imu_world":
        return {
            "shape": [batch_size, 64, int(in_ch or 45)],
            "kind": "normal",
            "scale": 0.2,
        }
    if modality in ("depth", "ir", "droi", "iroi", "droi224", "iroi224", "hroi"):
        side = 224 if modality.endswith("224") else 112
        return {"shape": [batch_size, 8, side, side], "kind": "uniform", "scale": 1.0}
    raise PackageError(f"cannot infer synthetic input for modality {modality!r}")


def normalize_spec(spec: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise PackageError("ensemble spec must be a JSON object")
    members_raw = spec.get("members")
    if not isinstance(members_raw, list) or not members_raw:
        raise PackageError("ensemble spec needs a non-empty members list")
    normalized: dict[str, Any] = {
        "name": str(spec.get("name", "custom_ensemble")),
        "fusion": copy.deepcopy(
            spec.get(
                "fusion",
                {
                    "type": "weighted_probability_sum",
                    "normalization": "softmax",
                },
            )
        ),
        "members": [],
    }
    if "notes" in spec:
        normalized["notes"] = str(spec["notes"])
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    total_weight = 0.0
    for index, raw in enumerate(members_raw):
        if not isinstance(raw, Mapping):
            raise PackageError(f"member #{index} is not an object")
        member_id = str(raw.get("id", "")).strip()
        checkpoint = str(raw.get("checkpoint", "")).strip()
        if not member_id or member_id in seen_ids:
            raise PackageError(f"member id {member_id!r} is empty or duplicated")
        if not checkpoint or checkpoint in seen_paths:
            raise PackageError(f"checkpoint {checkpoint!r} is empty or duplicated")
        seen_ids.add(member_id)
        seen_paths.add(checkpoint)
        weight = float(raw.get("weight", 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise PackageError(f"member {member_id!r} has invalid weight {weight}")
        total_weight += weight
        model = normalized_model_spec(raw)
        input_spec = copy.deepcopy(
            raw.get("input", inferred_input_spec(model, batch_size))
        )
        input_spec["shape"] = [int(x) for x in input_spec["shape"]]
        input_spec["shape"][0] = batch_size
        member: dict[str, Any] = {
            "id": member_id,
            "checkpoint": checkpoint,
            "weight": weight,
            "model": model,
            "input": input_spec,
        }
        for optional in ("tag", "fold", "role", "feature_config"):
            if optional in raw:
                member[optional] = copy.deepcopy(raw[optional])
        normalized["members"].append(member)
    if total_weight <= 0.0:
        raise PackageError("ensemble member weights sum to zero")
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise PackageError(f"ensemble member weights must sum to 1.0, got {total_weight:.12f}")
    return normalized


def preset_block8(gcn_weight: float = 0.30) -> dict[str, Any]:
    if not (0.0 <= gcn_weight <= 1.0):
        raise PackageError("GCN block weight must be in [0, 1]")
    members: list[dict[str, Any]] = []
    tcn_total = 0.8 * (1.0 - gcn_weight)
    gcn_total = 0.8 * gcn_weight
    for tag in TCN5:
        for fold in range(4):
            members.append(
                {
                    "id": f"{tag}_f{fold}",
                    "checkpoint": f"checkpoints/{tag}_f{fold}.pt",
                    "weight": tcn_total / (len(TCN5) * 4),
                    "tag": tag,
                    "fold": fold,
                    "role": "tcn_soup",
                    "model": copy.deepcopy(TAG_MODELS[tag]),
                    "input": copy.deepcopy(TAG_MODELS[tag]["input"]),
                }
            )
            members[-1]["model"].pop("input", None)
    for tag in GCN3:
        for fold in range(4):
            members.append(
                {
                    "id": f"{tag}_f{fold}",
                    "checkpoint": f"checkpoints/{tag}_f{fold}.pt",
                    "weight": gcn_total / (len(GCN3) * 4),
                    "tag": tag,
                    "fold": fold,
                    "role": "gcn_soup",
                    "model": copy.deepcopy(TAG_MODELS[tag]),
                    "input": copy.deepcopy(TAG_MODELS[tag]["input"]),
                }
            )
            members[-1]["model"].pop("input", None)
    for fold in range(4):
        tag = "imu_inv4"
        members.append(
            {
                "id": f"{tag}_f{fold}",
                "checkpoint": f"checkpoints/{tag}_f{fold}.pt",
                "weight": 0.2 / 4,
                "tag": tag,
                "fold": fold,
                "role": "imu",
                "model": copy.deepcopy(TAG_MODELS[tag]),
                "input": copy.deepcopy(TAG_MODELS[tag]["input"]),
            }
        )
        members[-1]["model"].pop("input", None)
    return {
        "name": "block8" if gcn_weight == 0.30 else f"block8_g{int(gcn_weight * 100):02d}",
        "fusion": {
            "type": "weighted_probability_sum",
            "normalization": "softmax",
            "description": (
                f"0.8 * ({1-gcn_weight:.2f} * mean(TCN5) + "
                f"{gcn_weight:.2f} * mean(GCN3)) + 0.2 * mean(IMU)"
            ),
        },
        "notes": "Known rules-clean block8 fold-probability ensemble.",
        "members": members,
    }


def preset_astgcn_a20() -> dict[str, Any]:
    """Package the accepted EXP-040 fixed-weight ensemble.

    Formula:
      0.825 * (0.80 * (0.925 * (0.50 * TCN5 + 0.50 * GCN3)
                              + 0.075 * MultiTCN)
               + 0.20 * AdaptiveSTGCN)
      + 0.175 * invariant-IMU
    """
    family_weights = {
        **{tag: 0.30525 / len(TCN5) for tag in TCN5},
        **{tag: 0.30525 / len(GCN3) for tag in GCN3},
        "skel_multitcn": 0.0495,
        "astgcn_v1": 0.165,
        "imu_inv4": 0.175,
    }
    roles = {
        **{tag: "tcn_soup" for tag in TCN5},
        **{tag: "gcn_soup" for tag in GCN3},
        "skel_multitcn": "multi_stream_tcn",
        "astgcn_v1": "adaptive_graph",
        "imu_inv4": "imu",
    }
    members: list[dict[str, Any]] = []
    for tag, family_weight in family_weights.items():
        for fold in range(4):
            model = copy.deepcopy(TAG_MODELS[tag])
            input_spec = model.pop("input")
            members.append(
                {
                    "id": f"{tag}_f{fold}",
                    "checkpoint": f"checkpoints/{tag}_f{fold}.pt",
                    "weight": family_weight / 4,
                    "tag": tag,
                    "fold": fold,
                    "role": roles[tag],
                    "model": model,
                    "input": input_spec,
                }
            )
    return {
        "name": "astgcn_a20",
        "fusion": {
            "type": "weighted_probability_sum",
            "normalization": "softmax",
            "description": (
                "0.825*(0.80*(0.925*(0.50*mean(TCN5)+0.50*mean(GCN3))"
                "+0.075*mean(MultiTCN))+0.20*mean(AdaptiveSTGCN))"
                "+0.175*mean(invariant-IMU)"
            ),
        },
        "notes": (
            "Fixed EXP-040 candidate used to generate sub_astgcn_a20.csv; "
            "center inference only."
        ),
        "members": members,
    }


def preset_astgcn_world25() -> dict[str, Any]:
    """Retain the accepted stack and add the all-fold-positive world IMU."""
    preset = preset_astgcn_a20()
    for member in preset["members"]:
        member["weight"] *= 0.75
    model = copy.deepcopy(TAG_MODELS["imu_world"])
    input_spec = model.pop("input")
    for fold in range(4):
        preset["members"].append(
            {
                "id": f"imu_world_f{fold}",
                "checkpoint": f"checkpoints/imu_world_f{fold}.pt",
                "weight": 0.25 / 4,
                "tag": "imu_world",
                "fold": fold,
                "role": "world_imu",
                "feature_config": {
                    "steps": 64,
                    "gyro_scale": 500.0,
                    "gravity_z": 1.0,
                },
                "model": copy.deepcopy(model),
                "input": copy.deepcopy(input_spec),
            }
        )
    preset["name"] = "astgcn_world25"
    preset["fusion"]["description"] = (
        "0.75*astgcn_a20 + 0.25*mean(world-frame-IMU)"
    )
    preset["notes"] = (
        "EXP-046 fixed candidate; world-frame IMU additive weight 0.25 "
        "confirmed positive on all four held-out subject folds."
    )
    return preset


def get_preset(name: str) -> dict[str, Any]:
    if name == "block8":
        return preset_block8(0.30)
    if name == "block8_g40":
        return preset_block8(0.40)
    if name == "astgcn_a20":
        return preset_astgcn_a20()
    if name == "astgcn_world25":
        return preset_astgcn_world25()
    raise PackageError(f"unknown preset {name!r}")


def example_spec() -> dict[str, Any]:
    model = copy.deepcopy(TAG_MODELS["skel_jvb_big"])
    input_spec = model.pop("input")
    return {
        "name": "example_single_member",
        "fusion": {
            "type": "weighted_probability_sum",
            "normalization": "softmax",
        },
        "members": [
            {
                "id": "skel_jvb_big_f0",
                "checkpoint": "checkpoints/skel_jvb_big_f0.pt",
                "weight": 1.0,
                "tag": "skel_jvb_big",
                "fold": 0,
                "model": model,
                "input": input_spec,
            }
        ],
    }


def member_state_hash(
    names_and_payloads: Iterable[tuple[str, bytes]],
) -> str:
    digest = hashlib.sha256()
    for name, payload in names_and_payloads:
        update_member_state_digest(digest, name, payload)
    return digest.hexdigest()


def update_member_state_digest(
    digest: Any,
    name: str,
    payload: bytes,
) -> None:
    """Update the canonical member hash without retaining prior payloads."""
    encoded = name.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def pack_member(
    member: Mapping[str, Any],
    root: pathlib.Path,
    member_index: int,
    verify_seed: int,
    verify: bool,
    min_argmax_match: float,
    max_mean_logit_error: float,
    min_cosine: float,
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    source = pathlib.Path(str(member["checkpoint"]))
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_size = source.stat().st_size
    source_digest = sha256_file(source)
    original = load_checkpoint(source)
    reconstructed: OrderedDict[str, torch.Tensor] = OrderedDict()
    entries: list[tuple[str, bytes]] = []
    tensors_meta: list[dict[str, Any]] = []
    safe_prefix = f"tensors/{member_index:04d}"
    for tensor_index, name in enumerate(sorted(original)):
        payload, tensor_meta, restored = quantize_tensor(original[name])
        entry = f"{safe_prefix}/{tensor_index:04d}.bin"
        tensor_meta.update(
            {
                "name": name,
                "entry": entry,
                "sha256": sha256_bytes(payload),
            }
        )
        tensors_meta.append(tensor_meta)
        entries.append((entry, payload))
        reconstructed[name] = restored

    # The model must always load strictly, even when numerical forward checking
    # was explicitly disabled.
    model = build_model(member["model"])
    model.load_state_dict(reconstructed, strict=True)
    del model
    metrics = None
    if verify:
        metrics = verification_metrics(
            original,
            reconstructed,
            member["model"],
            member["input"],
            verify_seed,
        )
        if metrics["argmax_match"] < min_argmax_match:
            raise PackageError(
                f"{member['id']}: int8 argmax match {metrics['argmax_match']:.3f} "
                f"is below {min_argmax_match:.3f}"
            )
        if metrics["logit_mean_abs_error"] > max_mean_logit_error:
            raise PackageError(
                f"{member['id']}: mean logit error "
                f"{metrics['logit_mean_abs_error']:.6f} exceeds "
                f"{max_mean_logit_error:.6f}"
            )
        if metrics["mean_cosine_similarity"] < min_cosine:
            raise PackageError(
                f"{member['id']}: logit cosine "
                f"{metrics['mean_cosine_similarity']:.6f} is below {min_cosine:.6f}"
            )

    output = {
        key: copy.deepcopy(value)
        for key, value in member.items()
        if key != "input"
    }
    try:
        output["source_checkpoint"] = str(source.relative_to(root))
    except ValueError:
        output["source_checkpoint"] = str(source)
    output.pop("checkpoint", None)
    output.update(
        {
            "state_sha256": member_state_hash(
                (meta["name"], payload)
                for meta, (_, payload) in zip(tensors_meta, entries)
            ),
            "tensors": tensors_meta,
        }
    )
    if metrics is not None:
        output["verification"] = metrics
    # Detect a checkpoint being overwritten by a concurrent training process.
    final_source_digest = sha256_file(source)
    if source.stat().st_size != source_size or final_source_digest != source_digest:
        raise PackageError(f"{source}: checkpoint changed while it was being packaged")
    output["source_sha256"] = source_digest
    output["source_size_bytes"] = source_size
    output["verification_input"] = copy.deepcopy(member["input"])
    return output, entries


def write_artifact(
    output: pathlib.Path,
    manifest: Mapping[str, Any],
    entries: Iterable[tuple[str, bytes]],
    max_bytes: int,
    force: bool,
) -> tuple[int, str]:
    sidecar = pathlib.Path(str(output) + SIDECAR_SUFFIX)
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")
    if sidecar.exists() and not force:
        raise FileExistsError(f"{sidecar} already exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(
            tmp_name,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=False,
        ) as archive:
            archive.writestr(deterministic_zip_info(MANIFEST_ENTRY), canonical_json(manifest))
            for entry, payload in sorted(entries, key=lambda item: item[0]):
                archive.writestr(deterministic_zip_info(entry), payload)
        size = os.path.getsize(tmp_name)
        if size > max_bytes:
            raise PackageError(
                f"artifact is {size:,} bytes, exceeding strict limit {max_bytes:,}; "
                "prune members or use a smaller compliant recipe"
            )
        digest = sha256_file(tmp_name)
        os.replace(tmp_name, output)
        atomic_text(sidecar, f"{digest}  {output.name}\n", force=force)
        return size, digest
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def build_artifact(args: argparse.Namespace) -> None:
    root = pathlib.Path(args.root).resolve()
    if args.spec:
        with open(args.spec, encoding="utf-8") as handle:
            source_spec = json.load(handle)
    else:
        source_spec = get_preset(args.preset)
    spec = normalize_spec(source_spec, batch_size=args.verify_batch)
    code_dir = pathlib.Path(__file__).resolve().parent
    model_sources = [code_dir / "har_models.py"]
    if any(
        member["model"]["modality"] == "imu_world"
        for member in spec["members"]
    ):
        model_sources.append(code_dir / "train_imu_world.py")
    model_source_digests = {
        source.name: sha256_file(source) for source in model_sources
    }
    max_bytes = int(round(args.max_mb * 1_000_000))
    if max_bytes <= 0 or max_bytes > RULE_LIMIT_BYTES:
        raise PackageError(
            f"--max-mb must be in (0, 100]; got {args.max_mb}. "
            "The utility will not emit a rules-noncompliant artifact."
        )
    print(
        f"packing {spec['name']}: {len(spec['members'])} members; "
        f"strict limit={max_bytes:,} bytes",
        flush=True,
    )
    packed_members: list[dict[str, Any]] = []
    entries: list[tuple[str, bytes]] = []
    for index, member in enumerate(spec["members"]):
        packed, member_entries = pack_member(
            member,
            root=root,
            member_index=index,
            verify_seed=args.verify_seed + index,
            verify=not args.no_verify,
            min_argmax_match=args.min_argmax_match,
            max_mean_logit_error=args.max_mean_logit_error,
            min_cosine=args.min_cosine,
        )
        packed_members.append(packed)
        entries.extend(member_entries)
        verify_text = ""
        if "verification" in packed:
            result = packed["verification"]
            verify_text = (
                f"; argmax={result['argmax_match']:.3f}; "
                f"mean|dlogit|={result['logit_mean_abs_error']:.5f}; "
                f"cos={result['mean_cosine_similarity']:.6f}"
            )
        payload_size = sum(len(payload) for _, payload in member_entries)
        print(
            f"[{index + 1:02d}/{len(spec['members']):02d}] {member['id']}: "
            f"{payload_size:,} packed bytes{verify_text}",
            flush=True,
        )
    for source in model_sources:
        if sha256_file(source) != model_source_digests[source.name]:
            raise PackageError(
                f"{source}: model source changed while the artifact was being built"
            )
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "quantization": {
            "floating_tensors": "symmetric_int8_per_tensor",
            "qmin": -127,
            "qmax": 127,
            "zero_point": 0,
            "non_floating_tensors": "lossless_raw",
        },
        "byte_order": sys.byteorder,
        "ensemble": {
            "name": spec["name"],
            "fusion": spec["fusion"],
            "notes": spec.get("notes", ""),
            "weight_sum": sum(float(member["weight"]) for member in packed_members),
        },
        "members": packed_members,
        "rules": {
            "maximum_artifact_bytes": max_bytes,
            "competition_ceiling_bytes": RULE_LIMIT_BYTES,
        },
        "loader": {
            "module": pathlib.Path(__file__).name,
            "function": "iter_ensemble_members",
            "archive_reader": "zipfile_per_tensor",
            "torch_load_used": False,
            "maximum_decoded_state_tensors": 1,
            "source_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        },
        "runtime_contract": {
            "torch_version": str(torch.__version__),
            "model_modules": [source.name for source in model_sources],
            "model_sources_sha256": model_source_digests,
        },
        "verification": {
            "enabled": not args.no_verify,
            "minimum_argmax_match": args.min_argmax_match,
            "maximum_mean_abs_logit_error": args.max_mean_logit_error,
            "minimum_mean_cosine_similarity": args.min_cosine,
        },
    }
    output = pathlib.Path(args.output).resolve()
    size, digest = write_artifact(
        output, manifest, entries, max_bytes=max_bytes, force=args.force
    )
    # Full readback catches archive corruption, bad metadata, and reconstruction
    # mistakes before the artifact is handed off.
    inspect_artifact(output, verify=True, quiet=True)
    print(f"wrote {output}", flush=True)
    print(f"size={size:,} bytes ({size / 1_000_000:.3f} MB)", flush=True)
    print(f"sha256={digest}", flush=True)
    print(f"sidecar={output}{SIDECAR_SUFFIX}", flush=True)


def read_manifest(
    artifact: os.PathLike[str] | str,
    archive: zipfile.ZipFile | None = None,
) -> dict[str, Any]:
    close = archive is None
    if archive is None:
        archive = zipfile.ZipFile(artifact, "r")
    try:
        try:
            raw = archive.read(MANIFEST_ENTRY)
        except KeyError as exc:
            raise PackageError(f"{artifact}: missing {MANIFEST_ENTRY}") from exc
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise PackageError(f"{artifact}: manifest root is not a JSON object")
        if manifest.get("format") != FORMAT_NAME:
            raise PackageError(f"{artifact}: unsupported format {manifest.get('format')!r}")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise PackageError(
                f"{artifact}: unsupported format version "
                f"{manifest.get('format_version')!r}"
            )
        if manifest.get("byte_order") != sys.byteorder:
            raise PackageError(
                f"{artifact}: artifact byte order {manifest.get('byte_order')!r} "
                f"does not match runtime {sys.byteorder!r}"
            )
        return manifest
    finally:
        if close:
            archive.close()


def select_member(
    manifest: Mapping[str, Any],
    member: int | str,
) -> Mapping[str, Any]:
    """Resolve one member by stable archive index or manifest ID."""
    members = manifest["members"]
    if isinstance(member, int):
        try:
            return members[member]
        except IndexError as exc:
            raise KeyError(f"member index {member} is out of range") from exc
    selected = next((item for item in members if item["id"] == member), None)
    if selected is None:
        raise KeyError(f"artifact has no member {member!r}")
    return selected


def _load_member_state_dict_from_archive(
    archive: zipfile.ZipFile,
    selected: Mapping[str, Any],
    *,
    device: torch.device | str,
    verify_payloads: bool,
) -> OrderedDict[str, torch.Tensor]:
    """Dequantize one member while holding at most one packed tensor buffer."""
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    state_digest = hashlib.sha256()
    for tensor_meta in selected["tensors"]:
        payload = archive.read(tensor_meta["entry"])
        if verify_payloads:
            actual = sha256_bytes(payload)
            if actual != tensor_meta["sha256"]:
                raise PackageError(
                    f"{selected['id']}:{tensor_meta['name']} payload SHA256 mismatch"
                )
            update_member_state_digest(
                state_digest, tensor_meta["name"], payload
            )
        state[tensor_meta["name"]] = tensor_from_payload(
            payload, tensor_meta
        ).to(device)
        # ``payload`` is replaced on the next iteration.  In particular, this
        # loader never retains the full int8 package (or even a whole member's
        # packed payload) in memory.
    if verify_payloads and state_digest.hexdigest() != selected["state_sha256"]:
        raise PackageError(f"{selected['id']}: state SHA256 mismatch")
    return state


def load_member_state_dict(
    artifact: os.PathLike[str] | str,
    member: int | str,
    *,
    device: torch.device | str = "cpu",
    verify_payloads: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    with zipfile.ZipFile(artifact, "r") as archive:
        manifest = read_manifest(artifact, archive)
        selected = select_member(manifest, member)
        return _load_member_state_dict_from_archive(
            archive,
            selected,
            device=device,
            verify_payloads=verify_payloads,
        )


def member_memory_profile(
    member: Mapping[str, Any],
    model: torch.nn.Module,
) -> dict[str, int]:
    """Return auditable logical byte counts for one reconstructed member."""
    named_parameters = dict(model.named_parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in named_parameters.values()
    )
    fp32_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in named_parameters.values()
        if parameter.dtype == torch.float32
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )
    packed_tensor_bytes = sum(
        int(tensor["nbytes"]) for tensor in member["tensors"]
    )
    largest_payload_bytes = max(
        (int(tensor["nbytes"]) for tensor in member["tensors"]),
        default=0,
    )
    restored_state_bytes = sum(
        math.prod(int(axis) for axis in tensor["shape"])
        * torch.empty((), dtype=resolve_dtype(str(tensor["dtype"]))).element_size()
        for tensor in member["tensors"]
    )
    largest_decoded_fp32_parameter_bytes = max(
        (
            math.prod(int(axis) for axis in tensor["shape"])
            * torch.empty(
                (), dtype=resolve_dtype(str(tensor["dtype"]))
            ).element_size()
            for tensor in member["tensors"]
            if tensor["name"] in named_parameters
            and resolve_dtype(str(tensor["dtype"])) == torch.float32
        ),
        default=0,
    )
    return {
        "parameter_bytes": int(parameter_bytes),
        "fp32_parameter_bytes": int(fp32_parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "model_bytes": int(parameter_bytes + buffer_bytes),
        "restored_state_bytes": int(restored_state_bytes),
        "packed_tensor_bytes": int(packed_tensor_bytes),
        "largest_payload_bytes": int(largest_payload_bytes),
        "largest_decoded_fp32_parameter_bytes": int(
            largest_decoded_fp32_parameter_bytes
        ),
        # The direct streamed loader below holds one initialized model plus at
        # most one decoded parameter.  This is a conservative logical bound on
        # live fp32 parameter/state bytes (allocator overhead and activations
        # are intentionally outside the model-size accounting).
        "stream_peak_live_fp32_parameter_bytes": int(
            fp32_parameter_bytes + largest_decoded_fp32_parameter_bytes
        ),
    }


def package_storage_profile(
    artifact: os.PathLike[str] | str,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Describe immutable archive storage without reconstructing a model."""
    if manifest is None:
        manifest = read_manifest(artifact)
    member_packed = [
        sum(int(tensor["nbytes"]) for tensor in member["tensors"])
        for member in manifest["members"]
    ]
    return {
        "package_bytes": int(os.path.getsize(artifact)),
        "package_tensor_payload_bytes": int(sum(member_packed)),
        "largest_member_packed_bytes": int(max(member_packed, default=0)),
        "largest_tensor_payload_bytes": int(
            max(
                (
                    int(tensor["nbytes"])
                    for member in manifest["members"]
                    for tensor in member["tensors"]
                ),
                default=0,
            )
        ),
        "member_count": int(len(member_packed)),
    }


def _load_member_model_from_archive(
    archive: zipfile.ZipFile,
    selected: Mapping[str, Any],
    *,
    device: torch.device | str,
    verify_payloads: bool,
) -> tuple[torch.nn.Module, dict[str, int]]:
    """Build one model and copy one decoded archive tensor at a time.

    Unlike ``load_member_state_dict``, this deployment path never owns a full
    reconstructed state dict alongside the initialized model.  It preserves
    strict state-schema checks and the canonical payload digest while bounding
    decoded state to one tensor.
    """
    model = build_model(selected["model"])
    target_state = model.state_dict()
    tensor_meta = list(selected["tensors"])
    manifest_names = [str(item["name"]) for item in tensor_meta]
    if len(manifest_names) != len(set(manifest_names)):
        raise PackageError(f"{selected['id']}: duplicate tensor name in manifest")
    missing = set(target_state) - set(manifest_names)
    unexpected = set(manifest_names) - set(target_state)
    if missing or unexpected:
        raise PackageError(
            f"{selected['id']}: state schema mismatch; "
            f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )

    state_digest = hashlib.sha256()
    with torch.no_grad():
        for item in tensor_meta:
            payload = archive.read(item["entry"])
            if verify_payloads:
                actual = sha256_bytes(payload)
                if actual != item["sha256"]:
                    raise PackageError(
                        f"{selected['id']}:{item['name']} payload SHA256 mismatch"
                    )
                update_member_state_digest(state_digest, item["name"], payload)
            restored = tensor_from_payload(payload, item)
            destination = target_state[item["name"]]
            if destination.shape != restored.shape:
                raise PackageError(
                    f"{selected['id']}:{item['name']} shape mismatch: "
                    f"model={tuple(destination.shape)}, "
                    f"artifact={tuple(restored.shape)}"
                )
            # This matches load_state_dict's copy semantics, including an
            # explicit dtype/device conversion if a future model requires it.
            destination.copy_(restored)
            del restored, payload

    if verify_payloads and state_digest.hexdigest() != selected["state_sha256"]:
        raise PackageError(f"{selected['id']}: state SHA256 mismatch")
    del target_state
    model.to(device).eval()
    return model, member_memory_profile(selected, model)


def iter_ensemble_members(
    artifact: os.PathLike[str] | str,
    *,
    device: torch.device | str = "cpu",
    verify_payloads: bool = True,
) -> Iterator[
    tuple[Mapping[str, Any], torch.nn.Module, dict[str, int]]
]:
    """Yield one reconstructed model at a time from the existing ZIP artifact.

    The archive is already a lazy per-tensor format: this path does not call
    ``torch.load``, never materializes the complete int8 payload, and copies
    only one decoded state tensor into one initialized model at a time.  The
    caller must not retain yielded models if bounded memory is required.
    """
    with zipfile.ZipFile(artifact, "r") as archive:
        manifest = read_manifest(artifact, archive)
        for member in manifest["members"]:
            model, memory = _load_member_model_from_archive(
                archive,
                member,
                device=device,
                verify_payloads=verify_payloads,
            )
            try:
                yield member, model, memory
            finally:
                # The generator drops its reference before constructing the
                # next model.  Streamed callers should also drop theirs at the
                # end of each loop body.
                del model


def load_ensemble(
    artifact: os.PathLike[str] | str,
    *,
    device: torch.device | str = "cpu",
    verify_payloads: bool = True,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], torch.nn.Module]]]:
    """Eager compatibility API; reconstruct and retain every packaged model."""
    manifest = read_manifest(artifact)
    loaded = [
        (dict(member), model)
        for member, model, _ in iter_ensemble_members(
            artifact,
            device=device,
            verify_payloads=verify_payloads,
        )
    ]
    return manifest, loaded


def verify_sidecar(path: pathlib.Path) -> tuple[bool | None, str]:
    sidecar = pathlib.Path(str(path) + SIDECAR_SUFFIX)
    digest = sha256_file(path)
    if not sidecar.exists():
        return None, digest
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    if not parts:
        raise PackageError(f"{sidecar}: empty SHA256 sidecar")
    expected = parts[0].lower()
    if expected != digest:
        raise PackageError(
            f"{path}: artifact SHA256 mismatch; expected {expected}, got {digest}"
        )
    return True, digest


def inspect_artifact(
    path: pathlib.Path,
    *,
    verify: bool,
    quiet: bool = False,
    dump_json: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    sidecar_ok, digest = verify_sidecar(path)
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip() if verify else None
        if bad:
            raise PackageError(f"{path}: CRC failure in ZIP entry {bad}")
        manifest = read_manifest(path, archive)
        max_bytes = int(manifest["rules"]["maximum_artifact_bytes"])
        ceiling = int(manifest["rules"]["competition_ceiling_bytes"])
        if size > max_bytes or size > ceiling:
            raise PackageError(
                f"{path}: {size:,} bytes exceeds manifest/rules size limit"
            )
        member_ids: set[str] = set()
        archive_name_list = archive.namelist()
        archive_names = set(archive_name_list)
        if len(archive_names) != len(archive_name_list):
            raise PackageError("artifact contains duplicate ZIP entry names")
        referenced_entries = {MANIFEST_ENTRY}
        weight_sum = 0.0
        if verify:
            for member in manifest["members"]:
                if member["id"] in member_ids:
                    raise PackageError(f"duplicate member id {member['id']!r}")
                member_ids.add(member["id"])
                weight_sum += float(member["weight"])
                state_payloads: list[tuple[str, bytes]] = []
                tensor_names: set[str] = set()
                for tensor_meta in member["tensors"]:
                    entry = tensor_meta["entry"]
                    if entry not in archive_names:
                        raise PackageError(f"missing tensor entry {entry}")
                    if entry in referenced_entries:
                        raise PackageError(f"tensor ZIP entry {entry!r} is referenced twice")
                    if tensor_meta["name"] in tensor_names:
                        raise PackageError(
                            f"{member['id']}: duplicate tensor name {tensor_meta['name']!r}"
                        )
                    tensor_names.add(tensor_meta["name"])
                    referenced_entries.add(entry)
                    payload = archive.read(entry)
                    if sha256_bytes(payload) != tensor_meta["sha256"]:
                        raise PackageError(
                            f"{member['id']}:{tensor_meta['name']} SHA256 mismatch"
                        )
                    # Exercise reconstruction and validate payload byte counts.
                    tensor_from_payload(payload, tensor_meta)
                    state_payloads.append((tensor_meta["name"], payload))
                if member_state_hash(state_payloads) != member["state_sha256"]:
                    raise PackageError(f"{member['id']}: state SHA256 mismatch")
            if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
                raise PackageError(f"artifact weights sum to {weight_sum}, not 1.0")
            extra_entries = archive_names - referenced_entries
            if extra_entries:
                raise PackageError(
                    f"artifact contains {len(extra_entries)} unreferenced ZIP entries; "
                    f"first={sorted(extra_entries)[0]}"
                )
    if not quiet:
        if dump_json:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            print(f"artifact={path}")
            print(f"format={manifest['format']} v{manifest['format_version']}")
            print(f"ensemble={manifest['ensemble']['name']}")
            print(f"members={len(manifest['members'])}")
            print(f"size={size:,} bytes ({size / 1_000_000:.3f} MB)")
            print(f"sha256={digest}")
            print(
                "sidecar="
                + ("verified" if sidecar_ok else "absent (artifact hash shown above)")
            )
            print(f"payload_verification={'passed' if verify else 'not requested'}")
    return manifest


def command_inspect(args: argparse.Namespace) -> None:
    inspect_artifact(
        pathlib.Path(args.artifact).resolve(),
        verify=args.verify,
        dump_json=args.json,
    )


def command_verify_models(args: argparse.Namespace) -> None:
    path = pathlib.Path(args.artifact).resolve()
    manifest, models = load_ensemble(path, device="cpu", verify_payloads=True)
    print(
        f"strictly reconstructed {len(models)} models from "
        f"{manifest['ensemble']['name']}",
        flush=True,
    )
    for index, (member, model) in enumerate(models):
        input_spec = copy.deepcopy(
            member.get(
                "verification_input",
                inferred_input_spec(member["model"], args.batch_size),
            )
        )
        input_spec["shape"][0] = args.batch_size
        x = synthetic_input(input_spec, args.seed + index)
        with torch.inference_mode():
            logits = model(x).float()
        if logits.shape != (args.batch_size, 40) or not torch.isfinite(logits).all():
            raise PackageError(
                f"{member['id']}: invalid reconstructed logits {tuple(logits.shape)}"
            )
        print(
            f"[{index + 1:02d}/{len(models):02d}] {member['id']}: "
            f"logits={tuple(logits.shape)} finite=yes",
            flush=True,
        )
def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package CUHK-X fold ensembles as verified symmetric-int8 artifacts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="quantize, verify, and package selected checkpoints")
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--preset",
        choices=("block8", "block8_g40", "astgcn_a20", "astgcn_world25"),
    )
    source.add_argument("--spec", help="JSON ensemble specification")
    build.add_argument("--output", required=True, help="artifact path (e.g. model.pth)")
    build.add_argument(
        "--root",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="base directory for relative checkpoint paths",
    )
    build.add_argument(
        "--max-mb",
        type=float,
        default=100.0,
        help="strict decimal-MB output limit; cannot exceed competition ceiling 100",
    )
    build.add_argument("--verify-batch", type=int, default=DEFAULT_VERIFY_BATCH)
    build.add_argument("--verify-seed", type=int, default=20260730)
    build.add_argument("--min-argmax-match", type=float, default=0.75)
    build.add_argument("--max-mean-logit-error", type=float, default=0.05)
    build.add_argument("--min-cosine", type=float, default=0.999)
    build.add_argument(
        "--no-verify",
        action="store_true",
        help="skip original-vs-int8 forward comparison (strict state loading remains)",
    )
    build.add_argument("--force", action="store_true", help="replace output and sidecar")
    build.set_defaults(func=build_artifact)

    inspect = sub.add_parser("inspect", help="inspect and optionally hash-check an artifact")
    inspect.add_argument("artifact")
    inspect.add_argument("--verify", action="store_true", help="verify every tensor payload")
    inspect.add_argument("--json", action="store_true", help="print the complete manifest")
    inspect.set_defaults(func=command_inspect)

    verify_models = sub.add_parser(
        "verify-models", help="strictly reconstruct every model and run a smoke forward"
    )
    verify_models.add_argument("artifact")
    verify_models.add_argument("--batch-size", type=int, default=2)
    verify_models.add_argument("--seed", type=int, default=20260730)
    verify_models.set_defaults(func=command_verify_models)

    example = sub.add_parser("example-spec", help="print a valid custom JSON spec")
    example.set_defaults(
        func=lambda _args: print(json.dumps(example_spec(), indent=2, sort_keys=True))
    )
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.command == "build":
        if args.verify_batch <= 0:
            raise PackageError("--verify-batch must be positive")
        if not (0.0 <= args.min_argmax_match <= 1.0):
            raise PackageError("--min-argmax-match must be in [0, 1]")
        if args.max_mean_logit_error < 0.0:
            raise PackageError("--max-mean-logit-error must be nonnegative")
        if not (-1.0 <= args.min_cosine <= 1.0):
            raise PackageError("--min-cosine must be in [-1, 1]")
    elif args.command == "verify-models" and args.batch_size <= 0:
        raise PackageError("--batch-size must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_cli_args(args)
        args.func(args)
        return 0
    except (PackageError, FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
