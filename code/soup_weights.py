#!/usr/bin/env python3
"""EXP-151 -- WEIGHT SOUP: average N checkpoints into one, so a bag costs one model's bytes.

The specific debt this is trying to pay. `sub_r2th20` scored **172** using a 4-fold thermal
BAG -- four models, ~104 MB of thermal alone, which cannot ship. The all-train thermal model
scores **171** (`sub_pkgship3`) and **170** once int5 pays for the detectors
(`sub_pkgship4`). So 2 public clips are sitting in a configuration that is not
reproducible, and Stage 2 verifies reproduction from the <=100 MB checkpoint.

A soup is the standard way to collect a bag's benefit into a single set of weights: models
fine-tuned from the SAME initialisation stay in one loss basin, so averaging them is a
valid point rather than nonsense (Wortsman et al., "Model Soups"). All our fold models
start from the same Kinetics-400 MViTv2-S, which is the precondition.

HONESTY ABOUT VALIDATION. A soup of the four fold models has collectively seen every
training clip, so there is no held-out set left to score it on -- the only models that never
saw subject u are fold_f(u) and loso_u. What CAN be measured without labels is agreement
with the bag's own test predictions, and the bag's test score is known (172). High
agreement means the soup inherits the behaviour that scored 172; it is a proxy, and the
confirmation is one submission. The seed-soup array (EXP-148) is the version that IS
fold-safe and measurable.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="checkpoint tags")
    ap.add_argument("--out", required=True, help="output tag")
    a = ap.parse_args()

    sds, ref = [], None
    for t in a.inputs:
        p = ROOT / "checkpoints" / f"{t}.pt"
        pkg = torch.load(p, map_location="cpu", weights_only=False)
        sd = pkg["state_dict"]
        if ref is None:
            ref, meta = sd, pkg
        assert set(sd) == set(ref), f"{t}: key mismatch"
        for k in sd:
            assert sd[k].shape == ref[k].shape, f"{t}/{k}: shape mismatch"
        sds.append(sd)
        print(f"  + {t}")

    out = {}
    for k in ref:
        v0 = ref[k]
        if v0.is_floating_point():
            out[k] = torch.stack([sd[k].float() for sd in sds]).mean(0).to(v0.dtype)
        else:
            # integer buffers (e.g. num_batches_tracked) are not averaged; take the first
            # and assert the others agree, so a silent semantic difference cannot hide
            for sd in sds[1:]:
                if not torch.equal(sd[k], v0):
                    print(f"    note: non-float buffer {k} differs across inputs; keeping first")
                    break
            out[k] = v0.clone()
    meta = dict(meta); meta["state_dict"] = out
    meta["soup_inputs"] = list(a.inputs)
    dest = ROOT / "checkpoints" / f"{a.out}.pt"
    torch.save(meta, dest)
    print(f"wrote {dest}  ({dest.stat().st_size/1e6:.1f} MB)  from {len(sds)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
