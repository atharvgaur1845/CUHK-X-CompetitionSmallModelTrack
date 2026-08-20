#!/usr/bin/env python3
"""Test inference for the EXP-094 MotionBERT skeleton member.

Trains with `cross_entropy(logits + log_prior)`, so the raw softmax here is already in
UNIFORM-prior space -- the space `fuse_general.py` expects on its `--visual` slot. Do not
re-add log_prior.

Row order comes from sample_submission.csv, never from a cache index.
"""
from __future__ import annotations
import argparse, csv, hashlib, os, sys
import numpy as np, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from train_skel_motionbert import SkelSet, build_actionnet, predict

ART = os.path.join(ROOT, "research", "artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
SAMPLE = os.path.join(ROOT, "Small-Model-Track", "Testing", "sample_submission.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="skel_mb_f2")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    device = torch.device("cuda")
    pkg = torch.load(os.path.join(CKPT, f"{args.tag}.pt"), map_location="cpu", weights_only=False)
    saved = pkg["args"]

    class A:  # rebuild exactly what the checkpoint was trained as
        dim_feat = int(saved.get("dim_feat", 512))
        dropout = float(saved.get("dropout", 0.5))
        checkpoint = ""          # weights come from our own state_dict, not the NTU file
    net = build_actionnet(A, 40)
    net.load_state_dict(pkg["state_dict"])
    net = net.to(device).eval()
    print(f"{args.tag}: frames={saved.get('n_frames')} depth_ch={saved.get('depth_channel')} "
          f"OOF micro={pkg.get('micro'):.5f} object={pkg.get('object'):.5f}")

    rows = list(csv.reader(open(SAMPLE)))
    header, body = rows[0], rows[1:]
    paths = [r[0] for r in body]
    sids = [q.strip("/").split("/")[-1] for q in paths]
    ds = SkelSet(sids, None, "test", int(saved.get("n_frames", 64)), False,
                 depth_channel=bool(saved.get("depth_channel", False)))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, pin_memory=False)
    probs, _ = predict(net, loader, device)
    probs = probs.astype(np.float32)
    assert probs.shape == (len(sids), 40), probs.shape
    out = os.path.join(ART, f"testprobs_{args.tag}.npz")
    np.savez_compressed(out, probs=probs, sids=np.array(sids, dtype="<U15"))
    print(f"wrote {out}\n  sha256={hashlib.sha256(open(out,'rb').read()).hexdigest()}")
    print(f"  distinct argmax classes = {len(set(probs.argmax(1).tolist()))}/40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
