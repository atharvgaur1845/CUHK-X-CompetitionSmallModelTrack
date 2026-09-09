#!/usr/bin/env python3
"""True sub-byte bit-packing for the quantised video weights (EXP-144).

Why. `quantize_checkpoint.py --bits 6` produces codes in [-32, 31], and the package has
always stored each one in a full int8 container. That is lossless and simple, and it wastes
exactly 25% of the video branch: 8.6 MB per MViTv2-S view. EXP-129 recorded that saving as
"not implemented rather than implemented and unused", because at 91.36 MB with two views
nothing needed it. EXP-143's nested CV then put `person + wrist + thermal` first (2074 vs
2062 for the shipping pair), and that is the 129 MB configuration that scored 172 against
the shippable 170. Three views only fit if the 25% is real, so it is now load-bearing.

The invariant that matters. This must be EXACTLY invertible on the code array -- not
"close", not "within a tolerance". `unpack_stage2.py --check weights` compares dequantised
tensors against `quantize_checkpoint.py` at bit-level equality and a single flipped code
fails it. The self-test below is exhaustive over every representable value.
"""
from __future__ import annotations
import numpy as np


def pack_codes(code: np.ndarray, bits: int) -> bytes:
    """int8 codes in [-2**(bits-1), 2**(bits-1)-1] -> dense little-endian bitstream."""
    if bits == 8:
        return np.ascontiguousarray(code, dtype=np.int8).tobytes()
    if not 1 <= bits < 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    off = 1 << (bits - 1)
    u = (np.asarray(code, dtype=np.int16).reshape(-1) + off).astype(np.uint8)
    if u.max(initial=0) >= (1 << bits):
        raise ValueError(f"code out of range for {bits} bits")
    # big-endian bit order per value, low `bits` bits only, then repack densely
    b = np.unpackbits(u[:, None], axis=1)[:, 8 - bits:]
    return np.packbits(b.reshape(-1)).tobytes()


def unpack_codes(raw: bytes, n: int, bits: int) -> np.ndarray:
    """Inverse of pack_codes. `n` is the number of codes, which the manifest carries."""
    if bits == 8:
        return np.frombuffer(raw, dtype=np.int8)
    off = 1 << (bits - 1)
    b = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[: n * bits].reshape(n, bits)
    pad = np.zeros((n, 8 - bits), dtype=np.uint8)
    u = np.packbits(np.concatenate([pad, b], axis=1), axis=1).reshape(-1)
    return (u.astype(np.int16) - off).astype(np.int8)


def packed_nbytes(n: int, bits: int) -> int:
    return n if bits == 8 else (n * bits + 7) // 8


def _self_test() -> None:
    rng = np.random.default_rng(0)
    for bits in range(1, 9):
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        # exhaustive over every representable value, in every phase alignment
        full = np.tile(np.arange(lo, hi + 1, dtype=np.int8), 7)
        for n in (0, 1, 2, 3, 4, 5, 7, 8, 33, len(full)):
            a = full[:n] if n <= len(full) else full
            got = unpack_codes(pack_codes(a, bits), len(a), bits)
            assert np.array_equal(got, a), f"bits={bits} n={n} round-trip failed"
            assert len(pack_codes(a, bits)) == packed_nbytes(len(a), bits)
        for _ in range(20):
            a = rng.integers(lo, hi + 1, size=int(rng.integers(1, 5000))).astype(np.int8)
            assert np.array_equal(unpack_codes(pack_codes(a, bits), len(a), bits), a)
    n = 34_500_000
    print(f"self-test PASS (bits 1..8, exhaustive values, random lengths)")
    print(f"  a 34.5M-parameter view at 6 bits: "
          f"{n/1e6:.1f} MB int8 -> {packed_nbytes(n, 6)/1e6:.1f} MB packed "
          f"({100*(1-packed_nbytes(n,6)/n):.0f}% saved)")


if __name__ == "__main__":
    _self_test()
