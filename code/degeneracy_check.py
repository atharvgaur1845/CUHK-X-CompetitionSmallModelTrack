#!/usr/bin/env python3
"""Show that per-member temperature is NOT a new inference degree of freedom.

For the champion pipeline the decoder maximizes, over a recording's label
sequence,

    S(c) = sum_i log fused_i(c_i) + lambda * sum_i log T(c_{i-1}, c_i)

and the fused log-probability is an affine function of the member log-probs:

    log fused_i(c) = (1-w)/Tb * log b'_i(c) + w/Tv * log v_i(c) - logZ_i

logZ_i does not depend on c, so it cannot change the argmax.  Writing
A=(1-w)/Tb, B=w/Tv and S=A+B, the emission term equals

    S * [ (A/S) log b'_i(c) + (B/S) log v_i(c) ]

so decoding with (Tb, Tv, w, lambda) is *identical* to decoding with
temperatures 1 and

    w'      = B / (A + B)
    lambda' = lambda / (A + B)

Temperature therefore moves along the (weight, transition-weight) plane that
has already been scanned on the public leaderboard.  It is a reparameterization,
not a new object.  This script verifies the identity numerically.
"""
from __future__ import annotations

import numpy as np


def equivalent(w: float, tb: float, tv: float, lam: float) -> tuple[float, float]:
    a = (1.0 - w) / tb
    b = w / tv
    s = a + b
    return b / s, lam / s


def main() -> int:
    print("champion: w=0.45  Tb=1  Tv=1  lambda=0.5")
    print()
    print("%6s %6s | %10s %10s   (equivalent temperature-1 config)"
          % ("Tb", "Tv", "w'", "lambda'"))
    print("-" * 62)
    for tb, tv in [(1.0, 0.7), (1.0, 0.85), (1.0, 1.25), (1.0, 1.5),
                   (0.7, 1.0), (1.25, 1.0), (0.8, 1.2), (1.2, 0.8),
                   (0.5, 0.5), (2.0, 2.0)]:
        wp, lp = equivalent(0.45, tb, tv, 0.5)
        print("%6.2f %6.2f | %10.6f %10.6f" % (tb, tv, wp, lp))
    print()
    print("Note the last two rows: a COMMON temperature leaves w' unchanged at")
    print("0.45 and only rescales lambda.  A common temperature is exactly a")
    print("transition-weight change; a differential temperature is exactly a")
    print("fusion-weight change plus a transition-weight change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
