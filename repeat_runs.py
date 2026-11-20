"""
How big is the noise?

Two runs of an identical configuration can differ by several knots, because
weight init, the balanced sampler and the augmentation are all random, and
because early stopping picks a checkpoint off a noisy validation curve. If
you do not know that spread, you cannot tell a real improvement from luck -
and you will waste days chasing differences that are not there.

This runs the SAME configuration N times with different training seeds (the
data split is held fixed) and reports mean, std and range.

Run:  python repeat_runs.py --n 5
      python repeat_runs.py --n 5 --extra --no-images        # any train.py flag
      python repeat_runs.py --n 5 --label crop101

Rule of thumb afterwards: only believe a difference bigger than about
2 x std. Anything smaller is noise.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIELDS = ["val_rmse", "test_rmse", "test_mae", "cat3_rmse", "cat3_bias"]


def one_run(seed: int, extra: list[str]) -> dict:
    cmd = [sys.executable, "train.py", "--seed", str(seed)] + extra
    print(f"  seed {seed}: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit("training failed")
    line = re.search(r"^RESULT .*$", r.stdout, re.M)
    if not line:
        raise SystemExit("no RESULT line - is train.py up to date?")
    vals = dict(kv.split("=") for kv in line.group(0).split()[1:])
    out = {k: float(vals[k]) for k in FIELDS}
    print("     " + "  ".join(f"{k}={v:.2f}" for k, v in out.items()))
    return out


def main(n, extra, label, epochs):
    extra = list(extra) + ["--epochs", str(epochs)]
    rows = [one_run(1000 + i, extra) for i in range(n)]
    df = pd.DataFrame(rows)

    print(f"\n{'='*66}\n{n} runs, identical config: {' '.join(extra)}\n{'='*66}")
    print(f"{'metric':<12}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'spread':>9}")
    for c in FIELDS:
        v = df[c]
        print(f"{c:<12}{v.mean():9.2f}{v.std():9.2f}{v.min():9.2f}"
              f"{v.max():9.2f}{v.max()-v.min():9.2f}")

    s = df["test_rmse"].std()
    print(f"\nTest RMSE std is {s:.2f} kt, so only trust differences larger "
          f"than about {2*s:.1f} kt.")

    out = ROOT / f"repeats_{label}.csv" if label else ROOT / "repeats.csv"
    df.insert(0, "seed", [1000 + i for i in range(n)])
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--label", default="")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is passed to train.py")
    a = p.parse_args()
    main(a.n, a.extra, a.label, a.epochs)
