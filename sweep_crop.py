"""
Controlled crop-size experiment.

The lesson from the last run: if you change the crop AND the dataset at the
same time, the comparison is meaningless. This script changes exactly one
thing at a time and keeps each result in its own folder, so the three runs
are genuinely comparable.

Run:  python sweep_crop.py
      python sweep_crop.py --crops 201 101 65 --epochs 60

It takes a while - each crop needs a full rebuild plus a training run.
Results land in sweep_results.csv.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import config

ROOT = Path(__file__).resolve().parent


def set_crop(value: int):
    """Rewrite the CROP line in config.py."""
    txt = (ROOT / "config.py").read_text()
    txt = re.sub(r"^CROP = \d+", f"CROP = {value}", txt, flags=re.M)
    (ROOT / "config.py").write_text(txt)


def run(cmd: list[str]) -> str:
    print(f"    $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return r.stdout


def parse_metrics() -> dict:
    m = pd.read_csv(config.FIG_DIR / "metrics.csv").set_index("band")
    out = {}
    for band, tag in [("ALL", "all"), ("Cat 3+     >=96 kt", "cat3")]:
        if band in m.index:
            out[f"{tag}_rmse"] = m.at[band, "RMSE_kt"]
            out[f"{tag}_mae"] = m.at[band, "MAE_kt"]
            out[f"{tag}_bias"] = m.at[band, "bias_kt"]
    return out


def main(crops, epochs, balance):
    rows = []
    original = config.CROP
    try:
        for crop in crops:
            print(f"\n{'='*60}\nCROP = {crop}  ({crop*0.07*111:.0f} km, "
                  f"{crop*0.07*111/config.IMG_SIZE:.1f} km/px)\n{'='*60}")
            set_crop(crop)

            shutil.rmtree(config.FRAME_DIR, ignore_errors=True)
            config.FRAME_DIR.mkdir(parents=True, exist_ok=True)
            config.MANIFEST.unlink(missing_ok=True)

            build = run([sys.executable, "step4_build_dataset.py"])
            n_frames = int(re.search(r"-> (\d+) unique", build).group(1))
            n_storms = int(re.search(r"Storms: (\d+)", build).group(1))

            train_out = run([sys.executable, "train.py", "--epochs", str(epochs),
                             "--balance", str(balance), "--workers", "2"])
            naive = float(re.search(r"naive baseline: *([\d.]+)", train_out).group(1))
            test_rmse = float(re.search(r"TEST RMSE *: *([\d.]+)", train_out).group(1))

            run([sys.executable, "evaluate.py"])
            row = {"crop": crop, "km_across": round(crop * 0.07 * 111),
                   "km_per_px": round(crop * 0.07 * 111 / config.IMG_SIZE, 1),
                   "storms": n_storms, "frames": n_frames,
                   "naive_rmse": naive, "test_rmse": test_rmse, **parse_metrics()}
            rows.append(row)
            print("\n  -> " + "  ".join(f"{k}={v}" for k, v in row.items()))

            # keep this crop's figures so they are not overwritten
            dest = config.FIG_DIR / f"crop_{crop}"
            dest.mkdir(exist_ok=True)
            for png in config.FIG_DIR.glob("*.png"):
                shutil.copy(png, dest / png.name)
    finally:
        set_crop(original)
        print(f"\nconfig.py restored to CROP = {original}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "sweep_results.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved {ROOT / 'sweep_results.csv'}")
    print("\nSanity check: 'storms' and 'frames' MUST be identical across rows. "
          "If they are not, the crops were run on different data and the "
          "comparison is invalid.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crops", type=int, nargs="+", default=[201, 145, 101, 65])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--balance", type=float, default=0.5)
    a = p.parse_args()
    main(a.crops, a.epochs, a.balance)
