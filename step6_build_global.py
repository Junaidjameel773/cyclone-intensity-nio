"""
STEP 6 - Build the transfer pre-training set from OTHER basins.

Why: the North Indian Ocean gives you only ~110 severe training windows.
The Atlantic and Western Pacific produce Category 4-5 storms routinely, and
you already have ~35,000 of their frames on disk - they came down with the
original (over-broad) download. An eyewall looks like an eyewall regardless
of which ocean it is over, so a backbone trained on those frames learns the
visual signature of intensity from thousands of examples instead of dozens.

This reuses the exact same reading and preprocessing code as step 4, so the
pre-training images are identical in format to the fine-tuning images.

Run:  python step6_build_global.py            # downloads labels, then builds
      python step6_build_global.py --limit 4000   # quick trial run

Makes: data/raw/ibtracs.ALL.list.v04r01.csv
       data/processed/tracks_global_clean.csv
       data/processed/frames_global/*.npy
       data/processed/manifest_global.csv
"""

import argparse
import urllib.request

import numpy as np
import pandas as pd

import config
from step2_explore_tracks import USE_COLS, add_features
from step4_build_dataset import read_ir, preprocess, FNAME_SID


def get_labels() -> pd.DataFrame:
    if not config.IBTRACS_ALL_CSV.exists():
        print("Downloading global IBTrACS (large file, a few hundred MB)...")
        urllib.request.urlretrieve(config.IBTRACS_ALL_URL, config.IBTRACS_ALL_CSV)
        print(f"  saved {config.IBTRACS_ALL_CSV}")

    if config.GLOBAL_TRACKS.exists():
        return pd.read_csv(config.GLOBAL_TRACKS, parse_dates=["ISO_TIME"])

    df = pd.read_csv(config.IBTRACS_ALL_CSV, skiprows=[1], usecols=USE_COLS,
                     low_memory=False, na_values=[" ", "", "NOT_NAMED"])
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for c in ["LAT", "LON", "USA_WIND", "USA_PRES", "SEASON"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ISO_TIME", "LAT", "LON", "USA_WIND"])
    df = df[(df["USA_WIND"] > 0) & (df["ISO_TIME"].dt.hour % 3 == 0)]
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    df = add_features(df)
    df.to_csv(config.GLOBAL_TRACKS, index=False)
    print(f"Global labels: {len(df)} rows, {df['SID'].nunique()} storms")
    return df


def main(limit):
    tracks = get_labels()
    ni_sids = set(pd.read_csv(config.TRACKS_CLEAN)["SID"].unique())
    global_sids = set(tracks["SID"].unique())

    # Everything on disk that is NOT in the North Indian Ocean set. Excluding
    # NI storms is essential: if they leaked into pre-training, the model
    # would meet the test storms before fine-tuning and the evaluation would
    # be worthless.
    files = []
    for f in sorted(config.HURSAT_DIR.rglob("*.nc")):
        m = FNAME_SID.match(f.name)
        if not m:
            continue
        sid = m.group(1)
        if sid in ni_sids or sid not in global_sids:
            continue
        files.append((f, sid))
    if limit:
        files = files[:limit]
    print(f"{len(files)} other-basin files to read "
          f"({len(ni_sids)} NI storms held out)")

    records, best_vza, skipped, dup = {}, {}, 0, 0
    by_sid = {s: g for s, g in tracks.groupby("SID")}

    for i, (f, sid) in enumerate(files, 1):
        if i % 2000 == 0:
            print(f"  {i}/{len(files)} ... {len(records)} unique timestamps")

        out = read_ir(f)
        if out is None:
            skipped += 1
            continue
        arr, ts, file_sid, name, vza = out
        cand = by_sid.get(file_sid or sid)
        if cand is None:
            skipped += 1
            continue

        img = preprocess(arr)
        if img is None:
            skipped += 1
            continue

        dt = (cand["ISO_TIME"] - ts).abs()
        j = dt.idxmin()
        if dt.loc[j] > pd.Timedelta(minutes=90):
            skipped += 1
            continue

        rec = cand.loc[j]
        key = f"{rec['SID']}_{rec['ISO_TIME']:%Y%m%d%H%M}.npy"
        if key in best_vza:
            dup += 1
            if vza >= best_vza[key]:
                continue
        best_vza[key] = vza

        np.save(config.GLOBAL_FRAME_DIR / key, img)
        records[key] = {
            "frame_path": key, "SID": rec["SID"], "NAME": rec["NAME"],
            "SEASON": rec["SEASON"], "ISO_TIME": rec["ISO_TIME"],
            "USA_WIND": rec["USA_WIND"], "SAT_VZA": vza,
            **{c: rec[c] for c in config.TAB_COLS},
        }

    man = pd.DataFrame(list(records.values())).sort_values(["SID", "ISO_TIME"])
    man.to_csv(config.GLOBAL_MANIFEST, index=False)

    cat3 = (man["USA_WIND"] >= config.CAT3_KT).mean()
    print(f"\nPre-training set: {len(man)} frames, {man['SID'].nunique()} storms")
    print(f"  {dup} duplicate views collapsed, {skipped} unusable")
    print(f"  wind range {man['USA_WIND'].min():.0f}-{man['USA_WIND'].max():.0f} kt")
    print(f"  Cat 3+ share: {100*cat3:.1f}%   "
          f"(North Indian Ocean was ~4.5% - this is the whole point)")
    print(f"Saved {config.GLOBAL_MANIFEST}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    main(p.parse_args().limit)
