"""
STEP 2 - Clean the track data and build the tabular features.

What this does, in plain English:
  1. Loads the raw IBTrACS CSV (skipping its weird units row).
  2. Keeps only the columns we need, and only storms in our year range.
  3. Throws away rows with no wind speed (we cannot learn from a missing answer).
  4. Adds a few extra numbers per row: month as sin/cos, and how fast /
     which way the storm is moving (computed from the previous position).
  5. Saves a tidy CSV + two sanity-check plots.

Run:  python step2_explore_tracks.py
Makes: data/processed/tracks_ni_clean.csv, figures/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

USE_COLS = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON",
            "USA_WIND", "USA_PRES", "NATURE"]


def load_raw() -> pd.DataFrame:
    # skiprows=[1] drops the units row ("degrees_north", "kts", ...)
    df = pd.read_csv(config.IBTRACS_CSV, skiprows=[1], usecols=USE_COLS,
                     low_memory=False, na_values=[" ", "", "NOT_NAMED"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for c in ["LAT", "LON", "USA_WIND", "USA_PRES", "SEASON"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["ISO_TIME", "LAT", "LON", "USA_WIND"])
    df = df[(df["SEASON"] >= config.START_YEAR) & (df["SEASON"] <= config.END_YEAR)]
    df = df[df["USA_WIND"] > 0]

    # IBTrACS reports every 3 h but some rows are interpolated 'spur' points.
    # Keeping only synoptic-ish times (multiples of 3 h) keeps things regular.
    df = df[df["ISO_TIME"].dt.hour % 3 == 0]
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    print(f"Rows: {n0} raw -> {len(df)} usable  ({df['SID'].nunique()} storms)")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row tabular features. All of these are knowable at prediction time."""
    month = df["ISO_TIME"].dt.month
    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

    g = df.groupby("SID")
    df["DLAT"] = g["LAT"].diff().fillna(0.0)      # movement since last fix
    df["DLON"] = g["LON"].diff().fillna(0.0)
    dt_h = g["ISO_TIME"].diff().dt.total_seconds().fillna(3 * 3600) / 3600
    # 1 degree ~ 60 nautical miles; translation speed in knots
    df["SPEED_KT"] = np.sqrt(df["DLAT"] ** 2 +
                             (df["DLON"] * np.cos(np.radians(df["LAT"]))) ** 2) * 60 / dt_h
    df["SPEED_KT"] = df["SPEED_KT"].clip(0, 40)
    df["NAME"] = df["NAME"].fillna("UNNAMED").str.upper().str.strip()
    return df


def plots(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(df["USA_WIND"], bins=30, color="#3b6ea5", edgecolor="white")
    ax[0].axvline(config.CAT3_KT, color="crimson", ls="--", label="Cat 3 (96 kt)")
    ax[0].set_xlabel("Max sustained wind (kt)"); ax[0].set_ylabel("count")
    ax[0].set_title("Label distribution - note how few strong storms there are")
    ax[0].legend()

    for _, s in df.groupby("SID"):
        ax[1].plot(s["LON"], s["LAT"], lw=0.6, alpha=0.5)
    ax[1].set_xlabel("Longitude"); ax[1].set_ylabel("Latitude")
    ax[1].set_title(f"North Indian Ocean tracks {config.START_YEAR}-{config.END_YEAR}")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "01_data_overview.png", dpi=130)
    print(f"Saved {config.FIG_DIR / '01_data_overview.png'}")


def main():
    df = add_features(clean(load_raw()))
    df.to_csv(config.TRACKS_CLEAN, index=False)
    print(f"Saved {config.TRACKS_CLEAN}")

    print("\nStorms per season:")
    print(df.groupby("SEASON")["SID"].nunique().to_string())

    print("\nStrongest storms in the file (these are your interesting test cases):")
    top = (df.groupby(["SID", "NAME", "SEASON"])["USA_WIND"].max()
             .sort_values(ascending=False).head(12))
    print(top.to_string())

    print(f"\nRows at Cat 3+ ({config.CAT3_KT} kt or more): "
          f"{(df['USA_WIND'] >= config.CAT3_KT).sum()} "
          f"({100 * (df['USA_WIND'] >= config.CAT3_KT).mean():.1f}% of data)")
    plots(df)


if __name__ == "__main__":
    main()
