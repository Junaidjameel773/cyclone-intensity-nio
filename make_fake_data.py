"""
Make FAKE data so you can run the whole training pipeline today,
before any real download has finished.

It writes the exact same files step 4 would write, just with made-up
storms: a swirl of cold cloud whose centre gets colder as the storm gets
stronger. If train.py works on this, your code is correct and the only
thing left is swapping in real data.

Run:  python make_fake_data.py --storms 40
Then: python train.py --epochs 3
"""

import argparse
import numpy as np
import pandas as pd
import config

RNG = np.random.default_rng(0)
NAMES = ["GONU", "PHAILIN", "HUDHUD", "CHAPALA", "NARGIS", "SIDR", "NILAM",
         "LAILA", "PHET", "MEGH", "VIYARU", "HELEN", "LEHAR", "MADI", "ASHOBAA"]


def fake_frame(wind: float) -> np.ndarray:
    """Stronger storm -> colder, tighter core, plus a warm eye above ~90 kt."""
    n = config.IMG_SIZE
    y, x = np.mgrid[0:n, 0:n] - n / 2
    r = np.sqrt(x ** 2 + y ** 2)

    tightness = 8 + 40 * np.exp(-wind / 45)          # strong storms are compact
    cloud = np.exp(-(r / tightness) ** 2)
    img = 1.0 - 0.85 * cloud                          # 0 = cold cloud, 1 = warm sea

    if wind > 90:                                     # punch an eye
        img += 0.6 * np.exp(-(r / (2 + 60 / wind)) ** 2)

    img += 0.05 * RNG.standard_normal((n, n))
    return np.clip(img, 0, 1).astype(np.float32)


def main(n_storms: int):
    rows = []
    for s in range(n_storms):
        sid = f"FAKE{s:04d}"
        name = NAMES[s % len(NAMES)]
        season = 2000 + s % 16
        t0 = pd.Timestamp(f"{season}-05-01") + pd.Timedelta(days=int(RNG.integers(0, 150)))
        steps = int(RNG.integers(16, 40))
        peak = float(RNG.uniform(35, 140))
        lat, lon = RNG.uniform(8, 18), RNG.uniform(62, 92)

        for k in range(steps):
            # a simple rise-and-fall life cycle
            frac = np.sin(np.pi * (k + 1) / (steps + 1)) ** 1.5
            wind = float(np.clip(25 + (peak - 25) * frac + RNG.normal(0, 3), 20, 160))
            ts = t0 + pd.Timedelta(hours=3 * k)
            dlat, dlon = RNG.normal(0.15, .1), RNG.normal(0.25, .1)
            lat, lon = lat + dlat, lon + dlon

            path = config.FRAME_DIR / f"{sid}_{ts:%Y%m%d%H%M}.npy"
            np.save(path, fake_frame(wind))
            rows.append({
                "frame_path": path.name, "SID": sid, "NAME": name, "SEASON": season,
                "ISO_TIME": ts, "USA_WIND": round(wind, 1),
                "LAT": lat, "LON": lon,
                "MONTH_SIN": np.sin(2 * np.pi * ts.month / 12),
                "MONTH_COS": np.cos(2 * np.pi * ts.month / 12),
                "DLAT": dlat, "DLON": dlon,
                "SPEED_KT": float(np.hypot(dlat, dlon) * 60 / 3),
            })

    pd.DataFrame(rows).to_csv(config.MANIFEST, index=False)
    print(f"Wrote {len(rows)} fake frames for {n_storms} storms -> {config.MANIFEST}")
    print("This is FAKE data. Delete data/processed/ before using real data.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--storms", type=int, default=40)
    main(p.parse_args().storms)
