"""
STEP 4 - Turn raw netCDF files into a clean, model-ready dataset.

For every satellite file we:
  1. open it and pull out the 11-micron infrared channel (variable IRWIN),
  2. centre-crop it and resize to 128 x 128,
  3. save it as a small .npy file,
  4. find the matching row in the track table (same storm, same time) so we
     know the true wind speed at that moment.

The output is ONE csv (the "manifest") where each line is:
    image path, storm id, time, wind speed, and the tabular features.
Everything after this step reads only that manifest. This is the single most
important idea in the project: do the slow, messy work once.

Run:  python step4_build_dataset.py
Makes: data/processed/frames/*.npy, data/processed/manifest.csv
"""

import re
import numpy as np
import pandas as pd
from skimage.transform import resize

import config

try:
    import xarray as xr
except ImportError:
    raise SystemExit("Please run:  pip install xarray netCDF4")

# Real HURSAT-B1 v06 filenames look like:
#   2007151N14072.GONU.2007.06.03.0600.23.FY2-C.024.hursat-b1.v06.nc
#   ^ IBTrACS SID  ^name ^ timestamp
FNAME_TIME = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})(\d{2})")
FNAME_SID = re.compile(r"^(\d{4}\d{3}[NS]\d{5})\.")

# Variables inside the file that are DERIVED FROM THE ANSWER. Never use these
# as model inputs: WindSpd and CentPrs are the intensity copied from IBTrACS,
# so feeding them in would be training a model to read its own answer sheet.
LEAKY_VARS = ("WindSpd", "CentPrs")


def read_ir(path):
    """Return (ir, timestamp, sid, name, vza) or None if the file is unusable.

    vza = view zenith angle: how far off-nadir the satellite was. A storm on
    the edge of a satellite's disc is smeared and foreshortened, so when two
    satellites saw the same storm at the same minute we keep the lower vza.
    """
    try:
        ds = xr.open_dataset(path, decode_timedelta=False)
    except Exception:
        return None

    with ds:
        # The IR window channel. Print ds.data_vars once if the name differs.
        var = next((v for v in ("IRWIN", "irwin", "IRWIN_2", "ch1")
                    if v in ds.variables), None)
        if var is None:
            return None
        arr = np.asarray(ds[var].values, dtype=np.float32).squeeze()
        if arr.ndim != 2:
            return None

        # time: prefer what is inside the file, fall back to the filename
        ts = None
        if "time" in ds.variables:
            try:
                ts = pd.to_datetime(np.asarray(ds["time"].values).ravel()[0])
            except Exception:
                ts = None
        if ts is None or pd.isna(ts):
            m = FNAME_TIME.search(path.name)
            if not m:
                return None
            ts = pd.Timestamp(*(int(x) for x in m.groups()))

        # HURSAT stores the IBTrACS storm ID in its attributes. This is an
        # exact join key - far safer than matching on storm names, which
        # repeat across seasons and are blank for unnamed systems.
        sid = str(ds.attrs.get("TC_serial_number", "")).strip()
        name = str(ds.attrs.get("TC_name",
                   ds.attrs.get("storm_name", ""))).upper().strip()
        if not sid:                      # fall back to the filename
            m_sid = FNAME_SID.match(path.name)
            sid = m_sid.group(1) if m_sid else ""
        if not name:
            parts = path.name.split(".")
            name = next((p.upper() for p in parts if p.isalpha() and len(p) > 3), "")

        # VZA is stored as a scalar variable; the filename also carries it.
        try:
            vza = float(np.asarray(ds["VZA"].values).ravel()[0])
        except Exception:
            m_v = re.search(r"\.(\d{3})\.hursat", path.name)
            vza = float(m_v.group(1)) if m_v else 99.0

    return arr, pd.Timestamp(ts), sid, name, vza


def preprocess(arr: np.ndarray) -> np.ndarray | None:
    """Centre-crop, resize to 128x128, scale kelvin -> 0..1."""
    h, w = arr.shape
    if min(h, w) < config.CROP:
        return None
    top, left = (h - config.CROP) // 2, (w - config.CROP) // 2
    crop = arr[top:top + config.CROP, left:left + config.CROP]

    # missing pixels -> warmest value (looks like "no cloud", a safe default)
    crop = np.where(np.isfinite(crop), crop, config.IR_MAX)
    if np.isfinite(crop).mean() < 0.5:
        return None

    small = resize(crop, (config.IMG_SIZE, config.IMG_SIZE),
                   order=1, anti_aliasing=True, preserve_range=True)
    scaled = (small - config.IR_MIN) / (config.IR_MAX - config.IR_MIN)
    return np.clip(scaled, 0, 1).astype(np.float32)


def main():
    tracks = pd.read_csv(config.TRACKS_CLEAN, parse_dates=["ISO_TIME"])
    all_files = sorted(config.HURSAT_DIR.rglob("*.nc"))

    # Cheap pre-filter: the SID is in the filename, so we can reject
    # other-basin storms without opening them. Opening a netCDF costs
    # milliseconds; doing it 35,000 times unnecessarily costs half an hour.
    # Reject on SID *and* name. HURSAT was built against IBTrACS v03r08, so a
    # few storms were renumbered in v04r01 - rejecting on SID alone silently
    # drops them. The name is field 2 of the filename, so we compare it
    # exactly (no substring matching, which is what caused MALA/MALAKAS).
    ni_sids = set(tracks["SID"].unique())
    ni_names = {n for n in tracks["NAME"].dropna().unique() if n != "UNNAMED"}

    files, rejected = [], 0
    for f in all_files:
        m = FNAME_SID.match(f.name)
        parts = f.name.split(".")
        fname_name = parts[1].upper() if len(parts) > 1 else ""
        known_sid = bool(m) and m.group(1) in ni_sids
        known_name = fname_name in ni_names
        if m and not known_sid and not known_name:
            rejected += 1          # confidently not ours
        else:
            files.append(f)        # ours, renumbered, or unparseable - open it
    print(f"{len(all_files)} netCDF files on disk")
    print(f"  {rejected} skipped by filename (other basins)")
    print(f"  {len(files)} to read, against {len(tracks)} track rows")
    if not files:
        raise SystemExit("No .nc files. Run step3_get_hursat.py first.")

    records, best_vza = {}, {}
    skipped, n_by_name, duplicates = 0, 0, 0
    for i, f in enumerate(files, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(files)} files read ... {len(records)} unique timestamps")

        out = read_ir(f)
        if out is None:
            skipped += 1
            continue
        arr, ts, sid, name, vza = out

        img = preprocess(arr)
        if img is None:
            skipped += 1
            continue

        # Match to the track table: prefer the exact SID, fall back to the
        # name. (HURSAT was built with IBTrACS v03r08, so a handful of SIDs
        # were renumbered in v04r01 - the name fallback catches those.)
        cand = tracks[tracks["SID"] == sid] if sid else tracks.iloc[0:0]
        if cand.empty and name:
            cand = tracks[(tracks["NAME"] == name) & (tracks["SEASON"] == ts.year)]
            if not cand.empty:
                n_by_name += 1
        if cand.empty:
            skipped += 1
            continue

        dt = (cand["ISO_TIME"] - ts).abs()
        j = dt.idxmin()
        if dt.loc[j] > pd.Timedelta(minutes=90):
            skipped += 1
            continue

        rec = tracks.loc[j]
        key = f"{rec['SID']}_{rec['ISO_TIME']:%Y%m%d%H%M}.npy"

        # Several satellites can see one storm at one moment. Keep the best
        # view only, otherwise the same instant appears many times and the
        # model sees near-identical duplicates.
        if key in best_vza:
            duplicates += 1
            if vza >= best_vza[key]:
                continue
        best_vza[key] = vza

        npy = config.FRAME_DIR / key
        np.save(npy, img)
        records[key] = {
            "frame_path": npy.name,
            "SID": rec["SID"], "NAME": rec["NAME"], "SEASON": rec["SEASON"],
            "ISO_TIME": rec["ISO_TIME"], "USA_WIND": rec["USA_WIND"],
            "SAT_VZA": vza,
            **{c: rec[c] for c in config.TAB_COLS},
        }

    man = pd.DataFrame(list(records.values()))
    man = man.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    man.to_csv(config.MANIFEST, index=False)

    print(f"\nDone. {len(files)} files read -> {len(man)} unique timestamps.")
    print(f"  {duplicates} extra satellite views collapsed (best VZA kept)")
    print(f"  {skipped} files unusable/unmatched")
    print(f"  matched by SID: {len(man)-n_by_name}, by name fallback: {n_by_name}")
    print(f"Storms: {man['SID'].nunique()}   "
          f"Wind range: {man['USA_WIND'].min():.0f}-{man['USA_WIND'].max():.0f} kt")
    print(f"Saved {config.MANIFEST}")


if __name__ == "__main__":
    main()
