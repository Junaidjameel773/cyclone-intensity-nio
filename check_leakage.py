"""
Did any North Indian Ocean storm leak into the pre-training set?

This matters more than any other check in the project. If a test storm
appeared during pre-training, the model met it before fine-tuning began and
the headline result is worthless. A large, surprising improvement is exactly
when you should look hardest for a bug.

Three independent checks, because one is not enough:
  1. SID overlap - the obvious one, and the one step 6 already guards.
  2. Name + season overlap - catches storms whose SID was renumbered between
     IBTrACS v03r08 (which HURSAT was built with) and v04r01. Step 4 matched
     211 frames by name fallback precisely because those SIDs did NOT match,
     which means an SID-only exclusion could have let them through.
  3. Geographic - any frame whose position sits inside the North Indian Ocean
     box, regardless of what it is called.

Run:  python check_leakage.py
"""

import pandas as pd
import config

# North Indian Ocean bounding box (Arabian Sea + Bay of Bengal)
NI_BOX = dict(lat_min=0, lat_max=31, lon_min=40, lon_max=100)


def main():
    ni = pd.read_csv(config.MANIFEST, parse_dates=["ISO_TIME"])
    gl = pd.read_csv(config.GLOBAL_MANIFEST, parse_dates=["ISO_TIME"])
    tracks = pd.read_csv(config.TRACKS_CLEAN)
    print(f"North Indian Ocean set : {len(ni):>6} frames, {ni['SID'].nunique()} storms")
    print(f"Pre-training set       : {len(gl):>6} frames, {gl['SID'].nunique()} storms\n")

    fails = []

    # --- 1. SID overlap -------------------------------------------------
    shared = set(gl["SID"]) & set(tracks["SID"])
    print(f"[1] SID overlap with the NI best-track table : {len(shared)}")
    if shared:
        fails.append(f"{len(shared)} shared SIDs: {sorted(shared)[:5]}")

    # --- 2. name + season overlap --------------------------------------
    ni_key = {(r.NAME, r.SEASON) for r in tracks.itertuples()
              if isinstance(r.NAME, str) and r.NAME != "UNNAMED"}
    gl_key = {(r.NAME, r.SEASON) for r in gl.itertuples()
              if isinstance(r.NAME, str) and r.NAME != "UNNAMED"}
    shared_names = ni_key & gl_key
    print(f"[2] (name, season) pairs shared              : {len(shared_names)}")
    if shared_names:
        fails.append(f"shared name/season: {sorted(shared_names)[:5]}")

    # --- 3. geographic --------------------------------------------------
    inbox = gl[(gl.LAT.between(NI_BOX["lat_min"], NI_BOX["lat_max"])) &
               (gl.LON.between(NI_BOX["lon_min"], NI_BOX["lon_max"]))]
    print(f"[3] pre-training frames inside the NI box    : {len(inbox)} "
          f"({100*len(inbox)/len(gl):.2f}%)")
    if len(inbox):
        fails.append(f"{len(inbox)} frames positioned inside the NI box")
        print("      storms involved:",
              sorted(inbox["NAME"].dropna().unique())[:10])

    # --- 4. the test storms specifically --------------------------------
    print(f"\n[4] named held-out storms: {config.TEST_STORM_NAMES}")
    hit = gl[gl["NAME"].isin(config.TEST_STORM_NAMES)]
    if len(hit):
        print(f"    !! {len(hit)} pre-training frames carry a held-out name")
        print(hit.groupby(["NAME", "SEASON"]).size().to_string())
        print("    (check the season - a same-named storm in another basin "
              "and another year is NOT leakage)")
    else:
        print("    none present")

    print("\n" + "=" * 62)
    if fails:
        print("LEAKAGE SUSPECTED. Do not report the transfer result until "
              "this is resolved:")
        for f in fails:
            print("  - " + f)
        print("\nFix: add the offending SIDs to the exclusion list in "
              "step6_build_global.py, rebuild, and re-run pre-training.")
    else:
        print("CLEAN. No North Indian Ocean storm appears in the pre-training "
              "set by ID, by name, or by position.")
        print("The transfer result can be reported as a genuine "
              "cross-basin generalisation.")
    print("=" * 62)


if __name__ == "__main__":
    main()
