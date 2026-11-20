"""
STEP 3 - Download the satellite imagery (HURSAT-B1 v06).

HURSAT-B1 is NOAA's storm-centred satellite archive: for each cyclone, one
netCDF file every 3 hours, already cropped and centred on the storm eye.
That is why we use it instead of raw satellite imagery - the hard work of
finding and centring the storm is already done for us.

IMPORTANT: HURSAT-B1 v06 covers 1978-2015 ONLY. There is no 2016+ imagery here.

The archive layout can change, so this script does not hard-code filenames.
It reads the directory listing for each year, finds the links, and grabs the
ones whose name matches a storm in our cleaned track table.

Run:  python step3_get_hursat.py --years 2007 2013 2014
      python step3_get_hursat.py                # all years in config
Makes: data/raw/hursat/<YEAR>/*.nc

TIP: open https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/
in a browser once, so you can see the folder structure with your own eyes.
"""

import argparse
import re
import tarfile
import urllib.request
from urllib.parse import urljoin

import pandas as pd
import config

HREF = re.compile(r'href="([^"?][^"]*)"', re.I)


def list_links(url: str):
    """Return the hyperlinks on an Apache-style directory page."""
    with urllib.request.urlopen(url, timeout=60) as r:
        html = r.read().decode("utf-8", "ignore")
    links = [h for h in HREF.findall(html) if not h.startswith(("/", "http", "?"))]
    return [l for l in links if l not in ("../",)]


def storm_ids_for(year: int) -> set:
    """The IBTrACS SIDs of every North Indian Ocean storm in this season.

    We filter on SID, not name. The year folders on NCEI hold storms from
    EVERY basin, and name matching is a substring test - so 'MALA' also
    matches the Pacific typhoon MALAKAS, and 'HELEN' matches the Atlantic
    HELENE. Filtering on SID is exact, and it also catches the unnamed
    depressions that a name filter silently drops.
    """
    df = pd.read_csv(config.TRACKS_CLEAN)
    return set(df.loc[df["SEASON"] == year, "SID"].dropna().unique())


def download(url: str, dest):
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return True


def fetch_year(year: int, wanted: set, limit: int | None):
    year_url = f"{config.HURSAT_BASE}/{year}/"
    out_dir = config.HURSAT_DIR / str(year)
    print(f"\n=== {year} ===\n  listing {year_url}")
    try:
        links = list_links(year_url)
    except Exception as e:
        print(f"  !! could not list that URL ({e}).")
        print("     Open it in a browser and check the current folder layout.")
        return

    # HURSAT filenames embed the SID:
    #   HURSAT_b1_v06_2007151N14072_GONU_c20170721.tar.gz
    def interesting(link: str) -> bool:
        return (not wanted) or any(sid in link for sid in wanted)

    picks = [l for l in links if interesting(l)]
    if limit:
        picks = picks[:limit]
    print(f"  {len(links)} links found ({len(wanted)} NI storms this season), "
          f"{len(picks)} matched")

    for link in picks:
        url = urljoin(year_url, link)

        if link.endswith("/"):                      # a per-storm sub-folder
            try:
                inner = [x for x in list_links(url) if x.endswith((".nc", ".nc.gz"))]
            except Exception:
                continue
            for f in inner:
                if download(urljoin(url, f), out_dir / f):
                    print(f"    + {f}")

        elif link.endswith(".nc"):
            if download(url, out_dir / link):
                print(f"    + {link}")

        elif link.endswith((".tar.gz", ".tgz", ".tar")):
            tmp = out_dir / link
            if download(url, tmp):
                print(f"    + {link} (extracting)")
                with tarfile.open(tmp) as t:
                    t.extractall(out_dir, filter="data")
                tmp.unlink()

    n = len(list(out_dir.rglob("*.nc"))) if out_dir.exists() else 0
    print(f"  total .nc files for {year}: {n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="only grab N storms per year (use 2 for a quick test)")
    p.add_argument("--all-storms", action="store_true",
                   help="take EVERY storm that year, all basins (huge - only "
                        "useful if you plan to pre-train on other basins)")
    a = p.parse_args()

    years = a.years or range(config.START_YEAR, config.END_YEAR + 1)
    for y in years:
        if y > 2015:
            print(f"Skipping {y}: HURSAT-B1 v06 stops at 2015.")
            continue
        wanted = set() if a.all_storms else storm_ids_for(y)
        fetch_year(y, wanted, a.limit)


if __name__ == "__main__":
    main()
