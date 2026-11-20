"""
STEP 1 - Download the ground truth (the answers your model must learn).

IBTrACS = the official global list of every tropical cyclone: where it was,
when, and how strong. We only want the North Indian Ocean ("NI") basin file.

Run:  python step1_get_ibtracs.py
Makes: data/raw/ibtracs.NI.list.v04r01.csv   (a few MB, takes seconds)
"""

import urllib.request
import config


def main():
    if config.IBTRACS_CSV.exists():
        print(f"Already downloaded: {config.IBTRACS_CSV}")
        return

    print(f"Downloading from:\n  {config.IBTRACS_URL}")
    urllib.request.urlretrieve(config.IBTRACS_URL, config.IBTRACS_CSV)

    size_mb = config.IBTRACS_CSV.stat().st_size / 1e6
    print(f"Saved {config.IBTRACS_CSV}  ({size_mb:.1f} MB)")
    print("\nPeek at the first 2 lines:")
    with open(config.IBTRACS_CSV) as f:
        for _ in range(2):
            print("  " + f.readline()[:160] + " ...")
    print("\nNote: row 0 is the column names, row 1 is the UNITS row - we drop it in step 2.")


if __name__ == "__main__":
    main()
