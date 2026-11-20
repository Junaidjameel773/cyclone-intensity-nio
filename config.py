"""
config.py
---------
One place for every setting. Change things HERE, not inside the other scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"                 # downloaded files, untouched
PROC = DATA / "processed"          # cleaned files we create
IBTRACS_CSV = RAW / "ibtracs.NI.list.v04r01.csv"
TRACKS_CLEAN = PROC / "tracks_ni_clean.csv"
HURSAT_DIR = RAW / "hursat"        # .nc files land here
FRAME_DIR = PROC / "frames"        # one .npy image per timestep
MANIFEST = PROC / "manifest.csv"   # the master table: image path + labels
CKPT_DIR = ROOT / "checkpoints"

# --- transfer pre-training on other basins -------------------------------
IBTRACS_ALL_CSV = RAW / "ibtracs.ALL.list.v04r01.csv"
GLOBAL_TRACKS = PROC / "tracks_global_clean.csv"
GLOBAL_FRAME_DIR = PROC / "frames_global"
GLOBAL_MANIFEST = PROC / "manifest_global.csv"
FIG_DIR = ROOT / "figures"

for d in (RAW, PROC, HURSAT_DIR, FRAME_DIR, CKPT_DIR, FIG_DIR,
          PROC / "frames_global"):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data sources
# Global (all-basin) labels, used only for transfer pre-training.
IBTRACS_ALL_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)
HURSAT_BASE = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"

# HURSAT-B1 v06 only goes up to 2015. Do not set END_YEAR above 2015.
START_YEAR = 2000
END_YEAR = 2015

# ---------------------------------------------------------------- images
RAW_GRID = 301        # HURSAT-B1 native grid is 301 x 301 at 0.07 deg
# Centre crop, in native 0.07-degree pixels, BEFORE resizing to IMG_SIZE.
#   201 px = 14.1 deg = 1562 km across = 12.2 km/pixel  (a 40 km eye = 3 px)
#   101 px =  7.1 deg =  785 km across =  6.1 km/pixel  (a 40 km eye = 7 px)
# Intensity lives in the eye and eyewall, so a wide crop spends most of its
# pixels on empty ocean and blurs the only structure that matters.
CROP = 101
IMG_SIZE = 128        # final size fed to the CNN
IR_MIN, IR_MAX = 180.0, 310.0   # kelvin range used to scale IR to 0..1

# ---------------------------------------------------------------- sequences
SEQ_LEN = 4           # 4 frames x 3 hours = 12 hours of history
MAX_GAP_HOURS = 3.5   # frames further apart than this break a sequence

# ---------------------------------------------------------------- labels + loss
TARGET = "USA_WIND"   # maximum sustained wind, knots
CAT3_KT = 96.0        # Saffir-Simpson Cat 3 threshold, knots
HIGH_WEIGHT = 3.0     # how much extra the loss cares about Cat 3+ errors

# tabular features fed into the MLP branch
TAB_COLS = ["LAT", "LON", "MONTH_SIN", "MONTH_COS", "DLAT", "DLON", "SPEED_KT"]

# ---------------------------------------------------------------- split
# Held-out storms. These are all <= 2015 so HURSAT imagery exists for them.
# (Tauktae 2021 and Biparjoy 2023 are NOT usable - no HURSAT-B1 imagery.)
# Headline case studies you want plots for. Keep this SHORT - if every severe
# storm is in the test set, the model never learns what a severe storm looks
# like and you are testing extrapolation, not skill.
TEST_STORM_NAMES = ["GONU", "PHAILIN", "HUDHUD"]

# ...plus this many randomly chosen storms, so the test set spans the whole
# intensity range instead of only the extreme tail.
TEST_EXTRA_RANDOM = 6

VAL_FRACTION = 0.15   # fraction of the remaining storms used for validation
MIN_VAL_STORMS = 4    # a 1-storm validation set is pure noise

# ---------------------------------------------------------------- training
AUGMENT = True        # random 90-degree rotations of the whole sequence
BATCH_SIZE = 16
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
PATIENCE = 8          # early stopping
SEED = 42
