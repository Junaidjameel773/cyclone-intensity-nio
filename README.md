# Cyclone Intensity Estimation — a build-it-from-scratch guide

Your proposal is solid. This repo is that proposal turned into working code, plus
three corrections you need to make before you start downloading anything.

---

## Read this first: three things in the proposal that will break

### 1. HURSAT-B1 stops in 2015, not 2024

The publicly downloadable version is **HURSAT-B1 v06, covering 1978–2015**. There is
a newer internal version (v07b) that NOAA used to build the ADT-HURSAT product
through 2024, but the raw imagery archive you can download is still the 2015 one.

So:

- Change your year range from **2000–2024 → 2000–2015**.
- **Tauktae (2021) and Biparjoy (2023) cannot be your test storms.** There is no
  imagery for them. Swap in **Gonu (2007), Nargis (2008), Phailin (2013),
  Hudhud (2014), Chapala (2015)** — all severe, all well inside coverage.

This is not a downgrade. Say so explicitly in your report: "HURSAT-B1 v06 provides a
homogenised, inter-satellite-calibrated record through 2015; we use the full
2000–2015 window to avoid mixing calibration regimes." That is a *better* sentence
than one about 2024.

### 2. IBTrACS is now v04r01, and your URL points at v04r00

Use this instead (already set in `config.py`):

```
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv
```

### 3. Do ERA5 last, not first

The Copernicus API moved in late 2024: new URL, new token format, and requests fail
with an unhelpful 403 until you manually click "accept terms" on each dataset page.
Downloads also sit in a queue. It is the single most likely thing to eat a week.

Your model works fine with satellite imagery plus position/motion features alone.
Get that working end to end, *then* add SST and wind shear as an improvement you can
show a before/after number for. That is also a better story in the report.

---

## The order to build things

The mistake almost everyone makes is downloading 50 GB of satellite data before
knowing whether their training loop runs. Do it backwards.

| # | Do this | Command | Time |
|---|---------|---------|------|
| 0 | Prove the model trains, on fake data | `python make_fake_data.py` then `python train.py --epochs 3` | 15 min |
| 1 | Get the labels | `python step1_get_ibtracs.py` | 1 min |
| 2 | Clean them, look at them | `python step2_explore_tracks.py` | 2 min |
| 3 | Get imagery for **two** storms only | `python step3_get_hursat.py --years 2007 --limit 2` | 10 min |
| 4 | Build the dataset from those two | `python step4_build_dataset.py` | 5 min |
| 5 | Train on two storms. Expect bad numbers. | `python train.py --epochs 5` | 10 min |
| 6 | *Now* download all years | `python step3_get_hursat.py` | hours |
| 7 | Rebuild, train properly, evaluate | `step4` → `train.py` → `evaluate.py` | hours |
| 8 | Optional: add ERA5 | `python step5_era5_optional.py --year 2007 --month 6` | days |

Step 0 matters more than it looks. `make_fake_data.py` writes exactly the files step 4
would write, but with synthetic storms — a cold cloud blob that gets tighter and grows
an eye as wind speed rises. If training works on that, every bug left is a *data* bug,
and you will know which half of the project to debug.

Setup:

```bash
pip install -r requirements.txt
```

---

## What each file does

```
config.py                 every setting, in one place. Change things here.
step1_get_ibtracs.py      download the ground truth (wind speed per storm per 3 h)
step2_explore_tracks.py   clean it, add tabular features, plot sanity checks
step3_get_hursat.py       download the satellite netCDF files
step4_build_dataset.py    netCDF -> 128x128 .npy images + one manifest.csv
step5_era5_optional.py    optional SST / wind shear
make_fake_data.py         synthetic data so you can test before downloading
dataset.py                builds 4-frame sequences, splits by storm
model.py                  CNN + LSTM + cross-attention + tabular fusion, and W-MSE
train.py                  training loop, warm-up, early stopping, baseline check
evaluate.py               metrics table + the figures for your report
```

Everything after step 4 reads **only** `data/processed/manifest.csv`. One table, one
row per satellite frame, containing the image filename, the true wind speed, and the
tabular features. Do the slow messy work once, then never touch netCDF again.

---

## How the model actually works

```
images (B, 4, 1, 128, 128)
        │
    [1] CNN            each frame → an 8×8 grid of 256-dim features
        │
    [2] LSTM           reads the 4 frames in order → one summary vector h
        │
    [3] Cross-attention
        query = h ("what I remember about how this storm has been developing")
        keys  = all 4×8×8 = 256 patches from all frames
        → a context vector, plus attention weights you can plot as a heat map
        │
tab ─ [4] MLP ──→ concat(context, h, tab) → dense → wind speed in knots
```

Two things worth saying out loud in your report:

**Why attend over patches and not just time steps.** Attending over the four time steps
alone gives you a bar chart of four numbers. Attending over every patch of every frame
gives you a `(4, 8, 8)` map you can overlay on the imagery — that is the "attention map
overlay" deliverable, and it lets you show the model locking onto the eyewall.

**Why the weighted loss.** In the North Indian Ocean, Cat 3+ observations are a small
single-digit percentage of your rows (step 2 prints the exact number). Plain MSE will
happily sacrifice every intense storm to shave error off the common weak ones — and
intense storms are the only ones anyone cares about. `WeightedMSE` multiplies the
squared error by a weight that ramps from 1× to 3× around 96 kt.

---

## Three ways to not fool yourself

**Split by storm, never by row.** Frames 3 hours apart are nearly identical. If Phailin
has some frames in train and some in test, your RMSE will look spectacular and mean
nothing. `dataset.py` splits whole storms. Your entire result depends on this one line.

**Always report the naive baseline.** `train.py` prints the RMSE you would get by
predicting the training mean for every frame. If your model is not clearly below it,
it has learned nothing, whatever the loss curve says.

**Report RMSE broken down by intensity band.** `evaluate.py` does this. An overall RMSE
of 12 kt hides whether you are at 8 kt on weak storms and 30 kt on Cat 4s — which is
the thing your grader will ask about.

---

## What to actually put in the report

The deliverables in your proposal map onto this:

1. **Ablation table.** Same data, four rows: image-only → + LSTM → + cross-attention →
   + tabular fusion. Each row RMSE / MAE overall and for Cat 3+. This is the single
   most convincing table you can produce, and `--plain-mse` gives you a fifth row for
   the loss function.
2. **Per-storm lifecycle plots** (`figures/03_storm_*.png`) — predicted vs actual across
   Gonu, Phailin, Hudhud. Discuss where it lags during rapid intensification. It will.
3. **Attention overlays** (`figures/04_attention_*.png`).
4. **Honest limitations.** The 2015 cutoff. Small Cat 3+ sample. That IBTrACS USA_WIND
   in this basin is itself largely Dvorak-estimated from satellite imagery — so you are
   partly learning to imitate the Dvorak technique, not measuring ground truth. Saying
   this yourself is worth more marks than hoping nobody notices.

---

## When something goes wrong

**`step3` finds no links / the URL 404s.** Open
`https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/` in a
browser and look at the folder layout with your own eyes, then adjust the filter in
`fetch_year`. NCEI is mid cloud-migration and paths do move.

**`step4` keeps all files but matches none.** The storm name inside the netCDF did not
match IBTrACS. Print `name` and `ts` inside `read_ir` for the first ten files and see
what you are actually getting.

**`read_ir` returns None for everything.** The IR variable is not called `IRWIN` in your
files. Open one file and print `ds.data_vars`, then add the real name to the tuple.

**Training loss goes to NaN.** Lower the learning rate to `1e-4`. Check no image is all
NaN (`np.isfinite(img).all()`).

**Val RMSE is far below train loss.** You probably rebuilt the manifest without deleting
`data/processed/frames/` first, and old fake frames are still in there.

**It is too slow on CPU.** It will be. Use Google Colab or Kaggle for a free GPU; both
mount fine if you upload `data/processed/` as a zip. The model is only 2.1 M parameters,
so a T4 is plenty.
