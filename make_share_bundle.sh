#!/usr/bin/env bash
# Build a small, runnable bundle of the cyclone project.
#
# The working folder is ~14-16 GB, but almost all of that is raw HURSAT
# netCDF files that nothing downstream reads. Everything after step 4 reads
# only the processed frames and the manifest, so the raw archive can be left
# out entirely and regenerated with the scripts if anyone wants it.
#
# Run from inside cyclone_project/:
#     bash make_share_bundle.sh

set -e
OUT=../cyclone_share
rm -rf "$OUT"
mkdir -p "$OUT"/{data/processed,checkpoints,figures}

echo "== code =="
cp *.py requirements.txt README.md "$OUT"/ 2>/dev/null || true

echo "== processed data (what the model actually reads) =="
cp -r data/processed/frames "$OUT"/data/processed/
cp data/processed/manifest.csv data/processed/tracks_ni_clean.csv "$OUT"/data/processed/

echo "== results =="
cp repeats_*.csv ablation_table.* sweep_results.csv "$OUT"/ 2>/dev/null || true
cp figures/*.png figures/metrics.csv "$OUT"/figures/ 2>/dev/null || true

echo "== trained model =="
# best.pt alone lets anyone reproduce every number without retraining.
# pretrained.pt is only needed to redo the fine-tuning stage.
cp checkpoints/best.pt "$OUT"/checkpoints/ 2>/dev/null \
  || echo "   (best.pt not found - run train.py first)"

cat > "$OUT"/RUN_ME.md <<'EOF'
# Cyclone intensity estimation — how to run this

Junaid Jameel · CS-121/2026

## What is in here

| Folder | What |
|---|---|
| `*.py` | the full pipeline, 17 scripts |
| `data/processed/frames/` | 2,341 preprocessed 128×128 infrared frames |
| `data/processed/manifest.csv` | the data contract: image path, labels, features |
| `checkpoints/best.pt` | the trained model |
| `figures/` | every figure in the report |
| `repeats_*.csv`, `ablation_table.md` | the multi-seed results |

The raw HURSAT netCDF archive (~14 GB) is **not** included. Nothing after
step 4 reads it. To rebuild it from scratch: `step1`, `step3`, `step4`.

## Setup

```bash
conda create -n cyclone python=3.11 -y
conda activate cyclone
pip install -r requirements.txt
```

## Reproduce the headline numbers (~2 minutes, no training)

```bash
python evaluate.py
```

Expect test RMSE around 12–13 kt against a naive baseline of 35.65 kt, and
the per-band table with Cat 3+ bias near −5 kt.

## Verify there is no data leakage

```bash
python check_leakage.py
```

Needs `data/processed/manifest_global.csv`, which is not shipped (it indexes
the other-basin pre-training set). The result from our run is in the report:
zero overlap by storm ID, by name and season, and by geographic position.

## Retrain from the provided data (~2 minutes on a GPU)

```bash
python train.py --epochs 60 --balance 0.5          # local training only
```

The transfer result needs `checkpoints/pretrained.pt`, which is 1.26 GB and
omitted for size. To rebuild it: `python step6_build_global.py` then
`python pretrain.py --epochs 25` — several hours, and it needs the raw
archive.

## Check the pipeline without any data

```bash
python make_fake_data.py
python train.py --epochs 3
```

This generates synthetic storms in the same format and trains on them. If it
runs, the code is working.
EOF

echo
du -sh "$OUT"
echo
echo "Bundle at: $OUT"
echo "Zip it with:  zip -r cyclone_share.zip $OUT"
