# Cyclone Intensity Estimation — North Indian Ocean

Predicting tropical cyclone maximum sustained wind speed from storm-centred
infrared satellite imagery, for the Arabian Sea and Bay of Bengal.

A multi-modal CNN–LSTM with cross-attention, trained on NOAA archives, with
cross-basin transfer learning to address the basin's severe class imbalance.

Deep Learning course project · MS Artificial Intelligence · NED University of
Engineering and Technology.

---

## Why

Aircraft reconnaissance is not flown in this basin, so intensity is estimated
from satellite imagery using the Dvorak technique — a manual procedure from
1975 whose output varies between analysts. This project automates that
estimate.

## Results

Every configuration was run with five random seeds; figures are mean ± std.
Welch's *t*-test against the full model.

| Model | Test RMSE (kt) | Cat 3+ RMSE (kt) | Cat 3+ bias (kt) |
|---|---|---|---|
| Predict-the-mean baseline | 35.65 | — | — |
| Position & season only | 31.95 ± 0.00 | 63.30 ± 0.00 | −61.50 |
| Imagery only | 25.03 ± 2.00 | 48.38 ± 7.25 | −44.63 |
| Full model, trained locally | 22.79 ± 1.19 | 46.91 ± 3.36 | −42.52 |
| Pre-trained, backbone frozen | 12.36 ± 0.41 | **11.97 ± 1.29** | **−4.25** |
| **Pre-trained, fine-tuned** | **12.15 ± 0.50** | 14.09 ± 1.93 | −7.17 |

Cross-basin pre-training reduces error by 10.64 kt
(p = 4.8 × 10⁻⁶). Freezing the transferred backbone entirely is
statistically indistinguishable from fine-tuning it (p = 0.49), which suggests
the visual signature of cyclone intensity is largely basin-independent.

## Architecture

```
4 infrared frames (12 h)  →  CNN backbone  ─┬─→ pooled → LSTM → state h
                                            └─→ 4×8×8 patch tokens
                                                      ↓
                              h (query) → cross-attention → context + attention map
                                                      ↓
lat, lon, month, motion  →  MLP  ──────────→ concat → dense → wind speed (kt)
```

2.1 M parameters, written from scratch. No pre-trained image classifier is used
as a backbone; the transfer weights come from this same architecture trained on
other basins.

## Data

| Source | Role |
|---|---|
| [IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive) | best-track labels (`USA_WIND`, position, time) |
| [HURSAT-B1 v06](https://www.ncei.noaa.gov/products/hurricane-satellite-data) | storm-centred 11 µm infrared imagery |

48 storms, 2,341 labelled frames, 2000–2015. Both archives are US government
works in the public domain. No Kaggle dataset and no synthetic data are used.

**The split is storm-disjoint.** Consecutive frames are three hours apart and
visually near-identical, so a random split over frames would place almost
identical images in both training and test.

## Quick start

```bash
pip install -r requirements.txt

# prove the pipeline works — no downloads, synthetic storms
python make_fake_data.py
python train.py --epochs 3
```

With the real data:

```bash
python step1_get_ibtracs.py      # labels
python step3_get_hursat.py       # imagery (several GB)
python step4_build_dataset.py    # → data/processed/manifest.csv
python train.py --init-from checkpoints/pretrained.pt --epochs 60 --balance 0.5
python evaluate.py
```

## Model weights and processed data

Weights and processed data are **not** in this repository — git is the wrong
place for either. Both are in a 109 MB bundle together with the processed
dataset (2,341 frames), the trained checkpoints and every figure:

**[Download the runnable bundle](https://drive.google.com/file/d/1sWiXUPsi5kEDgiKP_hz2mFNT_cbhGjM4/view?usp=sharing)**

It reproduces every number in the results table in about two minutes, without
retraining:

```bash
unzip cyclone_share.zip && cd cyclone_share
pip install -r requirements.txt
python evaluate.py
```

## Repository layout

| File | What it does |
|---|---|
| `step1_get_ibtracs.py` | downloads the best-track labels |
| `step2_explore_tracks.py` | cleans tracks, builds tabular features |
| `step3_get_hursat.py` | downloads satellite imagery by storm ID |
| `step4_build_dataset.py` | netCDF → 128×128 frames + manifest |
| `step6_build_global.py` | builds the other-basin pre-training set |
| `dataset.py` | sequence windows, storm-disjoint split, augmentation |
| `model.py` | the architecture and the weighted MSE loss |
| `pretrain.py` / `train.py` | the two training stages |
| `evaluate.py` | per-band metrics, lifecycle curves, attention maps |
| `check_leakage.py` | three-way leakage audit of the pre-training set |
| `repeat_runs.py` | multi-seed runs with mean ± std |
| `make_ablation_table.py` | the ablation table and chart |
| `make_fake_data.py` | synthetic data for testing without downloads |

## Reproducing the figures

```bash
python evaluate.py              # per-band metrics + attention maps
python check_leakage.py         # leakage audit
python make_ablation_table.py   # ablation table and chart
```

## Limitations

- The public HURSAT-B1 release ends in **2015**, so Tauktae (2021) and
  Biparjoy (2023) cannot be evaluated.
- Of 90 storms in the best-track record for the period, **48** had usable
  imagery.
- IBTrACS `USA_WIND` in this basin is itself largely **Dvorak-derived**, so the
  model is partly learning to reproduce the Dvorak technique rather than being
  trained against independent ground truth.
- The ERA5 environmental branch (sea-surface temperature, vertical wind shear)
  is specified in the design but **not implemented**.
- **This is a course project and is not validated for operational use.**

## Citation

```bibtex
@misc{jameel2026cyclone,
  author = {Jameel, Junaid},
  title  = {Multi-Modal Tropical Cyclone Intensity Estimation
            in the North Indian Ocean},
  year   = {2026},
  note   = {Deep Learning course project, NED University of Engineering
            and Technology},
  howpublished = {\url{https://github.com/Junaidjameel773/cyclone-intensity-nio}}
}
```

## Acknowledgements

Data courtesy of NOAA NCEI: IBTrACS (Knapp et al., 2010; Gahtan et al., 2024)
and HURSAT-B1 (Knapp & Kossin, 2007).

## Licence

Code: MIT. Data: public domain (US government work), redistributed under the
terms of the original archives.
