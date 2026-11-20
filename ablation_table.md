| configuration | n_seeds | Test RMSE (kt) | Test MAE (kt) | Cat3+ RMSE (kt) | Cat3+ bias (kt) | overall vs ref | Cat3+ vs ref |
|---|---|---|---|---|---|---|---|
| Tabular features only (no imagery) | 5 | 31.95 ± 0.00 | 25.07 ± 0.00 | 63.30 ± 0.00 | -61.50 ± 0.00 | +9.16 kt, p=6.7e-05 * | +16.39 kt, p=0.0004 * |
| Imagery only (no tabular) | 5 | 25.03 ± 2.00 | 18.14 ± 0.98 | 48.38 ± 7.25 | -44.63 ± 7.00 | +2.24 kt, p=0.071 ns | +1.46 kt, p=0.7 ns |
| Full model, plain MSE (no imbalance handling) | 5 | 24.46 ± 2.10 | 17.01 ± 1.20 | 53.92 ± 5.26 | -51.91 ± 5.45 | +1.67 kt, p=0.17 ns | +7.01 kt, p=0.041 * |
| Full multi-modal model (random init) | 5 | 22.79 ± 1.19 | 16.38 ± 0.66 | 46.91 ± 3.36 | -42.52 ± 3.78 | — reference — | — reference — |
| Pre-trained, CNN backbone frozen | 5 | 12.36 ± 0.41 | 9.76 ± 0.28 | 11.97 ± 1.29 | -4.25 ± 3.62 | -10.43 kt, p=9.2e-06 * | -34.94 kt, p=2.9e-06 * |
| Pre-trained on other basins, then fine-tuned | 5 | 12.15 ± 0.50 | 9.57 ± 0.44 | 14.09 ± 1.93 | -7.17 ± 4.52 | -10.64 kt, p=4.8e-06 * | -32.83 kt, p=7.4e-07 * |
