"""
Assemble every repeats_*.csv into the final ablation table.

Each row is mean +/- std over N seeds, not a single lucky run. Differences
are tested against a reference row with Welch's t-test, so the table says
which gaps are real and which are noise.

Run:  python make_ablation_table.py
      python make_ablation_table.py --ref baseline

Makes: ablation_table.csv, ablation_table.md, figures/05_ablation.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

ROOT = Path(__file__).resolve().parent

# Human-readable names, in the order they should appear in the report.
PRETTY = [
    ("naive",      "Naive baseline (predict the mean)"),
    ("tabular",    "Tabular features only (no imagery)"),
    ("noimg",      "Tabular features only (no imagery)"),
    ("imageonly",  "Imagery only (no tabular)"),
    ("notab",      "Imagery only (no tabular)"),
    ("plainmse",   "Full model, plain MSE (no imbalance handling)"),
    ("nobalance",  "Full model, no balanced sampling"),
    ("baseline",   "Full multi-modal model (random init)"),
    ("frozen",     "Pre-trained, CNN backbone frozen"),
    ("pretrained", "Pre-trained on other basins, then fine-tuned"),
]
ORDER = {k: i for i, (k, _) in enumerate(PRETTY)}
LABELS = dict(PRETTY)


def welch(a, b):
    """Welch's t-test. Returns (t, p) - p is None if scipy is unavailable."""
    try:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except Exception:
        na, nb = len(a), len(b)
        se = np.sqrt(a.var(ddof=1)/na + b.var(ddof=1)/nb)
        return (float((a.mean() - b.mean()) / se) if se > 0 else np.nan), None


def main(ref_key):
    files = sorted(ROOT.glob("repeats_*.csv"))
    if not files:
        raise SystemExit("No repeats_*.csv found. Run repeat_runs.py first.")

    runs = {}
    for f in files:
        key = re.sub(r"^repeats_|\.csv$", "", f.name)
        runs[key] = pd.read_csv(f)
        print(f"loaded {f.name}: {len(runs[key])} seeds")

    keys = sorted(runs, key=lambda k: ORDER.get(k, 99))
    ref = runs.get(ref_key)

    rows = []
    for k in keys:
        df = runs[k]
        row = {"configuration": LABELS.get(k, k), "key": k, "n_seeds": len(df)}
        for m, name in [("test_rmse", "Test RMSE"), ("test_mae", "Test MAE"),
                        ("cat3_rmse", "Cat3+ RMSE"), ("cat3_bias", "Cat3+ bias")]:
            row[f"{name} (kt)"] = f"{df[m].mean():.2f} ± {df[m].std():.2f}"
            row[f"_{m}_mean"] = df[m].mean()
        # Significance is reported for BOTH overall and Cat 3+ RMSE. The
        # headline claim for imbalance handling lives in the Cat 3+ column,
        # so testing only the overall metric would understate it.
        if ref is not None and k != ref_key:
            for m, col in [("test_rmse", "overall vs ref"),
                           ("cat3_rmse", "Cat3+ vs ref")]:
                t, p = welch(df[m], ref[m])
                delta = df[m].mean() - ref[m].mean()
                mark = "*" if (p is not None and p < 0.05) else "ns"
                row[col] = (f"{delta:+.2f} kt, p={p:.2g} {mark}" if p is not None
                            else f"{delta:+.2f} kt, t={t:.1f}")
        else:
            lbl = "— reference —" if k == ref_key else ""
            row["overall vs ref"] = lbl
            row["Cat3+ vs ref"] = lbl
        rows.append(row)

    tbl = pd.DataFrame(rows)
    show = [c for c in tbl.columns if not c.startswith("_") and c != "key"]
    tbl[show].to_csv(ROOT / "ablation_table.csv", index=False)

    md = ["| " + " | ".join(show) + " |",
          "|" + "|".join(["---"] * len(show)) + "|"]
    for _, r in tbl.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in show) + " |")
    (ROOT / "ablation_table.md").write_text("\n".join(md) + "\n")

    print("\n" + tbl[show].to_string(index=False))
    print(f"\nSaved ablation_table.csv and ablation_table.md")

    # ---- chart ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.9 + 0.62 * len(keys)))
    for ax, metric, title in [(axes[0], "test_rmse", "Overall test RMSE"),
                              (axes[1], "cat3_rmse", "Category 3+ RMSE")]:
        means = [runs[k][metric].mean() for k in keys]
        errs = [runs[k][metric].std() for k in keys]
        names = [LABELS.get(k, k) for k in keys]
        y = np.arange(len(keys))
        colors = ["#2E6F40" if k == "pretrained" else "#3b6ea5" for k in keys]
        ax.barh(y, means, xerr=errs, color=colors, height=0.6,
                error_kw=dict(ecolor="#333", lw=1.2, capsize=4))
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("knots (lower is better)")
        ax.set_title(title, fontsize=11)
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.text(m + e + 0.4, i, f"{m:.1f}", va="center", fontsize=8.5)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Ablation: mean ± std over multiple seeds, identical data",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "05_ablation.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="baseline",
                   help="which run key the others are compared against")
    main(p.parse_args().ref)
