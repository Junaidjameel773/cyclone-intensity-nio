"""
evaluate.py - produce the figures and tables that go in your report.

Run:  python evaluate.py

Makes:
  figures/02_scatter.png        predicted vs actual, all test points
  figures/03_storm_<NAME>.png   intensity over the storm's life, predicted vs true
  figures/04_attention_<NAME>.png  where the model was looking
  figures/metrics.csv           RMSE / MAE overall and per intensity band
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from dataset import make_datasets
from model import MultiModalCycloneModel


def load_model(device):
    ck = torch.load(config.CKPT_DIR / "best.pt", map_location=device, weights_only=False)
    m = MultiModalCycloneModel().to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def predict(model, ds, device):
    rows = []
    for b in DataLoader(ds, batch_size=16, shuffle=False):
        p = model(b["images"].to(device), b["tab"].to(device)).cpu().numpy()
        rows += [{"sid": s, "time": t, "true": float(y), "pred": float(q)}
                 for s, t, y, q in zip(b["sid"], b["time"], b["target"].numpy(), p)]
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values(["sid", "time"])


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    bands = [("Depression/Storm  <64 kt", 0, 64),
             ("Cat 1-2   64-95 kt", 64, 96),
             ("Cat 3+     >=96 kt", 96, 999)]
    out = []
    for label, lo, hi in [("ALL", 0, 999)] + bands:
        s = df[(df["true"] >= lo) & (df["true"] < hi)]
        if s.empty:
            continue
        e = s["pred"] - s["true"]
        out.append({"band": label, "n": len(s),
                    "RMSE_kt": np.sqrt((e ** 2).mean()),
                    "MAE_kt": e.abs().mean(),
                    "bias_kt": e.mean()})
    return pd.DataFrame(out).round(2)


def scatter(df):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["true"], df["pred"], s=12, alpha=.45, color="#3b6ea5")
    lim = [0, max(df["true"].max(), df["pred"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.axvline(config.CAT3_KT, color="crimson", ls=":", lw=1)
    ax.set_xlabel("Actual wind (kt)"); ax.set_ylabel("Predicted wind (kt)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    r = np.sqrt(((df["pred"] - df["true"]) ** 2).mean())
    ax.set_title(f"Test set - RMSE {r:.1f} kt")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "02_scatter.png", dpi=140)
    plt.close(fig)


def storm_curves(df, man):
    name_of = man.drop_duplicates("SID").set_index("SID")["NAME"].to_dict()
    for sid, g in df.groupby("sid"):
        if len(g) < 5:
            continue
        nm = name_of.get(sid, sid)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(g["time"], g["true"], "o-", label="actual", color="#111", ms=3)
        ax.plot(g["time"], g["pred"], "s--", label="predicted", color="#d1495b", ms=3)
        ax.axhline(config.CAT3_KT, color="grey", ls=":", lw=1)
        ax.set_ylabel("Max sustained wind (kt)")
        ax.set_title(f"{nm} ({sid}) - lifecycle intensity")
        ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(config.FIG_DIR / f"03_storm_{nm}.png", dpi=140)
        plt.close(fig)


@torch.no_grad()
def attention_figure(model, ds, device, k=0):
    """Overlay the cross-attention weights back onto the satellite frames."""
    if len(ds) == 0:
        return
    b = ds[min(k, len(ds) - 1)]
    imgs = b["images"].unsqueeze(0).to(device)
    pred, attn = model(imgs, b["tab"].unsqueeze(0).to(device), return_attention=True)
    attn = attn[0].cpu().numpy()                      # (T, 8, 8)
    attn = attn / (attn.max() + 1e-9)
    frames = b["images"].squeeze(1).numpy()           # (T,128,128)

    T = frames.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))
    for t in range(T):
        axes[0, t].imshow(frames[t], cmap="Greys", vmin=0, vmax=1)
        axes[0, t].set_title(f"t-{T-1-t}" if t < T - 1 else "t (now)")
        axes[1, t].imshow(frames[t], cmap="Greys", vmin=0, vmax=1)
        axes[1, t].imshow(np.kron(attn[t], np.ones((config.IMG_SIZE // attn.shape[1],) * 2)),
                          cmap="inferno", alpha=.5, vmin=0, vmax=1)
        for r in (0, 1):
            axes[r, t].axis("off")
    fig.suptitle(f"{b['sid']} {b['time']} - true {b['target']:.0f} kt, "
                 f"predicted {pred.item():.0f} kt   (bottom row = attention)")
    fig.tight_layout()
    out = config.FIG_DIR / f"04_attention_{b['sid']}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"Saved {out}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    _, _, test_ds = make_datasets()
    man = pd.read_csv(config.MANIFEST)

    df = predict(model, test_ds, device)
    df.to_csv(config.PROC / "test_predictions.csv", index=False)

    tbl = metrics_table(df)
    tbl.to_csv(config.FIG_DIR / "metrics.csv", index=False)
    print("\n" + tbl.to_string(index=False))
    print("\nThe Cat 3+ row is the one that matters. That is what the weighted "
          "loss was for, and what a plain-MSE ablation should do worse on.")

    scatter(df)
    storm_curves(df, man)
    strongest = int(np.argmax(test_ds.window_targets()))
    print(f"attention map: strongest test window "
          f"({test_ds.window_targets()[strongest]:.0f} kt)")
    attention_figure(model, test_ds, device, k=strongest)
    print(f"\nFigures written to {config.FIG_DIR}")


if __name__ == "__main__":
    main()
