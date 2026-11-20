"""
dataset.py - turns the manifest into 4-frame sequences PyTorch can train on.

One training example =
    images : 4 satellite frames, 3 hours apart  -> shape (4, 1, 128, 128)
    tab    : the weather/position numbers at the LAST frame
    target : the true wind speed at the LAST frame

We build a list of valid "windows" once, up front. A window is valid only if
the 4 frames belong to the same storm and are evenly spaced in time.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


def build_windows(man: pd.DataFrame):
    """Return a list of index-arrays, each of length SEQ_LEN."""
    windows = []
    for _, g in man.groupby("SID", sort=False):
        g = g.sort_values("ISO_TIME")
        idx = g.index.to_numpy()
        times = g["ISO_TIME"].to_numpy()
        for i in range(config.SEQ_LEN - 1, len(idx)):
            w = idx[i - config.SEQ_LEN + 1: i + 1]
            span = (times[i] - times[i - config.SEQ_LEN + 1]) / np.timedelta64(1, "h")
            if span <= config.MAX_GAP_HOURS * (config.SEQ_LEN - 1) + 0.1:
                windows.append(w)
    return windows


def split_storms(man: pd.DataFrame, seed: int = config.SEED):
    """Split BY STORM, never by row - otherwise frames 3 h apart leak between
    train and test and your RMSE will look amazing for the wrong reason."""
    named = list(man.loc[man["NAME"].isin(config.TEST_STORM_NAMES), "SID"].unique())
    rest = np.array(sorted(set(man["SID"]) - set(named)))

    rng = np.random.default_rng(seed)
    rng.shuffle(rest)

    # test = the named case studies + a random slice of ordinary storms
    n_extra = min(config.TEST_EXTRA_RANDOM, max(0, len(rest) - config.MIN_VAL_STORMS - 1))
    test_sids = set(named) | set(rest[:n_extra])
    rest = rest[n_extra:]

    n_val = min(len(rest) - 1, max(config.MIN_VAL_STORMS,
                                   int(len(rest) * config.VAL_FRACTION)))
    n_val = max(1, n_val)
    return {"train": set(rest[n_val:]), "val": set(rest[:n_val]), "test": test_sids}


class CycloneSequenceDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, sids: set, tab_stats=None,
                 augment: bool = False, frame_dir=None):
        self.augment = augment
        self.frame_dir = frame_dir or config.FRAME_DIR
        self.man = manifest[manifest["SID"].isin(sids)].copy()
        self.man = self.man.sort_values(["SID", "ISO_TIME"])
        # reindex so positional lookups are cheap
        self.man = self.man.reset_index(drop=True)
        self.windows = build_windows(self.man)

        tab = self.man[config.TAB_COLS].to_numpy(np.float32)
        if tab_stats is None:                       # compute on TRAIN only
            self.mu, self.sd = tab.mean(0), tab.std(0) + 1e-6
        else:
            self.mu, self.sd = tab_stats
        self.tab = (tab - self.mu) / self.sd
        self.target = self.man[config.TARGET].to_numpy(np.float32)
        self.paths = self.man["frame_path"].to_numpy()

    def stats(self):
        return self.mu, self.sd

    def window_targets(self) -> np.ndarray:
        """Wind speed at the last frame of every window - used for sampling."""
        return np.array([self.target[w[-1]] for w in self.windows], dtype=np.float32)

    def sample_weights(self, bin_kt: float = 10.0, power: float = 0.5) -> np.ndarray:
        """Inverse-frequency weights so rare intensities get drawn more often.

        Bin the targets by intensity, then weight each window by
        1 / count(bin) ** power. power=0 is no rebalancing, power=1 is full
        rebalancing (every intensity equally likely, which over-corrects and
        makes training noisy). 0.5 is the usual compromise.
        """
        y = self.window_targets()
        bins = np.floor(y / bin_kt).astype(int)
        counts = pd.Series(bins).map(pd.Series(bins).value_counts()).to_numpy()
        w = 1.0 / np.power(counts, power)
        return w / w.mean()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        imgs = np.stack([np.load(self.frame_dir / self.paths[j]) for j in w])

        # Augmentation: rotate the WHOLE sequence by the same multiple of 90
        # degrees. A cyclone has no preferred orientation on the grid, so this
        # is physically harmless - but never mirror-flip, because that would
        # reverse the storm's rotation direction.
        if self.augment:
            k = np.random.randint(4)
            if k:
                imgs = np.rot90(imgs, k, axes=(1, 2)).copy()

        imgs = torch.from_numpy(imgs).unsqueeze(1)          # (T, 1, H, W)
        last = w[-1]
        return {
            "images": imgs,
            "tab": torch.from_numpy(self.tab[last]),
            "target": torch.tensor(self.target[last]),
            "sid": self.man.at[last, "SID"],
            "time": str(self.man.at[last, "ISO_TIME"]),
        }


def make_datasets():
    man = pd.read_csv(config.MANIFEST, parse_dates=["ISO_TIME"])
    sp = split_storms(man)
    train = CycloneSequenceDataset(man, sp["train"], augment=config.AUGMENT)
    val = CycloneSequenceDataset(man, sp["val"], train.stats())
    test = CycloneSequenceDataset(man, sp["test"], train.stats())
    print(f"storms  train {len(sp['train'])} | val {len(sp['val'])} | test {len(sp['test'])}")
    print(f"windows train {len(train)} | val {len(val)} | test {len(test)}")
    return train, val, test


if __name__ == "__main__":
    tr, va, te = make_datasets()
    b = tr[0]
    print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in b.items()})


def make_global_datasets(val_fraction: float = 0.05):
    """Train/val split over the other-basin pre-training set.

    No test split here: this set exists only to learn a good visual
    representation. All reporting happens on the North Indian Ocean data.
    """
    man = pd.read_csv(config.GLOBAL_MANIFEST, parse_dates=["ISO_TIME"])
    sids = np.array(sorted(man["SID"].unique()))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(sids)
    n_val = max(1, int(len(sids) * val_fraction))

    kw = dict(frame_dir=config.GLOBAL_FRAME_DIR)
    train = CycloneSequenceDataset(man, set(sids[n_val:]), augment=config.AUGMENT, **kw)
    val = CycloneSequenceDataset(man, set(sids[:n_val]), train.stats(), **kw)
    print(f"global storms  train {len(sids)-n_val} | val {n_val}")
    print(f"global windows train {len(train)} | val {len(val)}")
    return train, val
