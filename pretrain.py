"""
Stage 1 of transfer learning: pre-train on other basins.

Trains the identical architecture on the global (non-North-Indian) dataset,
then saves the weights. train.py --init-from picks them up for stage 2.

Run:  python pretrain.py --epochs 25
Makes: checkpoints/pretrained.pt

Note the tabular normalisation statistics are NOT reused downstream: the
global set has a different position distribution, so fine-tuning recomputes
them on North Indian Ocean data. Only the learned weights transfer.
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset import make_global_datasets
from model import MultiModalCycloneModel, WeightedMSE
from train import rmse_mae, run_eval


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, val_ds = make_global_datasets()

    if args.balance > 0:
        w = train_ds.sample_weights(power=args.balance)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.workers)

    model = MultiModalCycloneModel().to(device)
    loss_fn = WeightedMSE()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=config.WEIGHT_DECAY)
    steps = max(1, len(train_dl))
    warm = config.WARMUP_EPOCHS * steps
    total = args.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for b in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(b["images"].to(device), b["tab"].to(device)),
                           b["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            running += loss.item()

        vp, vt = run_eval(model, val_dl, device)
        v_rmse, v_mae = rmse_mae(vp, vt)
        print(f"epoch {epoch:3d} | loss {running/steps:8.2f} | "
              f"val RMSE {v_rmse:6.2f} kt | val MAE {v_mae:6.2f} kt | "
              f"{time.time()-t0:5.1f}s")

        if v_rmse < best:
            best = v_rmse
            torch.save({"model": model.state_dict(), "val_rmse": best,
                        "source": "global pre-training"},
                       config.CKPT_DIR / "pretrained.pt")

    print(f"\nBest global val RMSE {best:.2f} kt")
    print(f"Saved {config.CKPT_DIR / 'pretrained.pt'}")
    print("\nNext:  python train.py --init-from checkpoints/pretrained.pt "
          "--epochs 60 --balance 0.5")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--balance", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=config.SEED)
    main(p.parse_args())
