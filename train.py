"""
train.py - train the model.

Run:  python train.py                 # uses everything in config.py
      python train.py --epochs 3      # quick smoke test
      python train.py --baseline      # image-only model, no tabular, no attention

Saves the best weights to checkpoints/best.pt (best = lowest VALIDATION RMSE,
not training loss - training loss always goes down and tells you nothing).
"""

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset import make_datasets
from model import MultiModalCycloneModel, WeightedMSE


def rmse_mae(pred, true):
    err = pred - true
    return float(np.sqrt((err ** 2).mean())), float(np.abs(err).mean())


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    P, T = [], []
    for b in loader:
        p = model(b["images"].to(device), b["tab"].to(device))
        P.append(p.cpu().numpy()); T.append(b["target"].numpy())
    return np.concatenate(P), np.concatenate(T)


def main(args):
    # NOTE: config.SEED controls the train/val/test SPLIT and must stay fixed.
    # args.seed controls weight init, sampling and augmentation only, so
    # repeated runs measure training variance on identical data.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, val_ds, test_ds = make_datasets()

    # Intensity-balanced sampling. Cat 3+ is ~4.5% of the data, so without
    # this the model almost never sees a severe storm and learns to play safe
    # by predicting the mean. Weighting the LOSS is not enough on its own -
    # we have to change how often severe storms appear in a batch.
    if args.balance > 0:
        w = train_ds.sample_weights(power=args.balance)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, drop_last=True)
        y = train_ds.window_targets()
        eff = float(np.average(y, weights=w))
        print(f"balanced sampling (power={args.balance}): mean intensity seen "
              f"{y.mean():.1f} -> {eff:.1f} kt; "
              f"Cat3+ share {100*(y>=config.CAT3_KT).mean():.1f}% -> "
              f"{100*np.average(y >= config.CAT3_KT, weights=w):.1f}%")
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)

    dl = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers)
    val_dl, test_dl = dl(val_ds), dl(test_ds)

    model = MultiModalCycloneModel(use_images=not args.no_images,
                                   use_tab=not args.no_tab).to(device)

    if args.init_from:
        # Stage 2 of transfer learning: start from weights learned on other
        # basins instead of from random initialisation.
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        n = sum(1 for _ in model.state_dict())
        print(f"initialised from {args.init_from} "
              f"({n - len(missing)}/{n} tensors loaded)")
        if args.freeze_cnn:
            for p_ in model.cnn.parameters():
                p_.requires_grad = False
            print("CNN backbone frozen - only the temporal, attention, "
                  "tabular and head layers will train")
    mode = ("images+tabular" if not (args.no_images or args.no_tab)
            else "tabular only" if args.no_images else "images only")
    print(f"modality: {mode}")
    loss_fn = (torch.nn.MSELoss() if args.plain_mse
               else WeightedMSE(high_weight=args.high_weight))
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                            lr=args.lr,
                            weight_decay=config.WEIGHT_DECAY)

    steps_per_epoch = max(1, len(train_dl))
    warmup = config.WARMUP_EPOCHS * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_at(step):                       # linear warm-up then cosine decay
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best, bad_epochs, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for b in train_dl:
            opt.zero_grad()
            pred = model(b["images"].to(device), b["tab"].to(device))
            loss = loss_fn(pred, b["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            running += loss.item()

        vp, vt = run_eval(model, val_dl, device)
        v_rmse, v_mae = rmse_mae(vp, vt)
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch,
                        "val_rmse": v_rmse, "val_mae": v_mae})
        print(f"epoch {epoch:3d} | loss {running/steps_per_epoch:8.2f} | "
              f"val RMSE {v_rmse:6.2f} kt | val MAE {v_mae:6.2f} kt | "
              f"{time.time()-t0:5.1f}s")

        if v_rmse < best - 0.05:
            best, bad_epochs = v_rmse, 0
            torch.save({"model": model.state_dict(),
                        "tab_mu": train_ds.mu, "tab_sd": train_ds.sd,
                        "val_rmse": best}, config.CKPT_DIR / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                print(f"early stop: no improvement for {config.PATIENCE} epochs")
                break

    (config.CKPT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    # ---- final numbers on the held-out storms -------------------------
    ckpt = torch.load(config.CKPT_DIR / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    tp, tt = run_eval(model, test_dl, device)
    t_rmse, t_mae = rmse_mae(tp, tt)

    # the dumbest possible baseline: always guess the training mean.
    # If your model cannot beat this, something is wrong.
    naive = np.full_like(tt, train_ds.target.mean())
    n_rmse, _ = rmse_mae(naive, tt)

    cat3 = tt >= config.CAT3_KT
    c_rmse, c_mae = rmse_mae(tp[cat3], tt[cat3]) if cat3.any() else (float("nan"),) * 2
    c_bias = float((tp[cat3] - tt[cat3]).mean()) if cat3.any() else float("nan")

    print("\n" + "=" * 52)
    print(f"best val RMSE : {ckpt['val_rmse']:.2f} kt")
    print(f"TEST RMSE     : {t_rmse:.2f} kt")
    print(f"TEST MAE      : {t_mae:.2f} kt")
    print(f"naive baseline: {n_rmse:.2f} kt  (predict the mean every time)")
    print(f"Cat3+ RMSE    : {c_rmse:.2f} kt   bias {c_bias:+.2f} kt  (n={cat3.sum()})")
    print("=" * 52)
    # one machine-readable line for repeat_runs.py
    print(f"RESULT seed={args.seed} val_rmse={ckpt['val_rmse']:.3f} "
          f"test_rmse={t_rmse:.3f} test_mae={t_mae:.3f} "
          f"cat3_rmse={c_rmse:.3f} cat3_bias={c_bias:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--init-from", default=None,
                   help="path to a pre-trained checkpoint (transfer learning)")
    p.add_argument("--freeze-cnn", action="store_true",
                   help="freeze the pre-trained CNN backbone during fine-tuning")
    p.add_argument("--seed", type=int, default=config.SEED,
                   help="training seed (the data split is NOT affected)")
    p.add_argument("--no-images", action="store_true",
                   help="ablation: blind the model, tabular features only")
    p.add_argument("--no-tab", action="store_true",
                   help="ablation: imagery only, no tabular features")
    p.add_argument("--balance", type=float, default=0.5,
                   help="intensity-rebalancing strength, 0 = off, 1 = full")
    p.add_argument("--high-weight", type=float, default=config.HIGH_WEIGHT,
                   help="loss multiplier for Cat 3+ errors")
    p.add_argument("--plain-mse", action="store_true",
                   help="ablation: use ordinary MSE instead of weighted MSE")
    main(p.parse_args())
