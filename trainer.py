"""
trainer.py
==========

Training loop. Two things worth knowing:

1. Every epoch reports RMSE and MAE in keV, not in standardised units. Because
   the offset cancels in a difference,

       pred_keV - act_keV = (pred_std - act_std) * target_std

   so the conversion is exact and costs nothing.

2. Early stopping runs on an exponentially smoothed validation loss. The raw
   validation curve on this dataset is noisy enough that unsmoothed patience
   stops training 20-30 epochs too early.
"""

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, value):
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def _epoch_stats(sq_sum, abs_sum, n, target_std):
    rmse = np.sqrt(sq_sum / max(n, 1)) * target_std
    mae = (abs_sum / max(n, 1)) * target_std
    return float(rmse), float(mae)


def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler,
                checkpoint_path, num_epochs, target_std, ema_alpha=0.3,
                patience=30, verbose_every=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logs = {"train_losses": [], "val_losses": [],
            "train_rmse_keV": [], "val_rmse_keV": [],
            "train_mae_keV": [], "val_mae_keV": []}

    best_val = float("inf")
    ema = None
    stopper = EarlyStopping(patience=patience)

    print(f"\n   {'epoch':>6}{'train loss':>13}{'val loss':>11}"
          f"{'train RMSE':>13}{'val RMSE':>12}{'val MAE':>11}   (keV)")
    print("   " + "-" * 70)

    for epoch in range(1, num_epochs + 1):
        # ---------------------------------------------------------- train
        model.train()
        tot, sq, ab, n = 0.0, 0.0, 0.0, 0
        for cat, cont, y in train_loader:
            cat, cont, y = cat.to(device), cont.to(device), y.to(device)
            optimizer.zero_grad()
            pred, _ = model(cat, cont)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tot += loss.item()
            d = (pred - y).detach()
            sq += float((d ** 2).sum()); ab += float(d.abs().sum()); n += y.numel()

        tr_loss = tot / max(len(train_loader), 1)
        tr_rmse, tr_mae = _epoch_stats(sq, ab, n, target_std)

        # ------------------------------------------------------------ val
        model.eval()
        tot, sq, ab, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for cat, cont, y in val_loader:
                cat, cont, y = cat.to(device), cont.to(device), y.to(device)
                pred, _ = model(cat, cont)
                tot += criterion(pred, y).item()
                d = pred - y
                sq += float((d ** 2).sum()); ab += float(d.abs().sum()); n += y.numel()

        va_loss = tot / max(len(val_loader), 1)
        va_rmse, va_mae = _epoch_stats(sq, ab, n, target_std)

        logs["train_losses"].append(tr_loss); logs["val_losses"].append(va_loss)
        logs["train_rmse_keV"].append(tr_rmse); logs["val_rmse_keV"].append(va_rmse)
        logs["train_mae_keV"].append(tr_mae); logs["val_mae_keV"].append(va_mae)

        ema = va_loss if ema is None else ema_alpha * va_loss + (1 - ema_alpha) * ema
        scheduler.step(ema)
        stopper(ema)

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"   {epoch:>6}{tr_loss:>13.5f}{va_loss:>11.5f}"
                  f"{tr_rmse:>13.1f}{va_rmse:>12.1f}{va_mae:>11.1f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"          checkpoint saved (val loss {best_val:.5f}, "
                  f"val RMSE {va_rmse:,.1f} keV)")

        if stopper.early_stop:
            print(f"   early stopping at epoch {epoch}")
            break

    return model, logs
