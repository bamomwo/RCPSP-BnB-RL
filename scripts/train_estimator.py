"""
Train the search-effort estimator.

Learns the regressor x(I) -> y = log(1 + N_baseline(I)) that PPO uses for dynamic
per-instance reward scaling (alpha(I) = C_target / N_hat).

The estimator is not asked for a precise N_hat, only for one accurate enough that
alpha lands in a usable range. That requirement is a multiplicative BAND,
band_lo <= N_hat/N <= band_hi, and the whole script is organised around it: the
loss is a hinge that is flat inside the band (so capacity goes to violators
instead of buying unusable precision) and the reported metrics are band membership
split by direction. Checkpoint selection uses the val band LOSS, not the hit rate:
the hit rate is a step function of the predictions, so it ties and plateaus, while
the loss still ranks two checkpoints that are both outside the band but at
different distances from it. See src/rcpsp_bb_rl/ml/estimator/model.py for the
loss rationale and metrics.py for why the breakdowns are stratified.

Per-epoch log: train/val band loss, pooled hit rate, and the two directional
violation rates. The stratified breakdown (by regime and by target decile) is
printed at the end, where it is read — a pooled hit rate hides failure on the
hard tail, which is where PPO spends its time.

Example:
  python3 scripts/train_estimator.py --config config/train_estimator.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.ml.estimator import (  # noqa: E402
    SearchEffortMLP,
    Standardizer,
    band_loss,
    band_metrics,
    format_band_report,
    load_estimator_dataset,
    save_estimator_checkpoint,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    # Data
    "data_csv": "data/estimator/data-120/data.csv",
    "val_frac": 0.2,
    # Model
    "hidden1": 64,
    "hidden2": 32,
    "dropout": 0.2,
    "negative_slope": 0.01,
    # Objective: the multiplicative band on N_hat/N the estimator is judged by.
    # [0.5, 2.0] = "within 2x either side", which bounds the realized node cost
    # alpha*N = c_target/ratio to [c_target/2, 2*c_target].
    "band_lo": 0.5,
    "band_hi": 2.0,
    # Asymmetry: under-prediction inflates alpha and drowns the incumbent/bound
    # channels, so it is penalised harder than over-prediction.
    "w_under": 2.0,
    "w_over": 1.0,
    # Only used to report the realized cost spread in node-cost units; does not
    # affect the fit.
    "c_target": 1.0,
    # Split: stratify by (regime, target decile) so the sparse hard deciles are
    # represented in val at their population rate.
    "stratify_split": True,
    # Sample weighting: when true, weight each training row's loss by its target
    # magnitude (w = y / mean(y)). Usually unnecessary — the hinge already drops
    # in-band rows from the loss, so it self-focuses on violators.
    "weight_by_target": False,
    # Optimisation
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "epochs": 300,
    "patience": 30,
    "min_epochs": 20,
    # Output
    "save_path": "models/estimator/estimator.pt",
    "seed": 42,
    "device": "cpu",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the RCPSP search-effort estimator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to JSON config file.")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: SearchEffortMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    band: Dict[str, float],
    solved: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Band loss plus the full band metric set for one split.

    The loss here is always UNWEIGHTED, even when training uses per-sample weights,
    so the number used for model selection means the same thing across runs.
    """
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = band_loss(
            pred, y,
            log_lo=band["log_lo"], log_hi=band["log_hi"],
            w_under=band["w_under"], w_over=band["w_over"],
        ).item()

    metrics: Dict[str, Any] = {"band_loss": loss}
    metrics.update(
        band_metrics(
            pred.cpu().numpy(), y.cpu().numpy(),
            solved=solved,
            band_lo=band["band_lo"], band_hi=band["band_hi"],
            c_target=band["c_target"],
        )
    )
    return metrics


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config.update(load_json(Path(args.config)))

    set_seed(int(config["seed"]))
    device = torch.device(
        "cpu" if (config["device"] == "cuda" and not torch.cuda.is_available())
        else config["device"]
    )
    print(f"Device: {device}")

    # --- Data ---
    stratify = bool(config["stratify_split"])
    split = load_estimator_dataset(
        config["data_csv"],
        val_frac=float(config["val_frac"]),
        seed=int(config["seed"]),
        stratify=stratify,
    )
    n_censored_val = int((split.solved_val == 0.0).sum())
    print(f"Loaded {config['data_csv']}: "
          f"{len(split.y_train)} train / {len(split.y_val)} val  "
          f"({len(split.feature_names)} features)")
    print(f"Split: {'stratified by (regime, target decile)' if stratify else 'random permutation'}"
          f"  |  val regimes: {len(split.y_val) - n_censored_val} exhausted / "
          f"{n_censored_val} censored")

    # --- Scaler: fit on TRAIN ONLY, then apply to both splits ---
    scaler = Standardizer.fit(split.x_train)
    x_train = torch.as_tensor(scaler.transform(split.x_train), dtype=torch.float32, device=device)
    y_train = torch.as_tensor(split.y_train, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(scaler.transform(split.x_val), dtype=torch.float32, device=device)
    y_val = torch.as_tensor(split.y_val, dtype=torch.float32, device=device)

    # --- Model ---
    model = SearchEffortMLP(
        in_dim=len(split.feature_names),
        hidden1=int(config["hidden1"]),
        hidden2=int(config["hidden2"]),
        dropout=float(config["dropout"]),
        negative_slope=float(config["negative_slope"]),
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    # Band config, resolved once. log_lo/log_hi are the loss's actual thresholds
    # on the residual d = y_hat - y; band_lo/band_hi stay in ratio units for
    # reporting and for the checkpoint's loss_config.
    band_lo = float(config["band_lo"])
    band_hi = float(config["band_hi"])
    if not (0.0 < band_lo < 1.0 <= band_hi):
        raise ValueError(
            f"Expected 0 < band_lo < 1 <= band_hi; got [{band_lo}, {band_hi}]. "
            f"The band must straddle a ratio of 1.0."
        )
    band = {
        "band_lo": band_lo,
        "band_hi": band_hi,
        "log_lo": math.log(band_lo),
        "log_hi": math.log(band_hi),
        "w_under": float(config["w_under"]),
        "w_over": float(config["w_over"]),
        "c_target": float(config["c_target"]),
    }
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Objective: band [{band_lo:g}, {band_hi:g}] on N_hat/N  "
          f"(w_under={band['w_under']:g}, w_over={band['w_over']:g})")

    # --- Per-sample training weights ---
    # Weighting emphasises the hard-instance tail during training. Eval loss is
    # always left UNWEIGHTED so val_band_loss stays comparable across runs.
    weight_by_target = bool(config["weight_by_target"])
    if weight_by_target:
        w_np = split.y_train / max(split.y_train.mean(), 1e-8)
        train_weights = torch.as_tensor(w_np, dtype=torch.float32, device=device)
        print(f"Sample weighting: ON (w = y / mean(y), range "
              f"[{w_np.min():.2f}, {w_np.max():.2f}])")
    else:
        train_weights = None
        print("Sample weighting: OFF (uniform)")

    save_path = Path(config["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Training loop with early stopping on val band loss ---
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    min_epochs = int(config["min_epochs"])
    n_train = x_train.shape[0]

    best_val = float("inf")
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    epochs_no_improve = 0

    print(f"\n{'='*96}")
    print(f"  Training estimator  epochs={epochs}  batch={batch_size}  patience={patience}")
    print(f"{'='*96}")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start: start + batch_size]
            pred = model(x_train[idx])
            w = train_weights[idx] if train_weights is not None else None
            loss = band_loss(
                pred, y_train[idx],
                log_lo=band["log_lo"], log_hi=band["log_hi"],
                w_under=band["w_under"], w_over=band["w_over"],
                weight=w,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val = evaluate(model, x_val, y_val, band, solved=split.solved_val)


        #Diagnostic
        train_eval = evaluate(model,x_train,y_train,band,solved=split.solved_train)

        # Selection on val band_loss rather than on hit rate: the hit rate is a
        # step function of the predictions, so it plateaus and ties constantly,
        # while the loss keeps ranking checkpoints that are still moving toward
        # the band. Both are logged.
        improved = val["band_loss"] < best_val
        if improved:
            best_val = val["band_loss"]
            best_epoch = epoch
            best_metrics = val
            epochs_no_improve = 0
            save_estimator_checkpoint(
                model, scaler,
                loss_config={
                    "band_lo": band["band_lo"],
                    "band_hi": band["band_hi"],
                    "w_under": band["w_under"],
                    "w_over": band["w_over"],
                    "c_target": band["c_target"],
                },
                path=str(save_path),
                feature_names=split.feature_names,
                extra={
                    "train_config": config,
                    "val_metrics": val,
                    "epoch": epoch,
                },
            )
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or improved or epoch == 1:
            print(
                f"[Epoch {epoch:4d}] "
                f"train_loss={train_loss:.4f}  "
                f"train_hit={train_eval['band_hit_rate']*100:5.1f}%  "
                f"val_loss={val['band_loss']:.4f}  "
                f"val_hit={val['band_hit_rate']*100:5.1f}%  "
                f"val_under={val['under_rate']*100:5.1f}%  "
                f"val_over={val['over_rate']*100:5.1f}%"
            )
            # print(
            #     f"[Epoch {epoch:4d}] "
            #     f"train={train_loss:.4f}  val={val['band_loss']:.4f}  "
            #     f"hit={val['band_hit_rate']*100:5.1f}%  "
            #     f"under={val['under_rate']*100:5.1f}%  "
            #     f"over={val['over_rate']*100:5.1f}%  "
            #     f"ratio_p50={val['ratio_p50']:.2f}  "
            #     f"mae_log={val['mae_log']:.3f}"
            #     f"{'  [best]' if improved else ''}"
            # )

        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    print(f"\n{'='*96}")
    print(f"  Training complete — best val_band_loss={best_val:.4f} at epoch {best_epoch}")
    print(f"{'='*96}")
    if best_metrics:
        print(format_band_report(best_metrics, band["band_lo"], band["band_hi"]))
    print(f"\n  best model -> {save_path}")
    print(f"{'='*96}")


if __name__ == "__main__":
    main()
