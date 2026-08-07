"""
Train the search-effort estimator.

Learns the regressor x(I) -> y = log(1 + N_baseline(I)) that PPO uses for
dynamic per-instance reward scaling (alpha(I) = C_target / N_hat). The loss is
pinball (quantile) loss at a low tau so the model leans toward UNDER-predicting
N: over-predicting N shrinks alpha and kills the efficiency signal, the more
dangerous failure. See src/rcpsp_bb_rl/ml/estimator/model.py for the rationale.

Reports metrics in two spaces every epoch:
  - log-space:  pinball loss + MAE on y = log(1+N)
  - node-space: median ratio N_hat/N and the OVER-PREDICTION RATE
                (fraction with N_hat > N) — the real success metric. tau is
                tuned to keep this modest (e.g. <= ~0.30).

Example:
  python3 scripts/train_estimator.py --config config/train_estimator.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim

# This file now lives at src/rcpsp_bb_rl/ml/estimator/; the repo root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.ml.estimator import (  # noqa: E402
    SearchEffortMLP,
    Standardizer,
    load_estimator_dataset,
    pinball_loss,
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
    # Loss — tau < 0.5 penalises over-prediction (1-tau)/tau harder.
    "tau": 0.25,
    # Sample weighting: when true, weight each training row's loss by its target
    # magnitude (w = y / mean(y)) so hard instances aren't swamped by the easy
    # crowd. Off by default (uniform weighting = original behaviour).
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


def node_space_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Diagnostics in node space (invert y = log(1+N)).

    over_pred_rate is the fraction of instances where N_hat > N — the failure the
    pinball tau is meant to suppress. median_ratio is the typical N_hat/N; a value
    slightly below 1.0 is the intended mild low-bias.
    """
    n_true = np.expm1(y_true)
    n_pred = np.expm1(y_pred)
    n_true_safe = np.maximum(n_true, 1.0)
    ratio = np.maximum(n_pred, 0.0) / n_true_safe
    over = float(np.mean(n_pred > n_true))
    return {
        "over_pred_rate": over,
        "median_ratio": float(np.median(ratio)),
        "mean_ratio": float(np.mean(ratio)),
    }


def evaluate(
    model: SearchEffortMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    tau: float,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = pinball_loss(pred, y, tau).item()
        mae = torch.mean(torch.abs(pred - y)).item()
    metrics = {"pinball": loss, "mae_log": mae}
    metrics.update(node_space_metrics(y.cpu().numpy(), pred.cpu().numpy()))
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
    split = load_estimator_dataset(
        config["data_csv"], val_frac=float(config["val_frac"]), seed=int(config["seed"])
    )
    print(f"Loaded {config['data_csv']}: "
          f"{len(split.y_train)} train / {len(split.y_val)} val  "
          f"({len(split.feature_names)} features)")

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
    tau = float(config["tau"])
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}  "
          f"tau={tau} (over-prediction penalised {(1 - tau) / tau:.1f}x)")

    # --- Per-sample training weights ---
    # Weighting emphasises the hard-instance tail during training. Eval loss is
    # always left UNWEIGHTED so val_pinball stays directly comparable to the
    # uniform-weighting baseline run.
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

    # --- Training loop with early stopping on val pinball loss ---
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    min_epochs = int(config["min_epochs"])
    n_train = x_train.shape[0]

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    print(f"\n{'='*84}")
    print(f"  Training estimator  epochs={epochs}  batch={batch_size}  patience={patience}")
    print(f"{'='*84}")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start: start + batch_size]
            pred = model(x_train[idx])
            w = train_weights[idx] if train_weights is not None else None
            loss = pinball_loss(pred, y_train[idx], tau, weight=w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val = evaluate(model, x_val, y_val, tau)

        improved = val["pinball"] < best_val
        if improved:
            best_val = val["pinball"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_estimator_checkpoint(
                model, scaler, tau, str(save_path),
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
                f"train_pinball={train_loss:.4f}  "
                f"val_pinball={val['pinball']:.4f}  "
                f"val_mae_log={val['mae_log']:.4f}  "
                f"over_pred_rate={val['over_pred_rate']*100:.1f}%  "
                f"median_ratio={val['median_ratio']:.3f}"
                f"{'  [best]' if improved else ''}"
            )

        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    print(f"\n{'='*84}")
    print(f"  Training complete")
    print(f"  best val_pinball={best_val:.4f} at epoch {best_epoch}")
    print(f"  best model -> {save_path}")
    print(f"{'='*84}")


if __name__ == "__main__":
    main()
