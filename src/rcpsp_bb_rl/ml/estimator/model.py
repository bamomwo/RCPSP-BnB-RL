"""
Search-effort estimator: MLP that maps an instance feature vector x(I) to
y = log(1 + N_baseline(I)), the log node count at the locked PPO episode horizon.

The estimator exists to enable dynamic, per-instance reward scaling in PPO. From
its prediction N_hat = exp(y_hat) - 1 we derive the node-cost coefficient
    alpha(I) = C_target / N_hat
so that easy instances (small trees) get a larger per-node penalty and hard
instances (large trees) get a smaller one — keeping the return magnitude
comparable across instances and stabilising the critic.

Objective: the estimator does not need a precise N_hat, only one accurate enough
that alpha lands in a usable range. The requirement is a MULTIPLICATIVE BAND

    L <= N_hat / N <= U

which, because the target is already in log space, is a plain interval on the
residual d = y_hat - y ~= log(N_hat / N):  log L <= d <= log U.

The band is not an arbitrary tolerance. Realized node cost over an episode is
alpha * N = C_target / (N_hat / N), so ratio in [L, U] is exactly the statement
that the node-cost channel lands within [C_target/U, C_target/L]. Picking
L = 0.5, U = 2.0 says "within 2x either side of C_target".

Loss: asymmetric squared-hinge band loss (see band_loss). Both directions of
violation are real but unequal failures:
  - N_hat too HIGH  -> alpha too small -> the efficiency signal dies on that
    instance.
  - N_hat too LOW   -> alpha too large -> the node-cost term drowns the
    incumbent/bound signals, which is the return blow-up dynamic scaling exists
    to prevent.
w_under > w_over encodes that asymmetry in interpretable units, replacing the
quantile-tau trick this module previously used (which leaned the wrong way).

Everything needed for faithful inference — weights, the input standardiser
statistics, the band configuration, and the feature order — is persisted in a
single checkpoint so PPO can reload the estimator and reproduce training-time
normalisation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from rcpsp_bb_rl.data.parsing import RCPSPInstance
from rcpsp_bb_rl.ml.estimator.data.features import FEATURE_NAMES, NUM_FEATURES, extract_features


# ---------------------------------------------------------------------------
# Input standardiser (numpy-backed; no sklearn dependency)
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    """
    Zero-mean/unit-variance feature standardiser.

    Fit on the training split ONLY, then persisted in the checkpoint and reapplied
    identically at inference. Fitting on all data (or refitting at inference) leaks
    statistics and silently corrupts every prediction, so the fitted stats travel
    with the model.
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        # Guard constant columns: a zero std would divide to inf/nan.
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def band_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_lo: float,
    log_hi: float,
    w_under: float = 2.0,
    w_over: float = 1.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Asymmetric squared-hinge loss on the log-residual d = pred - target.

    Zero penalty while d is inside [log_lo, log_hi] (i.e. N_hat/N inside the
    acceptable band); quadratically growing penalty outside it:

        over  = relu(d - log_hi)   -> N_hat too high, weighted by w_over
        under = relu(log_lo - d)   -> N_hat too low,  weighted by w_under
        loss  = mean(w_over * over^2 + w_under * under^2)

    Design notes:
      - HINGE, so a sample already inside the band contributes no gradient. All
        capacity goes to violators instead of buying precision the reward cannot
        use.
      - SQUARED rather than linear outside the band, so a 10x violation pulls
        harder than a 1.05x one (an L1 hinge pulls both equally), and the loss
        stays C^1 at the boundary. Early in training almost every sample is
        outside the band, so this does not starve gradient.
      - w_under > w_over because under-prediction inflates alpha and drowns the
        other reward channels — the more damaging of the two failures.

    The optional per-sample `weight` re-emphasises rows (e.g. by target
    magnitude). It is usually unnecessary here: the hinge is already
    self-focusing, since easy instances that land in-band drop out of the loss on
    their own. Returns the (weighted) mean over the batch.
    """
    d = pred - target
    over = torch.clamp(d - log_hi, min=0.0)
    under = torch.clamp(log_lo - d, min=0.0)
    per_sample = w_over * over.pow(2) + w_under * under.pow(2)
    if weight is not None:
        return torch.sum(weight * per_sample) / torch.sum(weight)
    return torch.mean(per_sample)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SearchEffortMLP(nn.Module):
    """
    MLP regressor: [NUM_FEATURES] -> 64 -> 32 -> 1.

    Hidden layers use LeakyReLU (guards against dead units) with dropout on the
    first hidden layer. The output is a single linear neuron predicting y in
    raw log-space (log(1 + N)); no output activation.
    """

    def __init__(
        self,
        in_dim: int = NUM_FEATURES,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.2,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_p = dropout
        self.negative_slope = negative_slope

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=self.negative_slope, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] -> y_hat: [B] (log-space)."""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_estimator_checkpoint(
    model: SearchEffortMLP,
    scaler: Standardizer,
    loss_config: dict,
    path: str,
    feature_names: Optional[List[str]] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Persist weights + scaler stats + loss config + feature order in one file.

    loss_config records the band the model was fitted against (band_lo, band_hi,
    w_under, w_over) so a checkpoint can be interpreted later without its config
    file — the numbers that define what "accurate enough" meant for this run.

    feature_names records the exact input contract so inference can detect a
    features.py drift (append/reorder) rather than silently misalign columns.
    """
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "in_dim": model.in_dim,
            "hidden1": model.hidden1,
            "hidden2": model.hidden2,
            "dropout": model.dropout_p,
            "negative_slope": model.negative_slope,
        },
        "scaler": scaler.to_dict(),
        "loss_config": dict(loss_config),
        "feature_names": list(feature_names) if feature_names is not None else list(FEATURE_NAMES),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_estimator_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[SearchEffortMLP, Standardizer, dict]:
    """
    Load (model, scaler, loss_config) from a checkpoint written by
    save_estimator_checkpoint.

    loss_config is metadata only — inference needs just the weights and scaler —
    but it is returned so callers can log or assert on the band the estimator was
    fitted against. Older checkpoints carrying a bare `tau` are reported as
    {"legacy_tau": ...} rather than silently presented as a band.
    """
    ckpt = torch.load(path, map_location=device)
    if "model_state" not in ckpt or "scaler" not in ckpt:
        raise ValueError(f"Checkpoint at {path} is not a valid estimator checkpoint.")

    cfg = ckpt.get("config", {})
    model = SearchEffortMLP(
        in_dim=int(cfg.get("in_dim", NUM_FEATURES)),
        hidden1=int(cfg.get("hidden1", 64)),
        hidden2=int(cfg.get("hidden2", 32)),
        dropout=float(cfg.get("dropout", 0.2)),
        negative_slope=float(cfg.get("negative_slope", 0.01)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Feature-contract guard: a checkpoint trained on a different feature set
    # would misalign columns at inference. Fail loud rather than predict garbage.
    saved_names = ckpt.get("feature_names")
    if saved_names is not None and list(saved_names) != list(FEATURE_NAMES):
        raise ValueError(
            "Estimator checkpoint feature order does not match the current "
            "features.py FEATURE_NAMES. Retrain the estimator against the "
            "updated features."
        )

    scaler = Standardizer.from_dict(ckpt["scaler"])
    if "loss_config" in ckpt:
        loss_config = dict(ckpt["loss_config"])
    elif "tau" in ckpt:
        loss_config = {"legacy_tau": float(ckpt["tau"])}
    else:
        loss_config = {}
    return model, scaler, loss_config


# ---------------------------------------------------------------------------
# Inference: the single call PPO makes per instance
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_log_effort(
    model: SearchEffortMLP,
    scaler: Standardizer,
    features: np.ndarray,
    device: torch.device | str = "cpu",
) -> float:
    """Predict y_hat = log(1 + N_hat) from a raw (unscaled) feature vector."""
    x = scaler.transform(np.asarray(features, dtype=np.float64).reshape(1, -1))
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_hat = float(model(xt).item())
    return y_hat


@torch.no_grad()
def predict_difficulty(
    model: SearchEffortMLP,
    scaler: Standardizer,
    instance: RCPSPInstance,
    device: torch.device | str = "cpu",
    min_nodes: float = 1.0,
) -> float:
    """
    Estimate baseline node count N_hat for an instance.

    Extracts the 18 static features, applies the persisted scaler, runs the MLP,
    and inverts the log target: N_hat = exp(y_hat) - 1, clamped to >= min_nodes so
    the downstream alpha = C_target / N_hat can never divide by ~0.
    """
    feats = extract_features(instance)
    y_hat = predict_log_effort(model, scaler, feats, device=device)
    n_hat = float(np.expm1(y_hat))
    return max(min_nodes, n_hat)
