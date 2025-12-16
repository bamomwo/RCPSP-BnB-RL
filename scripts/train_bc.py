import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader, random_split

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.featurize import collate_state_batch  # noqa: E402
from rcpsp_bb_rl.data.trajectory_dataset import load_trajectories  # noqa: E402
from rcpsp_bb_rl.models import PolicyMLP  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised imitation (behavior cloning) policy MLP.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file to override defaults.",
    )
    parser.add_argument(
        "--trajectories-dir",
        default="data/trajectories",
        help="Directory containing trajectory JSONL files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern for trajectory files (default: *.jsonl).",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer sizes for the policy MLP.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of data for validation.")
    parser.add_argument("--max-resources", type=int, default=4, help="Pad/truncate resource dimensions to this size.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "models" / "policy.pt"),
        help="Path to save the trained model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def parse_args_from_defaults() -> argparse.Namespace:
    """Helper to capture argparse defaults without CLI overrides."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--trajectories-dir", default="data/trajectories")
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "policy.pt"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args([])


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    trajectories_dir: Path,
    pattern: str,
    batch_size: int,
    val_frac: float,
    max_resources: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    dataset = load_trajectories(trajectories_dir, pattern=pattern, flatten_ready=False)

    val_size = max(1, int(len(dataset) * val_frac))
    train_size = max(len(dataset) - val_size, 1)
    train_ds, val_ds = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    collate = lambda batch: collate_state_batch(batch, max_resources=max_resources)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def evaluate(model: PolicyMLP, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch["lengths"].numel() == 0:
                continue
            cand = batch["candidate_feats"].to(device)
            glob = batch["global_feats"].to(device)
            lengths = batch["lengths"]
            targets = batch["targets"].to(device)
            logits = model(cand, glob)
            loss = model.listwise_nll(logits, lengths, targets)
            acc = model.listwise_accuracy(logits, lengths, targets)
            total_loss += loss.item()
            total_acc += acc.item()
            total_batches += 1
    if total_batches == 0:
        return 0.0, 0.0
    return total_loss / total_batches, total_acc / total_batches


def main() -> None:
    args = parse_args()
    defaults = parse_args_from_defaults()
    config = vars(defaults).copy()

    if args.config:
        cfg_path = Path(args.config)
        file_cfg = load_config(cfg_path)
        config.update(file_cfg)

    # CLI overrides config/defaults when differing from defaults.
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v != getattr(defaults, k):
            config[k] = v

    set_seed(int(config["seed"]))

    trajectories_dir = Path(config["trajectories_dir"])
    train_loader, val_loader = make_loaders(
        trajectories_dir=trajectories_dir,
        pattern=config["pattern"],
        batch_size=int(config["batch_size"]),
        val_frac=float(config["val_frac"]),
        max_resources=int(config["max_resources"]),
        seed=int(config["seed"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Peek a batch to infer feature dims.
    sample_batch = next(iter(train_loader))
    global_dim = sample_batch["global_feats"].shape[1]
    candidate_dim = sample_batch["candidate_feats"].shape[1]

    model = PolicyMLP(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        hidden_sizes=config["hidden_sizes"],
        dropout=float(config["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))

    best_val_loss = float("inf")
    best_val_acc = 0.0
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0
        for batch in train_loader:
            if batch["lengths"].numel() == 0:
                continue
            cand = batch["candidate_feats"].to(device)
            glob = batch["global_feats"].to(device)
            lengths = batch["lengths"]
            targets = batch["targets"].to(device)

            logits = model(cand, glob)
            loss = model.listwise_nll(logits, lengths, targets)
            acc = model.listwise_accuracy(logits, lengths, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        train_acc = epoch_acc / max(batches, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "global_dim": global_dim,
                        "candidate_dim": candidate_dim,
                        "hidden_sizes": [int(h) for h in config["hidden_sizes"]],
                        "dropout": float(config["dropout"]),
                    },
                },
                output_path,
            )

    print(f"Done. Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}. Saved to {output_path}")


if __name__ == "__main__":
    main()
