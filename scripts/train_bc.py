import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.parsing import RCPSPInstance, load_instance  # noqa: E402
from rcpsp_bb_rl.ml.il.featurize import (  # noqa: E402
    featurize_states,
    global_feature_dim,
    candidate_feature_dim,
)
from rcpsp_bb_rl.ml.il.generate_trajectories import TrajectoryRecord, _load_jsonl  # noqa: E402
from rcpsp_bb_rl.ml.models import BranchingTransformer, save_policy_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    # Data
    "trajectories_dir": "data/trajectories",
    "instances_dir": "data/j30h",
    "instances_pattern": "*.rcp",
    "pattern": "*.jsonl",
    "val_frac": 0.1,
    "max_resources": 4,
    # Model
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_dim": 128,
    "dropout": 0.1,
    # Training
    "batch_size": 64,
    "lr": 3e-4,
    "epochs": None,
    "patience": 5,
    "min_epochs": 0,
    "seed": 42,
    # Output
    "output": str(PROJECT_ROOT / "models" / "policy_bc.pt"),
}


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

InstancePair = Tuple[Path, Path]


def collect_instance_pairs(
    trajectories_dir: Path,
    instances_dir: Path,
    instances_pattern: str = "*.rcp",
    pattern: str = "*.jsonl",
) -> List[InstancePair]:
    """
    Match trajectory files to their source RCPSP instances.

    The returned list is the correct unit for train/validation splitting:
    one trajectory file corresponds to one instance and should stay wholly in
    either train or validation.
    """
    jsonl_paths = sorted(trajectories_dir.glob(pattern))
    if not jsonl_paths:
        raise FileNotFoundError(
            f"No trajectory files found under {trajectories_dir} matching {pattern}"
        )

    inst_lookup: Dict[str, Path] = {
        p.stem: p
        for p in instances_dir.rglob(instances_pattern)
    }

    pairs: List[InstancePair] = []
    missing = []
    for jsonl_path in jsonl_paths:
        inst_path = inst_lookup.get(jsonl_path.stem)
        if inst_path is None:
            missing.append(jsonl_path.stem)
            continue
        pairs.append((jsonl_path, inst_path))

    if missing:
        print(
            f"Warning: no instance file found for {len(missing)} trajectory file(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if not pairs:
        raise RuntimeError(
            "No matched trajectory-instance pairs found — check trajectories_dir and instances_dir."
        )

    return pairs


# ---------------------------------------------------------------------------
# Dataset — pairs each TrajectoryRecord with its RCPSPInstance
# ---------------------------------------------------------------------------

class InstanceTrajectoryDataset(Dataset):
    """
    Loads JSONL trajectory files from a pre-split list of matched
    (trajectory_path, instance_path) pairs.
    """

    def __init__(
        self,
        instance_pairs: List[InstancePair],
    ) -> None:
        self._items: List[Tuple[TrajectoryRecord, RCPSPInstance]] = []
        self.instance_pairs = list(instance_pairs)

        for jsonl_path, inst_path in self.instance_pairs:
            instance = load_instance(inst_path)
            for raw in _load_jsonl(jsonl_path):
                self._items.append((TrajectoryRecord(raw=raw, source=jsonl_path), instance))

        if not self._items:
            raise RuntimeError("Dataset is empty — check the provided instance_pairs split.")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[TrajectoryRecord, RCPSPInstance]:
        return self._items[idx]


# ---------------------------------------------------------------------------
# Collate — builds feature tensors from a batch of (record, instance) pairs
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[TrajectoryRecord, RCPSPInstance]],
    max_resources: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Group records by instance so NodeContext is built once per instance,
    then featurize all records in the batch.
    """
    from collections import defaultdict
    inst_map: Dict[int, RCPSPInstance] = {}
    rec_map: Dict[int, List[TrajectoryRecord]] = defaultdict(list)

    for rec, inst in batch:
        key = id(inst)
        inst_map[key] = inst
        rec_map[key].append(rec)

    all_cand: List[List[float]] = []
    all_glob: List[List[float]] = []
    all_lengths: List[int] = []
    all_targets: List[int] = []
    all_depths: List[int] = []

    for key, recs in rec_map.items():
        inst = inst_map[key]
        result = featurize_states(recs, inst, max_resources)
        if result["lengths"].numel() == 0:
            continue
        all_cand.extend(result["candidate_feats"].tolist())
        all_glob.extend(result["global_feats"].tolist())
        all_lengths.extend(result["lengths"].tolist())
        all_targets.extend(result["targets"].tolist())
        all_depths.extend(result["depths"].tolist())

    if not all_lengths:
        return None

    return {
        "candidate_feats": torch.tensor(all_cand, dtype=torch.float32),
        "global_feats": torch.tensor(all_glob, dtype=torch.float32),
        "lengths": torch.tensor(all_lengths, dtype=torch.long),
        "targets": torch.tensor(all_targets, dtype=torch.long),
        "depths": torch.tensor(all_depths, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a behaviour-cloning policy using the BranchingTransformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    config: Dict,
) -> Tuple[DataLoader, DataLoader]:
    instance_pairs = collect_instance_pairs(
        trajectories_dir=Path(config["trajectories_dir"]),
        instances_dir=Path(config["instances_dir"]),
        instances_pattern=config["instances_pattern"],
        pattern=config["pattern"],
    )
    import random

    rng = random.Random(int(config["seed"]))
    shuffled_pairs = list(instance_pairs)
    rng.shuffle(shuffled_pairs)

    num_instances = len(shuffled_pairs)
    if num_instances == 1:
        train_pairs = shuffled_pairs
        val_pairs: List[InstancePair] = []
    else:
        val_size = max(1, int(num_instances * float(config["val_frac"])))
        train_size = max(num_instances - val_size, 1)
        val_size = num_instances - train_size
        train_pairs = shuffled_pairs[:train_size]
        val_pairs = shuffled_pairs[train_size:]

    print(
        f"Instance split — train: {len(train_pairs)} | val: {len(val_pairs)} "
        f"(total matched trajectories: {num_instances})"
    )

    train_ds = InstanceTrajectoryDataset(train_pairs)
    val_ds = InstanceTrajectoryDataset(val_pairs) if val_pairs else None
    max_resources = int(config["max_resources"])
    collate = lambda batch: collate_fn(batch, max_resources)
    train_loader = DataLoader(
        train_ds, batch_size=int(config["batch_size"]),
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        [] if val_ds is None else val_ds, batch_size=int(config["batch_size"]),
        shuffle=False, collate_fn=collate,
    )
    return train_loader, val_loader


def evaluate(
    model: BranchingTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = total_acc = batches = 0.0
    with torch.no_grad():
        for batch in loader:
            if batch is None or batch["lengths"].numel() == 0:
                continue
            cand = batch["candidate_feats"].to(device)
            glob = batch["global_feats"].to(device)
            lengths = batch["lengths"]
            targets = batch["targets"].to(device)

            # global_feats in the batch are repeated per candidate — take one per state
            # for the transformer we need one global vector per state, not per candidate.
            # Reconstruct per-state global by taking the first row of each state's block.
            offsets = [0] + torch.cumsum(lengths, dim=0).tolist()[:-1]
            glob_per_state = glob[offsets]  # [B, Fg]

            logits_list = []
            offset = 0
            for i, length in enumerate(lengths.tolist()):
                c = cand[offset: offset + length]          # [R, Fc]
                g = glob_per_state[i]                      # [Fg]
                logits_i, _ = model(c, g)
                logits_list.append(logits_i)
                offset += length

            logits_flat = torch.cat(logits_list, dim=0)
            loss = BranchingTransformer.listwise_nll(logits_flat, lengths, targets)
            acc = BranchingTransformer.listwise_accuracy(logits_flat, lengths, targets)
            total_loss += loss.item()
            total_acc += acc.item()
            batches += 1

    if batches == 0:
        return 0.0, 0.0
    return total_loss / batches, total_acc / batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config.update(load_config(Path(args.config)))

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset...")
    train_loader, val_loader = make_loaders(config)

    max_resources = int(config["max_resources"])
    global_dim = global_feature_dim(max_resources)
    candidate_dim = candidate_feature_dim(max_resources)
    print(f"Feature dims — global: {global_dim}, candidate: {candidate_dim}")

    model = BranchingTransformer(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        ffn_dim=int(config["ffn_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve = 0
    patience = int(config["patience"])
    min_epochs = int(config["min_epochs"])
    max_epochs = config.get("epochs")
    max_epochs = None if max_epochs is None else int(max_epochs)
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epoch = 0
    while True:
        epoch += 1
        model.train()
        epoch_loss = epoch_acc = batches = 0.0

        for batch in train_loader:
            if batch is None or batch["lengths"].numel() == 0:
                continue

            cand = batch["candidate_feats"].to(device)
            glob = batch["global_feats"].to(device)
            lengths = batch["lengths"]
            targets = batch["targets"].to(device)

            # One global vector per state (first row of each candidate block)
            offsets = [0] + torch.cumsum(lengths, dim=0).tolist()[:-1]
            glob_per_state = glob[offsets]

            logits_list = []
            offset = 0
            for i, length in enumerate(lengths.tolist()):
                c = cand[offset: offset + length]
                g = glob_per_state[i]
                logits_i, _ = model(c, g)
                logits_list.append(logits_i)
                offset += length

            logits_flat = torch.cat(logits_list, dim=0)
            loss = BranchingTransformer.listwise_nll(logits_flat, lengths, targets)
            acc = BranchingTransformer.listwise_accuracy(logits_flat, lengths, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        train_acc = epoch_acc / max(batches, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve = 0
            save_policy_checkpoint(model, str(output_path), extra={"train_config": config})
        else:
            no_improve += 1

        if epoch >= min_epochs and no_improve >= patience:
            print(f"Early stopping after {epoch} epochs (patience={patience}).")
            break
        if max_epochs is not None and epoch >= max_epochs:
            print(f"Reached max_epochs={max_epochs}.")
            break

    print(
        f"Done. best_val_loss={best_val_loss:.4f}  best_val_acc={best_val_acc:.4f} "
        f"saved to {output_path}"
    )


if __name__ == "__main__":
    main()
