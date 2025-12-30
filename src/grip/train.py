import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fire import Fire

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vendor'))

from grip.model import Model
from grip.xin_feeder_baidu import Feeder

from waymo_data import GripPreprocessConfig, convert_waymo_to_grip

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_loader(
    data_path: str,
    graph_args: dict,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    split: str,
    num_workers: int,
    skip_split: bool = False,
    subset_fraction: float = 1.0,
):
    feeder = Feeder(
        data_path=data_path,
        graph_args=graph_args,
        train_val_test=split,
        skip_split=skip_split,
    )
    if subset_fraction < 1.0:
        num_samples = int(len(feeder) * subset_fraction)
        indices = torch.randperm(len(feeder))[:num_samples]
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        return torch.utils.data.DataLoader(
            dataset=feeder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=feeder,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )


def _preprocess_batch(raw_data: torch.Tensor, device: torch.device, rescale_xy: torch.Tensor):
    feature_id = [3, 4, 9, 10]
    ori_data = raw_data[:, feature_id].clone().to(device)
    data = ori_data.clone()

    # Train on absolute positions (not deltas) to avoid error accumulation over prediction horizon
    # Normalize by rescale_xy to keep magnitudes manageable for numerical stability
    data[:, :2] = data[:, :2] / rescale_xy

    object_type = raw_data[:, 2:3].to(device)
    return data, ori_data, object_type


def _compute_rmse(pred, target, mask, agent_mask, error_order: int = 2):
    # agent_mask: (batch_size, V) - 1.0 for real agents, 0.0 for padding (required)
    # mask: (batch_size, 1, T, V) - temporal validity per agent
    
    # Validate shapes and types
    batch_size, _, _, V = pred.shape
    assert agent_mask.shape == (batch_size, V), f"agent_mask shape {agent_mask.shape} != (batch, V) = ({batch_size}, {V})"
    assert mask.shape[-1] == V, f"mask last dim {mask.shape[-1]} != V = {V}"
    
    # Ensure agent_mask is float on same device as predictions
    if not isinstance(agent_mask, torch.Tensor):
        agent_mask = torch.tensor(agent_mask, dtype=torch.float32, device=pred.device)
    else:
        agent_mask = agent_mask.float().to(pred.device)
    
    # Combine both masks: agent must be real AND temporally valid
    combined_mask = mask * agent_mask.unsqueeze(1).unsqueeze(1)
    
    pred = pred * combined_mask
    target = target * combined_mask
    diff = torch.abs(pred - target) ** error_order
    sum_time = torch.sum(diff, dim=1).sum(dim=-1)
    num = combined_mask.sum(dim=1).sum(dim=-1)
    return sum_time, num


def _train_one_epoch(model, loader, optimizer, device, rescale_xy):
    model.train()
    total_loss = 0.0
    batches = 0

    print("\nStarting a new training epoch...")
    for batch_idx, batch_data in enumerate(loader):
        print(f"  Processing batch {batch_idx+1} out of {len(loader)}...")
        # Expect 4 items: ori_data, A, mean_xy, agent_mask
        ori_data, A, _, agent_mask = batch_data
            
        ori_data = ori_data.float().to(device)
        A = A.float().to(device)
        
        # Sanitize agent_mask: handle collated list-of-None or convert to tensor
        if isinstance(agent_mask, list):
            if all(item is None for item in agent_mask):
                raise ValueError("agent_mask is required but all batch items are None")
            agent_mask = torch.stack([torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in agent_mask])
        if not isinstance(agent_mask, torch.Tensor):
            agent_mask = torch.tensor(agent_mask, dtype=torch.float32)
        agent_mask = agent_mask.float().to(device)
        
        data, _, _ = _preprocess_batch(ori_data, device, rescale_xy)

        for now_history in range(1, data.shape[-2]):
            # print(f"    Training with history length {now_history} out of {data.shape[-2]-1}...")
            inputs = data[:, :, :now_history, :]
            targets = data[:, :2, now_history:, :]
            masks = data[:, -1:, now_history:, :]

            preds = model(
                pra_x=inputs,
                pra_A=A,
                pra_pred_length=targets.shape[-2],
                pra_teacher_forcing_ratio=0,
                pra_teacher_location=targets,
            )

            sum_time, num = _compute_rmse(preds, targets, masks, agent_mask, error_order=1)
            loss = torch.sum(sum_time) / torch.clamp(torch.sum(num), min=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

    print("Finished training epoch.")
    return total_loss / max(batches, 1)



def _evaluate(model, loader, device, rescale_xy, history_frames):
    model.eval()
    sum_arr = None
    num_arr = None
    future_steps = None

    print("\nStarting evaluation...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            print(f"  Evaluating batch {batch_idx+1} out of {len(loader)}...")
            # Expect 4 items: ori_data, A, mean_xy, agent_mask
            ori_data, A, _, agent_mask = batch_data
                
            ori_data = ori_data.float().to(device)
            A = A.float().to(device)
            
            # Sanitize agent_mask: handle collated list-of-None or convert to tensor
            if isinstance(agent_mask, list):
                if all(item is None for item in agent_mask):
                    raise ValueError("agent_mask is required but all batch items are None")
                agent_mask = torch.stack([torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in agent_mask])
            if not isinstance(agent_mask, torch.Tensor):
                agent_mask = torch.tensor(agent_mask, dtype=torch.float32)
            agent_mask = agent_mask.float().to(device)
            
            data, no_norm_loc, _ = _preprocess_batch(ori_data, device, rescale_xy)

            inputs = data[:, :, :history_frames, :]
            rel_targets = data[:, :2, history_frames:, :]
            future_steps = future_steps or rel_targets.shape[-2]
            masks = data[:, -1:, history_frames:, :]

            # If there are no future frames (e.g., Waymo test split), skip prediction evaluation
            if rel_targets.shape[-2] == 0:
                print("  [SKIP] No future frames in this split/batch; skipping prediction evaluation.")
                return np.zeros(0), 0.0

            abs_targets = no_norm_loc[:, :2, history_frames:, :]
            last_loc = no_norm_loc[:, :2, history_frames - 1 : history_frames, :]

            preds = model(
                pra_x=inputs,
                pra_A=A,
                pra_pred_length=abs_targets.shape[-2],
                pra_teacher_forcing_ratio=0,
                pra_teacher_location=None,  # Pure inference for evaluation
            )
            # Model predicts normalized absolute positions; denormalize for comparison
            preds = preds * rescale_xy

            sum_time, num = _compute_rmse(preds, abs_targets, masks, agent_mask)
            batch_sum = sum_time.cpu().numpy().sum(axis=0)
            batch_num = num.cpu().numpy().sum(axis=0)
            if sum_arr is None:
                sum_arr = batch_sum
                num_arr = batch_num
            else:
                sum_arr += batch_sum
                num_arr += batch_num

    if sum_arr is None or num_arr is None:
        return np.zeros(future_steps or 1), 0.0

    rmse = np.sqrt(np.divide(sum_arr, np.maximum(num_arr, 1e-6)))
    print("Finished evaluation.")
    return rmse, float(np.mean(rmse))


def _save_checkpoint(model, path: Path, epoch: int, metrics: float):
    payload = {
        "epoch": epoch,
        "metric": metrics,
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)


def train(
    train_data_path: str = "data/grip/train_data.pkl",
    val_data_path: str = "data/grip/val_data.pkl",
    preprocess_config: Optional[str] = None,
    work_dir: str = "outputs/grip",
    device: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 16,
    val_batch_size: int = 16,
    num_workers: int = 2,
    lr: float = 1e-3,
    max_x: float = 30.0,
    max_y: float = 30.0,
    seed: int = 42,
    num_nodes: Optional[int] = None,
    subset_fraction: float = 1.0,
):
    output_directory = Path(work_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    if preprocess_config is None:
        candidate = Path(train_data_path).resolve().parent / "preprocess_config.json"
        if candidate.exists():
            preprocess_config = str(candidate)

    # Default to train/val frame counts (10 history + 79 future)
    history_frames = 10
    if preprocess_config:
        with open(preprocess_config, "r", encoding="utf-8") as f:
            meta = json.load(f)
            num_nodes = num_nodes or meta.get("max_agents")
            # For backward compat: read from config if present, else use default
            history_frames = 10  # Fixed for Waymo Motion Challenge
    num_nodes = num_nodes or 64

    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)

    graph_args = {"max_hop": 2, "num_node": num_nodes}
    model = Model(
        in_channels=4,
        graph_args=graph_args,
        edge_importance_weighting=True,
        use_cuda=torch_device.type == "cuda",
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load pre-split data (skip_split=True to use all samples from each pickle)
    train_loader = _build_loader(
        train_data_path, graph_args, batch_size, True, True, "all", num_workers, skip_split=True, subset_fraction=subset_fraction
    )
    val_loader = _build_loader(
        val_data_path, graph_args, val_batch_size, False, False, "all", num_workers, skip_split=True, subset_fraction=subset_fraction
    )

    print(f"Training on device: {torch_device}")
    rescale_xy = torch.ones((1, 2, 1, 1), device=torch_device)
    rescale_xy[:, 0] = max_x
    rescale_xy[:, 1] = max_y

    best_metric = float("inf")
    for epoch in range(epochs):
        # Train on train_loader only (no test data leakage)
        train_loss = _train_one_epoch(model, train_loader, optimizer, torch_device, rescale_xy)
        val_rmse_curve, val_rmse_mean = _evaluate(model, val_loader, torch_device, rescale_xy, history_frames)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse_mean:.3f} "
            f"per step: {[round(x,3) for x in val_rmse_curve.tolist()]}"
        )

        ckpt_path = output_directory / f"checkpoint_epoch_{epoch+1:03d}.pt"
        _save_checkpoint(model, ckpt_path, epoch + 1, val_rmse_mean)
        if val_rmse_mean < best_metric:
            best_metric = val_rmse_mean
            _save_checkpoint(model, output_directory / "best.pt", epoch + 1, val_rmse_mean)


def prepare_data(
    train_hdf5: str = "vendor/erik-waymo/data/graphs/training/training_seqlen90.hdf5",
    val_hdf5: str = "vendor/erik-waymo/data/graphs/validation/validation_seqlen90.hdf5",
    test_hdf5: str = "vendor/erik-waymo/data/graphs/testing/testing_seqlen90.hdf5",
    output_dir: str = "data/grip",
    max_agents: int = 64,
    neighbor_distance: float = 10.0,
):
    """Prepare Waymo data for GRIP++ at native 10Hz (Waymo Motion Challenge standard).
    
    Frame allocation:
    - Train/Val: 10 history + 79 future = 89 frames @ 10Hz
    - Test: 10 history + 0 future = 10 frames @ 10Hz (futures hidden)
    
    Outputs: {train,val,test}_data.pkl + preprocess_config.json
    """
    # Build HDF5 paths dict
    hdf5_paths = {
        "train": train_hdf5,
        "val": val_hdf5,
        "test": test_hdf5,
    }

    # Create config for native 10Hz processing
    config = GripPreprocessConfig(
        max_agents=max_agents,
        neighbor_distance=neighbor_distance,
    )

    print(f"Using preprocess config: {config}")

    # Convert all three splits
    output_paths = convert_waymo_to_grip(hdf5_paths, output_dir, config)
    
    for split, path in output_paths.items():
        print(f"Wrote {split} data to {path}")


class GripCLI:
    def prepare(self, **kwargs):  # type: ignore[override]
        return prepare_data(**kwargs)

    def train(self, **kwargs):  # type: ignore[override]
        return train(**kwargs)


if __name__ == "__main__":
    Fire(GripCLI)