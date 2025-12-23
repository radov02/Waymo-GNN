# VectorNet Implementation for Waymo Open Motion Dataset

This module implements VectorNet, a hierarchical graph neural network for behavior prediction as described in:

> **VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation**  
> Gao et al., CVPR 2020  
> [arXiv:2005.04259](https://arxiv.org/abs/2005.04259)

## Architecture Overview

VectorNet uses a vectorized representation of HD maps and agent trajectories, avoiding lossy rendering and computationally intensive ConvNet encoding.

### Key Insight: Multi-Step Prediction (Not Autoregressive)

**VectorNet predicts the FULL future trajectory at once** - it does NOT require autoregressive fine-tuning like GCN/GAT approaches. This is the correct approach as per the original paper.

The model:
1. Encodes history timesteps (default: 10 steps = 1 second)
2. Aggregates temporal information using attention
3. **Predicts all 50 future timesteps in a single forward pass**

### Architecture Components

1. **Polyline Subgraph Network**: Aggregates vectors within each polyline using MLP + MaxPooling
2. **Global Interaction Graph**: Models interactions between polylines using self-attention
3. **Temporal Aggregation**: Attention-based aggregation across history timesteps
4. **Multi-Step Trajectory Decoder**: Predicts full future trajectory at once

## File Structure

```
src/vectornet/
├── __init__.py                      # Module exports
├── VectorNet.py                     # Model architectures
│   ├── VectorNetMultiStep           # For HDF5 dataset (legacy)
│   └── VectorNetTFRecord            # For TFRecord dataset (recommended)
├── vectornet_config.py              # Configuration parameters
├── vectornet_dataset.py             # HDF5 dataset handlers
├── vectornet_tfrecord_dataset.py    # TFRecord dataset (with map features)
├── vectornet_helpers.py             # Utility functions
├── training_vectornet.py            # Training for HDF5
├── training_vectornet_tfrecord.py   # Training for TFRecord (recommended)
├── testing_vectornet.py             # Testing/evaluation
└── README.md                        # This file
```

## Dataset Options

### Option 1: TFRecord Dataset (Recommended)

The TFRecord dataset (`VectorNetTFRecordDataset`) directly loads raw Waymo scenario files and extracts:
- **Agent trajectory polylines** with full history
- **Map feature polylines** (lanes, roads, crosswalks)
- **Multi-step future labels** (50 timesteps)

This is the recommended approach as it includes HD map features that are essential for VectorNet.

### Option 2: HDF5 Dataset (Legacy)

The HDF5 dataset (`VectorNetDataset`) uses pre-processed graph snapshots. Note that the current HDF5 files **do not contain map features**, which limits VectorNet's ability to reason about road structure.

## Usage

### Training with TFRecord Dataset (Recommended)

```bash
# Train VectorNet with direct TFRecord loading
python src/vectornet/training_vectornet_tfrecord.py --data_dir data/scenario

# With options
python src/vectornet/training_vectornet_tfrecord.py \
    --data_dir data/scenario \
    --batch_size 32 \
    --hidden_dim 128 \
    --epochs 100 \
    --max_train 10000
```

### Training with HDF5 Dataset (Legacy)

```bash
# Train VectorNet (multi-step prediction)
python src/vectornet/training_vectornet.py
```

### Testing

```bash
# Evaluate trained model
python src/vectornet/testing_vectornet.py

# Use specific checkpoint
python src/vectornet/testing_vectornet.py --checkpoint checkpoints/vectornet/best_vectornet_tfrecord.pt
```

### Using in Code (TFRecord - Recommended)

```python
from vectornet import VectorNetTFRecord, AGENT_VECTOR_DIM, MAP_VECTOR_DIM
from vectornet.vectornet_tfrecord_dataset import (
    VectorNetTFRecordDataset, 
    vectornet_collate_fn
)
from torch.utils.data import DataLoader

# Create dataset (with map features!)
dataset = VectorNetTFRecordDataset(
    tfrecord_dir='data/scenario',
    split='training',
    history_len=10,
    future_len=50,
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=vectornet_collate_fn,
    num_workers=4,
)

# Create model (predicts 50 future timesteps at once)
model = VectorNetTFRecord(
    agent_input_dim=AGENT_VECTOR_DIM,  # 16 features
    map_input_dim=MAP_VECTOR_DIM,      # 13 features
    hidden_dim=128,
    prediction_horizon=50,
)

# Forward pass - returns [B, 50, 2] predictions
for batch in dataloader:
    predictions = model(batch)  # [B, prediction_horizon, 2]
    targets = batch['future_positions']  # [B, prediction_horizon, 2]
```

### Using in Code (HDF5 - Legacy)

```python
from vectornet import VectorNetMultiStep, VectorNetDataset, get_vectornet_config

# Get configuration
config = get_vectornet_config()

# Create model (predicts 50 future timesteps at once)
model = VectorNetMultiStep(
    input_dim=15,
    hidden_dim=128,
    output_dim=2,
    prediction_horizon=50,
    num_polyline_layers=3,
    num_global_layers=1,
    num_heads=8
)

# Load dataset
dataset = VectorNetDataset(
    hdf5_path='data/graphs/training/scenario_graphs.h5',
    seq_len=60  # history + prediction horizon
)

# Forward pass - returns [N, 50, 2] predictions
predictions = model(x, edge_index)  # [N, prediction_horizon, 2]
```

## Configuration

Key parameters in `vectornet_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 15 | Node feature dimension |
| `hidden_dim` | 128 | Hidden layer dimension |
| `output_dim` | 2 | Output dimension (dx, dy) |
| `prediction_horizon` | 50 | Future timesteps to predict |
| `history_length` | 10 | Past timesteps to encode |
| `num_polyline_layers` | 3 | Polyline network layers |
| `num_global_layers` | 1 | Global graph layers |
| `num_heads` | 8 | Attention heads |
| `batch_size` | 48 | Training batch size |
| `learning_rate` | 0.001 | Initial learning rate |
| `epochs` | 30 | Training epochs |

## Model Architecture Details

### VectorNetMultiStep

```
Input: History graph sequence [T × (N, 15)]
          ↓
┌─────────────────────────────────┐
│  For each timestep t:           │
│  ┌────────────────────────────┐ │
│  │ Polyline Subgraph Network  │ │
│  │ (MLP + MaxPool)            │ │
│  └────────────────────────────┘ │
│          ↓                      │
│  ┌────────────────────────────┐ │
│  │ Global Interaction Graph   │ │
│  │ (Multi-head Self-Attention)│ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│ Temporal Attention Aggregation  │
│ (Aggregate across T timesteps)  │
└─────────────────────────────────┘
          ↓
┌─────────────────────────────────┐
│ Multi-Step Trajectory Decoder   │
│ (MLP: hidden → 50 × 2)          │
└─────────────────────────────────┘
          ↓
Output: Full trajectory [N, 50, 2]
```

## Metrics

The model is evaluated using:

- **ADE (Average Displacement Error)**: Mean L2 error across all predicted timesteps
- **FDE (Final Displacement Error)**: L2 error at the final predicted timestep
- **Miss Rate**: Percentage of predictions with FDE > 2.0m

## Why No Autoregressive Fine-tuning?

Unlike GCN/GAT models that predict one step at a time and need autoregressive fine-tuning to handle error accumulation, VectorNet:

1. **Encodes full history context** using attention over all past timesteps
2. **Predicts full future directly** from the encoded representation
3. **No error accumulation** since each future timestep is predicted from the same encoding

This makes VectorNet:
- Faster at inference (one forward pass vs 50 forward passes)
- More robust to prediction errors
- Simpler to train and deploy

## References

```bibtex
@inproceedings{gao2020vectornet,
  title={VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation},
  author={Gao, Jiyang and Sun, Chen and Zhao, Hang and Shen, Yi and Anguelov, Dragomir and Li, Congcong and Schmid, Cordelia},
  booktitle={CVPR},
  year={2020}
}
```
