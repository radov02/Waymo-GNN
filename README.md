# Prediction of agent trajectories on Waymo autonomous driving dataset using GNNs

Project for Stanford CS224W 2025 class at FRI.

## Project Structure

```
src/
├── config.py                    # All hyperparameters and configuration
├── dataset.py                   # HDF5 dataset loader
├── graph_creation_and_saving.py # TFRecord → HDF5 graph conversion
├── sweep_config.py              # W&B hyperparameter sweeps
│
├── autoregressive_predictions/  # GCN-based autoregressive approach
│   ├── SpatioTemporalGNN.py     # Base GCN + GRU model architecture
│   ├── training.py              # Single-step prediction training
│   ├── finetune_single_step_to_autoregressive.py  # Fine-tune for autoregressive rollout
│   └── testing.py               # Evaluation with visualization
│
├── gat_autoregressive/          # GAT-based autoregressive approach
│   ├── SpatioTemporalGAT.py     # GAT + GRU model architecture
│   ├── training.py              # Single-step prediction training
│   ├── finetune.py              # Fine-tune for autoregressive rollout
│   └── testing.py               # Evaluation with visualization
│
└── helper_functions/
    ├── graph_creation_functions.py
    ├── visualization_functions.py
    └── helpers.py               # Loss functions, metrics
```

**Checkpoint directories:**
- `checkpoints/` - Base GCN single-step model
- `checkpoints/autoregressive/` - Fine-tuned GCN autoregressive model
- `checkpoints/gat/` - Base GAT single-step model
- `checkpoints/gat/autoregressive/` - Fine-tuned GAT autoregressive model

**Visualization directories:**
- `visualizations/autoreg/` - GCN autoregressive visualizations
- `visualizations/autoreg/gat/` - GAT autoregressive visualizations

---

## Pipeline Overview

```
1. Data Download        → ./data/scenario/{training,validation,testing}/
2. Graph Creation       → ./data/graphs/{training,validation,testing}/*.hdf5
3. Choose Your Model Architecture:

   GCN (Graph Convolutional Networks):
    - Training          → checkpoints/best_model.pt
    - Fine-tuning       → checkpoints/autoregressive/
    - Testing           → visualizations/autoreg/

   GAT (Graph Attention Networks):
    - Training          → checkpoints/gat/best_model.pt
    - Fine-tuning       → checkpoints/gat/autoregressive/
    - Testing           → visualizations/autoreg/gat/
```

---

## Detailed Steps

### 1. Data Download

Download WOMD files from [Waymo Cloud Console](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario) into:
- `./data/scenario/training/`
- `./data/scenario/validation/`
- `./data/scenario/testing/`

### 2. Graph Creation

```powershell
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/graph_creation_and_saving.py
```

- Parses `.tfrecord` files from `./data/scenario/`
- Creates spatial graphs for each timestep (nodes = agents, edges = proximity)
- Stores compressed HDF5 files at `./data/graphs/{split}/{split}.hdf5`

### 3. Single-Step Training (Base Model)

```powershell
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/training.py
```

- Trains `SpatioTemporalGNN` model (GCN spatial encoder + GRU temporal encoder)
- Predicts next-step displacement (dx, dy) for each agent
- Saves checkpoint to `./checkpoints/best_model.pt`

---

## Two Model Architectures

### GCN: Graph Convolutional Networks

**Training the GCN model:**

```powershell
# Train base GCN model
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/training.py

# Fine-tune for autoregressive rollout
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/finetune_single_step_to_autoregressive.py

# Test and visualize
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/testing.py
```

### GAT: Graph Attention Networks

**Training the GAT model:**

```powershell
# Train base GAT model
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/training.py

# Fine-tune for autoregressive rollout
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/finetune.py

# Test and visualize
$env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/testing.py
```

### Comparison

| Aspect | GCN | GAT |
|--------|-----|-----|
| Spatial encoding | Graph convolution | Multi-head attention |
| Edge weights | Optional (distance-based) | Learned attention |
| Parameters | Fewer | More (attention heads) |
| Interpretability | Lower | Higher (attention weights) |

---

## Autoregressive Fine-Tuning

Both GCN and GAT models use **scheduled sampling** to transition from teacher forcing to autoregressive prediction:

1. Loads pre-trained single-step model
2. Uses **scheduled sampling** to transition from teacher forcing to autoregressive
3. Model learns to handle its own prediction errors
4. Predicts: t → t+1 → feed back → t+2 → ... → t+K

**Scheduled Sampling Strategies:**
- `linear`: Linearly increase use of predictions
- `exponential`: Slower start, faster transition
- `inverse_sigmoid`: S-curve

---

## Model Architecture

**SpatioTemporalGNN (GCN-based):**
1. **Spatial Encoder**: GCN layers process agent interactions at each timestep
2. **Temporal Encoder**: Per-agent GRU maintains hidden state across time
3. **Decoder**: MLP predicts displacement (dx, dy) for next timestep

**SpatioTemporalGAT (GAT-based):**
1. **Spatial Encoder**: Multi-head GAT layers with learned attention
2. **Temporal Encoder**: Per-agent GRU maintains hidden state across time
3. **Decoder**: MLP predicts displacement (dx, dy) for next timestep

---

## Configuration (`config.py`)

Key parameters:
```python
# Model
hidden_channels = 128       # GNN hidden dimension
num_layers = 3              # GCN layers
num_gru_layers = 1          # Temporal layers
sequence_length = 30        # Timesteps per scenario (0.1s each)

# Training
learning_rate = 0.001
epochs = 10                 # For single-step training
early_stopping_patience = 15

# Autoregressive fine-tuning
autoreg_num_rollout_steps = 20  # 2.0s horizon
autoreg_num_epochs = 10
autoreg_sampling_strategy = 'linear'

# GAT-specific
use_gat = True                  # Toggle between GAT and GCN
gat_num_heads = 4               # Number of attention heads
gat_checkpoint_dir = 'checkpoints/gat'
```

---

## Multi-GPU Support

Automatically detects and uses multiple GPUs via `DataParallel`:
- Multi-GPU: Data parallelism across all available GPUs
- Single GPU: Standard training
- CPU: Falls back gracefully

---
