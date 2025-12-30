# Prediction of Agent Trajectories on Waymo Autonomous Driving Dataset using GNNs

Project for Stanford CS224W 2025 class at FRI.

---

## Project Structure

```
waymo-project/
├── src/
│   ├── config.py                           # Hyperparameters and configuration
│   ├── dataset.py                          # HDF5 dataset loader
│   ├── graph_creation_and_saving.py        # TFRecord to HDF5 graph conversion
│   │
│   ├── autoregressive_predictions/
│   │   ├── SpatioTemporalGNN_batched.py    # GCN + GRU model architecture
│   │   ├── training_batched.py             # Single-step training
│   │   ├── finetune_single_step_to_autoregressive.py  # Autoregressive fine-tuning
│   │   └── testing.py                      # Evaluation and visualization
│   │
│   ├── gat_autoregressive/
│   │   ├── SpatioTemporalGAT_batched.py    # GAT + GRU model architecture
│   │   ├── training_gat_batched.py         # Single-step training
│   │   ├── finetune.py                     # Autoregressive fine-tuning
│   │   └── testing_gat.py                  # Evaluation and visualization
│   │
│   ├── vectornet/
│   │   ├── VectorNet.py                    # VectorNet model architecture
│   │   ├── prepare_vectornet_data.py       # TFRecord to VectorNet polyline format
│   │   ├── vectornet_tfrecord_dataset.py   # VectorNet dataset loader
│   │   ├── training_vectornet_tfrecord.py  # Training pipeline
│   │   ├── testing_vectornet.py            # Evaluation
│   │   └── vectornet_helpers.py            # Utility functions
│   │
│   └── helper_functions/
│       ├── graph_creation_functions.py     # Graph construction utilities
│       ├── visualization_functions.py      # Trajectory visualization
│       ├── helpers.py                      # Loss functions and metrics
│       └── cloud_tfrecord_downloader.py    # Google Cloud data downloader
│
├── data/
│   ├── scenario/
│   │   ├── training/                       # Training scenario files (.tfrecord)
│   │   ├── validation/                     # Validation scenario files
│   │   └── testing/                        # Testing scenario files
│   │
│   └── graphs/
│       ├── training/                       # Training graphs (.hdf5)
│       ├── validation/                     # Validation graphs
│       └── testing/                        # Testing graphs
│
├── checkpoints/
│   ├── gcn/                                # GCN model checkpoints
│   │   └── autoregressive/                 # GCN autoregressive models
│   ├── gat/
│   │   ├── best_model.pt                   # Best GAT single-step model
│   │   └── autoregressive/                 # GAT autoregressive models
│   └── vectornet/                          # VectorNet checkpoints
│
├── visualizations/
│   ├── autoreg/                            # Autoregressive predictions
│   │   ├── finetune/                       # Fine-tuning visualizations
│   │   ├── gat/                            # GAT model visualizations
│   │   └── vectornet/                      # VectorNet visualizations
│   ├── training/                           # Training progress visualizations
│   └── scenario_sequence/                  # Scenario replay visualizations
│
├── wandb/                                  # Weights and Biases logs
├── setup_remote.sh                         # Cloud instance setup script
├── download_visualizations.ps1             # PowerShell script to download viz from cloud
├── USE_TMUX.md                             # TMUX usage guide for cloud instances
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## Running on Cloud Training Instance

### Step 1: Generate SSH Key (Local Machine)

```powershell
ssh-keygen -t ed25519 -f C:\Users\YOURUSERNAME\.ssh\primeintellect_ed25519 -C "yourPublicKeyName"
```

This creates two files:
- `C:\Users\YOURUSERNAME\.ssh\primeintellect_ed25519` (private key)
- `C:\Users\YOURUSERNAME\.ssh\primeintellect_ed25519.pub` (public key)

Note: You can use any name instead of `primeintellect_ed25519` for your SSH key.

Add the public key to your cloud instance and proceed.

### Step 2: Connect to Instance

```powershell
ssh root@IP -p THEPORT -i C:\Users\YOURUSERNAME\.ssh\primeintellect_ed25519
```

### Step 3: Clone Repository and Setup

```bash
whoami
hostname
pwd
ls -la

git clone https://github.com/radov02/Waymo-GNN.git
cd Waymo-GNN
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release apt-transport-https software-properties-common
git pull
chmod +x setup_remote.sh
./setup_remote.sh
```

### Step 4: Configure Google Cloud SDK

Required for downloading Waymo Open Motion Dataset (WOMD):

```bash
apt-get update
apt-get install -y apt-transport-https ca-certificates gnupg curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update
apt-get install -y google-cloud-sdk

gcloud init
gcloud auth application-default login
```

Ensure billing is enabled in [Google Cloud Console](https://console.cloud.google.com/billing/projects)

### Step 5: Set Weights and Biases API Key (Optional)

```bash
export WANDB_API_KEY='your-key-here'
```

### Step 6: Data and Graph Preparation

**For GCN and GAT models:**

Download WOMD files from Google Cloud or transfer pre-created graphs. These models use graph-based datasets stored in HDF5 format.

Create graphs from scenarios:
```bash
python src/graph_creation_and_saving.py
```

**For VectorNet model:**

VectorNet requires a different data preparation process. It uses polyline-based encoding and reads directly from TFRecord files.

Prepare VectorNet data:
```bash
python src/vectornet/prepare_vectornet_data.py --source_dir data/scenario
```

---

## Training on Cloud Instance

### Multi-Terminal Setup with TMUX

Use TMUX for persistent sessions (see [USE_TMUX.md](USE_TMUX.md) for detailed guide):

```bash
# Start new tmux session
tmux new -s training

# Split panes: Ctrl+B then %
# Navigate panes: Ctrl+B then arrow keys
```

**Pane 1 - Training:**
```bash
python src/autoregressive_predictions/training_batched.py
```

**Pane 2 - GPU Monitoring:**
```bash
nvtop
```

Or use standard nvidia-smi:
```bash
watch -n 1 nvidia-smi
```

### File Transfer

Download files from instance to local machine:
```powershell
scp -P THEPORT -i C:\Users\YOURUSERNAME\.ssh\primeintellect_ed25519 root@IP:~/Waymo-GNN/visualizations/autoreg/epoch_001_progress.png C:\Users\radov\Downloads\
```

Upload files to instance:
```powershell
scp -P THEPORT -i C:\Users\YOURUSERNAME\.ssh\primeintellect_ed25519 C:\Users\radov\Downloads\best_model.pt root@IP:~/Waymo-GNN/checkpoints
```

Use `download_visualizations.ps1` for batch downloading visualizations.

---

## Training Locally

### Graph Creation

**For GCN and GAT:**
```bash
python src/graph_creation_and_saving.py
```

**For VectorNet:**
```bash
python src/vectornet/prepare_vectornet_data.py --source_dir data/scenario
```

### Training

Choose one model architecture:

**GCN (Graph Convolutional Networks):**
```bash
python src/autoregressive_predictions/training_batched.py
python src/autoregressive_predictions/finetune_single_step_to_autoregressive.py
python src/autoregressive_predictions/testing.py
```

**GAT (Graph Attention Networks):**
```bash
python src/gat_autoregressive/training_gat_batched.py
python src/gat_autoregressive/finetune.py
python src/gat_autoregressive/testing_gat.py
```

**VectorNet (Polyline-based Encoder):**
```bash
python src/vectornet/training_vectornet_tfrecord.py
python src/vectornet/testing_vectornet.py
```

---

## Model Architectures

### GCN: Graph Convolutional Networks with Autoregressive Prediction

**Spatial Encoding:** Graph Convolution layers process agent interactions based on proximity graphs at each timestep. Nodes represent agents, edges connect nearby agents within a configurable radius (default 35m).

**Temporal Encoding:** Per-agent GRU maintains hidden state across timesteps, capturing temporal dependencies.

**Single-step Training:** Predicts next displacement (dx, dy) given 90 timesteps of history.

**Autoregressive Fine-tuning:** Uses curriculum learning - starts with 10 steps and gradually increases to 50 steps. Scheduled sampling transitions from teacher forcing to autoregressive rollout. Model learns to handle its own prediction errors over up to 5.0s horizon.

**Dataset:** Uses HDF5 graph files created by `graph_creation_and_saving.py` from TFRecord scenarios.

### GAT: Graph Attention Networks with Autoregressive Prediction

**Spatial Encoding:** Multi-head attention mechanism (8 heads by default) learns which agents to attend to, replacing fixed proximity-based edges with learned attention weights. More expressive than GCN but requires more parameters.

**Temporal Encoding:** Per-agent GRU maintains hidden state across timesteps, identical to GCN approach.

**Single-step Training:** Predicts next displacement (dx, dy) given 90 timesteps of history.

**Autoregressive Fine-tuning:** Same curriculum learning approach as GCN but with learned spatial interactions. Attention weights provide interpretability of which agent relationships matter.

**Key Differences from GCN:**
- Learned attention over fixed graph structure: GAT computes attention coefficients between all agent pairs, allowing the model to learn which interactions are important rather than relying on distance-based edges.
- Higher capacity: Multiple attention heads enable diverse interaction patterns in parallel.
- Better interpretability: Attention weights reveal which agents influence predictions.
- Increased computational cost: O(n^2) attention computation versus sparse graph operations in GCN.

**Dataset:** Uses HDF5 graph files created by `graph_creation_and_saving.py` from TFRecord scenarios.

### VectorNet: Polyline-based Hierarchical Encoder

**Architecture:** VectorNet uses a hierarchical approach different from GCN/GAT. Instead of graph convolutions on agent proximity, it encodes polylines (sequences of vectors) representing agent trajectories and map features.

**Polyline Subgraph Network:** Each agent trajectory and map feature is encoded independently as a polyline using MLP layers. This captures local structure within each polyline.

**Global Interaction Graph:** After polyline encoding, a global attention-based graph (8 attention heads) aggregates interactions between all polylines (agents and map features) to capture scene-level context.

**Direct Multi-step Prediction:** Unlike GCN/GAT which predict one step and require autoregressive fine-tuning, VectorNet predicts the FULL future trajectory (50 timesteps, 5.0s) in a single forward pass using 10 timesteps (1.0s) of history. This avoids error accumulation from autoregressive rollout.

**Node Completion Task:** VectorNet includes an optional auxiliary self-supervised task where random nodes are masked (similar to BERT) and reconstructed, improving context understanding. Disabled by default.

**Key Advantages:**
- No autoregressive fine-tuning needed: Direct multi-step prediction avoids compounding errors.
- Explicit map feature encoding: Polyline representation naturally handles lane boundaries, crosswalks, and road geometry.
- Hierarchical structure: Local polyline encoding followed by global interactions mirrors the semantic structure of traffic scenes.

**Key Differences from GCN/GAT:**
- Input representation: Polylines (vector sequences) instead of proximity graphs.
- Prediction mode: Multi-step (full trajectory) instead of single-step autoregressive.
- Map features: Explicitly encoded as separate polylines, not just spatial context.

**Dataset:** Uses VectorNet-formatted TFRecord files created by `prepare_vectornet_data.py`, which extracts polylines from raw TFRecord scenarios. This is a different preprocessing pipeline than GCN/GAT.

---

## Testing Configuration

All models share common testing settings configured in [src/config.py](src/config.py):

```python
test_num_rollout_steps = 50     # 5.0s prediction horizon
test_max_scenarios = 100        # Maximum scenarios to evaluate
test_visualize = True           # Generate prediction visualizations
test_visualize_max = 20         # Max scenarios to visualize
test_use_wandb = True           # Log metrics to Weights & Biases
test_horizons = [10, 20, 30, 40, 50]  # Evaluation horizons (1s, 2s, 3s, 4s, 5s)
```

---

### Scheduled Sampling for Autoregressive Transition (GCN and GAT only)

Both GCN and GAT use curriculum learning and scheduled sampling during fine-tuning to safely transition from teacher forcing to autoregressive prediction:

- **Curriculum learning:** Training starts with 10 rollout steps and gradually increases to the maximum (50 steps)
- **Linear schedule:** Linearly increase the fraction of predictions used as inputs during training
- **Exponential schedule:** Slow start with faster transition mid-training
- **Inverse sigmoid schedule:** S-curve transition mimicking curriculum learning

This prevents exposure bias and allows models to learn recovery from prediction errors.

---

## Key Hyperparameters

Edit [src/config.py](src/config.py) to customize:

**GCN Model:**
```python
# Model architecture
hidden_channels = 256           # GNN hidden dimension
num_layers = 4                  # Number of GCN layers
num_gru_layers = 2              # Temporal GRU layers
sequence_length = 90            # Input sequence length (9.0s at 0.1s per step)
dropout = 0.1                   # Dropout rate

# Training
learning_rate = 0.001
epochs = 20
batch_size = 48
scheduler_patience = 5          # LR scheduler patience
scheduler_factor = 0.5          # LR reduction factor

# Loss weights (direction + magnitude balance)
loss_alpha = 0.4                # Angle weight (directional accuracy)
loss_beta = 0.3                 # MSE weight (positional accuracy)
loss_gamma = 0.1                # Velocity magnitude consistency
loss_delta = 0.4                # Cosine similarity

# Autoregressive fine-tuning (curriculum learning)
autoreg_num_rollout_steps = 50  # Max prediction horizon (5.0s)
autoreg_num_epochs = 20
autoreg_sampling_strategy = 'linear'
autoreg_learning_rate = 0.0005
```

**GAT Model:**
```python
# GAT-specific
gat_num_heads = 8               # Attention heads
gat_learning_rate = 0.0001      # Lower LR for stable attention training

# GAT loss weights (prioritize displacement accuracy)
gat_loss_alpha = 0.15           # Angle weight
gat_loss_beta = 0.6             # MSE weight
gat_loss_gamma = 0.1            # Velocity magnitude
gat_loss_delta = 0.15           # Cosine similarity

# Training (same batch size and epochs as GCN)
batch_size = 48
epochs = 20
```

**VectorNet Model:**
```python
# Model architecture
vectornet_hidden_dim = 128                  # Hidden dimension
vectornet_num_polyline_layers = 3           # Polyline encoder layers
vectornet_num_global_layers = 1             # Global interaction layers
vectornet_num_heads = 8                     # Attention heads
vectornet_prediction_horizon = 50           # Future steps (5.0s)
vectornet_history_length = 10               # Past steps (1.0s)
vectornet_num_agents_to_predict = 14        # Agents per scenario

# Training
vectornet_batch_size = 48
vectornet_learning_rate = 0.001
vectornet_epochs = 30

# Loss weights
vectornet_loss_alpha = 0.2                  # Angle loss
vectornet_loss_beta = 0.5                   # MSE loss (primary)
vectornet_loss_gamma = 0.1                  # Velocity consistency
vectornet_loss_delta = 0.2                  # Cosine similarity

# Auxiliary task
vectornet_use_node_completion = False       # Self-supervised masking task
vectornet_node_completion_ratio = 0.15      # Fraction of nodes to mask
```

**GPU Optimization (auto-detected):**
```python
use_amp = True                  # Mixed precision (Volta+ GPUs)
use_bf16 = True                 # BFloat16 (Ampere+ GPUs)
use_torch_compile = True        # torch.compile for 20-30% speedup
use_data_parallel = True        # Multi-GPU training (auto-detected)
```

---
