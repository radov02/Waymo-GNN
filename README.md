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
│   ├── sweep_config.py                     # W&B hyperparameter sweeps
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
│   │   ├── vectornet_dataset.py            # Alternative dataset implementation
│   │   ├── training_vectornet_tfrecord.py  # Training pipeline
│   │   ├── testing_vectornet.py            # Evaluation
│   │   ├── vectornet_helpers.py            # Utility functions
│   │   └── vectornet.md                    # VectorNet documentation
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
│   ├── best_model.pt                       # Best GCN single-step model
│   ├── best_autoregressive_*.pt            # GCN autoregressive (various rollout steps)
│   ├── gat/
│   │   ├── best_model.pt                   # Best GAT single-step model
│   │   └── best_autoregressive_*.pt        # GAT autoregressive models
│   │
│   └── vectornet/
│       └── best_vectornet_model.pt         # Best VectorNet model
│
├── visualizations/
│   ├── autoreg/                            # GCN autoregressive predictions
│   ├── training/                           # Training progress visualizations
│   └── scenario_sequence/                  # Scenario replay visualizations
│
├── wandb/                                  # Weights and Biases logs
├── setup_remote.sh                         # Cloud instance setup script
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

This creates VectorNet-formatted TFRecord files in `src/vectornet/data/{training,validation,testing}/`

---

## Training on Cloud Instance

### Multi-Terminal Setup

Open one terminal for training and another for monitoring:

**Terminal 1 - Training:**
```bash
python src/autoregressive_predictions/training_batched.py
```

**Terminal 2 - GPU Monitoring:**
```bash
apt-get update && apt-get install -y nvtop
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
python src/vectornet/training_vectornet_tfrecord.py --data_dir data/scenario
python src/vectornet/testing_vectornet.py
```

---

## Model Architectures

### GCN: Graph Convolutional Networks with Autoregressive Prediction

**Spatial Encoding:** Graph Convolution layers process agent interactions based on proximity graphs at each timestep. Nodes represent agents, edges connect nearby agents.

**Temporal Encoding:** Per-agent GRU maintains hidden state across timesteps, capturing temporal dependencies.

**Single-step Training:** Predicts next displacement (dx, dy) given 30 timesteps of history.

**Autoregressive Fine-tuning:** Transitions from teacher forcing to autoregressive rollout using scheduled sampling. Model learns to handle its own prediction errors over 3-50 timestep horizons.

**Dataset:** Uses HDF5 graph files created by `graph_creation_and_saving.py` from TFRecord scenarios.

### GAT: Graph Attention Networks with Autoregressive Prediction

**Spatial Encoding:** Multi-head attention mechanism learns which agents to attend to, replacing fixed proximity-based edges with learned attention weights. More expressive than GCN but requires more parameters.

**Temporal Encoding:** Per-agent GRU maintains hidden state across timesteps, identical to GCN approach.

**Single-step Training:** Predicts next displacement (dx, dy) given 30 timesteps of history.

**Autoregressive Fine-tuning:** Same scheduled sampling approach as GCN but with learned spatial interactions. Attention weights provide interpretability of which agent relationships matter.

**Key Differences from GCN:**
- Learned attention over fixed graph structure: GAT computes attention coefficients between all agent pairs, allowing the model to learn which interactions are important rather than relying on distance-based edges.
- Higher capacity: Multiple attention heads enable diverse interaction patterns in parallel.
- Better interpretability: Attention weights reveal which agents influence predictions.
- Increased computational cost: O(n^2) attention computation versus sparse graph operations in GCN.

**Dataset:** Uses HDF5 graph files created by `graph_creation_and_saving.py` from TFRecord scenarios.

### VectorNet: Polyline-based Hierarchical Encoder

**Architecture:** VectorNet uses a hierarchical approach different from GCN/GAT. Instead of graph convolutions on agent proximity, it encodes polylines (sequences of vectors) representing agent trajectories and map features.

**Polyline Subgraph Network:** Each agent trajectory and map feature is encoded independently as a polyline using MLP layers. This captures local structure within each polyline.

**Global Interaction Graph:** After polyline encoding, a global attention-based graph aggregates interactions between all polylines (agents and map features) to capture scene-level context.

**Direct Multi-step Prediction:** Unlike GCN/GAT which predict one step and require autoregressive fine-tuning, VectorNet predicts the FULL future trajectory (50 timesteps, 5.0s) in a single forward pass. This avoids error accumulation from autoregressive rollout.

**Node Completion Task:** VectorNet includes an auxiliary self-supervised task where random nodes are masked (similar to BERT) and reconstructed, improving context understanding.

**Key Advantages:**
- No autoregressive fine-tuning needed: Direct multi-step prediction avoids compounding errors.
- Explicit map feature encoding: Polyline representation naturally handles lane boundaries, crosswalks, and road geometry.
- Hierarchical structure: Local polyline encoding followed by global interactions mirrors the semantic structure of traffic scenes.

**Key Differences from GCN/GAT:**
- Input representation: Polylines (vector sequences) instead of proximity graphs.
- Prediction mode: Multi-step (full trajectory) instead of single-step autoregressive.
- Map features: Explicitly encoded as separate polylines, not just spatial context.

**Dataset:** Uses VectorNet-formatted TFRecord files created by `prepare_vectornet_data.py`, which extracts polylines from raw TFRecord scenarios. This is a different preprocessing pipeline than GCN/GAT.

### Scheduled Sampling for Autoregressive Transition (GCN and GAT only)

Both GCN and GAT use scheduled sampling during fine-tuning to safely transition from teacher forcing to autoregressive prediction:

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
hidden_channels = 128           # GNN hidden dimension
num_layers = 3                  # Number of GCN layers
num_gru_layers = 1              # Temporal GRU layers
sequence_length = 30            # Input history length (3.0s at 0.1s per step)

# Training
learning_rate = 0.001
epochs = 20
batch_size = 48
early_stopping_patience = 15

# Autoregressive fine-tuning
autoreg_num_rollout_steps = 20  # Prediction horizon (2.0s)
autoreg_num_epochs = 10
autoreg_sampling_strategy = 'linear'
```

**GAT Model:**
```python
# GAT-specific
use_gat = True
gat_num_heads = 4               # Attention heads
gat_dropout = 0.1

# Training (same as GCN)
learning_rate = 0.001
epochs = 20
batch_size = 48

# Autoregressive (same as GCN)
autoreg_num_rollout_steps = 20
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

# Training
vectornet_batch_size = 48
vectornet_learning_rate = 0.001
vectornet_epochs = 30

# Auxiliary task
vectornet_use_node_completion = True        # Self-supervised masking task
vectornet_node_completion_ratio = 0.15      # Fraction of nodes to mask
```
- CPU: Falls back gracefully

---
