# Docker Deployment Guide for PrimeIntellect RTX Pro 6000

This guide covers deploying the Waymo GNN Trajectory Prediction project on a **PrimeIntellect RTX Pro 6000 96GB** GPU instance.

## 🚀 Quick Start

### 1. Clone and Navigate
```bash
git clone <your-repo-url> waymo-project
cd waymo-project
```

### 2. Build and Deploy
```bash
# Make the deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### 3. Start Training
```bash
# Start interactive container
docker-compose up -d waymo-gnn
docker-compose exec waymo-gnn bash

# Inside container - run training
python src/autoregressive_predictions/training_batched.py
```

---

## 📋 Prerequisites

Before deploying, ensure your PrimeIntellect instance has:

1. **NVIDIA Driver** (>= 535.x for CUDA 12.4)
   ```bash
   nvidia-smi  # Verify GPU is detected
   ```

2. **Docker** (>= 24.0)
   ```bash
   docker --version
   ```

3. **NVIDIA Container Toolkit**
   ```bash
   # Install if not present
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

---

## 🐳 Docker Commands

### Build Image
```bash
docker build -t waymo-gnn-trajectory:latest .
```

### Run Interactive Container
```bash
docker run -it --gpus all --shm-size=32g \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/visualizations:/workspace/visualizations \
  -v $(pwd)/wandb:/workspace/wandb \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --name waymo-gnn-training \
  waymo-gnn-trajectory:latest
```

### Using Docker Compose
```bash
# Set Weights & Biases API key (optional)
export WANDB_API_KEY='your-wandb-api-key'

# Start main service
docker-compose up -d waymo-gnn

# Attach to container
docker-compose exec waymo-gnn bash

# Or run specific profiles:
docker-compose --profile preprocessing up create-graphs
docker-compose --profile training up train-gcn
docker-compose --profile training up train-gat
docker-compose --profile testing up test-model
```

---

## 📁 Data Management

### Mount Your Data
The container expects data at `/workspace/data`. Mount your local data directory:

```bash
# Directory structure expected:
data/
├── scenario/           # Raw TFRecord files from Waymo
│   ├── training/
│   ├── validation/
│   └── testing/
└── graphs/             # Processed HDF5 graphs (generated)
    ├── training/
    ├── validation/
    └── testing/
```

### Download Waymo Data (inside container)
```bash
# The graph creation script can download data automatically
python src/graph_creation_and_saving.py
```

---

## 🎯 Training Workflows

### 1. Graph Creation (First Run)
Convert Waymo TFRecord files to optimized HDF5 graphs:
```bash
export PYTHONWARNINGS=ignore
export TF_CPP_MIN_LOG_LEVEL=3
python src/graph_creation_and_saving.py
```

### 2. Train GCN Model
```bash
python src/autoregressive_predictions/training_batched.py
```

### 3. Train GAT Model
```bash
python src/gat_autoregressive/training_gat_batched.py
```

### 4. Fine-tune for Autoregressive Prediction
```bash
# GCN
python src/autoregressive_predictions/finetune_single_step_to_autoregressive.py

# GAT
python src/gat_autoregressive/finetune.py
```

### 5. Test and Evaluate
```bash
python src/autoregressive_predictions/testing.py
```

---

## 🔧 RTX Pro 6000 Optimizations

The Docker setup includes optimizations specifically for the **RTX Pro 6000 (Ada Lovelace, 96GB)**:

| Feature | Setting | Benefit |
|---------|---------|---------|
| **TF32** | Enabled | 3x faster matrix operations |
| **BF16 Mixed Precision** | Auto-enabled | 2x memory efficiency |
| **torch.compile()** | reduce-overhead mode | Optimized kernel fusion |
| **Shared Memory** | 32GB | Fast DataLoader multiprocessing |
| **GPU Memory** | 95% utilization | Maximum batch sizes |
| **cuDNN Benchmark** | Enabled | Auto-tuned convolutions |

### Configuration in `src/config.py`
```python
# These are auto-detected for Ada architecture (SM 8.9):
use_amp = True          # Automatic Mixed Precision
use_bf16 = True         # BFloat16 (better than FP16)
use_torch_compile = True  # PyTorch 2.0+ compilation
torch_compile_mode = "reduce-overhead"
```

---

## 📊 Monitoring with Weights & Biases

### Setup W&B
```bash
# Set API key before running container
export WANDB_API_KEY='your-api-key'

# Or login inside container
wandb login
```

### View Training Progress
Training metrics are logged to W&B automatically. Access your dashboard at: https://wandb.ai

---

## 🔄 Long-Running Training with tmux

For training sessions that survive SSH disconnections:

```bash
# Inside container
tmux new -s training

# Start training
python src/autoregressive_predictions/training_batched.py

# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t training

# List sessions
tmux ls
```

---

## 💾 Checkpoints & Outputs

| Path | Content |
|------|---------|
| `/workspace/checkpoints/` | Model weights (.pt files) |
| `/workspace/checkpoints/autoregressive/` | Fine-tuned GCN models |
| `/workspace/checkpoints/gat/` | GAT models |
| `/workspace/visualizations/` | Training visualizations |
| `/workspace/wandb/` | W&B logs |

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker info | grep -i nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory
```python
# In src/config.py, reduce batch size
batch_size = 2  # Try smaller values

# Or enable gradient checkpointing
use_gradient_checkpointing = True
```

### Slow DataLoader
```python
# In src/config.py
num_workers = 8  # Increase for RTX Pro 6000
prefetch_factor = 4
```

### Container Cleanup
```bash
# Stop and remove
docker-compose down

# Remove all containers and images
docker system prune -a
```

---

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | Weights & Biases API key | - |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | all |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow log level | 2 |

---

## 📌 Quick Reference

```bash
# Build
docker build -t waymo-gnn-trajectory .

# Run interactive
docker-compose up -d waymo-gnn && docker-compose exec waymo-gnn bash

# Train GCN
python src/autoregressive_predictions/training_batched.py

# Train GAT
python src/gat_autoregressive/training_gat_batched.py

# Stop
docker-compose down
```
