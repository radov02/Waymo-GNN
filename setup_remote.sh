#!/bin/bash
# =============================================================================
# Remote Instance Setup Script (No Docker Required)
# For instances with pytorch/pytorch:2.4.0-cuda12.4.1-cudnn8-runtime base
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Setting up Waymo GNN Training Environment"
echo "=========================================="

# Check if we're on a GPU instance
echo ""
echo "[1/6] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. GPU might not be available."
fi

# Check PyTorch and CUDA
echo ""
echo "[2/6] Verifying PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install system dependencies (if running as root or with sudo)
echo ""
echo "[3/6] Installing system dependencies..."
if [ "$EUID" -eq 0 ]; then
    apt-get update && apt-get install -y --no-install-recommends \
        git wget curl vim htop tmux \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
else
    echo "Not running as root. Skipping system packages (may already be installed)."
fi

# Install PyTorch Geometric
echo ""
echo "[4/6] Installing PyTorch Geometric..."
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install --no-cache-dir torch-geometric

# Install other Python dependencies
echo ""
echo "[5/6] Installing Python dependencies..."
pip install --no-cache-dir \
    h5py>=3.10.0 \
    pandas>=2.1.0 \
    wandb>=0.16.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    tqdm>=4.66.0 \
    scipy>=1.11.0 \
    protobuf>=4.25.0 \
    Pillow>=10.0.0 \
    pyyaml>=6.0

# Install TensorFlow CPU (for Waymo data parsing only - much smaller than GPU version)
echo ""
echo "[6/6] Installing TensorFlow (CPU-only for data parsing)..."
pip install --no-cache-dir tensorflow-cpu>=2.15.0

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p checkpoints/autoregressive checkpoints/gat checkpoints/gat/autoregressive
mkdir -p visualizations/autoreg visualizations/autoreg/gat visualizations/autoreg/finetune
mkdir -p data/graphs/training data/graphs/validation data/graphs/testing
mkdir -p data/scenario/training data/scenario/validation data/scenario/testing
mkdir -p wandb

# Set environment variables
echo ""
echo "Setting environment variables..."
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Add to bashrc for persistence
echo 'export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"' >> ~/.bashrc
echo 'export PYTHONUNBUFFERED=1' >> ~/.bashrc
echo 'export TF_CPP_MIN_LOG_LEVEL=2' >> ~/.bashrc
echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python src/autoregressive_predictions/training_batched.py"
echo ""
echo "For GAT model training:"
echo "  python src/gat_autoregressive/training_gat_batched.py"
echo ""
echo "Optional: Set your W&B API key:"
echo "  export WANDB_API_KEY='your-key-here'"
echo ""
