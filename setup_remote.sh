#!/bin/bash
# =============================================================================
# Remote Instance Setup Script for Blackwell GPUs (sm_120)
# RTX PRO 6000 Blackwell Server Edition requires PyTorch 2.5+
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Setting up Waymo GNN Training Environment"
echo "For Blackwell GPU (sm_120) Support"
echo "=========================================="

# Check if we're on a GPU instance
echo ""
echo "[1/7] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
else
    echo "WARNING: nvidia-smi not found. GPU might not be available."
fi

# Uninstall old PyTorch
echo ""
echo "[2/7] Removing old PyTorch (incompatible with Blackwell)..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch with Blackwell support
# Blackwell (sm_120) requires PyTorch nightly as of Dec 2025
echo ""
echo "[3/7] Installing PyTorch with Blackwell (sm_120) support..."
echo "Trying PyTorch nightly (required for Blackwell architecture)..."

# Try to install torch+torchvision from the nightly index.
# Use --no-deps to avoid pip dependency resolution failures between dev wheels;
# if that fails, attempt a safer two-step install.
if pip install --no-cache-dir --no-deps --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124; then
    echo "Installed torch + torchvision (no-deps) from nightly index."
else
    echo "Initial no-deps install failed; trying two-step install (torch then torchvision)."
    if pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124; then
        if pip install --no-cache-dir --no-deps --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124; then
            echo "Installed torch and torchvision (torch full, torchvision no-deps)."
        else
            echo "Failed to install torchvision from nightly index. Will try PyPI fallback for torchvision."
            pip install --no-cache-dir torchvision || echo "WARNING: torchvision install failed; continue and debug manually."
        fi
    else
        echo "ERROR: Failed to install torch from nightly index."
        exit 1
    fi
fi

# Try torchaudio on the same index; fallback to PyPI; warn and continue if still not available
if ! pip install --no-cache-dir --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124; then
    echo "torchaudio not found on nightly index; trying PyPI..."
    if ! pip install --no-cache-dir torchaudio; then
        echo "WARNING: torchaudio install failed. Continuing without torchaudio."
        echo "If you need torchaudio, install manually: https://pytorch.org/audio/stable/installation.html"
    fi
fi

# Verify PyTorch installation
echo ""
echo "[4/7] Verifying PyTorch and CUDA compatibility..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    # Test actual CUDA operation
    try:
        x = torch.randn(10, 10, device='cuda')
        y = x @ x.T
        print('CUDA tensor operations: OK')
    except Exception as e:
        print(f'CUDA test failed: {e}')
"

# Install system dependencies (if running as root or with sudo)
echo ""
echo "[5/7] Installing system dependencies..."
if [ "$EUID" -eq 0 ]; then
    apt-get update && apt-get install -y --no-install-recommends \
        git wget curl vim htop tmux \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
else
    echo "Not running as root. Skipping system packages (may already be installed)."
fi

# Install PyTorch Geometric (must match new PyTorch version)
echo ""
echo "[6/7] Installing PyTorch Geometric..."
# For nightly PyTorch, we need to install from source or use compatible wheels
# Try multiple approaches
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "Detected PyTorch version: $TORCH_VERSION"

# Try PyG wheels, fallback to pip install which will build from source
pip install --no-cache-dir torch-geometric
pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv || \
    echo "Note: Some PyG extensions may need to be built from source for nightly PyTorch"

# Install other Python dependencies
echo ""
echo "[7/7] Installing Python dependencies..."
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

# Install TensorFlow CPU (for Waymo data parsing only)
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
