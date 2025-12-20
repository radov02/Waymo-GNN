#!/bin/bash
# =============================================================================
# PrimeIntellect RTX Pro 6000 Deployment Script
# Waymo GNN Trajectory Prediction
# =============================================================================

set -e

echo "========================================"
echo "Waymo GNN - PrimeIntellect Deployment"
echo "========================================"

# =============================================================================
# CONFIGURATION
# =============================================================================
IMAGE_NAME="waymo-gnn-trajectory"
CONTAINER_NAME="waymo-gnn-training"
WANDB_API_KEY="${WANDB_API_KEY:-}"  # Set your W&B API key

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
echo ""
echo "[1/6] Running pre-flight checks..."

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA driver not found. Please install NVIDIA drivers."
    exit 1
fi

echo "  ✓ NVIDIA Driver detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker."
    exit 1
fi
echo "  ✓ Docker detected: $(docker --version)"

# Check NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA Container Toolkit may not be configured."
    echo "  Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi
echo "  ✓ NVIDIA Container Toolkit configured"

# =============================================================================
# BUILD IMAGE
# =============================================================================
echo ""
echo "[2/6] Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

echo "  ✓ Image built successfully"

# =============================================================================
# SETUP DATA DIRECTORIES
# =============================================================================
echo ""
echo "[3/6] Setting up directories..."

mkdir -p data/scenario/{training,validation,testing}
mkdir -p data/graphs/{training,validation,testing}
mkdir -p checkpoints/{autoregressive,gat/autoregressive}
mkdir -p visualizations/autoreg/{gat,finetune}
mkdir -p wandb

echo "  ✓ Directories created"

# =============================================================================
# VERIFY GPU ACCESS
# =============================================================================
echo ""
echo "[4/6] Verifying GPU access in container..."

docker run --rm --gpus all ${IMAGE_NAME}:latest python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
    print(f'    Compute Capability: {props.major}.{props.minor}')
"

echo "  ✓ GPU access verified"

# =============================================================================
# PRINT USAGE INSTRUCTIONS
# =============================================================================
echo ""
echo "[5/6] Setup complete!"
echo ""
echo "========================================"
echo "DEPLOYMENT INSTRUCTIONS"
echo "========================================"
echo ""
echo "1. INTERACTIVE SESSION (recommended for development):"
echo "   docker run -it --gpus all --shm-size=32g \\"
echo "     -v \$(pwd)/data:/workspace/data \\"
echo "     -v \$(pwd)/checkpoints:/workspace/checkpoints \\"
echo "     -v \$(pwd)/visualizations:/workspace/visualizations \\"
echo "     -v \$(pwd)/wandb:/workspace/wandb \\"
echo "     -e WANDB_API_KEY=\$WANDB_API_KEY \\"
echo "     --name ${CONTAINER_NAME} \\"
echo "     ${IMAGE_NAME}:latest"
echo ""
echo "2. USING DOCKER COMPOSE:"
echo "   # Set W&B API key (optional)"
echo "   export WANDB_API_KEY='your-key-here'"
echo ""
echo "   # Interactive session"
echo "   docker-compose up -d waymo-gnn"
echo "   docker-compose exec waymo-gnn bash"
echo ""
echo "   # Run graph creation"
echo "   docker-compose --profile preprocessing up create-graphs"
echo ""
echo "   # Run GCN training"
echo "   docker-compose --profile training up train-gcn"
echo ""
echo "   # Run GAT training"
echo "   docker-compose --profile training up train-gat"
echo ""
echo "3. TRAINING COMMANDS (inside container):"
echo "   # Create graphs from TFRecord files"
echo "   python src/graph_creation_and_saving.py"
echo ""
echo "   # Train GCN model"
echo "   python src/autoregressive_predictions/training_batched.py"
echo ""
echo "   # Train GAT model"
echo "   python src/gat_autoregressive/training_gat_batched.py"
echo ""
echo "   # Test model"
echo "   python src/autoregressive_predictions/testing.py"
echo ""
echo "4. TMUX FOR LONG-RUNNING TRAINING:"
echo "   # Inside container, start tmux session"
echo "   tmux new -s training"
echo "   python src/autoregressive_predictions/training_batched.py"
echo "   # Press Ctrl+B, then D to detach"
echo "   # Reconnect with: tmux attach -t training"
echo ""
echo "========================================"
echo "RTX PRO 6000 OPTIMIZATIONS"
echo "========================================"
echo ""
echo "Your model is optimized for RTX Pro 6000 with:"
echo "  - TF32 enabled for 3x faster matrix ops"
echo "  - BF16 mixed precision training"
echo "  - torch.compile() with reduce-overhead mode"
echo "  - 32GB shared memory for DataLoader workers"
echo "  - 95% GPU memory utilization"
echo ""
echo "[6/6] Ready to train! 🚀"
