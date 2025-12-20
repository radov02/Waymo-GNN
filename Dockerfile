# Dockerfile for Waymo GNN Trajectory Prediction
# Optimized for PrimeIntellect RTX Pro 6000 96GB instance
# Based on NVIDIA CUDA 12.4 with cuDNN 9 for Ada Lovelace architecture

# =============================================================================
# BASE IMAGE: Official PyTorch with CUDA 12.4 and cuDNN 9 (much smaller than NGC)
# =============================================================================
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # CUDA configuration for RTX Pro 6000 (Ada Lovelace - SM 8.9)
    TORCH_CUDA_ARCH_LIST="8.9" \
    # Enable TF32 for faster matrix operations on Ada architecture
    NVIDIA_TF32_OVERRIDE=1 \
    # Disable TensorFlow GPU memory pre-allocation (PyTorch is primary)
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=2 \
    # cuDNN optimizations
    CUDNN_V8_API_ENABLED=1 \
    # Path configuration
    PYTHONPATH=/workspace/src:/workspace \
    # Weights & Biases configuration (set your key at runtime)
    WANDB_DIR=/workspace/wandb \
    # HDF5 thread-safety
    HDF5_USE_FILE_LOCKING=FALSE

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    screen \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# WORKING DIRECTORY
# =============================================================================
WORKDIR /workspace

# =============================================================================
# PYTHON DEPENDENCIES
# =============================================================================
# Copy requirements first for better Docker layer caching
COPY requirements.txt /workspace/requirements.txt

# Install PyTorch Geometric and other dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    # Install PyTorch Geometric dependencies
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.5.0+cu124.html && \
    pip install --no-cache-dir torch-geometric && \
    # Install remaining requirements
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
        pyyaml>=6.0 && \
    # Install TensorFlow for Waymo dataset parsing
    pip install --no-cache-dir tensorflow>=2.15.0

# =============================================================================
# PROJECT FILES
# =============================================================================
# Copy the entire project
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/checkpoints \
             /workspace/checkpoints/autoregressive \
             /workspace/checkpoints/gat \
             /workspace/checkpoints/gat/autoregressive \
             /workspace/visualizations/autoreg \
             /workspace/visualizations/autoreg/gat \
             /workspace/visualizations/autoreg/finetune \
             /workspace/data/graphs/training \
             /workspace/data/graphs/validation \
             /workspace/data/graphs/testing \
             /workspace/data/scenario/training \
             /workspace/data/scenario/validation \
             /workspace/data/scenario/testing \
             /workspace/wandb

# =============================================================================
# NVIDIA CONTAINER RUNTIME CONFIGURATION
# =============================================================================
# This enables GPU access in the container
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# =============================================================================
# HEALTHCHECK
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}, Count: {torch.cuda.device_count()}')" || exit 1

# =============================================================================
# DEFAULT COMMAND
# =============================================================================
# Default: Start interactive shell
CMD ["/bin/bash"]
