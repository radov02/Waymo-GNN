import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== DataLoader Configuration ==============
# num_workers: Parallel data loading processes
# - Set to 0 on Windows if you get multiprocessing errors
# - Set to 2-4 on Linux for parallel loading
# - Higher values = more RAM usage but faster loading
#num_workers = min(4, max(0, torch.cuda.device_count() * 2)) if torch.cuda.is_available() else 0
num_workers = 0

# pin_memory: Faster CPU→GPU transfer (only useful with CUDA)
pin_memory = torch.cuda.is_available()

# prefetch_factor: How many batches to prefetch per worker
# Higher values improve GPU utilization but use more RAM
prefetch_factor = 4 if num_workers > 0 else None

debug_mode = False

# ============== Memory Optimization ==============
# Gradient checkpointing: Trade compute for memory (useful for large models/sequences)
use_gradient_checkpointing = False  # Enable if running out of GPU memory

# Validation sampling: randomly sample N scenarios from validation set each epoch
# This speeds up validation while still monitoring model performance
max_validation_scenarios = 20  # Max scenarios to randomly sample for validation per epoch

# Cache validation scenarios in RAM for faster validation
cache_validation_data = True
max_validation_cache_size = max_validation_scenarios  # Cache size matches sample size

# ============== CUDA Optimization ==============
# Enable cuDNN optimizations for better performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
    torch.backends.cudnn.enabled = True    # Enable cuDNN
    
    # TF32 mode: Use TensorFloat-32 for matmuls on Ampere+ (RTX 30xx, A100, H100)
    # ~3x faster than FP32 with minimal precision loss
    if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set optimal CUDA memory allocation strategy
    # Reduces fragmentation for variable-size graphs
    if hasattr(torch.cuda, 'memory'):
        try:
            torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
        except:
            pass  # Not available in older PyTorch versions

# ============== Mixed Precision Training ==============
# Automatic Mixed Precision (AMP) for faster training on modern GPUs (Volta+)
# Uses float16 for forward/backward, float32 for optimizer
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7  # Volta or newer

# Use BF16 instead of FP16 on Ampere+ GPUs (better numerical stability, native on H100)
# BF16 has same exponent range as FP32, so no gradient scaling needed
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

# torch.compile() optimization (PyTorch 2.0+)
# Requires Triton backend which needs CUDA Capability >= 7.0 (Volta+)
# Auto-disabled on Pascal (GTX 10xx) and older GPUs
#use_torch_compile = (
#    torch.cuda.is_available() and 
#    torch.cuda.get_device_capability()[0] >= 7 and  # Volta or newer (Triton requirement)
#    hasattr(torch, 'compile')  # PyTorch 2.0+
#)
# Compile mode: 'default', 'reduce-overhead', 'max-autotune'
# - 'default': Good balance of compile time and speedup
# - 'reduce-overhead': Lower kernel launch overhead (good for small batches)
# - 'max-autotune': Maximum optimization (longer compile, best runtime)
torch_compile_mode = "reduce-overhead"  # Best for batch_size=1 with many small kernels
use_torch_compile = False

# ============== Multi-GPU Configuration ==============
# Automatically detect available GPUs
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
use_data_parallel = num_gpus > 1  # Only use DataParallel if multiple GPUs available


def print_gpu_info():
    """Print detailed GPU information for debugging."""
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("GPU CONFIGURATION")
        print(f"{'='*60}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-processor Count: {props.multi_processor_count}")
            # Identify GPU generation
            if props.major >= 9:
                print(f"  Architecture: Hopper (H100) - FP8/BF16 optimized")
            elif props.major >= 8:
                print(f"  Architecture: Ampere (A100/RTX 30xx/40xx) - TF32/BF16 enabled")
            elif props.major >= 7:
                print(f"  Architecture: Volta/Turing - FP16 Tensor Cores")
        
        # Print optimization status
        print(f"\n--- Optimizations ---")
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  TF32 Matmul: {torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends.cuda.matmul, 'allow_tf32') else 'N/A'}")
        print(f"  TF32 cuDNN: {torch.backends.cudnn.allow_tf32 if hasattr(torch.backends.cudnn, 'allow_tf32') else 'N/A'}")
        print(f"  Mixed Precision (AMP): {use_amp}")
        print(f"  BF16 Mode: {use_bf16}")
        print(f"  torch.compile: {use_torch_compile} (mode={torch_compile_mode if use_torch_compile else 'N/A'})")
        print(f"{'='*60}\n")
    else:
        print("\nNo CUDA GPU available - running on CPU\n")


def setup_model_parallel(model, device):
    """
    Wrap model with DataParallel if multiple GPUs are available.
    
    Args:
        model: PyTorch model
        device: Target device
    
    Returns:
        model: Wrapped model (or original if single GPU/CPU)
        is_parallel: Boolean indicating if model is wrapped
    """
    if use_data_parallel:
        print(f"\nMulti-GPU Training: Using {num_gpus} GPUs with DataParallel")
        print(f"   GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        model = nn.DataParallel(model)
        model = model.to(device)
        return model, True
    else:
        if torch.cuda.is_available():
            print(f"\nSingle-GPU Training: {torch.cuda.get_device_name(0)}")
        else:
            print(f"\nCPU Training (no GPU detected)")
        model = model.to(device)
        return model, False


def get_model_for_saving(model, is_parallel):
    """
    Get the underlying model for saving (unwrap DataParallel if needed).
    DataParallel adds 'module.' prefix to state_dict keys.
    
    Args:
        model: Model (possibly wrapped in DataParallel)
        is_parallel: Whether model is wrapped
    
    Returns:
        The underlying model for state_dict saving
    """
    if is_parallel:
        return model.module
    return model


def load_model_state(model, state_dict, is_parallel):
    """
    Load state dict into model, handling DataParallel prefix.
    
    Args:
        model: Model to load into
        state_dict: State dict (may have 'module.' prefix)
        is_parallel: Whether current model is wrapped in DataParallel
    """
    # Check if state_dict has 'module.' prefix
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_parallel and not has_module_prefix:
        # Model is parallel but checkpoint isn't - add prefix
        new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    elif not is_parallel and has_module_prefix:
        # Model is not parallel but checkpoint is - remove prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        # Both match
        model.load_state_dict(state_dict)

# ============== Physics Constants ==============
TIMESTEP_DT = 0.1           # seconds per timestep
MAX_SPEED = 30.0            # m/s (~108 km/h) - max speed for normalization
MAX_ACCEL = 10.0            # m/s² (~1g) - max acceleration for normalization
MAX_DIST_SDC = 100.0        # meters - max distance to ego vehicle
MAX_DIST_NEAREST = 50.0     # meters - max distance to nearest neighbor
CM_TO_METERS = 100.0        # Waymo uses centimeters, convert to meters

# data download:
number_of_training_tfrecord_files = 5
number_of_validation_tfrecord_files = 5
number_of_testing_tfrecord_files = 5

# graph creation:
radius = 35.0
graph_creation_method = 'radius'    # 'radius' or 'star'
sequence_length = 90    # Max timesteps for training/validation. Testing data only has 11 timesteps (auto-detected)
max_num_scenarios_per_tfrecord_file = None  # Use ALL scenarios (was 1 - causing data starvation!)
use_edge_weights = False  # False to disable distance-based edge weights

# model (SpatioTemporalGNN):
input_dim = 15      # 11 properties (vx, vy, speed, heading, valid, ax, ay, rel_x_sdc, rel_y_sdc, dist_sdc, dist_nearest) + 4 one-hot object type
output_dim = 2      # predicting (dx, dy) for next timestep
hidden_channels = 128  # Increased from 64 - more capacity for complex patterns
num_layers = 3      # Number of GCN layers for spatial encoding
num_gru_layers = 1  # Number of GRU layers for temporal encoding
dropout = 0.1       # Reduced from 0.2 - model might be underfitting
use_gat = False     # Set to True to use Graph Attention Networks instead of GCN

# ============== GAT-specific Configuration ==============
gat_num_heads = 4                   # Number of attention heads in GAT
gat_checkpoint_dir = 'checkpoints/gat'              # GAT model checkpoints
gat_checkpoint_dir_autoreg = 'checkpoints/gat'  # GAT autoregressive checkpoints
gat_viz_dir = 'visualizations/autoreg/gat'          # GAT training visualizations
gat_viz_dir_testing = 'visualizations/autoreg/gat/testing'  # GAT test visualizations

# training:
# batch_size: Number of scenarios processed in parallel
# - For 8GB GPU: use 2 (GAT uses more memory than GCN due to attention)
# - For 16GB GPU: use 4-8
# - For 24GB+ GPU: use 8-16
batch_size = 16
learning_rate = 0.001
epochs = 20
gradient_clip_value = 1.0
# Learning rate scheduler settings
scheduler_patience = 5  # Wait 5 epochs before reducing LR
scheduler_factor = 0.5  # Reduce LR by 50% when triggered
min_lr = 1e-5  # Don't go below 0.00001 (previous min_lr=1e-7 was too small)
# Early stopping settings
early_stopping_patience = 15  # Stop if no improvement for N epochs
early_stopping_min_delta = 0.001  # Minimum improvement to count as progress
# Loss weights: Balance direction and magnitude for accurate trajectory prediction
# Higher MSE weight ensures model learns correct displacement magnitudes
loss_alpha = 0.2    # Angle loss weight (directional accuracy)
loss_beta = 0.5     # MSE weight (positional accuracy) - PRIMARY loss for magnitude
loss_gamma = 0.1    # Velocity magnitude consistency
loss_delta = 0.2    # Cosine similarity (directional signal)

# wandb:
project_name = "waymo-project"
dataset_name = "waymo open motion dataset v 1_3_0"

# visualization during training:
visualize_every_n_epochs = 1
visualize_first_batch_only = False
max_nodes_per_graph_viz = 9  # Max nodes to show per graph in visualization
show_timesteps_viz = 9  # Show 9 evenly-spaced timesteps instead of all 90
viz_vehicles_only = True  # Only show vehicles (not pedestrians/cyclists) in training visualization
viz_base_dir = 'visualizations'
viz_training_dir = 'visualizations/autoreg'  # Training visualizations for GCN model (GAT module overrides this)
viz_scenario_dir = 'visualizations/scenario_sequence'

# model checkpoints:
checkpoint_dir = 'checkpoints'
checkpoint_dir_autoreg = 'checkpoints/autoregressive'  # Autoregressive model checkpoints

# ============== Autoregressive Fine-tuning ==============
autoreg_num_rollout_steps = 89       # Number of steps to roll out (20 = 2.0s horizon, max ~19 for T=29)
autoreg_num_epochs = 10             # Number of fine-tuning epochs
autoreg_sampling_strategy = 'linear'  # 'linear', 'exponential', or 'inverse_sigmoid'
autoreg_visualize_every_n_epochs = 1  # Visualize every N epochs during fine-tuning
autoreg_viz_dir = 'visualizations/autoreg'  # Directory for autoregressive visualizations (GCN model)
autoreg_skip_map_features = False    # Skip loading scenario map features for visualization (faster but no roads)
