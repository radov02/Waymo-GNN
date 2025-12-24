import torch
import torch.nn as nn


# ============== Physics Constants ==============
MAX_SPEED = 30.0            # max speed (108 km/h) for normalization
MAX_ACCEL = 10.0            # max acceleration for normalization in m/s^2
MAX_DIST_SDC = 100.0        # max distance in m to SDC vehicle
MAX_DIST_NEAREST = 50.0     # max distance in m to nearest neighbor
POSITION_SCALE = 100.0      # scale factor for positions (m <-> cm)

# ============== Data Download ==============
number_of_training_tfrecord_files = 5
number_of_validation_tfrecord_files = 5
number_of_testing_tfrecord_files = 5


# ============== Graph Creation for SpatioTemporal GCN and GAT ==============
radius = 35.0
graph_creation_method = 'radius'    # 'radius' or 'star'
sequence_length = 90    # length of created graph sequence (max 90)
max_num_scenarios_per_tfrecord_file = None      # Use ALL scenarios (set None)
use_edge_weights = False    # False to disable distance-based edge weights


# ============== SpatioTemporalGNN model (GCN and GAT) ==============
input_dim = 15      # 11 properties (vx, vy, speed, heading, valid, ax, ay, rel_x_sdc, rel_y_sdc, dist_sdc, dist_nearest) + 4 one-hot object type
output_dim = 2      # predicting (dx, dy) for next timestep
hidden_channels = 128   # capacity for complex patterns
num_layers = 3      # number of GCN layers for spatial encoding
num_gru_layers = 1  # number of GRU layers for temporal encoding
dropout = 0.1       # to prevent overfitting
gcn_checkpoint_dir = 'checkpoints/gcn'
gcn_checkpoint_dir_autoreg = 'checkpoints/gcn/autoregressive'
gcn_viz_dir = 'visualizations/gcn'
gcn_viz_dir_autoreg = 'visualizations/gcn/autoreg'
gcn_viz_dir_testing = 'visualizations/gcn/autoreg/testing'
# and GAT-specific configuration:
gat_num_heads = 4                   # number of attention heads in GAT
gat_checkpoint_dir = 'checkpoints/gat'
gat_checkpoint_dir_autoreg = 'checkpoints/gat/autoregressive'
gat_viz_dir = 'visualizations/gat'
gat_viz_dir_autoreg = 'visualizations/gat/autoreg'
gat_viz_dir_testing = 'visualizations/gat/autoreg/testing'
# ============== Training Configuration: ==============
epochs = 20
batch_size = 48     # number of graph sequences per batch
learning_rate = 0.001
gradient_clip_value = 1.0  # gradient clipping threshold to prevent exploding gradients
# Learning rate scheduler settings:
scheduler_patience = 5  # wait scheduler_patience epochs before reducing LR
scheduler_factor = 0.5  # reduce LR by scheduler_factor*100 % when triggered
min_lr = 1e-5         # minimum learning rate
# Early stopping settings:
early_stopping_patience = 10  # stop if no improvement for N epochs
early_stopping_min_delta = 0.00000001  # minimum improvement to count as progress
# Loss weights - balance direction and magnitude for accurate trajectory prediction:
loss_alpha = 0.2    # angle weight (directional accuracy)
loss_beta = 0.5     # MSE weight (positional accuracy) - the primary loss for magnitude
loss_gamma = 0.1    # Velocity magnitude consistency
loss_delta = 0.2    # Cosine similarity (directional signal)
# Training:
visualize_every_n_epochs = 10
visualize_first_batch_only = False
max_nodes_per_graph_viz = 9  # max nodes to show per graph in visualization
show_timesteps_viz = 9      # show show_timesteps_viz evenly-spaced timesteps out of all scenario timespan
viz_vehicles_only = True    # only show vehicles (not pedestrians/cyclists) in training visualization
max_scenario_files_for_viz = 2  # index first n tfrecord files for faster scenario loading for visualization of map features
# ============== Autoregressive Fine-tuning ==============
# CURRICULUM LEARNING: Training starts with 10 steps and gradually increases
# to max_rollout_steps over training. This helps model learn progressively.
autoreg_num_rollout_steps = 50       # Max rollout steps (50 = 5.0s horizon) - curriculum starts at 10
autoreg_num_epochs = 40
autoreg_sampling_strategy = 'linear'  # 'linear', 'exponential', or 'inverse_sigmoid'
autoreg_visualize_every_n_epochs = 1
autoreg_skip_map_features = False


# ============== VectorNet model ==============
# VectorNet predicts FULL trajectory at once (not autoregressive like GCN/GAT)
# Uses polyline encoding of agent trajectories and map features
# Dataset Settings:
vectornet_sequence_length = 90        # max sequence length in dataset
vectornet_max_validation_scenarios = 20     # sample size for validation
vectornet_cache_validation = True       # cache validation data in RAM
# Model Architecture:
vectornet_input_dim = 15
vectornet_hidden_dim = 128
vectornet_output_dim = 2
vectornet_num_polyline_layers = 3  # Polyline Subgraph Network layers
vectornet_num_global_layers = 1      # Global Interaction Graph layers
vectornet_num_heads = 8             # Attention heads in global graph
vectornet_num_gru_layers = 1         # GRU layers for VectorNetTemporal variant
vectornet_dropout = 0.1
vectornet_checkpoint_dir = 'checkpoints/vectornet'
vectornet_checkpoint_prefix = 'vectornet'
vectornet_best_model = 'best_vectornet_model.pt'
vectornet_final_model = 'final_vectornet_model.pt'
vectornet_viz_dir = 'visualizations/vectornet'
vectornet_viz_dir_testing = 'visualizations/vectornet/testing'
# Prediction Settings:
vectornet_mode = 'multi_step'     # 'multi_step' - full trajectory at once
vectornet_prediction_horizon = 50
vectornet_history_length = 10       # past timesteps to encode (1.0s)
vectornet_num_agents_to_predict = 14    # Limit predictions per scenario (None = all)
# ============== VectorNet Training Configuration ==============
vectornet_epochs = 30
vectornet_visualize_every_n_epochs = 5
vectornet_viz_scenarios = 2      # scenarios to visualize during training
vectornet_batch_size = 48        # number of scenario sequences per batch
vectornet_learning_rate = 0.001
vectornet_gradient_clip = 1.0
vectornet_scheduler_patience = 5
vectornet_scheduler_factor = 0.5
vectornet_min_lr = 1e-6
vectornet_early_stopping_patience = 10
vectornet_early_stopping_min_delta = 0.0000001
# Loss Weights:
vectornet_loss_alpha = 0.2    # Angle loss weight
vectornet_loss_beta = 0.5     # MSE loss weight (primary)
vectornet_loss_gamma = 0.1    # Velocity consistency weight
vectornet_loss_delta = 0.2    # Cosine similarity weight
# VectorNet auxiliary task settings:
vectornet_use_node_completion = False  # Node completion pretraining task
vectornet_node_completion_ratio = 0.15  # Fraction of nodes to mask


# ============== Testing Configuration ==============
# Common testing settings for all models (GCN, GAT, VectorNet)
test_hdf5_path = 'data/graphs/testing'  # Path to test HDF5 files
test_num_rollout_steps = 50  # Number of autoregressive steps for testing (5.0s horizon)
test_max_scenarios = 100  # Maximum scenarios to test
test_visualize = True  # Generate visualizations during testing
test_visualize_max = 10  # Maximum scenarios to visualize
test_use_wandb = True  # Log test metrics to wandb
test_horizons = [10, 20, 30, 40, 50]  # Evaluation horizons (timesteps) for metrics


# OTHER CONFIGS:
# ============== Wandb ==============
project_name = "waymo-project"
dataset_name = "waymo open motion dataset v 1_3_0"
vectornet_wandb_project = project_name
vectornet_wandb_name = 'vectornet-training'
vectornet_wandb_tags = ['vectornet', 'waymo', 'trajectory-prediction']

# ============== Device Configuration ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== DataLoader Configuration ==============
pin_memory = torch.cuda.is_available()  # Faster CPU->GPU transfer (CUDA only)
debug_mode = False
enable_debug_viz = False  # Enable [DEBUG VIZ] print statements for debugging visualizations

# GCN Model DataLoader Settings:
gcn_num_workers = 4  # Parallel data loading for GCN (set 0 on Windows if multiprocessing errors)
gcn_prefetch_factor = 8 if gcn_num_workers > 0 else None  # Batches to prefetch per worker (higher = better GPU utilization)
prefetch_factor = gcn_prefetch_factor  # Alias for backward compatibility
# GAT Model DataLoader Settings:
gat_num_workers = 4
gat_prefetch_factor = 8 if gat_num_workers > 0 else None
# VectorNet Model DataLoader Settings:
vectornet_num_workers = 6  # larger batches, more preprocessing per sample
vectornet_prefetch_factor = 10 if vectornet_num_workers > 0 else None  # More prefetch for polyline encoding overhead

# ============== Memory Optimization ==============
use_gradient_checkpointing = False  # Trade compute for memory - Enable if running out of GPU memory
max_validation_scenarios = 20  # Max scenarios to randomly sample for validation per epoch
cache_validation_data = True    # Cache validation scenarios in RAM for faster validation
max_validation_cache_size = max_validation_scenarios  # Cache size matches sample size

# ============== CUDA Optimization ==============
# Enable cuDNN optimizations for better performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
    torch.backends.cudnn.enabled = True    # Enable cuDNN
    if torch.cuda.get_device_capability()[0] >= 8:  # TF32 mode: Use TensorFloat-32 for matmuls on Ampere+ (RTX 30xx, A100, H100), ~3x faster than FP32 with minimal precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.cuda, 'memory'):  # Set optimal CUDA memory allocation strategy, reduces fragmentation for variable-size graphs
        try: torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
        except: pass  # Not available in older PyTorch versions

# ============== Mixed Precision Training ==============
# Automatic Mixed Precision (AMP) for faster training on modern GPUs (Volta+)
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7  # Volta or newer
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
# torch.compile mode: 'default' is most stable with dynamic shapes, 'reduce-overhead' can cause warnings
torch_compile_mode = "default"  # Compile mode: 'default', 'reduce-overhead', 'max-autotune'
use_torch_compile = True  # Enabled for 20-30% speedup on compatible GPUs

# ============== Multi-GPU Configuration ==============
# Automatically detect available GPUs
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
use_data_parallel = num_gpus > 1  # Only use DataParallel if multiple GPUs available

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"\n{'-'*60}")
        print("GPU CONFIGURATION:")
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
    
    # Check if state_dict has '_orig_mod.' prefix from torch.compile()
    has_orig_mod_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    
    # Strip _orig_mod. prefix if present (from torch.compile)
    if has_orig_mod_prefix:
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
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
