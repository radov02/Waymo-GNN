import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0     # leave 0

debug_mode = False

# ============== Physics Constants ==============
TIMESTEP_DT = 0.1           # seconds per timestep
MAX_SPEED = 30.0            # m/s (~108 km/h) - max speed for normalization
MAX_ACCEL = 10.0            # m/s² (~1g) - max acceleration for normalization
MAX_DIST_SDC = 100.0        # meters - max distance to ego vehicle
MAX_DIST_NEAREST = 50.0     # meters - max distance to nearest neighbor
CM_TO_METERS = 100.0        # Waymo uses centimeters, convert to meters

# data download:
number_of_training_tfrecord_files = 5
number_of_validation_tfrecord_files = 10
number_of_testing_tfrecord_files = 10

# graph creation:
radius = 30.0
graph_creation_method = 'radius'    # 'radius' or 'star'
sequence_length = 30    # Max timesteps for training/validation. Testing data only has 11 timesteps (auto-detected)
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

# training:
batch_size = 1  # must be 1 - each scenario needs separate temporal evolution
learning_rate = 0.001
epochs = 300  # Increased from 100 - model needs more time to converge
gradient_clip_value = 1.0
# Learning rate scheduler settings
scheduler_patience = 5  # Wait 5 epochs before reducing LR
scheduler_factor = 0.5  # Reduce LR by 50% when triggered
min_lr = 1e-5  # Don't go below 0.00001 (previous min_lr=1e-7 was too small)
# Early stopping settings
early_stopping_patience = 15  # Stop if no improvement for N epochs
early_stopping_min_delta = 0.001  # Minimum improvement to count as progress
# Loss weights: Prioritize direction (cosine + angle) over magnitude (MSE)
loss_alpha = 0.4   # Angle loss weight (directional accuracy) - INCREASED
loss_beta = 0.2    # MSE weight (positional accuracy) - DECREASED
loss_gamma = 0.05  # Velocity magnitude consistency - REDUCED (only for low-accel agents)
loss_delta = 0.35  # Cosine similarity (directional signal) - INCREASED significantly

# wandb:
project_name = "waymo-project"
dataset_name = "waymo open motion dataset v 1_3_0"

# visualization during training:
visualize_every_n_epochs = 1
visualize_first_batch_only = False
max_nodes_per_graph_viz = 8  # Max nodes to show per graph in visualization
show_timesteps_viz = 9  # Show 9 evenly-spaced timesteps instead of all 90
viz_vehicles_only = True  # Only show vehicles (not pedestrians/cyclists) in training visualization
viz_base_dir = 'visualizations'
viz_training_dir = 'visualizations/training'
viz_scenario_dir = 'visualizations/scenario_sequence'

# model checkpoints:
checkpoint_dir = 'checkpoints'
