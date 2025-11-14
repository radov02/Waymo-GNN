import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0

debug_mode = False

# data download:
number_of_training_tfrecord_files = 10
number_of_validation_tfrecord_files = 10
number_of_testing_tfrecord_files = 10

# graph creation:
radius = 30.0  # 30m is more reasonable for local interactions
graph_creation_method = 'radius'  # 'radius' or 'star'
sequence_length = 91
max_num_scenarios_per_tfrecord_file = 1  # set to None if all
use_edge_weights = False  # Set to False to disable distance-based edge weights

# model:
input_dim = 9       # 5 properties (x, y, vx, vy, valid) + 4 one-hot object type
output_dim = 2      # predicting (x, y) position for next timestep
hidden_channels = 64
num_layers = 3
topk = 7
dropout = 0.3

# training:
batch_size = 1
learning_rate = 0.005  # Lower LR to avoid mode collapse with directional loss
epochs = 30  # Extended for better convergence with stronger directional loss
gradient_clip_value = 5.0  # Less aggressive clipping
loss_alpha = 0.6  # ANGLE weight: 60% - DOMINANT! (correct direction is critical)
loss_beta = 0.15  # MSE weight: 15% (magnitude)
loss_gamma = 0.15  # Velocity consistency: 15% (physics prior - increased, diversity removed)
loss_delta = 0.1  # Cosine similarity: 10% (backup directional signal)

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
