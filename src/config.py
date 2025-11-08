import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4

# data download:
number_of_training_tfrecord_files = 10
number_of_validation_tfrecord_files = 10
number_of_testing_tfrecord_files = 10

# graph creation:
radius = 30.0  # 30m is more reasonable for local interactions
graph_creation_method = 'radius'  # 'radius' or 'star'
sequence_length = 11

# model:
input_dim = 9       # 5 properties (x, y, vx, vy, valid) + 4 one-hot object type
output_dim = 2      # predicting (x, y) position for next timestep
hidden_channels = 64
num_layers = 3
topk = 7
dropout = 0.3

# training:
batch_size = 10     # divisible by sequence_length for clean batching
learning_rate = 0.001  # FIXED: Reduced from 0.01 to 0.001 (10x smaller)
epochs = 20

# wandb:
project_name = "waymo-project"
dataset_name = "waymo open motion dataset v 1_3_0"

# visualization during training:
visualize_every_n_epochs = 1  # Generate visualizations every 5 epochs
visualize_first_batch_only = False  # FIXED: Only visualize first batch to save time
max_nodes_per_graph_viz = 15  # Max nodes to show per graph in visualization
show_timesteps_viz = 10  # Number of timesteps to show in visualization (reduced from sequence_length-1)
# visualization paths:
viz_base_dir = 'visualizations'  # Base directory for all visualizations
viz_training_dir = 'visualizations/training'  # Training progress visualizations
viz_scenario_dir = 'visualizations/scenario_sequence'  # Graph sequence creation visualizations
