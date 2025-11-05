import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 30     # divisible by 10 so that batches contain full graph sequences
num_workers = 4
num_layers = 3
input_dim = 9       # 5 properties (x, y, vx, vy, valid) + 4 one-hot object type
output_dim = 2      # predicting (x, y) position for next timestep
topk = 7
epochs = 20
radius = 30.0
graph_creation_method = 'radius'
sequence_length = 11