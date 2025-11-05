import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 33     # divisible by 11
num_workers = 1
num_layers = 3
input_dim = 5
output_dim = 5
topk = 7
epochs = 20
radius = 30.0
graph_creation_method = 'radius'
sequence_length = 11