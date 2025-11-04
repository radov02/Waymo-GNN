import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 33     # divisible by 11
num_workers = 1