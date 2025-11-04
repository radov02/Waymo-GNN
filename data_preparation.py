import torch
from torch_geometric.data import HeteroData
from graph_creation import 

data = HeteroData()     # PyG's object representing a heterogeneous graph (multiple node and/or edge types)

# node types (assigning features to them):
num_cars = 10
num_car_features = 3
data['car'].x = torch.randn(num_cars, num_car_features)
num_pedestrians = 10
num_pedestrian_features = 3
data['pedestrian'].x = torch.tensor([[ 0.2263, -0.1631, -0.3624],
                                    [-0.7786, -0.8193,  1.4414],
                                    [ 0.4437,  0.6856,  1.5653],
                                    [-0.8561, -0.6644,  0.2692],
                                    [ 1.7984, -0.4106,  0.2282],
                                    [ 0.8712,  2.4511,  0.1243],
                                    [-0.8279,  1.3517, -0.1583],
                                    [-0.3681,  0.4416, -0.9410],
                                    [-2.1176,  0.8650, -1.3122],
                                    [-1.0826,  1.3534,  0.0815]])
print(data['pedestrian'].x, data['pedestrian'].x.shape)


# %%
# edge types:
# edge type (author, writes, paper):
data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]

# %%
# PyTorch tensor functionality:
device = 'cpu'
if torch.cuda.is_available():
    data = data.pin_memory()
    device = 'cuda'
data = data.to(device, non_blocking=True)

print(f"Device: {device}")

# %% [markdown]
# ### Saving graphs:
# 

# %%
torch.save(data, 'graph.pt')

# %% [markdown]
# ### Dataset:

# %%
from torch_geometric.data import Data, InMemoryDataset, Dataset

# suppose we have three graphs and want to create a dataset out of them:
g1 = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0,1,2],[1,2,0]]))
g2 = Data(x=torch.randn(4, 2), edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]))
g3 = Data(x=torch.randn(5, 2), edge_index=torch.tensor([[0,1,2,3,4],[1,2,3,4,0]]))

# use InMemoryDataset:
class MyGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(".")
        self.data, self.slices = self.collate(data_list)
    
    def __len__(self):
        return self.data.num_graphs
    

# create the dataset:
dataset = MyGraphDataset([g1,g2,g3])

# %% [markdown]
# NOTE: ```InMemoryDataset``` base class loads EVERYTHING into memory at once by collecting all the Data objects into a single big tensor (with slice indices so PyG knows how to split them back out).
# 
# If you have limited memory, then use ```Dataset``` base class.
# 
# Or even better for our purpose, we can use the ```TemporalDataset``` class, which handles dataset in a way to use lazy loading (one graph snapshot at a time), optional preprocessing and clean iteration:

# %%
class LazyTemporalDataset:
    def __init__(self, snapshot_paths):
        """snapshot_paths is a list of filepaths
           
            (node_features, edge_index, edge_weight, targets_y)"""
        self.snapshot_paths = snapshot_paths

    def __len__(self):
        return len(self.snapshot_paths)

    def __getitem__(self, idx):
        snapshot_path_string = self.snapshot_paths[idx]
        data = torch.load(snapshot_path_string, map_location=torch.device(device), 
                          weights_only=False)    # loads tensors to the device
        
        # Handle both HeteroData and dict formats
        if isinstance(data, dict):
            x = data['x']
            edge_index = data['edge_index']
            edge_weight = data.get('edge_weight')
            y = data.get('y')
            return (x, edge_index, edge_weight, y)
        else:
            # Assume it's a PyG Data or HeteroData object
            return data

    def snapshots(self):    # generator interface
        for idx in range(len(self)):
            yield self[idx]     # self[idx] triggers __getitem__()


# %% [markdown]
# How does ```yield``` work?
# - when it is executed, python suspends (pauses) the function and hands control back to the caller
# - caller then has to call ```next(theGeneratorObject)```, then python will resume exactly after the last yield (note: for loop calls ```next(theGeneratorObject)``` by itself)
# 
# Why does ```self[idx]``` trigger ```self.__getitem__(key)```?
# - the first is just syntactic sugar for the latter
# - the special methods in python have names like ```__method__(...)```
#     - they are hooks, which the interpreter calls automatically for certain operations, examples:
#         - ```__init__``` is called on instance creation (constructor)
#         - ```__str__``` is called by ```str()``` or ```print()``` (to string)
#         - ```__getitem__``` is called by ```obj[key]``` (indexing using [])
#     - they can be overriden/implemented (not necessarily pre-implemented) to enable the operations (the printout, the subscription using [], ...)
#     - not all classes must have them

# %%
# USAGE:
snapshot_paths = ["graph.pt"]  # saved tensors """snap0.pt", "snap1.pt", "snap2.pt"""
dataset = LazyTemporalDataset(snapshot_paths)

# direct iteration:
for data in dataset.snapshots():
    # Handle both tuple and PyG object formats
    if isinstance(data, tuple):
        x, edge_index, edge_weight, y = data
        print(x.shape, edge_index.shape, y)
    else:
        # PyG Data or HeteroData object
        print(f"Loaded PyG object: {type(data)}")
        print(f"Node types: {data.node_types if hasattr(data, 'node_types') else 'N/A'}")

# separate iteration:
"""gen = dataset.snapshots()
first_snapshot = next(gen)"""


# %% [markdown]
# ### Dataloader:

# %%
from torch_geometric.loader import DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    print(batch)
