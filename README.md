# Prediction of agent trajectories on Waymo autonomous driving dataset using GNNs

Project for Stanford CS224W 2025 class at FRI.

### Description:

- download WOMD files from [Waymo Cloud Console](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))) into `./data/scenario/training`, `./data/scenario/validation` and `./data/scenario/testing`

- run ```graph_creation_and_saving.py```, which:
    - gets the locally downloaded scenario `.tfrecord` filepaths (at `./data/scenario`)
    - creates visualization of first graph sequence
    - goes through files (using filepaths from above) one at a time, parses it from TFRecord to scenario object, creates graphs for the scenario sequence and stores it compressed into HDF5 file with structure `scenarios/{scenario_id}/snapshot_graphs/{timestep}/x|edge_index|edge_weight|y`
    - tests lazy loading from HDF5 file using HDF5TemporalDataset dataset object, and batching

- run ```training.py```, which:
    - logs in to wandb using ```.env``` and initializes wandb run
    - registers given parameters from file ```config.py``` to wandb run
    - instantiates GNN model
    - defines optimizer, loss function
    - instantiates dataset object from HDF5 file
    - instantiates dataloader for loading graphs into batches
        > Graph minibatching:
        > - to parallelize the processing, PyG combines more graphs into single graph with many disconnected components (torch_geometric.data.Batch)
        > - the batch attribute of torch_geometric.data.Batch object is a vector, mapping each node to the index of corresponding graph like: [0, ..., 0, 1, ..., n - 2, n - 1, ..., n - 1]
    - runs training loop for defined epochs:
        - go through batches from dataloader:
            - go through all T timesteps sequentially (not parallelized on GPU due to temporal dependencies that have to be learned by GRU RNN in EvolveGCN!)
                - do forward pass on each of B graphs in batch for current timestep in PARALLEL on GPU
                - compute loss for batch at current timestep and add to accumulated loss
            - get average loss through all timesteps and perform backpropagation (learning)
        - calculate loss for epoch and log it to wandb
    - finishes wandb run
