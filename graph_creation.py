import torch
import h5py
from config import device, batch_size, num_workers, radius, graph_creation_method
from graph_creation_functions import initialize, get_data_files, create_scenario_dataset_dict, get_graphs_for_scenarios, save_scenarios_to_hdf5_streaming, test_hdf5_and_lazy_loading

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python graph_creation.py

if __name__ == '__main__':

    initialize()

    training_files = get_data_files(".\\data\\scenario\\training")

    save_scenarios_to_hdf5_streaming(training_files, 'data/graphs/training.hdf5', radius, graph_creation_method)

    test_hdf5_and_lazy_loading()