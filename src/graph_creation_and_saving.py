import torch
import h5py
from config import viz_scenario_dir
from helper_functions.graph_creation_functions import (initialize, get_data_files, save_scenarios_to_hdf5_streaming, 
                                       test_hdf5_and_lazy_loading, parse_scenario_file)
from helper_functions.visualization_functions import create_graph_sequence_visualization

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/graph_creation_and_saving.py

if __name__ == '__main__':

    initialize()

    training_files_filepaths = get_data_files("../data/scenario/training")

    create_graph_sequence_visualization(parse_scenario_file(training_files_filepaths[0])[0], save_dir='../' + viz_scenario_dir)

    save_scenarios_to_hdf5_streaming(training_files_filepaths, '../data/graphs/training/training.hdf5')

    test_hdf5_and_lazy_loading('../data/graphs/training/training.hdf5')
