import torch
import h5py
from config import device, batch_size, num_workers, radius, graph_creation_method
from graph_creation_functions import initialize, get_data_files, create_scenario_dataset_dict, get_graphs_for_scenarios, save_scenarios_to_hdf5, test_hdf5_and_lazy_loading

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python graph_creation.py

if __name__ == '__main__':
    
    initialize()

    training_files = get_data_files(".\\data\\scenario\\training")
    validation_files = get_data_files(".\\data\\scenario\\validation")
    testing_files = get_data_files(".\\data\\scenario\\testing")

    #print(training_files)


    training_dataset_dict = create_scenario_dataset_dict(training_files)    # {fileID: [scenarios...]}
    validation_dataset_dict = create_scenario_dataset_dict(validation_files)
    testing_dataset_dict = create_scenario_dataset_dict(testing_files)


    graphs_of_training_scenarios = get_graphs_for_scenarios(training_dataset_dict, radius, graph_creation_method)
    #graphs_of_validation_scenarios = get_graphs_for_scenarios(validation_dataset_dict, radius, graph_creation_method)
    #graphs_of_testing_scenarios = get_graphs_for_scenarios(testing_dataset_dict, radius, graph_creation_method)


    # we use HDF5 file to efficiently store the graphs locally:
    save_scenarios_to_hdf5(graphs_of_training_scenarios, 'data/graphs/training.hdf5')
    #save_scenarios_to_hdf5(graphs_of_validation_scenarios, 'data\graphs\validation')
    #save_scenarios_to_hdf5(graphs_of_testing_scenarios, 'data\graphs\testing')

    test_hdf5_and_lazy_loading()