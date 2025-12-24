from config import number_of_training_tfrecord_files, number_of_validation_tfrecord_files, number_of_testing_tfrecord_files, max_num_scenarios_per_tfrecord_file, sequence_length
from helper_functions.cloud_tfrecord_downloader import ensure_shards
from helper_functions.graph_creation_functions import (initialize, get_data_files, save_scenarios_to_hdf5_streaming, 
                                       test_hdf5_and_lazy_loading, parse_scenario_file)
from helper_functions.visualization_functions import create_graph_sequence_visualization

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/graph_creation_and_saving.py

if __name__ == '__main__':

    initialize()

    ensure_shards(number_of_training_tfrecord_files, datasets=["training"])    # downloads tfrecord files
    ensure_shards(number_of_validation_tfrecord_files, datasets=["validation"])
    ensure_shards(number_of_testing_tfrecord_files, datasets=["testing"])

    training_files_filepaths = get_data_files("./data/scenario/training")
    validation_files_filepaths = get_data_files("./data/scenario/validation")
    testing_files_filepaths = get_data_files("./data/scenario/testing")

    #create_graph_sequence_visualization(parse_scenario_file(training_files_filepaths[0])[0], save_dir='./visualizations/scenario_sequence', num_timesteps=15)

    save_scenarios_to_hdf5_streaming(training_files_filepaths, f'./data/graphs/training/training_seqlen{sequence_length}.hdf5', max_num_scenarios_per_tfrecord_file=max_num_scenarios_per_tfrecord_file)
    save_scenarios_to_hdf5_streaming(validation_files_filepaths, f'./data/graphs/validation/validation_seqlen{sequence_length}.hdf5', max_num_scenarios_per_tfrecord_file=max_num_scenarios_per_tfrecord_file)
    save_scenarios_to_hdf5_streaming(testing_files_filepaths, f'./data/graphs/testing/testing_seqlen{sequence_length}.hdf5', max_num_scenarios_per_tfrecord_file=max_num_scenarios_per_tfrecord_file)

    #test_hdf5_and_lazy_loading(f'./data/graphs/training/training_seqlen{sequence_length}.hdf5')
