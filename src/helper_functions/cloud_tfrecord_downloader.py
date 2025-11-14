from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
from google.cloud import storage
import json
from google.oauth2.credentials import Credentials as UserCredentials
from config import number_of_training_tfrecord_files, number_of_validation_tfrecord_files, number_of_testing_tfrecord_files

gs_template = "uncompressed/scenario/{dataset}/{dataset}.tfrecord-{i}-of-{length}"    # gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training/testing.tfrecord-00000-of-01000
dataset_name = "scenario"

project_id = "waymo-gnn-475616"
bucket_name = "waymo_open_dataset_motion_v_1_3_0"
local_data_prefix = "data/"

def get_gs_filename(dataset, i):
    length = ""
    if dataset == "training":
        length = "01000"
    elif dataset == "validation" or dataset == "testing":
        length = "00150"
    else:
        raise ValueError("Invalid dataset name")

    return gs_template.format(dataset=dataset, i=str(i).zfill(5), length=length)

def get_local_filename(dataset, i):
    return Path(local_data_prefix) / dataset_name / dataset / f"{str(i).zfill(5)}.tfrecord"

def get_waymo_bucket():
    adc_path = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    
    if not adc_path.exists():
        raise RuntimeError("Missing credentials. Please run: gcloud auth application-default login")
    
    with open(adc_path, 'r') as f:
        adc_info = json.load(f)
    
    credentials = UserCredentials.from_authorized_user_info(adc_info)
    storage_client = storage.Client(project=project_id, credentials=credentials)    

    bucket = storage_client.bucket(bucket_name)
    return bucket

def download_scenario_file(bucket, dataset, i):
    gs_filename = get_gs_filename(dataset, i)
    blob = bucket.blob(gs_filename)
    
    local_filename = get_local_filename(dataset, i)
    local_filename.parent.mkdir(parents=True, exist_ok=True)
    if local_filename.exists():
        print("Already exists:", local_filename)
        return
    
    try:
        import time
        start_time = time.perf_counter()
        print(f"Downloading {gs_filename} by executing: blob.download_to_filename(filename=str({local_filename}))")
        blob.download_to_filename(filename=str(local_filename))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"\tDownloaded blob {local_filename} in {elapsed_time:.1f} seconds.")
    except (Exception, KeyboardInterrupt) as e:
        print("Failed to write", local_filename, ":", repr(e))
        if local_filename.exists():
            local_filename.unlink()
            print(f"\tDeleted incomplete file: {local_filename}")
        raise

def ensure_shards(num_shards: int, datasets=["training"]) -> None:
    """Download the first ``num_shards`` TFRecords per split and build indices."""

    bucket = get_waymo_bucket()

    for split in datasets:
        start_time = time.perf_counter()
        
        for shard_idx in range(num_shards):
            download_scenario_file(bucket, split, shard_idx)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Downloaded {split} files in {elapsed_time:.1f} seconds.\n")