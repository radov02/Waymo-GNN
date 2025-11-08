from pathlib import Path
from google.cloud import storage
import json
from google.oauth2.credentials import Credentials as UserCredentials

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
    adc_path = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"   # Try to load credentials from default ADC file location
    
    if not adc_path.exists():
        raise RuntimeError("Missing credentials. Please run: gcloud auth application-default login")
    
    with open(adc_path, 'r') as f:
        adc_info = json.load(f)
    
    credentials = UserCredentials.from_authorized_user_info(adc_info)
    storage_client = storage.Client(project=project_id, credentials=credentials)    

    bucket = storage_client.bucket(bucket_name)      # Don't specify user_project - this was causing the permission error
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
        print(f"Downloading {gs_filename} by executing: blob.download_to_filename(filename=str({local_filename}))")
        blob.download_to_filename(filename=str(local_filename))
        print("\tDownloaded:", local_filename)
    except Exception as e:
        print("Failed to write", local_filename, ":", repr(e))

def ensure_shards(num_shards: int) -> None:
    """Download the first ``num_shards`` TFRecords per split and build indices."""

    bucket = get_waymo_bucket()

    for split in ("training", "validation", "testing"):
        for shard_idx in range(num_shards):
            download_scenario_file(bucket, split, shard_idx)



"""
def download_tfrecord_files(num_files=None, datasets=["training"], max_workers=8):
    #Download TFRecord files for specified datasets using multithreading. If num_files is None, uses values from config.py.
    bucket = get_waymo_bucket()
    
    dataset_configs = []
    if "training" in datasets:
        count = num_files if num_files is not None else number_of_training_tfrecord_files
        dataset_configs.append(("training", count))
    
    if "validation" in datasets:
        count = num_files if num_files is not None else number_of_validation_tfrecord_files
        dataset_configs.append(("validation", count))
    
    if "testing" in datasets:
        count = num_files if num_files is not None else number_of_testing_tfrecord_files
        dataset_configs.append(("testing", count))
    
    for dataset_type, file_count in dataset_configs:
        print(f"\nDownloading {file_count} files for {dataset_type} dataset with {max_workers} threads...")
        
        tasks = [(bucket, dataset_type, shard_idx) for shard_idx in range(file_count)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_shard = {
                executor.submit(download_scenario_file, *task): task[2] 
                for task in tasks
            }
            
            # Process completed downloads
            completed = 0
            for future in as_completed(future_to_shard):
                shard_idx = future_to_shard[future]
                try:
                    future.result()  # This will raise an exception if the download failed
                    completed += 1
                    if completed % 10 == 0 or completed == file_count:
                        print(f"Progress: {completed}/{file_count} files processed")
                except Exception as e:
                    print(f"Error downloading shard {shard_idx}: {repr(e)}")
        
        print(f"Completed downloading {dataset_type} dataset")
"""