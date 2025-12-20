from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import base64
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
    """
    Get the Waymo bucket using various authentication methods (in order of priority):
    1. GOOGLE_APPLICATION_CREDENTIALS env var (path to service account JSON file)
    2. ADC_B64 env var (base64-encoded JSON credentials)
    3. ADC_JSON_PATH env var (path to credentials JSON file)
    4. Default gcloud credentials file (Windows/Linux paths)
    5. Application Default Credentials (ADC) - for GCP instances with attached service account
    """
    from google.auth import default as google_auth_default
    from google.oauth2 import service_account
    
    credentials = None
    
    # Method 1: GOOGLE_APPLICATION_CREDENTIALS (standard Google Cloud method)
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path and Path(gac_path).exists():
        print(f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {gac_path}")
        credentials = service_account.Credentials.from_service_account_file(gac_path)
    
    # Method 2: ADC_JSON_PATH - explicit path to JSON credentials
    if credentials is None:
        adc_json_path = os.environ.get("ADC_JSON_PATH")
        if adc_json_path and Path(adc_json_path).exists():
            print(f"Using credentials from ADC_JSON_PATH: {adc_json_path}")
            with open(adc_json_path, 'r') as f:
                adc_info = json.load(f)
            if adc_info.get("type") == "service_account":
                credentials = service_account.Credentials.from_service_account_info(adc_info)
            else:
                credentials = UserCredentials.from_authorized_user_info(adc_info)
    
    # Method 3: ADC_B64 - base64-encoded JSON credentials
    if credentials is None:
        adc_b64 = os.environ.get("ADC_B64")
        if adc_b64:
            try:
                adc_json = base64.b64decode(adc_b64).decode("utf-8")
                adc_info = json.loads(adc_json)
                print("Using credentials from ADC_B64 environment variable")
                if adc_info.get("type") == "service_account":
                    credentials = service_account.Credentials.from_service_account_info(adc_info)
                else:
                    credentials = UserCredentials.from_authorized_user_info(adc_info)
            except Exception as e:
                print(f"WARNING: Failed to decode ADC_B64: {e}")
                print("ADC_B64 must be a base64-encoded JSON credentials file.")
                print("Generate it with: base64 -w0 /path/to/credentials.json")
    
    # Method 4: Default gcloud credentials file
    if credentials is None:
        # Try Windows path
        adc_path = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
        # Try Linux/Mac path if Windows doesn't exist
        if not adc_path.exists():
            adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        
        if adc_path.exists():
            print(f"Using credentials from gcloud default location: {adc_path}")
            with open(adc_path, 'r') as f:
                adc_info = json.load(f)
            if adc_info.get("type") == "service_account":
                credentials = service_account.Credentials.from_service_account_info(adc_info)
            else:
                credentials = UserCredentials.from_authorized_user_info(adc_info)
    
    # Method 5: Application Default Credentials (works on GCP instances)
    if credentials is None:
        try:
            print("Attempting to use Application Default Credentials (ADC)...")
            credentials, _ = google_auth_default()
            print("Using Application Default Credentials")
        except Exception as e:
            raise RuntimeError(
                f"Could not find valid credentials. Tried:\n"
                f"  1. GOOGLE_APPLICATION_CREDENTIALS env var\n"
                f"  2. ADC_JSON_PATH env var\n"
                f"  3. ADC_B64 env var (base64-encoded JSON)\n"
                f"  4. gcloud default credentials file\n"
                f"  5. Application Default Credentials\n\n"
                f"To fix, either:\n"
                f"  - Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json\n"
                f"  - Set ADC_JSON_PATH=/path/to/credentials.json\n"
                f"  - Run: gcloud auth application-default login\n"
                f"  - On GCP: attach a service account to the instance\n\n"
                f"Last error: {e}"
            )
    
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