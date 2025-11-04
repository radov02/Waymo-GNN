FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
# pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
# nvcr.io/nvidia/tensorflow:25.02-tf2-py3
# us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cu118.2-13.py310
# tensorflow/tensorflow:2.13.0-gpu-jupyter

# Install Google Cloud CLI
RUN apt-get -y update && apt-get -y install apt-transport-https ca-certificates gnupg curl && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y

WORKDIR /app

# --- Dependency Caching Layer ---
# Copy only the file needed for dependency installation.
COPY requirements.txt ./

# Install project dependencies into the system environment (Python 3.10).
# This layer is cached and only re-run when requirements.txt changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# --- Project Source Layer ---
# Copy the rest of the project source code.
COPY . .

# This requires setup.py or pyproject.toml in the project root, which we don't have.
# Install the project itself in editable mode, without re-installing dependencies.
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install --no-deps -e .

# Cloud: run the training script.
# Local: start JupyterLab for development defined in compose.yaml.
# CMD ["python", "src/train.py"]