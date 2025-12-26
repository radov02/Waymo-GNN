"""Fine-tune single-step GCN model for autoregressive prediction using scheduled sampling."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (autoreg_num_rollout_steps, autoreg_num_epochs, autoreg_sampling_strategy)
from gat_autoregressive.finetune import run_autoregressive_finetuning

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/finetune_single_step_to_autoregressive.py

if __name__ == '__main__':
    model = run_autoregressive_finetuning(
        model_type="gcn",
        pretrained_checkpoint="./checkpoints/best_model.pt",
        num_rollout_steps=autoreg_num_rollout_steps,
        num_epochs=autoreg_num_epochs,
        sampling_strategy=autoreg_sampling_strategy
    )
