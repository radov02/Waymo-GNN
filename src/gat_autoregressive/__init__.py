"""GAT Autoregressive Predictions Module.

This module contains GAT (Graph Attention Network) based models and scripts
for trajectory prediction with autoregressive multi-step rollout.

Files:
- SpatioTemporalGAT.py: GAT-only model architecture
- training.py: Train GAT model from scratch
- finetune.py: Fine-tune GAT model for autoregressive prediction
- testing.py: Evaluate GAT model on test scenarios
"""

from .SpatioTemporalGAT import SpatioTemporalGAT

__all__ = ['SpatioTemporalGAT']
