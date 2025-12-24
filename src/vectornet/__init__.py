"""VectorNet Module for Waymo Open Motion Dataset.

This module implements VectorNet: Encoding HD Maps and Agent Dynamics 
from Vectorized Representation (Gao et al., 2020).

VectorNet is a hierarchical graph neural network that:
1. Encodes spatial locality of individual road components using vectors
2. Models high-order interactions among all components using self-attention

IMPORTANT: VectorNet predicts the FULL future trajectory at once.
It does NOT require autoregressive fine-tuning like GCN/GAT approaches.

Components:
- VectorNetMultiStep: Main model for HDF5 dataset (legacy)
- VectorNetTFRecord: New model for TFRecord dataset with map features
- VectorNetTFRecordDataset: Direct TFRecord loading with agent + map polylines
- VectorNetDataset: Legacy dataset wrapper for HDF5 scenario graphs

Usage:
    # For TFRecord dataset (recommended, has map features):
    from vectornet import VectorNetTFRecord
    from vectornet.vectornet_tfrecord_dataset import VectorNetTFRecordDataset
    
    # For HDF5 dataset (legacy):
    from vectornet import VectorNetMultiStep
    from vectornet import VectorNetDataset

Example (TFRecord - recommended):
    from vectornet import VectorNetTFRecord, AGENT_VECTOR_DIM, MAP_VECTOR_DIM
    from vectornet.vectornet_tfrecord_dataset import VectorNetTFRecordDataset
    
    # Create dataset
    dataset = VectorNetTFRecordDataset(
        tfrecord_dir='data/scenario',
        split='training',
        history_len=10,
        future_len=50
    )
    
    # Create model (predicts 50 future timesteps at once)
    model = VectorNetTFRecord(
        agent_input_dim=AGENT_VECTOR_DIM,  # 16
        map_input_dim=MAP_VECTOR_DIM,      # 13
        hidden_dim=128,
        prediction_horizon=50
    )
    
    # Forward pass - returns [B, 50, 2] predictions
    predictions = model(batch)
"""

from .VectorNet import (
    VectorNet,
    VectorNetMultiStep,
    VectorNetTemporal,
    VectorNetTFRecord,
    PolylineSubgraphNetwork,
    GlobalInteractionGraph,
    MultiStepTrajectoryDecoder,
    AGENT_VECTOR_DIM,
    MAP_VECTOR_DIM,
)

from .vectornet_dataset import (
    VectorNetDataset,
    VectorNetPolylineDataset,
    collate_vectornet_batch,
    worker_init_fn,
)

# TFRecord dataset imports (handle gracefully if dependencies missing)
try:
    from .vectornet_tfrecord_dataset import (
        VectorNetTFRecordDataset,
        vectornet_collate_fn,
        create_vectornet_dataloaders,
        VectorNetDatasetWrapper,
    )
    HAS_TFRECORD_DATASET = True
except ImportError as e:
    HAS_TFRECORD_DATASET = False
    print(f"Warning: TFRecord dataset not available: {e}")

from .vectornet_helpers import (
    compute_vectornet_metrics,
    compute_ade_fde,
    visualize_vectornet_predictions,
    count_parameters,
    print_model_summary,
    MetricTracker,
)

__version__ = '3.0.0'  # TFRecord dataset version with map features

__all__ = [
    # Models
    'VectorNet',
    'VectorNetMultiStep',
    'VectorNetTemporal',
    'VectorNetTFRecord',
    'PolylineSubgraphNetwork',
    'GlobalInteractionGraph',
    'MultiStepTrajectoryDecoder',
    'AGENT_VECTOR_DIM',
    'MAP_VECTOR_DIM',
    
    # HDF5 Dataset (legacy)
    'VectorNetDataset',
    'VectorNetPolylineDataset',
    'collate_vectornet_batch',
    'worker_init_fn',
    
    # TFRecord Dataset (recommended)
    'VectorNetTFRecordDataset',
    'vectornet_collate_fn',
    'create_vectornet_dataloaders',
    'VectorNetDatasetWrapper',
    'HAS_TFRECORD_DATASET',
    
    # Config
    'get_vectornet_config',
    'create_vectornet_model',
    'print_vectornet_config',
    
    # Helpers
    'compute_vectornet_metrics',
    'compute_ade_fde',
    'visualize_vectornet_predictions',
    'count_parameters',
    'print_model_summary',
    'MetricTracker',
]
