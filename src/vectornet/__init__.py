from .VectorNet import (
    VectorNetTFRecord,
    PolylineSubgraphNetwork,
    GlobalInteractionGraph,
    MultiStepTrajectoryDecoder,
    AGENT_VECTOR_DIM,
    MAP_VECTOR_DIM,
)

try:
    from .vectornet_tfrecord_dataset import (
        VectorNetTFRecordDataset,
        vectornet_collate_fn,
        create_vectornet_dataloaders
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
    'VectorNetTFRecord',
    'PolylineSubgraphNetwork',
    'GlobalInteractionGraph',
    'MultiStepTrajectoryDecoder',
    'AGENT_VECTOR_DIM',
    'MAP_VECTOR_DIM',
    
    'VectorNetTFRecordDataset',
    'vectornet_collate_fn',
    'create_vectornet_dataloaders',
    'HAS_TFRECORD_DATASET',
    
    'get_vectornet_config',
    'create_vectornet_model',
    'print_vectornet_config',
    
    'compute_vectornet_metrics',
    'compute_ade_fde',
    'visualize_vectornet_predictions',
    'count_parameters',
    'print_model_summary',
    'MetricTracker',
]
