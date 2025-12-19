"""
YOLO Training Utilities Package for Databricks
Author: may.merkletan@databricks.com
Last Updated: 2025Oct22

This package contains modular utilities for YOLO instance segmentation training,
inference, and visualization on Databricks.
"""

# Import all utility functions for easy access
from .yolo_utils import (
    set_seeds,
    path_exists,
    get_organized_paths,
    setup_yolo_paths,
    check_yolo_environment,
    get_yolo_paths,
    get_inference_output_path,
    validate_data_yaml,
    get_split_info,
    copy_to_uc_volumes_with_yaml,
)

from .mlflow_callbacks import (
    mlflow_epoch_logger,
    configure_checkpoint_logging,
    copy_training_artifacts,
    log_training_artifacts_to_mlflow,
    finalize_training_run
)

from .inference_utils import (
    find_model_by_run_id,
    load_model_from_run,
    run_inference_with_metrics,
    inspect_inference_output
)

from .visualization_utils import (
    visualize_inference_results,
    visualize_predictions_vs_ground_truth
)

from .summary_utils import (
    # clear_cuda_cache,
    # clear_cuda_cache_aggressive,
    # gpu_status,
    print_inference_summary,
    print_multi_split_summary,
    export_inference_summary_markdown
)

from .cache_utils import (
    clear_cuda_cache,
    clear_cuda_cache_aggressive,
    gpu_status,
    clear_all_caches
)

__all__ = [
    # yolo_utils
    'set_seeds',
    'path_exists',
    'get_organized_paths',
    'setup_yolo_paths',
    'check_yolo_environment',
    'get_yolo_paths',
    'get_inference_output_path',
    'validate_data_yaml',
    'get_split_info',
    'copy_to_uc_volumes_with_yaml',
    
    # mlflow_callbacks
    'mlflow_epoch_logger',
    'configure_checkpoint_logging',
    'copy_training_artifacts',
    'log_training_artifacts_to_mlflow',
    'finalize_training_run',
    
    # inference_utils
    'find_model_by_run_id',
    'load_model_from_run',
    'run_inference_with_metrics',
    'inspect_inference_output',
    
    # visualization_utils
    'visualize_inference_results',
    'visualize_predictions_vs_ground_truth',
    
    # summary_utils
    'print_inference_summary',
    'print_multi_split_summary',
    'export_inference_summary_markdown',

    # cache_utils
    'clear_cuda_cache',
    'clear_cuda_cache_aggressive',
    'gpu_status',
    'clear_all_caches'
]

__version__ = '0.3.3'