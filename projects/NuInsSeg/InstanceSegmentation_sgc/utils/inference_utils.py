"""
YOLO Inference Module for Databricks
Handles model inference with comprehensive validation and tracking
Compatible with Ultralytics YOLO 8.3.x

Author: may.merkletan@databricks.com
Last Updated: 2025Oct23 
"""

import os
import json
import torch
import mlflow
from datetime import datetime
from ultralytics import YOLO


def find_model_by_run_id(run_id, model_type='best', search_volumes=True, try_mlflow=True, use_local_cache=True):
    """
    Find a model file by MLflow run_id with local caching.
    
    Args:
        run_id: MLflow run ID
        model_type: 'best' or 'last'
        search_volumes: Search UC Volumes for matching directory
        try_mlflow: Try downloading from MLflow artifacts
        use_local_cache: Cache downloaded model to /local_disk0/ (recommended for serverless)
    
    Returns:
        str: Path to model file, or None if not found
    """
    import builtins
    import shutil
    
    print(f"Searching for {model_type}.pt with run_id: {run_id}")
    
    # 1. Check local cache first (fastest)
    if use_local_cache:
        local_cache_dir = "/local_disk0/tmp/nuinsseg/models/"
        os.makedirs(local_cache_dir, exist_ok=True)
        cached_model = os.path.join(local_cache_dir, f"{run_id}_{model_type}.pt")
        
        if os.path.exists(cached_model):
            print(f"  ✓ Found in local cache: {cached_model}")
            return cached_model
    
    # 2. Try MLflow artifacts (most reliable)
    if try_mlflow:
        try:
            artifact_path = f"train/weights/{model_type}.pt"
            model_uri = f"runs:/{run_id}/{artifact_path}"
            print(f"  Trying MLflow: {model_uri}")
            
            # MLflow downloads to /tmp/ by default
            mlflow_path = mlflow.artifacts.download_artifacts(model_uri)
            
            if os.path.exists(mlflow_path):
                print(f"  ✓ Downloaded from MLflow to: {mlflow_path}")
                
                # Cache to local_disk0 for faster subsequent access
                if use_local_cache:
                    try:
                        shutil.copy(mlflow_path, cached_model)
                        print(f"  ✓ Cached to local disk: {cached_model}")
                        return cached_model
                    except Exception as e:
                        print(f"  ⚠ Could not cache to local disk: {e}")
                        return mlflow_path
                
                return mlflow_path
        except Exception as e:
            print(f"  ✗ MLflow download failed: {e}")
    
    # 3. Search UC Volumes (only if MLflow failed)
    if search_volumes:
        YOLO_ARTIFACTS_UCVol_path = getattr(builtins, 'YOLO_ARTIFACTS_UCVol_path', None)
        
        if YOLO_ARTIFACTS_UCVol_path and os.path.exists(YOLO_ARTIFACTS_UCVol_path):
            print(f"  Searching UC Volumes: {YOLO_ARTIFACTS_UCVol_path}")
            
            try:
                for item in os.listdir(YOLO_ARTIFACTS_UCVol_path):
                    if f'runid_{run_id}' in item:
                        model_path = os.path.join(
                            YOLO_ARTIFACTS_UCVol_path, 
                            item, 
                            'train', 
                            'weights', 
                            f'{model_type}.pt'
                        )
                        
                        if os.path.exists(model_path):
                            print(f"  ✓ Found in UC Volumes: {model_path}")
                            
                            # Optionally cache to local disk
                            if use_local_cache:
                                try:
                                    shutil.copy(model_path, cached_model)
                                    print(f"  ✓ Cached to local disk: {cached_model}")
                                    return cached_model
                                except Exception as e:
                                    print(f"  ⚠ Could not cache: {e}")
                            
                            return model_path
                
                print(f"  ✗ Not found in UC Volumes")
            except Exception as e:
                print(f"  ✗ UC Volumes search error: {e}")
        else:
            print(f"  ⚠ UC Volumes not available")
    
    print(f"  ✗ Model not found")
    return None


def load_model_from_run(run_id, model_type='best', use_mlflow=False):
    """
    Load a YOLO model from MLflow run or UC Volumes.
    
    Args:
        run_id: MLflow run ID
        model_type: 'best' or 'last'
        use_mlflow: If True, load from MLflow artifacts; if False, load from UC Volumes
    
    Returns:
        YOLO model
    """
    if use_mlflow:
        # Load from MLflow artifacts
        artifact_path = f"train/weights/{model_type}.pt"
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            local_path = mlflow.artifacts.download_artifacts(model_uri)
            model = YOLO(local_path)
            print(f"✓ Loaded {model_type} model from MLflow: {run_id}")
            return model
        except Exception as e:
            print(f"✗ Failed to load from MLflow: {e}")
            raise
    else:
        # Load from UC Volumes using find_model_by_run_id
        model_path = find_model_by_run_id(run_id, model_type=model_type, try_mlflow=False)
        
        if model_path is None:
            raise FileNotFoundError(f"Model not found for run_id: {run_id}")
        
        model = YOLO(model_path)
        print(f"✓ Loaded {model_type} model from: {model_path}")
        return model


def run_inference_with_metrics(
    model_path=None,
    data_yaml_path=None,
    split="val",
    device=None,
    batch_size=None,
    img_size=640,
    save_txt=True,
    save_conf=True,
    save_json=True,
    run_id=None,
    output_base=None,
    log_to_mlflow=True,
    skip_existing_params=False,
    validate_config=True,
    has_labels=None,
    use_nested_run='auto',  # | True, False
    debug=False,
    **kwargs
):
    """
    Run inference and calculate metrics with comprehensive performance tracking.
    
    Auto-detects paths, device, and batch size from globals if not provided.
    Validates data.yaml configuration before running inference.
    Works with any split: 'train', 'val', or 'test'.
    Automatically uses model.val() for labeled data or model.predict() for unlabeled data.
    
    Compatible with Ultralytics YOLO 8.3.200
    
    Args:
        model_path: Path to model weights (.pt file). If None, uses run_id to find model
        data_yaml_path: Path to dataset YAML. If None, uses YOLO_DATA_UCVol_path
        split: Dataset split ('train', 'val', 'test')
        device: Device to use ('cuda', 'cpu', 'auto', or None for 'auto')
        batch_size: Batch size (None for auto-detect based on device)
        img_size: Image size for inference
        save_txt: Save label files
        save_conf: Save confidence scores
        save_json: Save predictions as JSON
        run_id: MLflow run ID (used to find model AND log results)
        output_base: Base output directory (None to auto-detect from current_run_paths)
        log_to_mlflow: Whether to log to MLflow (requires run_id)
        skip_existing_params: Skip logging params if they already exist in MLflow run (ignored if use_nested_run=True)
        validate_config: Whether to validate data.yaml before inference
        has_labels: None (auto-detect), True (use model.val()), False (use model.predict())
        use_nested_run: 'auto' (detect if needed), True (always use), False (never use)
        debug: Enable debug output for troubleshooting
        **kwargs: Additional arguments for visualization
    
    Returns:
        dict: Comprehensive summary with metrics, performance, and paths
    """
    import builtins
    import yaml
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*70}")
    print(f"INFERENCE SETUP: {split.upper()} Split")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # 1. AUTO-DETECT PATHS AND RUN_ID
    # ========================================================================
    print("1. Auto-detecting paths and run_id...")
    
    current_run_paths = getattr(builtins, 'current_run_paths', None)
    
    # Priority for run_id: explicit parameter > auto-detect from current_run_paths
    detected_run_id = None
    if run_id is None:
        if current_run_paths:
            base_path = current_run_paths.get('base', '')
            if 'runid_' in base_path:
                try:
                    detected_run_id = base_path.split('runid_')[1].split('/')[0]
                    print(f"   Run ID (auto-detected): {detected_run_id}")
                except:
                    print(f"   Run ID: Not auto-detected")
            else:
                print(f"   Run ID: Not found in path")
        else:
            print(f"   Run ID: Not provided")
    else:
        print(f"   Run ID (provided): {run_id}")
    
    # Use provided run_id or detected run_id
    active_run_id = run_id if run_id is not None else detected_run_id
    
    # ========================================================================
    # MODEL PATH RESOLUTION
    # ========================================================================
    
    if model_path is None:
        if active_run_id:
            print(f"   Searching for model with run_id: {active_run_id}")
            
            for model_type in ['best', 'last']:
                model_path = find_model_by_run_id(
                    active_run_id,
                    model_type=model_type,
                    search_volumes=True,
                    try_mlflow=True
                )
                if model_path:
                    print(f"   ✓ Model ({model_type}): {model_path}")
                    break
            
            if model_path is None:
                error_msg = f"Could not find model for run_id: {active_run_id}\n\n"
                error_msg += "Searched:\n"
                error_msg += "  1. MLflow artifacts (runs:/{run_id}/train/weights/)\n"
                error_msg += "  2. UC Volumes (YOLO_ARTIFACTS_UCVol_path)\n\n"
                error_msg += "Solutions:\n"
                error_msg += "  1. Verify run_id is correct\n"
                error_msg += "  2. Check if training completed and saved weights\n"
                error_msg += "  3. Provide explicit model_path parameter\n"
                raise FileNotFoundError(error_msg)
        
        elif current_run_paths:
            print(f"   No run_id available, using current_run_paths...")
            possible_paths = [
                os.path.join(current_run_paths.get('train_weights', ''), 'best.pt'),
                os.path.join(current_run_paths.get('train_weights', ''), 'last.pt'),
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    model_path = path
                    model_type = 'best' if 'best.pt' in path else 'last'
                    print(f"   Model ({model_type}): {model_path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"No model found in current_run_paths\n"
                    f"Please provide either run_id or model_path parameter"
                )
        else:
            raise ValueError(
                "model_path not provided and cannot auto-detect\n\n"
                "Please provide either:\n"
                "  - run_id parameter, or\n"
                "  - model_path parameter, or\n"
                "  - set current_run_paths global"
            )
    else:
        print(f"   Model (provided): {model_path}")
    
    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Use active_run_id for the rest of the function
    run_id = active_run_id
    
    # Data YAML path
    if data_yaml_path is None:
        YOLO_DATA_UCVol_path = getattr(builtins, 'YOLO_DATA_UCVol_path', None)
        if YOLO_DATA_UCVol_path:
            data_yaml_path = os.path.join(YOLO_DATA_UCVol_path, "data.yaml")
            print(f"   Data YAML (auto): {data_yaml_path}")
        else:
            raise ValueError("data_yaml_path not provided and YOLO_DATA_UCVol_path not available")
    else:
        print(f"   Data YAML: {data_yaml_path}")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    # Output base path
    if output_base is None:
        if run_id:
            try:
                from .yolo_utils import get_inference_output_path
                output_base = get_inference_output_path(run_id, split)
                print(f"   Output base (from run_id): {output_base}")
            except Exception as e:
                print(f"   ⚠ Could not determine output path: {e}")
                output_base = None
        
        if output_base is None and current_run_paths:
            output_base = (
                current_run_paths.get(f'inference_{split}') or 
                current_run_paths.get(split) or
                os.path.join(current_run_paths.get('base', ''), split)
            )
            print(f"   Output base (from current_run_paths): {output_base}")
        
        if output_base is None:
            output_base = os.path.join(os.getcwd(), "inference_output", split)
            print(f"   ⚠ Output base (workspace fallback): {output_base}")
    else:
        print(f"   Output base (provided): {output_base}")
    
    # Validate run_id if MLflow logging is requested
    if log_to_mlflow and run_id:
        try:
            mlflow.get_run(run_id)
            print(f"   ✓ Run ID validated for MLflow logging: {run_id}")
        except Exception as e:
            print(f"   ✗ Warning: Could not validate run_id: {e}")
            print(f"   MLflow logging will be skipped")
            log_to_mlflow = False
    elif log_to_mlflow and not run_id:
        print(f"   ⚠ Warning: log_to_mlflow=True but no run_id available")
        print(f"   MLflow logging will be skipped")
        log_to_mlflow = False
    
    print()
    
    # ========================================================================
    # 2. VALIDATE DATA.YAML CONFIGURATION & DETECT LABELS
    # ========================================================================
    num_images_expected = None
    split_full_path = None
    
    if validate_config:
        print("2. Validating data.yaml configuration...")
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        available_splits = [k for k in ['train', 'val', 'test'] if k in data_config]
        
        if split not in data_config:
            raise ValueError(
                f"✗ Split '{split}' not found in data.yaml!\n"
                f"   Available splits: {available_splits}\n"
                f"   Data YAML: {data_yaml_path}"
            )
        
        print(f"   ✓ Split '{split}' found in data.yaml")
        
        split_rel_path = data_config[split]
        base_path = data_config.get('path', '')
        
        if split_rel_path.startswith('/Volumes/'):
            split_full_path = split_rel_path
        else:
            split_full_path = os.path.join(base_path, split_rel_path)
        
        print(f"   Split path: {split_full_path}")
        
        if not os.path.exists(split_full_path):
            raise FileNotFoundError(
                f"✗ Split path does not exist: {split_full_path}\n"
                f"   Please verify your dataset is properly copied to UC Volumes."
            )
        
        print(f"   ✓ Path exists")
        
        try:
            image_files = [f for f in os.listdir(split_full_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            num_images_expected = len(image_files)
            print(f"   ✓ Found {num_images_expected} images in {split} split")
        except Exception as e:
            print(f"   ⚠ Warning: Could not count images: {e}")
            num_images_expected = None
        
        if has_labels is None:
            labels_path = split_full_path.replace('/images', '/labels')
            if os.path.exists(labels_path):
                label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]
                has_labels = len(label_files) > 0
                print(f"   {'✓' if has_labels else '✗'} Labels detected: {len(label_files)} label files")
            else:
                has_labels = False
                print(f"   ✗ No labels directory found: {labels_path}")
        
        print(f"   Classes: {data_config.get('nc', 'N/A')}")
        print(f"   Names: {data_config.get('names', 'N/A')}")
        print()
    else:
        print("2. Skipping data.yaml validation (validate_config=False)")
        if has_labels is None:
            has_labels = True
            print("   Assuming labels exist (has_labels not specified)\n")
    
    use_validation = has_labels
    inference_method = "model.val() [with metrics]" if use_validation else "model.predict() [no metrics]"
    print(f"   Inference method: {inference_method}\n")
    
    # ========================================================================
    # 3. AUTO-DETECT DEVICE AND BATCH SIZE
    # ========================================================================
    print("3. Configuring compute resources...")
    
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device (auto): {device}")
    else:
        print(f"   Device: {device}")
    
    if batch_size is None or batch_size == 'auto':
        if device == 'cuda':
            try:
                free_memory = (torch.cuda.get_device_properties(0).total_memory - 
                              torch.cuda.memory_reserved(0)) / 1024**3
                batch_size = 16 if free_memory > 10 else (8 if free_memory > 6 else (4 if free_memory > 3 else 2))
                print(f"   Batch size (auto): {batch_size} (GPU free memory: {free_memory:.1f} GB)")
            except:
                batch_size = 8
                print(f"   Batch size (auto): {batch_size} (default)")
        else:
            batch_size = 4
            print(f"   Batch size (auto): {batch_size} (CPU)")
    else:
        print(f"   Batch size: {batch_size}")
    
    print(f"   Image size: {img_size}")
    print()
    
    # ========================================================================
    # 4. SETUP OUTPUT DIRECTORY
    # ========================================================================
    print("4. Setting up output directory...")
    output_path = os.path.join(output_base, f"inference_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    print(f"   Output: {output_path}\n")
    
    # ========================================================================
    # 5. LOAD MODEL
    # ========================================================================
    print("5. Loading model...")
    if model_path.startswith('/Volumes/'):
        print(f"   Loading from UC Volumes")
    elif '/tmp/' in model_path or '/artifacts/' in model_path:
        print(f"   Loading from MLflow download cache")
    
    model = YOLO(model_path)
    print("   ✓ Model loaded successfully\n")
    
    # ========================================================================
    # 6. RUN INFERENCE
    # ========================================================================
    print(f"{'='*70}")
    print(f"RUNNING INFERENCE: {split.upper()} Split")
    if run_id:
        print(f"Run ID: {run_id}")
    print(f"Method: {inference_method}")
    print(f"{'='*70}\n")
    
    print(f"Starting inference on {split} split...")
    print(f"   Data YAML: {data_yaml_path}")
    print(f"   Split: {split}")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print()
    
    if use_validation:
        results = model.val(
            data=data_yaml_path,
            split=split,
            save_json=save_json,
            save_txt=save_txt,
            save_conf=save_conf,
            project=output_path,
            name="results",
            exist_ok=True,
            device=device,
            batch=batch_size,
            imgsz=img_size,
            plots=True,
            rect=False,
        )
    else:
        print("   Note: Using predict mode (no ground truth labels)")
        
        if split_full_path is None:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            split_rel_path = data_config[split]
            base_path = data_config.get('path', '')
            if split_rel_path.startswith('/Volumes/'):
                split_full_path = split_rel_path
            else:
                split_full_path = os.path.join(base_path, split_rel_path)
        
        results = model.predict(
            source=split_full_path,
            save=True,
            save_txt=save_txt,
            save_conf=save_conf,
            project=output_path,
            name="results",
            exist_ok=True,
            device=device,
            batch=batch_size,
            imgsz=img_size,
        )
    
    print("\n✓ Inference complete\n")
    
    # ========================================================================
    # 7. EXTRACT METRICS AND PERFORMANCE
    # ========================================================================
    print("7. Extracting metrics and performance...")
    
    metrics_dict = {}
    
    if use_validation:
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            
            metrics_dict['box_map50_95'] = results_dict.get('metrics/mAP50-95(B)', 0)
            metrics_dict['box_map50'] = results_dict.get('metrics/mAP50(B)', 0)
            metrics_dict['box_precision'] = results_dict.get('metrics/precision(B)', 0)
            metrics_dict['box_recall'] = results_dict.get('metrics/recall(B)', 0)
            
            metrics_dict['mask_map50_95'] = results_dict.get('metrics/mAP50-95(M)', 0)
            metrics_dict['mask_map50'] = results_dict.get('metrics/mAP50(M)', 0)
            metrics_dict['mask_precision'] = results_dict.get('metrics/precision(M)', 0)
            metrics_dict['mask_recall'] = results_dict.get('metrics/recall(M)', 0)
            
            box_map50_95 = metrics_dict.get('box_map50_95', 0)
            box_map50 = metrics_dict.get('box_map50', 0)
            metrics_dict['fitness'] = 0.1 * box_map50 + 0.9 * box_map50_95
            
            print(f"   Metrics:")
            print(f"      Box mAP50-95:  {metrics_dict.get('box_map50_95', 0):.4f}")
            print(f"      Mask mAP50-95: {metrics_dict.get('mask_map50_95', 0):.4f}")
            print(f"      Fitness:       {metrics_dict.get('fitness', 0):.5f}")
        else:
            print(f"   ⚠ Warning: Validation mode but no results_dict found")
            metrics_dict = {
                'mode': 'validation_no_metrics',
                'note': 'Validator object missing results_dict attribute'
            }
    else:
        print(f"   No metrics available (prediction mode)")
        metrics_dict = {
            'mode': 'prediction_only',
            'note': 'No ground truth labels available for metric calculation'
        }
    
    # Extract performance metrics
    performance = {}
    
    if isinstance(results, list) and len(results) > 0:
        if hasattr(results[0], 'speed'):
            speed = results[0].speed
            performance = {
                'preprocess': speed.get('preprocess', 0),
                'inference': speed.get('inference', 0),
                'postprocess': speed.get('postprocess', 0),
            }
    elif hasattr(results, 'speed'):
        speed = results.speed
        performance = {
            'preprocess': speed.get('preprocess', 0),
            'inference': speed.get('inference', 0),
            'postprocess': speed.get('postprocess', 0),
        }
    
    if performance:
        performance['total'] = sum(performance.values())
        performance['fps'] = 1000.0 / performance['total'] if performance['total'] > 0 else 0
        
        print(f"\n   Performance:")
        print(f"      Preprocess:  {performance['preprocess']:.2f}ms")
        print(f"      Inference:   {performance['inference']:.2f}ms")
        print(f"      Postprocess: {performance['postprocess']:.2f}ms")
        print(f"      Total:       {performance['total']:.2f}ms ({performance['fps']:.1f} FPS)")
    else:
        print(f"\n   ⚠ Warning: Could not extract performance metrics")
    
    # Count processed images
    num_images = 0
    
    if isinstance(results, list):
        num_images = len(results)
    elif hasattr(results, 'seen'):
        num_images = results.seen
    elif use_validation and num_images_expected:
        num_images = num_images_expected
        print(f"\n   Note: Metrics calculated on all {num_images} images")
    
    if num_images == 0:
        print(f"   ⚠ Warning: Could not determine image count")
    
    print(f"\n   Processed {num_images} images")
    
    if num_images_expected is not None:
        if num_images == num_images_expected:
            print(f"   ✓ Image count matches expected ({num_images_expected})")
        else:
            print(f"   ⚠ Warning: Processed {num_images} images but expected {num_images_expected}")
    
    print()
    
    # ========================================================================
    # 8. SAVE SUMMARY
    # ========================================================================
    print("8. Saving metrics summary...")
    
    results_path = os.path.join(output_path, "results")
    
    if not os.path.exists(results_path):
        for subdir in ['predict', 'val', 'test']:
            alt_path = os.path.join(output_path, subdir, "results")
            if os.path.exists(alt_path):
                results_path = alt_path
                print(f"   Note: YOLO created '{subdir}/' subdirectory")
                break
        
        if not os.path.exists(results_path):
            results_path = os.path.join(output_path, "results")
            os.makedirs(results_path, exist_ok=True)
            print(f"   Created results directory: {results_path}")
    
    metrics_json_path = os.path.join(results_path, "metrics_summary.json")
    
    summary_data = {
        'timestamp': timestamp,
        'split': split,
        'inference_method': 'validation' if use_validation else 'prediction',
        'has_labels': use_validation,
        'num_images': num_images,
        'num_images_expected': num_images_expected,
        'device': device,
        'batch_size': batch_size,
        'img_size': img_size,
        'metrics': metrics_dict,
        'performance': performance,
        'model_path': model_path,
        'data_yaml_path': data_yaml_path,
        'run_id': run_id,
        'yolo_version': '8.3.200',
    }
    
    with open(metrics_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"   ✓ Saved to: {metrics_json_path}\n")
    
    # ========================================================================
    # 9. LOG TO MLFLOW - WITH NESTED RUN SUPPORT
    # ========================================================================
    nested_run_id = None
    
    if log_to_mlflow and run_id:
        print(f"9. Logging to MLflow (run_id: {run_id})...")
        
        try:
            # Check run status
            try:
                run_info = mlflow.get_run(run_id)
                run_status = run_info.info.status
                lifecycle_stage = run_info.info.lifecycle_stage
                
                print(f"   Run status: {run_status}")
                print(f"   Lifecycle stage: {lifecycle_stage}")
                
                if lifecycle_stage == 'deleted':
                    print(f"   ✗ Run has been deleted - cannot log to it")
                    print(f"   Skipping MLflow logging")
                    log_to_mlflow = False
                elif run_status in ['KILLED', 'FAILED']:
                    print(f"   ⚠ Run is {run_status} - attempting to log anyway")
            except mlflow.exceptions.MlflowException as e:
                print(f"   ✗ Cannot access run: {e}")
                print(f"   Skipping MLflow logging")
                log_to_mlflow = False
            
            if not log_to_mlflow:
                print()
            else:
                # Auto-detect if nested run is needed
                if use_nested_run == 'auto':
                    try:
                        existing_tags = mlflow.get_run(run_id).data.tags
                        inference_key = f'{split}_inference_complete'
                        use_nested_run = inference_key in existing_tags
                        
                        if use_nested_run:
                            print(f"   ⓘ Detected previous {split} inference, using nested run")
                        else:
                            print(f"   ⓘ First {split} inference, logging to parent run")
                    except:
                        use_nested_run = False
                
                # End any active run first
                active = mlflow.active_run()
                if active:
                    active_id = active.info.run_id
                    if active_id != run_id:
                        print(f"   ⚠ Ending different active run: {active_id}")
                        mlflow.end_run()
                
                # ============================================================
                # NESTED RUN APPROACH
                # ============================================================
                if use_nested_run:
                    try:
                        with mlflow.start_run(run_id=run_id):
                            with mlflow.start_run(nested=True, run_name=f"inference_{split}_{timestamp}"):
                                nested_run_id = mlflow.active_run().info.run_id
                                print(f"   ✓ Created nested run: {nested_run_id}")
                                
                                # Log parameters (no conflicts in nested run)
                                mlflow.log_params({
                                    'split': split,
                                    'device': str(device),
                                    'batch_size': str(batch_size),
                                    'img_size': str(img_size),
                                    'num_images': str(num_images),
                                    'inference_method': 'validation' if use_validation else 'prediction',
                                    'has_labels': str(use_validation),
                                    'parent_run_id': run_id,
                                    'timestamp': timestamp,
                                })
                                print(f"   ✓ Logged parameters")
                                
                                # Log metrics (only if validation mode)
                                if use_validation and metrics_dict.get('mode') != 'prediction_only':
                                    mlflow_metrics = {k: v for k, v in metrics_dict.items() 
                                                     if isinstance(v, (int, float))}
                                    if mlflow_metrics:
                                        mlflow.log_metrics(mlflow_metrics)
                                        print(f"   ✓ Logged {len(mlflow_metrics)} metrics")
                                
                                # Log performance metrics
                                if performance:
                                    mlflow.log_metrics({
                                        'speed_preprocess_ms': performance['preprocess'],
                                        'speed_inference_ms': performance['inference'],
                                        'speed_postprocess_ms': performance['postprocess'],
                                        'speed_total_ms': performance['total'],
                                        'speed_fps': performance['fps'],
                                    })
                                    print(f"   ✓ Logged performance metrics")
                                
                                # Log artifacts
                                artifact_count = 0
                                labels_count = 0
                                
                                if os.path.exists(results_path):
                                    # Log root-level files
                                    for item in os.listdir(results_path):
                                        item_path = os.path.join(results_path, item)
                                        if os.path.isfile(item_path):
                                            mlflow.log_artifact(item_path)
                                            artifact_count += 1
                                    
                                    # Handle labels directory
                                    labels_dir = os.path.join(results_path, "labels")
                                    if os.path.exists(labels_dir):
                                        try:
                                            label_files = [f for f in os.listdir(labels_dir) 
                                                          if f.endswith('.txt')]
                                            labels_count = len(label_files)
                                            
                                            # Log sample labels
                                            sample_size = min(5, labels_count)
                                            for i, label_file in enumerate(sorted(label_files)):
                                                if i < sample_size:
                                                    mlflow.log_artifact(
                                                        os.path.join(labels_dir, label_file),
                                                        artifact_path="labels/samples"
                                                    )
                                                    artifact_count += 1
                                            
                                            # Log metadata as tags (not params, to avoid conflicts)
                                            mlflow.set_tag('labels_count', str(labels_count))
                                            mlflow.set_tag('labels_path', labels_dir)
                                            
                                            print(f"   ✓ Logged {sample_size} sample labels (of {labels_count} total)")
                                        except Exception as e:
                                            print(f"   ⚠ Warning: Could not process labels: {e}")
                                    
                                    print(f"   ✓ Logged {artifact_count} artifacts")
                                
                                # Log metrics summary
                                mlflow.log_artifact(metrics_json_path)
                                print(f"   ✓ Logged metrics summary JSON")
                                
                                # Set tags
                                mlflow.set_tags({
                                    'inference_complete': 'true',
                                    'results_path': results_path,
                                })
                                print(f"   ✓ Set tags")
                        
                        print(f"\n   ✓ Successfully logged to nested run: {nested_run_id}")
                        print(f"   Parent run: {run_id}")
                        
                        if labels_count > 0:
                            print(f"   Note: Full dataset ({labels_count} labels) at: {labels_dir}")
                        
                        print()
                        
                    except mlflow.exceptions.MlflowException as e:
                        print(f"\n   ✗ Nested run creation failed: {e}")
                        print(f"   Falling back to parent run logging")
                        use_nested_run = False
                
                # ============================================================
                # PARENT RUN APPROACH (fallback or first-time)
                # ============================================================
                if not use_nested_run:
                    try:
                        with mlflow.start_run(run_id=run_id):
                            actual_run_id = mlflow.active_run().info.run_id
                            
                            if actual_run_id != run_id:
                                raise RuntimeError(
                                    f"MLflow context mismatch!\n"
                                    f"  Expected: {run_id}\n"
                                    f"  Got:      {actual_run_id}"
                                )
                            
                            print(f"   ✓ Resumed run: {run_id}")
                            
                            # Get existing params
                            existing_params = set()
                            existing_tags = {}
                            try:
                                run_data = mlflow.get_run(run_id)
                                existing_params = set(run_data.data.params.keys())
                                existing_tags = run_data.data.tags
                                print(f"   Found {len(existing_params)} existing parameters")
                            except Exception as e:
                                print(f"   Warning: Could not fetch existing params: {e}")
                            
                            # Log parameters (skip existing)
                            params_to_log = {
                                f'{split}_device': str(device),
                                f'{split}_batch_size': str(batch_size),
                                f'{split}_img_size': str(img_size),
                                f'{split}_num_images': str(num_images),
                                f'{split}_timestamp': timestamp,
                                f'{split}_inference_method': 'validation' if use_validation else 'prediction',
                                f'{split}_has_labels': str(use_validation),
                            }
                            
                            params_to_log = {k: v for k, v in params_to_log.items() 
                                            if k not in existing_params}
                            
                            if params_to_log:
                                mlflow.log_params(params_to_log)
                                print(f"   ✓ Logged {len(params_to_log)} parameters")
                            else:
                                print(f"   Skipped parameters (all exist)")
                            
                            # Log metrics
                            if use_validation and metrics_dict.get('mode') != 'prediction_only':
                                mlflow_metrics = {f'{split}_{k}': v for k, v in metrics_dict.items() 
                                                 if isinstance(v, (int, float))}
                                if mlflow_metrics:
                                    mlflow.log_metrics(mlflow_metrics)
                                    print(f"   ✓ Logged {len(mlflow_metrics)} metrics")
                            else:
                                print(f"   Skipped metrics (prediction mode)")
                            
                            # Log performance
                            if performance:
                                perf_metrics = {
                                    f'{split}_speed_preprocess_ms': performance['preprocess'],
                                    f'{split}_speed_inference_ms': performance['inference'],
                                    f'{split}_speed_postprocess_ms': performance['postprocess'],
                                    f'{split}_speed_total_ms': performance['total'],
                                    f'{split}_speed_fps': performance['fps'],
                                }
                                mlflow.log_metrics(perf_metrics)
                                print(f"   ✓ Logged performance metrics")
                            
                            # Log artifacts
                            artifact_count = 0
                            labels_count = 0
                            
                            if os.path.exists(results_path):
                                # Log root-level files
                                for item in os.listdir(results_path):
                                    item_path = os.path.join(results_path, item)
                                    if os.path.isfile(item_path):
                                        mlflow.log_artifact(item_path, artifact_path=f"inference/{split}")
                                        artifact_count += 1
                                
                                # Handle labels
                                labels_dir = os.path.join(results_path, "labels")
                                if os.path.exists(labels_dir):
                                    try:
                                        label_files = [f for f in os.listdir(labels_dir) 
                                                      if f.endswith('.txt')]
                                        labels_count = len(label_files)
                                        
                                        # Check if labels_count param exists
                                        labels_param_key = f'{split}_labels_count'
                                        if labels_param_key not in existing_params:
                                            mlflow.log_param(labels_param_key, labels_count)
                                        else:
                                            # Use tag instead (can be updated)
                                            mlflow.set_tag(f'{split}_labels_count_latest', labels_count)
                                        
                                        mlflow.set_tag(f'{split}_labels_path', labels_dir)
                                        
                                        # Log sample labels
                                        sample_size = min(5, labels_count)
                                        for i, label_file in enumerate(sorted(label_files)):
                                            if i < sample_size:
                                                mlflow.log_artifact(
                                                    os.path.join(labels_dir, label_file),
                                                    artifact_path=f"inference/{split}/labels/samples"
                                                )
                                                artifact_count += 1
                                        
                                        print(f"   ✓ Logged {sample_size} sample labels (of {labels_count} total)")
                                        print(f"   ✓ Tracked labels location: {labels_dir}")
                                        
                                    except Exception as e:
                                        print(f"   ⚠ Warning: Could not process labels: {e}")
                                
                                print(f"   ✓ Logged {artifact_count} artifacts")
                            
                            # Log metrics summary
                            mlflow.log_artifact(metrics_json_path, artifact_path=f"inference/{split}")
                            print(f"   ✓ Logged metrics summary JSON")
                            
                            # Set tags
                            mlflow.set_tags({
                                f'{split}_inference_timestamp': timestamp,
                                f'{split}_inference_complete': 'true',
                                f'{split}_inference_method': 'validation' if use_validation else 'prediction',
                                f'{split}_num_images': str(num_images),
                                f'{split}_results_path': results_path,
                            })
                            print(f"   ✓ Set tags")
                        
                        print(f"\n   ✓ Successfully logged to MLflow run: {run_id}")
                        
                        if existing_params:
                            print(f"   Note: Skipped {len(existing_params)} existing parameters")
                        
                        if labels_count > 0:
                            print(f"   Note: Full dataset ({labels_count} labels) at: {labels_dir}")
                        
                        print()
                        
                    except mlflow.exceptions.MlflowException as e:
                        if "INVALID_STATE" in str(e) or "cannot transition" in str(e).lower():
                            print(f"\n   ✗ Run is in invalid state for logging: {e}")
                            print(f"   This usually means the run was aborted/killed")
                            print(f"   Skipping MLflow logging")
                        else:
                            raise
            
        except Exception as e:
            print(f"\n   ✗ MLflow logging failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    elif log_to_mlflow and not run_id:
        print("9. Skipping MLflow logging (no run_id available)\n")
    else:
        print("9. Skipping MLflow logging (log_to_mlflow=False)\n")
    
    # ========================================================================
    # 10. PREPARE RETURN SUMMARY
    # ========================================================================
    inference_summary = {
        'split': split,
        'inference_method': 'validation' if use_validation else 'prediction',
        'has_labels': use_validation,
        'num_images': num_images,
        'num_images_expected': num_images_expected,
        'device': device,
        'batch_size': batch_size,
        'img_size': img_size,
        'timestamp': timestamp,
        'run_id': run_id,
        'nested_run_id': nested_run_id,  # NEW: track nested run if created
        'predictions_path': output_path,
        'metrics_path': results_path,
        'metrics_json_path': metrics_json_path,
        'data_yaml_path': data_yaml_path,
        'model_path': model_path,
        'metrics': metrics_dict,
        'performance': performance,
        'validation_passed': num_images == num_images_expected if num_images_expected else True,
        'mlflow_logged': log_to_mlflow and run_id is not None,
        'used_nested_run': nested_run_id is not None,  # NEW: flag for nested run
        'yolo_version': '8.3.200',
        'visualization': {
            'show_boxes': kwargs.get('show_boxes', True),
            'show_masks': kwargs.get('show_masks', True),
            'show_labels': kwargs.get('show_labels', True),
            'show_conf': kwargs.get('show_conf', True),
            'box_color': kwargs.get('box_color', (0, 255, 0)),
            'mask_color': kwargs.get('mask_color', (0, 0, 255)),
        }
    }
    
    print(f"{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"Split: {split}")
    print(f"Method: {inference_method}")
    print(f"Images: {num_images}")
    print(f"Output: {output_path}")
    print(f"Metrics: {metrics_json_path}")
    if run_id:
        print(f"MLflow Run: {run_id}")
        if nested_run_id:
            print(f"MLflow Nested Run: {nested_run_id}")
        print(f"MLflow Logged: {'Yes' if inference_summary['mlflow_logged'] else 'No'}")
    print(f"{'='*70}\n")
    
    return inference_summary
    


def inspect_inference_output(inference_summary):
    """
    Inspect the inference output directory structure.
    
    Args:
        inference_summary: Dictionary returned from run_inference_with_metrics
    """
    from pathlib import Path
    
    print(f"\n{'='*70}")
    print("INFERENCE OUTPUT INSPECTION")
    print('='*70)
    
    predictions_path = inference_summary.get('predictions_path')
    
    if not predictions_path:
        print(f"✗ No predictions path in summary")
        return
    
    print(f"\nPredictions path: {predictions_path}")
    
    if not os.path.exists(predictions_path):
        print(f"✗ Path does not exist!")
        return
    
    # Display summary info
    print(f"\nSummary:")
    print(f"  Split: {inference_summary.get('split', 'N/A')}")
    print(f"  Method: {inference_summary.get('inference_method', 'N/A')}")
    print(f"  Has labels: {inference_summary.get('has_labels', 'N/A')}")
    print(f"  Images processed: {inference_summary.get('num_images', 0)}")
    print(f"  Images expected: {inference_summary.get('num_images_expected', 'N/A')}")
    print(f"  Validation passed: {inference_summary.get('validation_passed', 'N/A')}")
    print(f"  MLflow logged: {inference_summary.get('mlflow_logged', 'N/A')}")
    if inference_summary.get('run_id'):
        print(f"  Run ID: {inference_summary.get('run_id')}")
    
    # List directory structure
    print(f"\nDirectory structure:")
    pred_path = Path(predictions_path)
    
    print(f"\n{pred_path.name}/")
    for item in sorted(pred_path.iterdir()):
        if item.is_dir():
            file_count = len(list(item.iterdir()))
            print(f"  📁 {item.name}/ ({file_count} items)")
            
            # Show first few files in each directory
            if file_count > 0:
                for subitem in sorted(item.iterdir())[:3]:
                    size_kb = subitem.stat().st_size / 1024 if subitem.is_file() else 0
                    print(f"     - {subitem.name} ({size_kb:.1f} KB)")
                if file_count > 3:
                    print(f"     ... and {file_count - 3} more")
        else:
            size_kb = item.stat().st_size / 1024
            print(f"  📄 {item.name} ({size_kb:.1f} KB)")
    
    # Find all images recursively
    all_images = list(pred_path.rglob("*.png")) + list(pred_path.rglob("*.jpg"))
    
    print(f"\n✓ Total images found (recursive): {len(all_images)}")
    
    # Check for expected split name in filenames
    split = inference_summary.get('split', 'val')
    split_images = [img for img in all_images if split in img.name.lower()]
    
    if split_images:
        print(f"✓ Found {len(split_images)} images with '{split}' in filename")
    else:
        print(f"⚠ Warning: No images found with '{split}' in filename")
        print(f"  This may indicate the wrong split was processed")
    
    if len(all_images) > 0:
        print(f"\nSample image paths (relative to predictions_path):")
        for img in sorted(all_images)[:10]:
            rel_path = img.relative_to(pred_path)
            size_kb = img.stat().st_size / 1024
            print(f"  {rel_path} ({size_kb:.1f} KB)")
        if len(all_images) > 10:
            print(f"  ... and {len(all_images) - 10} more")
    
    # Check for labels directory
    labels_dir = pred_path / "results" / "labels"
    if not labels_dir.exists():
        # Check in test subdirectory (YOLO's convention)
        labels_dir = pred_path / "test" / "results" / "labels"
    if not labels_dir.exists():
        # Check in predict subdirectory
        labels_dir = pred_path / "predict" / "results" / "labels"
    if not labels_dir.exists():
        # Check in val subdirectory
        labels_dir = pred_path / "val" / "results" / "labels"
    
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        print(f"\n✓ Found {len(label_files)} label files in {labels_dir.relative_to(pred_path)}/")
    
    # Check for metrics file
    metrics_path = inference_summary.get('metrics_json_path')
    if metrics_path and os.path.exists(metrics_path):
        print(f"\n✓ Metrics file exists: {Path(metrics_path).name}")
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            
            # Show metrics if available
            if metrics_data.get('inference_method') == 'validation':
                metrics = metrics_data.get('metrics', {})
                if metrics.get('mode') != 'validation_no_metrics':
                    print(f"  Box mAP50-95: {metrics.get('box_map50_95', 'N/A')}")
                    print(f"  Mask mAP50-95: {metrics.get('mask_map50_95', 'N/A')}")
                    print(f"  Fitness: {metrics.get('fitness', 'N/A')}")
                else:
                    print(f"  Mode: Validation (metrics extraction failed)")
            else:
                print(f"  Mode: Prediction only (no metrics)")
    
    print('='*70)