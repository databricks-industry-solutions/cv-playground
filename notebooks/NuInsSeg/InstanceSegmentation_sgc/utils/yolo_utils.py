"""
YOLO Training and Inference Utilities for Databricks
Author: may.merkletan@databricks.com
Last Updated: 2025Oct22
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seeds(seed_value=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    print(f"✓ Random seeds set to {seed_value}")


def path_exists(path):
    """Check if a path exists (works for both local and dbfs paths)."""
    # For UC Volumes paths (start with /Volumes/)
    if path.startswith('/Volumes/'):
        return os.path.exists(path)
    
    # For dbfs paths, try dbutils
    try:
        import dbutils
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def setup_yolo_paths(project_path=None, set_artifacts_path=True, verbose=True):
    """
    Setup YOLO paths for training and inference.
    
    Configures global paths and Ultralytics config directory:
    - YOLO_DATA_UCVol_path: Dataset location
    - YOLO_ARTIFACTS_UCVol_path: Training artifacts location
    - ULTRALYTICS_CONFIG_DIR: Ultralytics config/cache location
    
    Args:
        project_path: Base project path (e.g., '/Volumes/mmt/cv/projects/NuInsSeg')
                     If None, tries to auto-detect from existing YOLO_DATA_UCVol_path
        set_artifacts_path: Whether to set YOLO_ARTIFACTS_UCVol_path global
        verbose: Print configuration details
    
    Returns:
        dict: Configured paths including 'ultralytics_config'
    
    Example:
        from utils import setup_yolo_paths
        
        paths = setup_yolo_paths(
            project_path='/Volumes/mmt/cv/projects/NuInsSeg',
            set_artifacts_path=True
        )
        
        # Now safe to import YOLO
        from ultralytics import YOLO
    """
    import builtins
    from pathlib import Path
    
    # Auto-detect project path if not provided
    if project_path is None:
        data_path = getattr(builtins, 'YOLO_DATA_UCVol_path', None)
        if data_path:
            project_path = str(Path(data_path).parent)
            if verbose:
                print("Auto-detected project path from YOLO_DATA_UCVol_path")
        else:
            raise ValueError(
                "project_path not provided and cannot auto-detect.\n"
                "Please provide: setup_yolo_paths(project_path='/Volumes/...')\n\n"
                "Example:\n"
                "  paths = setup_yolo_paths(\n"
                "      project_path='/Volumes/mmt/cv/projects/NuInsSeg'\n"
                "  )"
            )
    
    # Construct standard paths
    paths = {
        'project': project_path,
        'data': f"{project_path}/yolo_dataset_on_vols",
        'artifacts': f"{project_path}/yolo_artifacts_sgc",
        'ultralytics_config': f"{project_path}/.config/Ultralytics"
    }
    
    # Verify project path exists
    if not os.path.exists(paths['project']):
        raise FileNotFoundError(
            f"Project path does not exist: {paths['project']}\n"
            f"Please verify the path is correct."
        )
    
    # Warn if data path doesn't exist
    if not os.path.exists(paths['data']) and verbose:
        print(f"[WARNING] Data path does not exist: {paths['data']}")
        print("          You may need to run copy_to_uc_volumes_with_yaml() first")
    
    # Set global variables for YOLO paths
    builtins.YOLO_DATA_UCVol_path = paths['data']
    
    if set_artifacts_path:
        builtins.YOLO_ARTIFACTS_UCVol_path = paths['artifacts']
        os.makedirs(paths['artifacts'], exist_ok=True)
        if verbose:
            print(f"[OK] Artifacts directory ready: {paths['artifacts']}")
    
    # Configure Ultralytics config directory
    os.makedirs(paths['ultralytics_config'], exist_ok=True)
    os.environ['ULTRALYTICS_CONFIG_DIR'] = paths['ultralytics_config']
    os.environ['YOLO_VERBOSE'] = 'True'
    
    if verbose:
        print(f"[OK] Ultralytics config directory ready: {paths['ultralytics_config']}")
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print("YOLO PATHS CONFIGURED")
        print(f"{'='*70}")
        print(f"Project:             {paths['project']}")
        print(f"Data:                {paths['data']}")
        if set_artifacts_path:
            print(f"Artifacts:           {paths['artifacts']}")
        print(f"Ultralytics Config:  {paths['ultralytics_config']}")
        print(f"{'='*70}\n")
        
        print("Global variables set:")
        print(f"  builtins.YOLO_DATA_UCVol_path = '{paths['data']}'")
        if set_artifacts_path:
            print(f"  builtins.YOLO_ARTIFACTS_UCVol_path = '{paths['artifacts']}'")
        print(f"\nEnvironment variables set:")
        print(f"  os.environ['ULTRALYTICS_CONFIG_DIR'] = '{paths['ultralytics_config']}'")
        print(f"  os.environ['YOLO_VERBOSE'] = 'True'")
        print()
    
    return paths


def get_organized_paths(run_id, timestamp, project_path):
    """
    Create organized path structure for training artifacts and inference results.
    
    Structure:
    /Volumes/{catalog}/{schema}/projects/{project}/yolo_artifacts_sgc/
    └── yolo_training_{timestamp}_runid_{run_id}/
        ├── train/                 # All training artifacts
        │   ├── weights/
        │   ├── checkpoints/
        │   └── best_model.pt
        ├── test/                  # Test inference results
        └── val/                   # Val inference results
    
    Args:
        run_id: MLflow run ID
        timestamp: Training timestamp string
        project_path: Base project path
    
    Returns:
        dict: Dictionary of organized paths
    """
    base_run_path = f"{project_path}/yolo_artifacts_sgc/yolo_training_{timestamp}_runid_{run_id}"
    
    paths = {
        'base': base_run_path,
        'train': f"{base_run_path}/train",
        'train_weights': f"{base_run_path}/train/weights",
        'train_checkpoints': f"{base_run_path}/train/checkpoints",
        'val': f"{base_run_path}/val",
        'test': f"{base_run_path}/test",
        'inference_val': f"{base_run_path}/val",
        'inference_test': f"{base_run_path}/test",
    }
    
    return paths


def validate_data_yaml(data_yaml_path, required_splits=None, verbose=True):
    """
    Validate data.yaml configuration and verify paths exist.
    
    Args:
        data_yaml_path: Path to data.yaml file
        required_splits: List of required splits (e.g., ['train', 'val', 'test'])
        verbose: Print detailed information
    
    Returns:
        dict: Validation results with split information
    
    Raises:
        FileNotFoundError: If data.yaml doesn't exist
        ValueError: If required splits are missing
    """
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    if verbose:
        print(f"\n{'='*70}")
        print("DATA.YAML VALIDATION")
        print(f"{'='*70}")
        print(f"File: {data_yaml_path}\n")
    
    # Check for required splits
    available_splits = [k for k in ['train', 'val', 'test'] if k in data_config]
    
    if required_splits:
        missing_splits = [s for s in required_splits if s not in data_config]
        if missing_splits:
            raise ValueError(
                f"Missing required splits: {missing_splits}\n"
                f"Available splits: {available_splits}"
            )
    
    # Validate each split
    validation_results = {
        'data_yaml_path': data_yaml_path,
        'available_splits': available_splits,
        'splits': {},
        'num_classes': data_config.get('nc', 0),
        'class_names': data_config.get('names', []),
    }
    
    base_path = data_config.get('path', '')
    
    for split in available_splits:
        split_rel_path = data_config[split]
        
        # Handle both absolute and relative paths
        if split_rel_path.startswith('/Volumes/'):
            split_full_path = split_rel_path
        else:
            split_full_path = os.path.join(base_path, split_rel_path)
        
        # Check if path exists
        exists = os.path.exists(split_full_path)
        
        # Count images if path exists
        num_images = 0
        if exists:
            try:
                image_files = [f for f in os.listdir(split_full_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                num_images = len(image_files)
            except Exception as e:
                if verbose:
                    print(f"   ⚠ Warning: Could not count images in {split}: {e}")
        
        validation_results['splits'][split] = {
            'path': split_full_path,
            'exists': exists,
            'num_images': num_images,
        }
        
        if verbose:
            status = "✓" if exists else "✗"
            print(f"{status} {split:5s}: {split_full_path}")
            if exists:
                print(f"         {num_images} images")
            else:
                print(f"         PATH DOES NOT EXIST!")
    
    if verbose:
        print(f"\nClasses: {validation_results['num_classes']}")
        print(f"Names: {validation_results['class_names']}")
        print(f"{'='*70}\n")
    
    return validation_results


def copy_to_uc_volumes_with_yaml(ws_proj_dir, yolo_data_ucvol_path, validate=True):
    """
    Copy dataset to UC Volumes and create appropriate data.yaml
    
    Args:
        ws_proj_dir: Source workspace project directory
        yolo_data_ucvol_path: Destination UC Volumes path
        validate: Whether to validate the copied data
    
    Returns:
        tuple: (data_yaml_path, validation_results)
    """
    print(f"\n{'='*70}")
    print("COPYING DATASET TO UC VOLUMES")
    print(f"{'='*70}")
    print(f"Source: {ws_proj_dir}/datasets/")
    print(f"Destination: {yolo_data_ucvol_path}\n")
    
    # 1: Create destination directory
    print("1. Creating destination directory...")
    os.makedirs(yolo_data_ucvol_path, exist_ok=True)
    print(f"   ✓ Created: {yolo_data_ucvol_path}\n")
    
    # 2: Copy all dataset files
    print("2. Copying dataset files...")
    result = os.system(f"rsync -av --progress {ws_proj_dir}/datasets/ {yolo_data_ucvol_path}/")
    
    if result != 0:
        print(f"   ⚠ Warning: rsync returned non-zero exit code: {result}")
    else:
        print(f"   ✓ Copy completed\n")
    
    # 3: Detect available splits
    print("3. Detecting available splits...")
    available_splits = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(yolo_data_ucvol_path, split, 'images')
        if os.path.exists(split_path):
            available_splits[split] = f"{yolo_data_ucvol_path}/{split}/images"
            print(f"   ✓ Found {split} split")
        else:
            print(f"   ✗ Missing {split} split")
    
    if not available_splits:
        raise ValueError(f"No valid splits found in {yolo_data_ucvol_path}")
    
    print()
    
    # 4: Create data.yaml with UC Volumes paths
    print("4. Creating data.yaml...")
    data_yaml_content = {
        "path": yolo_data_ucvol_path,  # Add base path
        "train": f"{yolo_data_ucvol_path}/train/images",
        "val": f"{yolo_data_ucvol_path}/val/images",
        "test": f"{yolo_data_ucvol_path}/test/images",
        "nc": 1,
        "names": ["Nuclei"]
    }
    
    # Only include splits that exist
    data_yaml_content = {k: v for k, v in data_yaml_content.items() 
                        if k in ['path', 'nc', 'names'] or 
                        (k in available_splits or k in ['train', 'val', 'test'] and 
                         any(k in path for path in available_splits.values()))}
    
    # Write data.yaml
    data_yaml_path = f"{yolo_data_ucvol_path}/data.yaml"
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✓ Created: {data_yaml_path}\n")
    
    # 5: Show data.yaml contents
    print("5. data.yaml contents:")
    print("-" * 70)
    with open(data_yaml_path, 'r') as f:
        print(f.read())
    print("-" * 70)
    
    # 6: Validate if requested
    validation_results = None
    if validate:
        print()
        validation_results = validate_data_yaml(data_yaml_path, verbose=True)
        
        # Check for issues
        issues = []
        for split, info in validation_results['splits'].items():
            if not info['exists']:
                issues.append(f"Split '{split}' path does not exist")
            elif info['num_images'] == 0:
                issues.append(f"Split '{split}' has no images")
        
        if issues:
            print("⚠ VALIDATION WARNINGS:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print("✓ All validations passed!")
    
    print(f"\n{'='*70}")
    print("COPY COMPLETE")
    print(f"{'='*70}\n")
    
    return data_yaml_path, validation_results


def get_split_info(data_yaml_path, split='val'):
    """
    Get information about a specific split from data.yaml
    
    Args:
        data_yaml_path: Path to data.yaml
        split: Split name ('train', 'val', or 'test')
    
    Returns:
        dict: Split information
    """
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    if split not in data_config:
        available = [k for k in ['train', 'val', 'test'] if k in data_config]
        raise ValueError(f"Split '{split}' not found. Available: {available}")
    
    split_path = data_config[split]
    base_path = data_config.get('path', '')
    
    # Handle absolute vs relative paths
    if split_path.startswith('/Volumes/'):
        full_path = split_path
    else:
        full_path = os.path.join(base_path, split_path)
    
    # Count images
    num_images = 0
    if os.path.exists(full_path):
        image_files = [f for f in os.listdir(full_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        num_images = len(image_files)
    
    return {
        'split': split,
        'path': full_path,
        'exists': os.path.exists(full_path),
        'num_images': num_images,
        'num_classes': data_config.get('nc', 0),
        'class_names': data_config.get('names', []),
    }


def check_yolo_environment(verbose=True, create_missing=False):
    """
    Check if YOLO environment is properly configured.
    
    Args:
        verbose: Print detailed status
        create_missing: Create missing directories if True
    
    Returns:
        dict: Environment status with 'ready' boolean
    """
    import builtins
    from pathlib import Path
    
    data_path = getattr(builtins, 'YOLO_DATA_UCVol_path', None)
    artifacts_path = getattr(builtins, 'YOLO_ARTIFACTS_UCVol_path', None)
    ultralytics_config = os.environ.get('ULTRALYTICS_CONFIG_DIR', None)
    
    # Extract parent paths for comprehensive checking
    project_path = None
    if data_path and '/yolo_dataset_on_vols' in data_path:
        project_path = str(Path(data_path).parent)
    elif artifacts_path and '/yolo_artifacts_sgc' in artifacts_path:
        project_path = str(Path(artifacts_path).parent)
    
    # Paths to check
    paths_to_check = {}
    if project_path:
        paths_to_check['project'] = project_path
    
    if data_path:
        paths_to_check['data'] = data_path
    
    if artifacts_path:
        paths_to_check['artifacts'] = artifacts_path
    
    if ultralytics_config:
        paths_to_check['ultralytics_config'] = ultralytics_config
    
    # Initialize status
    status = {
        'data_path_set': bool(data_path),
        'data_path_exists': False,
        'artifacts_path_set': bool(artifacts_path),
        'artifacts_path_exists': False,
        'ultralytics_config_set': bool(ultralytics_config),
        'ultralytics_config_exists': False,
        'data_yaml_exists': False,
        'data_yaml_path': None,
        'ready': False,
        'paths_checked': {},
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("YOLO ENVIRONMENT STATUS")
        print(f"{'='*70}\n")
        
        if paths_to_check:
            print("Checking directory structure:")
    
    # Check each path
    for path_name, path_value in paths_to_check.items():
        exists = os.path.exists(path_value)
        status['paths_checked'][path_name] = {
            'path': path_value,
            'exists': exists,
            'created': False
        }
        
        if verbose:
            status_icon = "[OK]" if exists else "[MISSING]"
            print(f"  {status_icon} {path_name:20s}: {path_value}")
        
        # Create if missing and requested
        if not exists and create_missing:
            try:
                os.makedirs(path_value, exist_ok=True)
                status['paths_checked'][path_name]['created'] = True
                status['paths_checked'][path_name]['exists'] = True
                if verbose:
                    print(f"    -> Created successfully")
            except Exception as e:
                if verbose:
                    print(f"    [ERROR] Failed to create: {e}")
    
    if verbose and paths_to_check:
        print()
    
    # Update main status flags
    if data_path:
        status['data_path_exists'] = status['paths_checked'].get('data', {}).get('exists', False)
        
        # Check for data.yaml
        if status['data_path_exists']:
            data_yaml = os.path.join(data_path, 'data.yaml')
            if os.path.exists(data_yaml):
                status['data_yaml_exists'] = True
                status['data_yaml_path'] = data_yaml
    
    if artifacts_path:
        status['artifacts_path_exists'] = status['paths_checked'].get('artifacts', {}).get('exists', False)
    
    if ultralytics_config:
        status['ultralytics_config_exists'] = status['paths_checked'].get('ultralytics_config', {}).get('exists', False)
    
    # Overall readiness
    status['ready'] = (
        status['data_path_set'] and 
        status['data_path_exists'] and 
        status['data_yaml_exists'] and
        status['artifacts_path_set'] and
        status['ultralytics_config_set']
    )
    
    if verbose:
        print("Configuration status:")
        
        # Data path
        if data_path:
            print(f"  [OK] YOLO_DATA_UCVol_path: {data_path}")
            print(f"    {'[OK]' if status['data_path_exists'] else '[MISSING]'} Path exists")
            print(f"    {'[OK]' if status['data_yaml_exists'] else '[MISSING]'} data.yaml found")
            if status['data_yaml_exists']:
                print(f"         {status['data_yaml_path']}")
        else:
            print(f"  [MISSING] YOLO_DATA_UCVol_path: NOT SET")
        
        print()
        
        # Artifacts path
        if artifacts_path:
            print(f"  [OK] YOLO_ARTIFACTS_UCVol_path: {artifacts_path}")
            if status['artifacts_path_exists']:
                print(f"    [OK] Path exists")
            elif status['paths_checked'].get('artifacts', {}).get('created'):
                print(f"    [OK] Path created")
            else:
                print(f"    [WARNING] Path will be created on first use")
        else:
            print(f"  [MISSING] YOLO_ARTIFACTS_UCVol_path: NOT SET")
        
        print()
        
        # Ultralytics config
        if ultralytics_config:
            print(f"  [OK] ULTRALYTICS_CONFIG_DIR: {ultralytics_config}")
            if status['ultralytics_config_exists']:
                print(f"    [OK] Path exists")
            elif status['paths_checked'].get('ultralytics_config', {}).get('created'):
                print(f"    [OK] Path created")
            else:
                print(f"    [MISSING] Path does not exist")
        else:
            print(f"  [MISSING] ULTRALYTICS_CONFIG_DIR: NOT SET")
        
        print(f"\n{'='*70}")
        if status['ready']:
            print("[OK] ENVIRONMENT READY")
        else:
            print("[NOT READY] ENVIRONMENT NOT READY")
        
        if not status['ready']:
            print("\nTo setup:")
            print("  from utils import setup_yolo_paths")
            print("  setup_yolo_paths(project_path='/Volumes/...')")
            
            if create_missing:
                print("\nNote: create_missing=True was used")
                created_paths = [k for k, v in status['paths_checked'].items() if v.get('created')]
                if created_paths:
                    print(f"Created: {', '.join(created_paths)}")
        
        print(f"{'='*70}\n")
    
    return status


def get_yolo_paths():
    """
    Get current YOLO paths from global variables and environment.
    
    Returns:
        dict: Current paths or None if not set
    """
    import builtins
    
    data_path = getattr(builtins, 'YOLO_DATA_UCVol_path', None)
    artifacts_path = getattr(builtins, 'YOLO_ARTIFACTS_UCVol_path', None)
    ultralytics_config = os.environ.get('ULTRALYTICS_CONFIG_DIR', None)
    
    if not data_path and not artifacts_path:
        return None
    
    paths = {}
    
    if data_path:
        paths['data'] = data_path
        if '/yolo_dataset_on_vols' in data_path:
            paths['project'] = str(Path(data_path).parent)
    
    if artifacts_path:
        paths['artifacts'] = artifacts_path
        if 'project' not in paths and '/yolo_artifacts_sgc' in artifacts_path:
            paths['project'] = str(Path(artifacts_path).parent)
    
    if ultralytics_config:
        paths['ultralytics_config'] = ultralytics_config
    
    return paths


def get_inference_output_path(run_id, split='test'):
    """
    Get the proper output path for inference results.
    
    Args:
        run_id: MLflow run ID
        split: Split name ('train', 'val', 'test')
    
    Returns:
        str: Output path for inference results
    
    Raises:
        ValueError: If paths not configured
    """
    import builtins
    
    YOLO_ARTIFACTS_UCVol_path = getattr(builtins, 'YOLO_ARTIFACTS_UCVol_path', None)
    
    if not YOLO_ARTIFACTS_UCVol_path:
        raise ValueError(
            "YOLO_ARTIFACTS_UCVol_path not set.\n"
            "Please run setup_yolo_paths() first."
        )
    
    # Try to find existing run directory
    if os.path.exists(YOLO_ARTIFACTS_UCVol_path):
        for item in os.listdir(YOLO_ARTIFACTS_UCVol_path):
            if f'runid_{run_id}' in item:
                return os.path.join(YOLO_ARTIFACTS_UCVol_path, item, split)
    
    # Create new directory if not found
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"yolo_inference_{timestamp}_runid_{run_id}"
    return os.path.join(YOLO_ARTIFACTS_UCVol_path, run_dir, split)