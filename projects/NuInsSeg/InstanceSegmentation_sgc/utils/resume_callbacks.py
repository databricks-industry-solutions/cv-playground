# `setup_resume()` helper function(s) to add to callback module:

def setup_resume(checkpoint_path, trainer=None):
    """
    Setup for resuming training from a checkpoint.
    Restores mlflow_epoch_logger state variables and optionally loads trainer state.
    
    Args:
        checkpoint_path (str): Path to checkpoint file (.pt)
        trainer: Optional YOLO trainer instance to load state into
    
    Returns:
        dict: Checkpoint data for reference
    
    Example:
        >>> checkpoint = setup_resume('checkpoints/checkpoint_epoch_050.pt', trainer)
        >>> # Now mlflow_epoch_logger state is restored
        >>> # Continue training with model.train(resume=True, callbacks={'on_fit_epoch_end': mlflow_epoch_logger})
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n{'='*70}")
    print("SETTING UP RESUME FROM CHECKPOINT")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Validate checkpoint structure
    if 'epoch' not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'epoch' field")
    
    resume_epoch = checkpoint['epoch']
    print(f"✓ Resuming from epoch {resume_epoch + 1}")
    
    # Restore mlflow_epoch_logger state variables
    print("\nRestoring callback state...")
    
    # Best fitness tracking
    if 'best_fitness' in checkpoint:
        mlflow_epoch_logger.best_fitness = checkpoint['best_fitness']
        print(f"  ✓ best_fitness: {mlflow_epoch_logger.best_fitness:.5f}")
    elif 'fitness' in checkpoint and checkpoint['fitness'] is not None:
        # Fallback to current fitness if best_fitness not saved
        mlflow_epoch_logger.best_fitness = checkpoint['fitness']
        print(f"  ✓ best_fitness: {mlflow_epoch_logger.best_fitness:.5f} (from current)")
    else:
        mlflow_epoch_logger.best_fitness = float('-inf')
        print(f"  ⚠ best_fitness: not found, initialized to -inf")
    
    # Best epoch tracking
    if 'best_epoch' in checkpoint:
        mlflow_epoch_logger.best_epoch = checkpoint['best_epoch']
        print(f"  ✓ best_epoch: {mlflow_epoch_logger.best_epoch + 1}")
    else:
        mlflow_epoch_logger.best_epoch = resume_epoch
        print(f"  ⚠ best_epoch: not found, set to resume epoch {resume_epoch + 1}")
    
    # Checkpoints logged count
    if 'checkpoints_logged' in checkpoint:
        mlflow_epoch_logger.checkpoints_logged = checkpoint['checkpoints_logged']
        print(f"  ✓ checkpoints_logged: {mlflow_epoch_logger.checkpoints_logged}")
    else:
        mlflow_epoch_logger.checkpoints_logged = 0
        print(f"  ⚠ checkpoints_logged: not found, initialized to 0")
    
    # Display metrics if available
    if 'metrics' in checkpoint and checkpoint['metrics']:
        print("\nCheckpoint metrics:")
        metrics = checkpoint['metrics']
        
        # Key metrics to display
        key_metrics = [
            'val_box_map50_95',
            'val_box_map50', 
            'val_mask_map50_95',
            'val_mask_map50',
            'fitness',
            'learning_rate'
        ]
        
        for metric in key_metrics:
            if metric in metrics and metrics[metric] is not None:
                print(f"  {metric}: {metrics[metric]:.5f}")
    
    # Optionally load trainer state
    if trainer is not None:
        print("\nLoading trainer state...")
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            try:
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                print("  ✓ Model state loaded")
            except Exception as e:
                print(f"  ⚠ Failed to load model state: {e}")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                try:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("  ✓ Optimizer state loaded")
                except Exception as e:
                    print(f"  ⚠ Failed to load optimizer state: {e}")
        
        # Load EMA state if available
        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            if hasattr(trainer, 'ema') and trainer.ema is not None:
                try:
                    trainer.ema.ema.load_state_dict(checkpoint['ema'])
                    print("  ✓ EMA state loaded")
                except Exception as e:
                    print(f"  ⚠ Failed to load EMA state: {e}")
        
        # Set epoch
        trainer.epoch = resume_epoch
        print(f"  ✓ Trainer epoch set to {resume_epoch}")
    
    # Log to MLflow if active run exists
    try:
        active_run = mlflow.active_run()
        if active_run:
            print("\nLogging resume info to MLflow...")
            mlflow.log_param("resumed_from_checkpoint", os.path.basename(checkpoint_path))
            mlflow.log_param("resume_epoch", resume_epoch + 1)
            mlflow.log_param("resume_best_fitness", mlflow_epoch_logger.best_fitness)
            mlflow.log_param("resume_best_epoch", mlflow_epoch_logger.best_epoch + 1)
            print("  ✓ Resume info logged to MLflow")
    except Exception as e:
        print(f"  ⚠ MLflow logging skipped: {e}")
    
    print(f"\n{'='*70}")
    print("RESUME SETUP COMPLETE")
    print(f"{'='*70}\n")
    
    return checkpoint


def find_latest_checkpoint(checkpoints_dir):
    """
    Find the most recent checkpoint in a directory.
    
    Args:
        checkpoints_dir (str): Directory containing checkpoint files
    
    Returns:
        str: Path to latest checkpoint, or None if no checkpoints found
    
    Example:
        >>> latest = find_latest_checkpoint('/Volumes/catalog/schema/volume/run_20250121_143022/train/checkpoints')
        >>> if latest:
        >>>     setup_resume(latest)
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Find all .pt files
    checkpoints = [
        f for f in os.listdir(checkpoints_dir) 
        if f.endswith('.pt') and f.startswith('checkpoint_epoch_')
    ]
    
    if not checkpoints:
        print(f"No checkpoints found in: {checkpoints_dir}")
        return None
    
    # Sort by epoch number (extracted from filename)
    def extract_epoch(filename):
        # Extract number from 'checkpoint_epoch_XXX.pt'
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except:
            return -1
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    latest = os.path.join(checkpoints_dir, checkpoints[0])
    
    print(f"Found {len(checkpoints)} checkpoint(s)")
    print(f"Latest: {checkpoints[0]}")
    
    return latest


def find_best_checkpoint(train_dir):
    """
    Find the best model checkpoint.
    
    Args:
        train_dir (str): Training directory containing best_model.pt
    
    Returns:
        str: Path to best model checkpoint, or None if not found
    
    Example:
        >>> best = find_best_checkpoint('/Volumes/catalog/schema/volume/run_20250121_143022/train')
        >>> if best:
        >>>     setup_resume(best)
    """
    best_path = os.path.join(train_dir, "best_model.pt")
    
    if os.path.exists(best_path):
        print(f"Found best model: {best_path}")
        return best_path
    else:
        print(f"Best model not found: {best_path}")
        return None


def get_resume_checkpoint(run_paths, prefer_best=False, specific_epoch=None):
    """
    Intelligently find the appropriate checkpoint for resuming.
    
    Args:
        run_paths (dict): Dictionary of organized paths from get_organized_paths()
        prefer_best (bool): If True, prefer best model over latest checkpoint
        specific_epoch (int): If provided, load checkpoint from specific epoch
    
    Returns:
        str: Path to checkpoint, or None if not found
    
    Example:
        >>> # Resume from latest checkpoint
        >>> ckpt = get_resume_checkpoint(run_paths)
        >>> 
        >>> # Resume from best model
        >>> ckpt = get_resume_checkpoint(run_paths, prefer_best=True)
        >>> 
        >>> # Resume from specific epoch
        >>> ckpt = get_resume_checkpoint(run_paths, specific_epoch=50)
        >>> 
        >>> if ckpt:
        >>>     setup_resume(ckpt, trainer)
    """
    print(f"\n{'='*70}")
    print("SEARCHING FOR RESUME CHECKPOINT")
    print(f"{'='*70}")
    
    # Specific epoch requested
    if specific_epoch is not None:
        checkpoints_dir = run_paths.get('train_checkpoints')
        if checkpoints_dir:
            specific_path = os.path.join(
                checkpoints_dir, 
                f"checkpoint_epoch_{specific_epoch:03d}.pt"
            )
            if os.path.exists(specific_path):
                print(f"✓ Found specific epoch checkpoint: epoch {specific_epoch}")
                return specific_path
            else:
                print(f"✗ Checkpoint for epoch {specific_epoch} not found")
                return None
    
    # Prefer best model
    if prefer_best:
        train_dir = run_paths.get('train')
        if train_dir:
            best_path = find_best_checkpoint(train_dir)
            if best_path:
                return best_path
            print("Best model not found, falling back to latest checkpoint...")
    
    # Find latest checkpoint
    checkpoints_dir = run_paths.get('train_checkpoints')
    if checkpoints_dir:
        latest_path = find_latest_checkpoint(checkpoints_dir)
        if latest_path:
            return latest_path
    
    print("✗ No checkpoints found")
    return None



# ## Usage Examples

# ### Example 1: Resume from Latest Checkpoint

# from ultralytics import YOLO
# import mlflow

# # Your run paths
# run_paths = {
#     'train': '/Volumes/catalog/schema/volume/run_20250121_143022/train',
#     'train_checkpoints': '/Volumes/catalog/schema/volume/run_20250121_143022/train/checkpoints',
#     # ... other paths
# }

# # Find and load latest checkpoint
# checkpoint_path = get_resume_checkpoint(run_paths)

# if checkpoint_path:
#     # Start MLflow run
#     with mlflow.start_run(run_name="resumed_training"):
#         # Setup resume (restores callback state)
#         checkpoint = setup_resume(checkpoint_path)
        
#         # Initialize model and train
#         model = YOLO('yolov8n-seg.pt')
        
#         # YOLO's built-in resume
#         results = model.train(
#             data='data.yaml',
#             epochs=100,
#             resume=True,  # YOLO will handle model/optimizer loading
#             project='yolo_training',
#             callbacks={'on_fit_epoch_end': mlflow_epoch_logger}
#         )
# else:
#     print("No checkpoint found, starting fresh training")


# ### Example 2: Resume from Best Model

# checkpoint_path = get_resume_checkpoint(run_paths, prefer_best=True)

# if checkpoint_path:
#     with mlflow.start_run(run_name="resume_from_best"):
#         setup_resume(checkpoint_path)
        
#         model = YOLO('yolov8n-seg.pt')
#         results = model.train(
#             data='data.yaml',
#             epochs=100,
#             resume=checkpoint_path,  # Explicitly specify checkpoint
#             callbacks={'on_fit_epoch_end': mlflow_epoch_logger}
#         )

# ### Example 3: Resume from Specific Epoch

# checkpoint_path = get_resume_checkpoint(run_paths, specific_epoch=50)

# if checkpoint_path:
#     with mlflow.start_run(run_name="resume_from_epoch_50"):
#         setup_resume(checkpoint_path)
        
#         model = YOLO('yolov8n-seg.pt')
#         results = model.train(
#             data='data.yaml',
#             epochs=100,
#             resume=checkpoint_path,
#             callbacks={'on_fit_epoch_end': mlflow_epoch_logger}
#         )


# ### Example 4: Manual Resume with Trainer Access

# from ultralytics.models.yolo.segment import SegmentationTrainer

# checkpoint_path = get_resume_checkpoint(run_paths)

# if checkpoint_path:
#     with mlflow.start_run(run_name="manual_resume"):
#         # Create trainer
#         model = YOLO('yolov8n-seg.pt')
#         trainer = SegmentationTrainer(overrides={'data': 'data.yaml', 'epochs': 100})
        
#         # Setup resume with trainer (loads model/optimizer state)
#         setup_resume(checkpoint_path, trainer=trainer)
        
#         # Train
#         trainer.train()


# ## Key Features

# 1. **Restores callback state**: `best_fitness`, `best_epoch`, `checkpoints_logged`
# 2. **Optionally loads trainer state**: model, optimizer, EMA
# 3. **Logs resume info to MLflow**
# 4. **Helper functions** to find latest/best/specific checkpoints
# 5. **Comprehensive error handling** and user feedback
# 6. **Works with YOLO's built-in resume** functionality

# The state variables are now properly restored, so `mlflow_epoch_logger` will continue tracking from where it left off!