"""
MLflow Callbacks for YOLO Training
Author: may.merkletan@databricks.com
Last Updated: 2025Oct21
"""

import os
import torch
import mlflow
import pandas as pd
import numpy as np


# Global configuration
CHECKPOINT_LOG_FREQUENCY = 1 #default unless overwritten/updated
LOG_BEST_MODEL = True
LOG_FINAL_EPOCH = True
LOG_FIRST_EPOCH = True


def _flatten_metrics(d: dict, prefix: str = "") -> dict:
    """Flatten metric dictionary."""
    flat = {}
    for k, v in d.items():
        if k.startswith("metrics/"):
            k = k.split("/", 1)[1]
        k = k.replace("(B)", "").replace("(M)", "")
        flat[f"{prefix}{k}"] = v
    return flat


def _should_log_checkpoint(epoch, total_epochs=None):
    """Determine if checkpoint should be logged."""
    if LOG_FIRST_EPOCH and epoch == 0:
        return True
    if LOG_FINAL_EPOCH and total_epochs is not None and epoch == total_epochs - 1:
        return True
    if (epoch + 1) % CHECKPOINT_LOG_FREQUENCY == 0:
        return True
    return False


def _generate_training_summary(run_id, timestamp, n_epochs, batch_sz, run_paths):
    """Generate comprehensive training summary."""
    summary_lines = []
    summary_lines.append(f"YOLO Training Summary")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Run ID: {run_id}")
    summary_lines.append(f"Timestamp: {timestamp}")
    summary_lines.append(f"Epochs: {n_epochs}")
    summary_lines.append(f"Batch Size: {batch_sz}")
    summary_lines.append("")
    
    # Checkpoint info
    if hasattr(mlflow_epoch_logger, 'checkpoints_logged'):
        summary_lines.append(f"Checkpoints: {mlflow_epoch_logger.checkpoints_logged}")
        if hasattr(mlflow_epoch_logger, 'best_fitness') and mlflow_epoch_logger.best_fitness > float('-inf'):
            summary_lines.append(f"Best Epoch: {mlflow_epoch_logger.best_epoch + 1}")
            summary_lines.append(f"Best Fitness: {mlflow_epoch_logger.best_fitness:.5f}")
    
    # Paths
    summary_lines.append(f"\nPaths:")
    for key, path in run_paths.items():
        summary_lines.append(f"  {key}: {path}")
    
    return "\n".join(summary_lines)


def mlflow_epoch_logger(trainer):
    """
    Called AFTER validation completes each epoch
    Logs metrics, checkpoints, and tracks best model
    """
    epoch = trainer.epoch
    total_epochs = getattr(trainer.args, 'epochs', None)

    # Get validation metrics
    val_metrics = {}
    cur_fitness = None
    
    if hasattr(trainer, 'metrics') and trainer.metrics is not None:
        metrics_dict = trainer.metrics
        
        # Extract box metrics
        box_map50_95 = metrics_dict.get('metrics/mAP50-95(B)')
        box_map50 = metrics_dict.get('metrics/mAP50(B)')
        
        # Extract mask metrics (segmentation)
        mask_map50_95 = metrics_dict.get('metrics/mAP50-95(M)')
        mask_map50 = metrics_dict.get('metrics/mAP50(M)')
        
        # Store all validation metrics
        val_metrics = {
            'val_box_map50_95': box_map50_95,
            'val_box_map50': box_map50,
            'val_box_precision': metrics_dict.get('metrics/precision(B)'),
            'val_box_recall': metrics_dict.get('metrics/recall(B)'),
            'val_mask_map50_95': mask_map50_95,
            'val_mask_map50': mask_map50,
            'val_mask_precision': metrics_dict.get('metrics/precision(M)'),
            'val_mask_recall': metrics_dict.get('metrics/recall(M)'),
        }
        
        # Calculate fitness using YOLO's formula
        if box_map50_95 is not None and box_map50 is not None:
            cur_fitness = 0.1 * float(box_map50) + 0.9 * float(box_map50_95)
            val_metrics['fitness'] = cur_fitness
    
    # Fallback to validator if metrics not in trainer
    elif hasattr(trainer, 'validator') and trainer.validator is not None:
        validator = trainer.validator
        
        if hasattr(validator, 'metrics') and validator.metrics is not None:
            metrics_obj = validator.metrics
            
            # Box metrics
            if hasattr(metrics_obj, 'box'):
                box = metrics_obj.box
                box_map50_95 = box.map if hasattr(box, 'map') else None
                box_map50 = box.map50 if hasattr(box, 'map50') else None
                
                val_metrics['val_box_map50_95'] = box_map50_95
                val_metrics['val_box_map50'] = box_map50
                val_metrics['val_box_precision'] = box.mp if hasattr(box, 'mp') else None
                val_metrics['val_box_recall'] = box.mr if hasattr(box, 'mr') else None
            
            # Mask metrics
            if hasattr(metrics_obj, 'seg'):
                seg = metrics_obj.seg
                mask_map50_95 = seg.map if hasattr(seg, 'map') else None
                mask_map50 = seg.map50 if hasattr(seg, 'map50') else None
                
                val_metrics['val_mask_map50_95'] = mask_map50_95
                val_metrics['val_mask_map50'] = mask_map50
                val_metrics['val_mask_precision'] = seg.mp if hasattr(seg, 'mp') else None
                val_metrics['val_mask_recall'] = seg.mr if hasattr(seg, 'mr') else None
            
            # Calculate fitness
            if box_map50_95 is not None and box_map50 is not None:
                cur_fitness = 0.1 * float(box_map50) + 0.9 * float(box_map50_95)
                val_metrics['fitness'] = cur_fitness

    # Training metrics
    train_metrics = {}
    
    # Get from trainer.loss_items (current batch losses)
    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
        loss_items = trainer.loss_items
        if len(loss_items) >= 4:
            train_metrics['train_box_loss'] = float(loss_items[0])
            train_metrics['train_seg_loss'] = float(loss_items[1])
            train_metrics['train_cls_loss'] = float(loss_items[2])
            train_metrics['train_dfl_loss'] = float(loss_items[3])
            train_metrics['train_total_loss'] = float(sum(loss_items))
    
    # Also get epoch-averaged losses from CSV
    csv_path = os.path.join(trainer.save_dir, "results.csv")
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and len(df) > epoch:
                row = df.iloc[epoch].to_dict()
                for k, v in row.items():
                    if k.strip() != "epoch" and pd.notna(v):
                        clean_key = k.strip().replace('/', '_').replace('(', '').replace(')', '')
                        train_metrics[f'epoch_avg_{clean_key}'] = float(v)
        except Exception as e:
            print(f"[Callback] CSV warning: {e}")

    # Learning rate
    lr = None
    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        lr = trainer.optimizer.param_groups[0]["lr"]

    # Log all metrics to MLflow
    merged = {**train_metrics, **val_metrics}
    if lr is not None:
        merged["learning_rate"] = lr
    if cur_fitness is not None:
        merged["fitness"] = cur_fitness
    
    # Remove None values
    merged = {k: v for k, v in merged.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    
    if merged:
        mlflow.log_metrics(merged, step=epoch)

    # Initialize tracking variables
    if not hasattr(mlflow_epoch_logger, "best_fitness"):
        mlflow_epoch_logger.best_fitness = float("-inf")
        mlflow_epoch_logger.best_epoch = -1
        mlflow_epoch_logger.checkpoints_logged = 0

    # Checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict() if hasattr(trainer, 'optimizer') else None,
        'fitness': cur_fitness,
        'metrics': merged,
        'lr': lr,
        'total_epochs': total_epochs
    }

    # Regular checkpoint logging
    should_log_regular = _should_log_checkpoint(epoch, total_epochs)
    
    if should_log_regular:
        # Import tmp_project_location from globals
        import builtins
        tmp_project_location = getattr(builtins, 'tmp_project_location', '/local_disk0/tmp/nuinsseg/')
        
        checkpoints_dir = os.path.join(tmp_project_location, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        epoch_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
        try:
            torch.save(checkpoint_data, epoch_checkpoint_path)
            mlflow.log_artifact(epoch_checkpoint_path, artifact_path="checkpoints")
            mlflow_epoch_logger.checkpoints_logged += 1
            print(f"[MLflow] Logged checkpoint for epoch {epoch+1}")
        except Exception as e:
            print(f"[Callback] Checkpoint save failed: {e}")

    # Best model tracking (only update if fitness improves)
    if cur_fitness is not None and LOG_BEST_MODEL:
        if cur_fitness > mlflow_epoch_logger.best_fitness:
            mlflow_epoch_logger.best_fitness = cur_fitness
            mlflow_epoch_logger.best_epoch = epoch

            # Import tmp_project_location from globals
            import builtins
            tmp_project_location = getattr(builtins, 'tmp_project_location', '/local_disk0/tmp/nuinsseg/')
            
            best_path = os.path.join(tmp_project_location, "best_model.pt")
            try:
                best_data = checkpoint_data.copy()
                best_data['is_best_model'] = True
                best_data['best_fitness'] = cur_fitness
                
                torch.save(best_data, best_path)
                mlflow.log_artifact(best_path, artifact_path="best")
                
                mlflow.log_metrics({
                    "best_fitness": cur_fitness,
                    "best_epoch": epoch + 1,
                }, step=epoch)
                
                print(f"[MLflow] New best model (fitness={cur_fitness:.5f}) at epoch {epoch+1}")
                
            except Exception as e:
                print(f"[Callback] Best model save failed: {e}")
    
    # Track epochs since best improvement
    if mlflow_epoch_logger.best_epoch >= 0:
        epochs_since_best = epoch - mlflow_epoch_logger.best_epoch
        mlflow.log_metric("epochs_since_best", epochs_since_best, step=epoch)

    # Console output
    def _fmt(val, fmt="{:.4f}"):
        if val is None:
            return "-"
        if isinstance(val, (int, float)) and not np.isnan(val):
            return fmt.format(val)
        return "-"

    status = ""
    if should_log_regular:
        status = " [CKPT]"
    
    print(
        f"[Epoch {epoch+1}/{total_epochs or '?'}]{status} "
        f"train_loss={_fmt(train_metrics.get('train_total_loss'))} "
        f"val_mAP50-95={_fmt(val_metrics.get('val_box_map50_95'))} "
        f"fitness={_fmt(cur_fitness, '{:.5f}')} "
        f"best={_fmt(mlflow_epoch_logger.best_fitness, '{:.5f}')} "
        f"lr={_fmt(lr, '{:.6f}')}"
    )


def configure_checkpoint_logging(frequency=10, log_best=True, log_final=True, log_first=True):
    """
    Configure checkpoint logging parameters.
    
    Args:
        frequency (int): Log checkpoint every N epochs
        log_best (bool): Always log best model
        log_final (bool): Always log final epoch
        log_first (bool): Always log first epoch
    """
    global CHECKPOINT_LOG_FREQUENCY, LOG_BEST_MODEL, LOG_FINAL_EPOCH, LOG_FIRST_EPOCH
    
    CHECKPOINT_LOG_FREQUENCY = frequency
    LOG_BEST_MODEL = log_best
    LOG_FINAL_EPOCH = log_final
    LOG_FIRST_EPOCH = log_first
    
    print(f"Updated checkpoint logging configuration:")
    print(f"  - Frequency: every {frequency} epochs")
    print(f"  - Log best model: {log_best}")
    print(f"  - Log final epoch: {log_final}")
    print(f"  - Log first epoch: {log_first}")


def copy_training_artifacts(tmp_project_location, run_paths):
    """
    Copy training artifacts from temp location to organized UC Volumes structure.
    
    Args:
        tmp_project_location: Temporary training directory
        run_paths: Dictionary of organized paths from get_organized_paths()
    
    Returns:
        bool: True if successful, False otherwise
    """
    import shutil
    
    print("Copying training artifacts...")
    try:
        yolo_training_path = os.path.join(tmp_project_location, "yolo_training")
        
        if os.path.exists(yolo_training_path):
            train_dest = run_paths['train']
            
            # Copy each item directly
            for item in os.listdir(yolo_training_path):
                src_item = os.path.join(yolo_training_path, item)
                dst_item = os.path.join(train_dest, item)
                
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                elif os.path.isdir(src_item):
                    if os.path.exists(dst_item):
                        shutil.rmtree(dst_item)
                    shutil.copytree(src_item, dst_item)
            
            print(f"SUCCESS: Copied training artifacts to {train_dest}")
            
            # Copy checkpoints
            checkpoints_src = os.path.join(tmp_project_location, "checkpoints")
            checkpoints_dst = run_paths['train_checkpoints']
            
            if os.path.exists(checkpoints_src):
                checkpoint_count = 0
                for ckpt_file in os.listdir(checkpoints_src):
                    if ckpt_file.endswith('.pt'):
                        shutil.copy2(
                            os.path.join(checkpoints_src, ckpt_file),
                            os.path.join(checkpoints_dst, ckpt_file)
                        )
                        checkpoint_count += 1
                print(f"SUCCESS: Copied {checkpoint_count} checkpoints")
            
            # Copy best model
            best_model_src = os.path.join(tmp_project_location, "best_model.pt")
            if os.path.exists(best_model_src):
                shutil.copy2(best_model_src, os.path.join(train_dest, "best_model.pt"))
                print(f"SUCCESS: Copied best model")
            
            return True
            
    except Exception as e:
        print(f"WARNING: Failed to copy artifacts: {e}")
        return False


def log_training_artifacts_to_mlflow(run_paths, data_yaml_path=None):
    """
    Log training artifacts to MLflow.
    
    Args:
        run_paths: Dictionary of organized paths from get_organized_paths()
        data_yaml_path: Optional path to data.yaml file to log
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\nLogging artifacts to MLflow...")
    try:
        train_path = run_paths['train']
        
        if os.path.exists(train_path):
            # Log root files
            for item in os.listdir(train_path):
                item_path = os.path.join(train_path, item)
                
                if os.path.isfile(item_path):
                    if item.endswith(('.png', '.jpg', '.csv', '.yaml', '.txt', '.pt')):
                        mlflow.log_artifact(item_path, artifact_path="train")
                
                elif os.path.isdir(item_path):
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isfile(subitem_path):
                            mlflow.log_artifact(subitem_path, artifact_path=f"train/{item}")
            
            print("SUCCESS: Logged training artifacts to MLflow")
        
        # Log dataset config if provided
        if data_yaml_path and os.path.exists(data_yaml_path):
            mlflow.log_artifact(data_yaml_path, artifact_path="dataset")
            print("SUCCESS: Logged dataset config to MLflow")
        
        return True
        
    except Exception as e:
        print(f"WARNING: Failed to log artifacts: {e}")
        return False


def finalize_training_run(run, timestamp, n_epochs, batch_sz, run_paths):
    """
    Finalize training run with metrics, summary, and cleanup.
    
    Args:
        run: MLflow run object
        timestamp: Training timestamp string
        n_epochs: Number of training epochs
        batch_sz: Batch size used
        run_paths: Dictionary of organized paths
    """
    # Log final metrics
    print(f"\n{'='*70}")
    print("FINALIZING TRAINING")
    print(f"{'='*70}\n")
    
    if hasattr(mlflow_epoch_logger, 'checkpoints_logged'):
        mlflow.log_metric("total_checkpoints_logged", mlflow_epoch_logger.checkpoints_logged)
        
        if hasattr(mlflow_epoch_logger, 'best_epoch') and mlflow_epoch_logger.best_epoch >= 0:
            mlflow.log_metric("final_best_epoch", mlflow_epoch_logger.best_epoch)
            
        if hasattr(mlflow_epoch_logger, 'best_fitness') and mlflow_epoch_logger.best_fitness > float('-inf'):
            mlflow.log_metric("final_best_fitness", mlflow_epoch_logger.best_fitness)
    
    # Generate and log training summary
    print("\nGenerating training summary...")
    try:
        train_dir = run_paths['train']
        os.makedirs(train_dir, exist_ok=True)
        
        summary_path = os.path.join(train_dir, "training_summary.txt")
        
        summary_text = _generate_training_summary(
            run.info.run_id, 
            timestamp, 
            n_epochs, 
            batch_sz, 
            run_paths
        )
        
        with open(summary_path, "w") as f:
            f.write(summary_text)
        
        if os.path.exists(summary_path):
            print(f"SUCCESS: Training summary saved to: {summary_path}")
            mlflow.log_artifact(summary_path, artifact_path="train")
            print(f"SUCCESS: Training summary logged to MLflow")
        
    except Exception as e:
        print(f"WARNING: Failed to create training summary: {e}")
    
    # Display completion info
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Run ID: {run.info.run_id}")
    print(f"Timestamp: {timestamp}")
    
    if hasattr(mlflow_epoch_logger, 'checkpoints_logged'):
        print(f"Checkpoints: {mlflow_epoch_logger.checkpoints_logged}")
        if hasattr(mlflow_epoch_logger, 'best_fitness') and mlflow_epoch_logger.best_fitness > float('-inf'):
            print(f"Best Epoch: {mlflow_epoch_logger.best_epoch + 1}")
            print(f"Best Fitness: {mlflow_epoch_logger.best_fitness:.5f}")
    
    print(f"\nPaths:")
    print(f"  Training: {run_paths['train']}")
    print(f"  Weights: {run_paths['train_weights']}")
    print("="*70)


## NOTE
# Important: YOLO (Ultralytics) has built-in checkpoint resume functionality:

# YOLO automatically saves 'last.pt' after each epoch
# To resume, simply:
# model.train(resume=True)  # Resumes from 'last.pt' in the last run directory

# Or specify a checkpoint:
# model.train(resume='path/to/checkpoint.pt')
# The saved callbacks would work alongside this, but please ensure the MLflow tracking variables are restored when resuming.
# Recommendation: Add a setup_resume() function that restores the mlflow_epoch_logger state variables when resuming training.

