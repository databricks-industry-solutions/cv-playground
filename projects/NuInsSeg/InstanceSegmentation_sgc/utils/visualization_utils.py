"""
YOLO Visualization Utilities
Author: may.merkletan@databricks.com
Last Updated: 2025Oct21
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import mlflow


def visualize_inference_results(
    inference_summary,
    num_samples=25,
    show_boxes=True,
    show_masks=True,
    show_labels=False,
    show_conf=False,
    box_color=(255, 255, 255),
    mask_color=(255, 255, 255),
    line_width=2,
    mask_alpha=0.5,
    font_scale=0.7,
    font_thickness=2,
    figsize=(15, 15),
    save_figure=True,
    show_figure=True,
    log_to_mlflow=False
):
    """
    Visualize inference results with custom overlays.
    Re-runs inference on random samples with custom visualization settings.
    
    Args:
        inference_summary: Output from run_inference_with_metrics()
        num_samples: Number of random images to display
        show_boxes: Show bounding boxes
        show_masks: Show segmentation masks
        show_labels: Show class labels
        show_conf: Show confidence scores
        box_color: RGB color for boxes
        mask_color: RGB color for masks
        line_width: Box line width
        mask_alpha: Mask transparency (0-1)
        font_scale: Label font size
        font_thickness: Label font thickness
        figsize: Figure size
        save_figure: Save visualization to file
        show_figure: Display figure
        log_to_mlflow: Log visualization to MLflow
    
    Returns:
        fig: Matplotlib figure object
    """
    from .inference_utils import load_model_from_run
    import builtins
    
    print(f"\n{'='*70}")
    print(f"VISUALIZING INFERENCE RESULTS")
    print('='*70)
    
    split = inference_summary['split']
    run_id = inference_summary['run_id']
    
    # Load model
    print(f"Loading model...")
    model = load_model_from_run(run_id, model_type='best', use_mlflow=False)
    class_names = model.names
    print(f"✓ Model loaded with {len(class_names)} classes")
    
    # Get images
    YOLO_DATA_UCVol_path = getattr(builtins, 'YOLO_DATA_UCVol_path', None)
    images_path = f"{YOLO_DATA_UCVol_path}/{split}/images"
    all_images = [
        os.path.join(images_path, img) 
        for img in os.listdir(images_path) 
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    print(f"✓ Found {len(all_images)} images")
    
    # Sample images
    num_samples = min(num_samples, len(all_images))
    selected_images = random.sample(all_images, num_samples)
    
    # Load and resize images
    resized_images = []
    image_names = []
    for img_path in selected_images:
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (640, 640))
        resized_images.append(resized_img)
        image_names.append(Path(img_path).name)
    
    print(f"✓ Loaded {len(resized_images)} images")
    
    # Run inference
    print(f"\nRunning inference with custom visualization...")
    print(f"  Boxes: {show_boxes} (RGB{box_color})")
    print(f"  Masks: {show_masks} (RGB{mask_color}, alpha={mask_alpha})")
    print(f"  Labels: {show_labels}, Confidence: {show_conf}")
    print(f"  Font: scale={font_scale}, thickness={font_thickness}")
    
    results = model.predict(resized_images, conf=0.25, verbose=False)
    print(f"✓ Inference complete")
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    print(f"\nDisplaying {num_samples} samples in {grid_size}x{grid_size} grid")
    
    # Plot images with custom overlays
    for idx, (img, result, ax, img_name) in enumerate(zip(resized_images, results, axes, image_names)):
        if idx >= num_samples:
            ax.axis('off')
            continue
        
        img_display = img.copy()
        
        # Draw masks first (if enabled)
        if show_masks and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            
            for mask in masks:
                mask_resized = cv2.resize(mask, (640, 640))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                colored_mask = np.zeros_like(img_display)
                colored_mask[mask_binary == 1] = mask_color
                img_display = cv2.addWeighted(img_display, 1.0, colored_mask, mask_alpha, 0)
        
        # Draw boxes (if enabled)
        if show_boxes and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_display, (x1, y1), (x2, y2), box_color, line_width)
                
                # Add label
                label_parts = []
                if show_labels:
                    label_parts.append(class_names[cls])
                if show_conf:
                    label_parts.append(f"{conf:.2f}")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        font_thickness
                    )
                    
                    # Draw label background
                    padding = 3
                    cv2.rectangle(
                        img_display,
                        (x1, y1 - text_height - baseline - padding * 2),
                        (x1 + text_width + padding, y1),
                        box_color, -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        img_display, 
                        label, 
                        (x1 + padding // 2, y1 - baseline - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        (0, 0, 0), 
                        font_thickness, 
                        cv2.LINE_AA
                    )
        
        # Display
        ax.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(img_name, fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = None
    if save_figure and inference_summary.get('predictions_path'):
        pred_path = inference_summary['predictions_path']
        save_path = os.path.join(pred_path, "custom_visualization.png")
        os.makedirs(pred_path, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    
    # Log to MLflow
    if log_to_mlflow and run_id and save_path:
        try:
            active_run = mlflow.active_run()
            
            if active_run and active_run.info.run_id == run_id:
                mlflow.log_artifact(save_path, artifact_path=f"inference/{split}/visualizations")
                print(f"✓ Logged visualization to active MLflow run")
            else:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(save_path, artifact_path=f"inference/{split}/visualizations")
                    print(f"✓ Logged visualization to MLflow run: {run_id}")
        except Exception as e:
            print(f"⚠ Warning: Could not log to MLflow: {e}")
    
    # Show figure
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    
    print('='*70)
    
    return fig


def visualize_predictions_vs_ground_truth(
    inference_summary,
    num_samples=9,
    show_boxes=True,
    show_masks=True,
    show_labels=False,
    show_conf=False,
    box_color=(255, 255, 255),
    mask_color=(255, 255, 255),
    line_width=2,
    mask_alpha=0.5,
    font_scale=0.7,
    font_thickness=2,
    figsize=(18, 12),
    save_figure=True,
    show_figure=True,
    log_to_mlflow=False
):
    """
    Show predictions alongside ground truth for comparison.
    
    Args:
        inference_summary: Output from run_inference_with_metrics()
        num_samples: Number of samples to show
        show_boxes: Show bounding boxes
        show_masks: Show segmentation masks
        show_labels: Show class labels
        show_conf: Show confidence scores
        box_color: RGB color for boxes
        mask_color: RGB color for masks
        line_width: Box line width
        mask_alpha: Mask transparency (0-1)
        font_scale: Label font size
        font_thickness: Label font thickness
        figsize: Figure size
        save_figure: Save visualization to file
        show_figure: Display figure
        log_to_mlflow: Log comparison to MLflow
    
    Returns:
        fig: Matplotlib figure object
    """
    from .inference_utils import load_model_from_run
    import builtins
    
    print(f"\n{'='*70}")
    print(f"PREDICTIONS vs GROUND TRUTH COMPARISON")
    print('='*70)
    
    split = inference_summary['split']
    run_id = inference_summary['run_id']
    
    # Load model
    print(f"Loading model...")
    model = load_model_from_run(run_id, model_type='best', use_mlflow=False)
    class_names = model.names
    print(f"✓ Model loaded with {len(class_names)} classes")
    
    # Get images
    YOLO_DATA_UCVol_path = getattr(builtins, 'YOLO_DATA_UCVol_path', None)
    original_images_path = f"{YOLO_DATA_UCVol_path}/{split}/images"
    
    print(f"Original images: {original_images_path}")
    
    if not os.path.exists(original_images_path):
        print(f"✗ Original images path not found: {original_images_path}")
        return None
    
    # Get all original images
    all_images = [
        os.path.join(original_images_path, img) 
        for img in os.listdir(original_images_path) 
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    print(f"✓ Found {len(all_images)} original images")
    
    if len(all_images) == 0:
        print(f"✗ No images found")
        return None
    
    # Randomly sample
    num_samples = min(num_samples, len(all_images))
    selected_images = random.sample(all_images, num_samples)
    
    # Load original images
    original_imgs = []
    resized_imgs = []
    image_names = []
    
    for img_path in selected_images:
        # Original
        orig_img = Image.open(img_path).convert('RGB')
        original_imgs.append(np.array(orig_img))
        
        # Resized for inference
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (640, 640))
        resized_imgs.append(resized_img)
        
        image_names.append(Path(img_path).name)
    
    print(f"✓ Loaded {len(original_imgs)} images")
    
    # Run inference
    print(f"\nRunning inference with custom visualization...")
    print(f"  Boxes: {show_boxes} (RGB{box_color})")
    print(f"  Masks: {show_masks} (RGB{mask_color}, alpha={mask_alpha})")
    print(f"  Labels: {show_labels}, Confidence: {show_conf}")
    print(f"  Font: scale={font_scale}, thickness={font_thickness}")
    
    results = model.predict(resized_imgs, conf=0.25, verbose=False)
    print(f"✓ Inference complete")
    
    # Create figure
    rows = num_samples
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nComparing {num_samples} samples")
    
    # Plot comparisons
    for idx, (orig_img, resized_img, result, img_name) in enumerate(zip(original_imgs, resized_imgs, results, image_names)):
        
        # Left: Original image
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"Original: {img_name}", fontsize=9)
        axes[idx, 0].axis('off')
        
        # Right: Prediction with custom overlays
        pred_img = resized_img.copy()
        
        # Draw masks first (if enabled)
        if show_masks and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            
            for mask in masks:
                mask_resized = cv2.resize(mask, (640, 640))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                colored_mask = np.zeros_like(pred_img)
                colored_mask[mask_binary == 1] = mask_color
                pred_img = cv2.addWeighted(pred_img, 1.0, colored_mask, mask_alpha, 0)
        
        # Draw boxes (if enabled)
        if show_boxes and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), box_color, line_width)
                
                # Add label
                label_parts = []
                if show_labels:
                    label_parts.append(class_names[cls])
                if show_conf:
                    label_parts.append(f"{conf:.2f}")
                
                if label_parts:
                    label = " ".join(label_parts)
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale,
                        font_thickness
                    )
                    
                    # Draw label background
                    padding = 3
                    cv2.rectangle(
                        pred_img,
                        (x1, y1 - text_height - baseline - padding * 2),
                        (x1 + text_width + padding, y1),
                        box_color, -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        pred_img, 
                        label, 
                        (x1 + padding // 2, y1 - baseline - padding),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                        cv2.LINE_AA
                    )
        
        # Display prediction
        axes[idx, 1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f"Prediction: {img_name}", fontsize=9)
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    
    # Save comparison figure
    save_path = None
    if save_figure and inference_summary.get('predictions_path'):
        pred_path = inference_summary['predictions_path']
        save_path = os.path.join(pred_path, "custom_comparison.png")
        os.makedirs(pred_path, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison saved to: {save_path}")
    
    # Log to MLflow
    if log_to_mlflow and run_id and save_path:
        try:
            active_run = mlflow.active_run()
            
            if active_run and active_run.info.run_id == run_id:
                mlflow.log_artifact(save_path, artifact_path=f"inference/{split}/comparisons")
                print(f"✓ Logged comparison to active MLflow run")
            else:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(save_path, artifact_path=f"inference/{split}/comparisons")
                    print(f"✓ Logged comparison to MLflow run: {run_id}")
        except Exception as e:
            print(f"⚠ Warning: Could not log to MLflow: {e}")
    
    # Show figure
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    
    print('='*70)
    
    return fig

