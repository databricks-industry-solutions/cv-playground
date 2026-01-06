"""
YOLO Summary and Reporting Utilities
Author: may.merkletan@databricks.com
Last Updated: 2025Oct21
"""

import os
# import gc
# import torch
import pandas as pd
from pathlib import Path


# def clear_cuda_cache():
#     """Clear CUDA cache and print memory stats."""
#     if torch.cuda.is_available():
#         gc.collect()
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
        
#         allocated = torch.cuda.memory_allocated(0) / 1024**3
#         reserved = torch.cuda.memory_reserved(0) / 1024**3
        
#         print(f"✓ CUDA cache cleared")
#         print(f"  Allocated: {allocated:.2f} GB")
#         print(f"  Reserved:  {reserved:.2f} GB\n")
#     else:
#         print("⚠ CUDA not available\n")


# # Enhanced CUDA cache clearing before inference
# import gc
# import torch

# def clear_cuda_cache_aggressive():
#     """Aggressively clear CUDA cache and free memory."""
#     if torch.cuda.is_available():
#         # Collect Python garbage
#         gc.collect()
        
#         # Empty CUDA cache
#         torch.cuda.empty_cache()
        
#         # Synchronize CUDA operations
#         torch.cuda.synchronize()
        
#         # Reset peak memory stats
#         torch.cuda.reset_peak_memory_stats()
#         torch.cuda.reset_accumulated_memory_stats()
        
#         # Get memory stats
#         allocated = torch.cuda.memory_allocated(0) / 1024**3
#         reserved = torch.cuda.memory_reserved(0) / 1024**3
#         total = torch.cuda.get_device_properties(0).total_memory / 1024**3
#         free = total - reserved
        
#         print(f"CUDA cache cleared aggressively")
#         print(f"  Allocated: {allocated:.2f} GB")
#         print(f"  Reserved:  {reserved:.2f} GB")
#         print(f"  Free:      {free:.2f} GB")
#         print(f"  Total:     {total:.2f} GB")
#         print(f"  Utilization: {(reserved/total)*100:.0f}%\n")
#     else:
#         print("CUDA not available\n")

# # Run before inference
# # clear_cuda_cache_aggressive()


# def gpu_status():
#     """Quick GPU status check."""
#     if not torch.cuda.is_available():
#         print("⚠ CUDA not available")
#         return
    
#     total = torch.cuda.get_device_properties(0).total_memory / 1024**3
#     reserved = torch.cuda.memory_reserved(0) / 1024**3
#     free = total - reserved
    
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"Total: {total:.1f} GB | Used: {reserved:.1f} GB | Free: {free:.1f} GB")
#     print(f"Utilization: {reserved/total*100:.0f}%")


def print_inference_summary(inference_summary, include_paths=True, include_performance=True):
    """
    Print a formatted summary of inference results.
    
    Args:
        inference_summary: Output from run_inference_with_metrics()
        include_paths: Include file paths in summary
        include_performance: Include performance metrics
    """
    print(f"\n{'='*70}")
    print("INFERENCE SUMMARY")
    print('='*70)
    
    # Basic info
    print(f"\nSplit: {inference_summary['split']}")
    print(f"Images: {inference_summary['num_images']}")
    print(f"Device: {inference_summary['device'].upper()}")
    print(f"Batch Size: {inference_summary['batch_size']} (auto-detected)")
    print(f"Image Size: {inference_summary['img_size']}")
    print(f"Timestamp: {inference_summary['timestamp']}")
    
    # Metrics
    if inference_summary.get('metrics'):
        metrics = inference_summary['metrics']
        print(f"\nMetrics:")
        print(f"  Box mAP50-95:  {metrics.get('box_map50_95', 0):.4f}")
        print(f"  Box mAP50:     {metrics.get('box_map50', 0):.4f}")
        print(f"  Box Precision: {metrics.get('box_precision', 0):.4f}")
        print(f"  Box Recall:    {metrics.get('box_recall', 0):.4f}")
        print(f"")
        print(f"  Mask mAP50-95: {metrics.get('mask_map50_95', 0):.4f}")
        print(f"  Mask mAP50:    {metrics.get('mask_map50', 0):.4f}")
        print(f"  Mask Precision: {metrics.get('mask_precision', 0):.4f}")
        print(f"  Mask Recall:   {metrics.get('mask_recall', 0):.4f}")
        print(f"")
        print(f"  Fitness:       {metrics.get('fitness', 0):.5f}")
    
    # Performance
    if include_performance:
        perf = inference_summary.get('performance', {})
        
        if perf and any(k in perf for k in ['preprocess', 'inference', 'postprocess']):
            print(f"\nPerformance:")
            
            preprocess = perf.get('preprocess', 0)
            inference = perf.get('inference', 0)
            postprocess = perf.get('postprocess', 0)
            
            print(f"  Preprocess:   {preprocess:.2f}ms per image")
            print(f"  Inference:    {inference:.2f}ms per image")
            print(f"  Postprocess:  {postprocess:.2f}ms per image")
            
            total = preprocess + inference + postprocess
            print(f"  Total:        {total:.2f}ms per image")
            
            if total > 0:
                fps = 1000.0 / total
                print(f"  Throughput:   {fps:.1f} FPS")
        else:
            print(f"\nPerformance: Not available")
    
    # Visualization settings
    if inference_summary.get('visualization'):
        viz = inference_summary['visualization']
        print(f"\nVisualization Settings:")
        print(f"  Boxes: {viz.get('show_boxes', False)}")
        print(f"  Masks: {viz.get('show_masks', False)}")
        print(f"  Labels: {viz.get('show_labels', False)}")
        print(f"  Confidence: {viz.get('show_conf', False)}")
        if viz.get('show_boxes') or viz.get('show_masks'):
            print(f"  Box Color: RGB{viz.get('box_color', (0, 255, 0))}")
            print(f"  Mask Color: RGB{viz.get('mask_color', (0, 0, 255))}")
    
    # Paths
    if include_paths:
        print(f"\nOutput Locations:")
        if inference_summary.get('predictions_path'):
            print(f"  Predictions: {inference_summary['predictions_path']}")
        if inference_summary.get('metrics_path'):
            print(f"  Metrics: {inference_summary['metrics_path']}")
        
        # Check what files exist
        predictions_path = inference_summary.get('predictions_path')
        if predictions_path and os.path.exists(predictions_path):
            # Look for images
            images = list(Path(predictions_path).glob("*.jpg")) + \
                    list(Path(predictions_path).glob("*.png"))
            
            # Check results subdirectory
            results_dir = os.path.join(predictions_path, "results")
            if os.path.exists(results_dir):
                images += list(Path(results_dir).glob("*.jpg")) + \
                         list(Path(results_dir).glob("*.png"))
            
            # Check for labels
            labels_dir = os.path.join(predictions_path, "labels")
            labels = list(Path(labels_dir).glob("*.txt")) if os.path.exists(labels_dir) else []
            
            print(f"\nSaved Files:")
            print(f"  Prediction images: {len(images)}")
            print(f"  Label files: {len(labels)}")
            
            # Check for custom visualizations
            custom_viz = os.path.join(predictions_path, "custom_visualization.png")
            custom_comp = os.path.join(predictions_path, "custom_comparison.png")
            
            if os.path.exists(custom_viz):
                print(f"  Custom visualization: {Path(custom_viz).name}")
            if os.path.exists(custom_comp):
                print(f"  Custom comparison: {Path(custom_comp).name}")
    
    print('='*70)


def print_multi_split_summary(summaries_dict):
    """
    Print summary for multiple splits (train, val, test).
    
    Args:
        summaries_dict: Dict of {split: inference_summary}
    """
    print(f"\n{'='*70}")
    print("MULTI-SPLIT INFERENCE SUMMARY")
    print('='*70)
    
    # Create comparison table
    rows = []
    for split, summary in summaries_dict.items():
        metrics = summary.get('metrics', {})
        perf = summary.get('performance', {})
        
        row = {
            'Split': split.upper(),
            'Images': summary.get('num_images', 0),
            'Box mAP50-95': f"{metrics.get('box_map50_95', 0):.4f}",
            'Box mAP50': f"{metrics.get('box_map50', 0):.4f}",
            'Mask mAP50-95': f"{metrics.get('mask_map50_95', 0):.4f}",
            'Mask mAP50': f"{metrics.get('mask_map50', 0):.4f}",
            'Fitness': f"{metrics.get('fitness', 0):.5f}",
        }
        
        # Add performance if available
        if perf and 'inference' in perf:
            row['Inference (ms)'] = f"{perf['inference']:.2f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")
    
    # Detailed info for each split
    for split, summary in summaries_dict.items():
        print(f"\n{'-'*70}")
        print(f"{split.upper()} SPLIT DETAILS")
        print(f"{'-'*70}")
        print(f"Device: {summary.get('device', 'N/A')}")
        print(f"Batch Size: {summary.get('batch_size', 'N/A')}")
        print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
        
        # Performance details
        perf = summary.get('performance', {})
        if perf and any(k in perf for k in ['preprocess', 'inference', 'postprocess']):
            total = sum([perf.get('preprocess', 0), perf.get('inference', 0), perf.get('postprocess', 0)])
            print(f"Performance: {total:.2f}ms per image ({1000/total:.1f} FPS)")
        
        if summary.get('predictions_path'):
            print(f"Output: {summary['predictions_path']}")
    
    print('='*70)


def export_inference_summary_markdown(inference_summary, output_path=None):
    """
    Export inference summary as a markdown file.
    
    Args:
        inference_summary: Output from run_inference_with_metrics()
        output_path: Path to save markdown file (auto-generated if None)
    """
    if output_path is None:
        predictions_path = inference_summary.get('predictions_path', '.')
        output_path = os.path.join(predictions_path, "inference_summary.md")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Inference Summary\n\n")
        
        # Basic info
        f.write("## Configuration\n\n")
        f.write(f"- **Split**: {inference_summary['split']}\n")
        f.write(f"- **Images**: {inference_summary['num_images']}\n")
        f.write(f"- **Device**: {inference_summary['device'].upper()}\n")
        f.write(f"- **Batch Size**: {inference_summary['batch_size']}\n")
        f.write(f"- **Image Size**: {inference_summary['img_size']}\n")
        f.write(f"- **Timestamp**: {inference_summary['timestamp']}\n")
        f.write(f"- **Run ID**: {inference_summary.get('run_id', 'N/A')}\n\n")
        
        # Metrics
        if inference_summary.get('metrics'):
            metrics = inference_summary['metrics']
            f.write("## Metrics\n\n")
            f.write("### Box Detection\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| mAP50-95 | {metrics.get('box_map50_95', 0):.4f} |\n")
            f.write(f"| mAP50 | {metrics.get('box_map50', 0):.4f} |\n")
            f.write(f"| Precision | {metrics.get('box_precision', 0):.4f} |\n")
            f.write(f"| Recall | {metrics.get('box_recall', 0):.4f} |\n\n")
            
            f.write("### Mask Segmentation\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| mAP50-95 | {metrics.get('mask_map50_95', 0):.4f} |\n")
            f.write(f"| mAP50 | {metrics.get('mask_map50', 0):.4f} |\n")
            f.write(f"| Precision | {metrics.get('mask_precision', 0):.4f} |\n")
            f.write(f"| Recall | {metrics.get('mask_recall', 0):.4f} |\n\n")
            
            f.write(f"**Fitness Score**: {metrics.get('fitness', 0):.5f}\n\n")
        
        # Performance
        perf = inference_summary.get('performance', {})
        if perf and any(k in perf for k in ['preprocess', 'inference', 'postprocess']):
            f.write("## Performance\n\n")
            f.write(f"| Stage | Time (ms) |\n")
            f.write(f"|-------|----------|\n")
            f.write(f"| Preprocess | {perf.get('preprocess', 0):.2f} |\n")
            f.write(f"| Inference | {perf.get('inference', 0):.2f} |\n")
            f.write(f"| Postprocess | {perf.get('postprocess', 0):.2f} |\n")
            
            total = sum([perf.get('preprocess', 0), perf.get('inference', 0), perf.get('postprocess', 0)])
            fps = 1000.0 / total if total > 0 else 0
            f.write(f"| **Total** | **{total:.2f}** |\n\n")
            f.write(f"**Throughput**: {fps:.1f} FPS\n\n")
        
        # Visualization settings
        if inference_summary.get('visualization'):
            viz = inference_summary['visualization']
            f.write("## Visualization Settings\n\n")
            f.write(f"- **Show Boxes**: {viz.get('show_boxes', False)}\n")
            f.write(f"- **Show Masks**: {viz.get('show_masks', False)}\n")
            f.write(f"- **Show Labels**: {viz.get('show_labels', False)}\n")
            f.write(f"- **Show Confidence**: {viz.get('show_conf', False)}\n")
            f.write(f"- **Box Color**: RGB{viz.get('box_color', (0, 255, 0))}\n")
            f.write(f"- **Mask Color**: RGB{viz.get('mask_color', (0, 0, 255))}\n\n")
        
        # Paths
        f.write("## Output Locations\n\n")
        if inference_summary.get('predictions_path'):
            f.write(f"- **Predictions**: `{inference_summary['predictions_path']}`\n")
        if inference_summary.get('metrics_path'):
            f.write(f"- **Metrics**: `{inference_summary['metrics_path']}`\n")
    
    print(f"✓ Markdown summary saved to: {output_path}")
    return output_path

