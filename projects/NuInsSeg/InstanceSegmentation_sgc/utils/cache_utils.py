"""
Cache Clearning Utilities
Author: may.merkletan@databricks.com
Last Updated: 2025Oct22
"""

import os
import gc
import torch
import pandas as pd
from pathlib import Path


def clear_cuda_cache():
    """Clear CUDA cache and print memory stats."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"✓ CUDA cache cleared")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB\n")
    else:
        print("⚠ CUDA not available\n")


# Enhanced CUDA cache clearing before inference
import gc
import torch

def clear_cuda_cache_aggressive():
    """Aggressively clear CUDA cache and free memory."""
    if torch.cuda.is_available():
        # Collect Python garbage
        gc.collect()
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        print(f"CUDA cache cleared aggressively")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Free:      {free:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Utilization: {(reserved/total)*100:.0f}%\n")
    else:
        print("CUDA not available\n")

# Run before inference
# clear_cuda_cache_aggressive()


def gpu_status():
    """Quick GPU status check."""
    if not torch.cuda.is_available():
        print("⚠ CUDA not available")
        return
    
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free = total - reserved
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total: {total:.1f} GB | Used: {reserved:.1f} GB | Free: {free:.1f} GB")
    print(f"Utilization: {reserved/total*100:.0f}%")



def clear_all_caches():
    """
    Clear all caches on serverless compute
    
    Note: To restart Python, run dbutils.library.restartPython() after this
    """
    import gc
    import sys
    import torch
    import shutil
    import os

    print("Clearing caches...")

    # 1. Clear Python modules
    custom_modules = [
        k for k in sys.modules.keys()
        if k.startswith(('utils', 'yolo_', 'custom_'))
    ]
    for module in custom_modules:
        del sys.modules[module]
    print(f"[OK] Cleared {len(custom_modules)} Python modules")

    # 2. Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[OK] Cleared CUDA cache")

    # 3. Clear Ultralytics cache
    ultralytics_config = os.environ.get('ULTRALYTICS_CONFIG_DIR')
    if ultralytics_config and os.path.exists(ultralytics_config):
        for subdir in ['weights', 'datasets', 'runs']:
            path = os.path.join(ultralytics_config, subdir)
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
        print("[OK] Cleared Ultralytics cache")

    # 4. Garbage collection
    gc.collect()
    print("[OK] Garbage collection complete")
    
    print("\n" + "="*70)
    print("CACHE CLEARING COMPLETE")
    print("="*70)
    print("\nFor complete reset, run in next cell:")
    print("  dbutils.library.restartPython()")
    print("="*70)

