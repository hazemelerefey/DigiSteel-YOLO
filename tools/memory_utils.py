"""
Memory Management Utilities for DigiSteel-YOLO Training
Fixes RAM/VRAM issues during training runs.
"""
import gc
import os
import torch
import psutil
from pathlib import Path


def get_memory_status():
    """Get current memory status."""
    ram = psutil.virtual_memory()
    status = {
        'ram_total_gb': ram.total / 1024**3,
        'ram_used_gb': ram.used / 1024**3,
        'ram_available_gb': ram.available / 1024**3,
        'ram_percent': ram.percent,
    }
    
    if torch.cuda.is_available():
        status['gpu_name'] = torch.cuda.get_device_name(0)
        status['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_mem / 1024**3
        status['gpu_free_gb'] = torch.cuda.mem_get_info(0)[0] / 1024**3
        status['gpu_used_gb'] = status['gpu_total_gb'] - status['gpu_free_gb']
    
    return status


def print_memory_status():
    """Print current memory status."""
    status = get_memory_status()
    print("\n💾 MEMORY STATUS:")
    print("=" * 50)
    print(f"  RAM: {status['ram_used_gb']:.1f}/{status['ram_total_gb']:.1f} GB ({status['ram_percent']:.1f}%)")
    if 'gpu_name' in status:
        print(f"  GPU: {status['gpu_name']}")
        print(f"  VRAM: {status['gpu_used_gb']:.1f}/{status['gpu_total_gb']:.1f} GB")
    else:
        print("  GPU: Not available")


def clear_memory():
    """Clear memory aggressively."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("✅ Memory cleared")


def get_safe_batch_size(imgsz=640, model_size='n'):
    """
    Calculate safe batch size based on available memory.
    
    Args:
        imgsz: Input image size
        model_size: Model size ('n', 's', 'm', 'l', 'x')
    
    Returns:
        Recommended batch size
    """
    status = get_memory_status()
    
    # Base batch sizes for different model sizes at 640px
    base_batch = {
        'n': 16, 's': 12, 'm': 8, 'l': 6, 'x': 4
    }
    
    # Scale by available memory
    if 'gpu_free_gb' in status:
        # GPU available
        free_gb = status['gpu_free_gb']
        if free_gb < 2:
            return 2
        elif free_gb < 4:
            return 4
        elif free_gb < 8:
            return base_batch.get(model_size, 8) // 2
        else:
            return base_batch.get(model_size, 8)
    else:
        # CPU only - use smaller batch
        return 2


def optimize_training_config(config: dict) -> dict:
    """
    Optimize training config for memory efficiency.
    
    Args:
        config: Training configuration dict
    
    Returns:
        Optimized config
    """
    status = get_memory_status()
    
    # If low memory, enable optimizations
    if status['ram_percent'] > 80:
        config['cache'] = False  # Disable caching
        config['workers'] = min(config.get('workers', 8), 2)  # Reduce workers
        print("⚠️ Low RAM: Disabled cache, reduced workers")
    
    if 'gpu_free_gb' in status and status['gpu_free_gb'] < 4:
        config['batch'] = min(config.get('batch', 16), 4)
        config['amp'] = True  # Enable mixed precision
        print("⚠️ Low VRAM: Reduced batch, enabled AMP")
    
    return config


class MemoryTracker:
    """Track memory usage during training."""
    
    def __init__(self):
        self.history = []
    
    def log(self, step: str):
        """Log current memory status."""
        status = get_memory_status()
        status['step'] = step
        self.history.append(status)
        return status
    
    def print_summary(self):
        """Print memory usage summary."""
        if not self.history:
            print("No memory data collected")
            return
        
        print("\n📊 MEMORY USAGE SUMMARY:")
        print("=" * 60)
        for entry in self.history:
            ram = entry['ram_percent']
            gpu = entry.get('gpu_used_gb', 0)
            print(f"  {entry['step']:30s}: RAM={ram:.1f}% GPU={gpu:.1f}GB")


if __name__ == "__main__":
    print_memory_status()
    print(f"\nRecommended batch size: {get_safe_batch_size()}")
