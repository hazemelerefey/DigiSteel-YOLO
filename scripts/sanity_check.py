import os
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from digisteel.modules import GhostConv, WFCA, EMA

# Register custom modules explicitly
tasks.GhostConv = GhostConv
tasks.WFCA = WFCA
tasks.EMA = EMA

def main():
    print("=== DigiSteel-YOLO v2 Sanity Check ===")
    
    # Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
    yaml_path = "configs/models/digisteel_v2.yaml"
    print(f"Loading architecture from {yaml_path}...")
    
    try:
        # Load the custom architecture
        model = YOLO(yaml_path)
        print("\n✅ Model built successfully!")
        
        # Print model info
        print(f"Parameters: {sum(p.numel() for p in model.model.parameters())}")
        
        # Test forward pass with dummy data
        print("\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 640, 640)
        # Ultralytics models usually expect inputs on the correct device
        device = next(model.model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            output = model.model(dummy_input)
            
        print("✅ Forward pass successful!")
        
        # Run 1 epoch of real training to test backward pass and loss
        print("\nTesting 1 epoch of training on NEU-DET...")
        
        # We need absolute path for the dataset
        data_yaml = "configs/data/neu_det.yaml"
        
        model.train(
            data=data_yaml,
            epochs=1,
            imgsz=640,
            batch=4,
            device=0 if torch.cuda.is_available() else "cpu",
            project="runs",
            name="sanity_check_v2",
            exist_ok=True,
            workers=0  # Set workers=0 to avoid multiprocessing issues on Windows
        )
        print("\n✅ Training loop (1 epoch) completed successfully!")
        print("SANITY CHECK PASSED. We are ready for full training.")
        
    except Exception as e:
        print(f"\n❌ Sanity Check Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
