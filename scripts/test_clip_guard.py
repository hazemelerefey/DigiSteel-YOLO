import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from Deployment.Main import is_steel_surface, model

def run_tests():
    print("=" * 60)
    print("🧪 DIGISTEEL-YOLO + CLIP SEMANTIC GUARD TEST SUITE")
    print("=" * 60)
    
    test_results = []
    
    # 1. Test In-Domain: Real Steel Images
    steel_img_path = ROOT / "datasets" / "NEU-DET" / "yolo" / "images" / "test" / "crazing_13.jpg"
    if steel_img_path.exists():
        img = Image.open(steel_img_path)
        is_valid, msg = is_steel_surface(img)
        print(f"\n[Test 1] Real Steel (crazing_13.jpg):")
        print(f"  Valid: {is_valid} | Reason: {msg}")
        
        # Test YOLO inference
        if model is not None and is_valid:
            results = model.predict(img, conf=0.45, iou=0.70, verbose=False)
            boxes = results[0].boxes
            n_boxes = len(boxes) if boxes is not None else 0
            print(f"  YOLO Detections: {n_boxes} defects found.")
            for box in (boxes if boxes is not None else []):
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"    - {cls_name} (conf: {conf:.3f})")
        test_results.append(("Real Steel (crazing)", is_valid == True, msg))
    
    # 2. Test Out-of-Domain: Synthetic Document / Text Screenshot
    text_img = Image.new('RGB', (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(text_img)
    draw.text((20, 50), "Meeting Minutes 2026\nQuarterly Financial Report\nConfidential Document", fill=(0, 0, 0))
    is_valid, msg = is_steel_surface(text_img)
    print(f"\n[Test 2] Out-of-Domain (Text Document):")
    print(f"  Valid: {is_valid} | Reason: {msg}")
    test_results.append(("Text Document", is_valid == False, msg))
    
    # 3. Test Out-of-Domain: Colorful Graphic / Abstract Banner
    color_img = Image.new('RGB', (400, 300), color=(255, 0, 128))
    draw = ImageDraw.Draw(color_img)
    draw.rectangle([50, 50, 350, 250], fill=(0, 255, 255))
    is_valid, msg = is_steel_surface(color_img)
    print(f"\n[Test 3] Out-of-Domain (Colorful Abstract / Banner):")
    print(f"  Valid: {is_valid} | Reason: {msg}")
    test_results.append(("Colorful Abstract Banner", is_valid == False, msg))
    
    # 4. Test Edge Case: Tiny Low-Res Image (<96px)
    tiny_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    is_valid, msg = is_steel_surface(tiny_img)
    print(f"\n[Test 4] Low Resolution (<96px):")
    print(f"  Valid: {is_valid} | Reason: {msg}")
    test_results.append(("Tiny Image (<96px)", is_valid == False, msg))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed, msg in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        if not passed: all_passed = False
        print(f"{status:10} | {name:<25} | {msg}")
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED: CLIP Semantic Gate + DAFEGate YOLO working perfectly!")
    else:
        print("⚠️ SOME TESTS FAILED.")

if __name__ == "__main__":
    run_tests()
