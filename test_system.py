#!/usr/bin/env python
# coding: utf-8

"""
System Test Script for Vehicle Detection System
===============================================

This script tests all major components of the vehicle detection system
to ensure everything is working correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('ultralytics', 'Ultralytics'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'TQDM')
    ]
    
    all_good = True
    for package, name in packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {name} ({package})")
        except ImportError as e:
            print(f"   ❌ {name} ({package}): {e}")
            all_good = False
    
    return all_good

def test_model_files():
    """Test if model files exist and are accessible"""
    print("\n🔍 Testing model files...")
    
    model_paths = [
        ('yolov8n.pt', 'Pre-trained YOLO model'),
        ('runs/detect/train6/weights/best.pt', 'Trained model (best)'),
        ('runs/detect/train6/weights/last.pt', 'Trained model (last)')
    ]
    
    all_good = True
    for model_path, description in model_paths:
        if Path(model_path).exists():
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            print(f"   ✅ {description}: {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ⚠️  {description}: {model_path} (not found)")
            if 'yolov8n.pt' in model_path:
                all_good = False
    
    return all_good

def test_dataset_structure():
    """Test if dataset directories exist and contain files"""
    print("\n🔍 Testing dataset structure...")
    
    dataset_dirs = [
        ('Dataset', 'Original dataset'),
        ('train', 'Training data'),
        ('test', 'Test data')
    ]
    
    all_good = True
    for dir_name, description in dataset_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            img_count = len(list(dir_path.glob("**/*.jpg")))
            label_count = len(list(dir_path.glob("**/*.txt")))
            print(f"   ✅ {description}: {dir_name}/ (images: {img_count}, labels: {label_count})")
        else:
            print(f"   ❌ {description}: {dir_name}/ (not found)")
            all_good = False
    
    return all_good

def test_detector_initialization():
    """Test if the vehicle detector can be initialized"""
    print("\n🔍 Testing detector initialization...")
    
    try:
        from simple_enhanced_detection import SimpleVehicleDetector
        
        # Try to initialize with available model
        model_path = 'runs/detect/train6/weights/best.pt' if Path('runs/detect/train6/weights/best.pt').exists() else 'yolov8n.pt'
        
        detector = SimpleVehicleDetector(
            model_path=model_path,
            confidence_threshold=0.5
        )
        
        print(f"   ✅ Detector initialized successfully with model: {model_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to initialize detector: {e}")
        return False

def test_image_processing():
    """Test if image processing works"""
    print("\n🔍 Testing image processing...")
    
    # Check if test images exist
    test_dir = Path("test/images")
    if not test_dir.exists():
        print("   ⚠️  No test images found in 'test/images' directory")
        return False
    
    test_images = list(test_dir.glob("*.jpg"))[:1]  # Test with first image
    if not test_images:
        print("   ⚠️  No JPG images found in test directory")
        return False
    
    try:
        from simple_enhanced_detection import SimpleVehicleDetector
        
        # Initialize detector
        model_path = 'runs/detect/train6/weights/best.pt' if Path('runs/detect/train6/weights/best6/weights/best.pt').exists() else 'yolov8n.pt'
        
        detector = SimpleVehicleDetector(
            model_path=model_path,
            confidence_threshold=0.5
        )
        
        # Process test image
        test_image = test_images[0]
        print(f"   📸 Processing test image: {test_image.name}")
        
        result = detector.process_image(
            image_path=str(test_image),
            output_path=f'test_output_{test_image.name}',
            show_result=False  # Don't show during testing
        )
        
        print(f"   ✅ Image processing successful!")
        print(f"      Vehicles detected: {len(result['detections'])}")
        print(f"      Processing time: {result['processing_time']:.3f} seconds")
        
        # Clean up test output
        if Path(f'test_output_{test_image.name}').exists():
            os.remove(f'test_output_{test_image.name}')
        
        return True
        
    except Exception as e:
        print(f"   ❌ Image processing failed: {e}")
        return False

def run_system_tests():
    """Run all system tests"""
    print("🚗 VEHICLE DETECTION SYSTEM - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Files", test_model_files),
        ("Dataset Structure", test_dataset_structure),
        ("Detector Initialization", test_detector_initialization),
        ("Image Processing", test_image_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

def main():
    """Main function"""
    try:
        success = run_system_tests()
        
        if success:
            print("\n🚀 You can now run the main system with:")
            print("   python main.py")
        else:
            print("\n🔧 To fix issues, try:")
            print("   uv add -r requirements.txt")
            print("   python main.py")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
