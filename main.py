#!/usr/bin/env python
# coding: utf-8

"""
Main Entry Point for Vehicle Detection System
=============================================

This is the main entry point for the vehicle detection system.
It provides a unified interface to access all functionality.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'cv2', 'numpy', 
        'matplotlib', 'pandas', 'sklearn', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("   uv add -r requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True

def show_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("🚗 VEHICLE DETECTION SYSTEM - MAIN MENU")
    print("="*60)
    print("1. Process Image")
    print("2. Process Video")
    print("3. Run Examples")
    print("4. Check System Status")
    print("5. Exit")
    print("="*60)

def process_image():
    """Process a single image"""
    print("\n📸 IMAGE PROCESSING")
    print("-" * 30)
    
    # Check if test images exist
    test_dir = Path("test/images")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))[:5]  # Get first 5 images
        print(f"Found {len(test_images)} test images")
        
        for i, img_path in enumerate(test_images):
            print(f"{i+1}. {img_path.name}")
        
        choice = input("\nSelect image number (or press Enter for first image): ").strip()
        
        try:
            if choice == "":
                selected_image = test_images[0]
            else:
                selected_image = test_images[int(choice) - 1]
            
            print(f"\nProcessing: {selected_image.name}")
            
            # Import and use the detector
            from simple_enhanced_detection import SimpleVehicleDetector
            
            detector = SimpleVehicleDetector(
                model_path='runs/detect/train6/weights/best.pt' if Path('runs/detect/train6/weights/best.pt').exists() else 'yolov8n.pt'
            )
            
            result = detector.process_image(
                image_path=str(selected_image),
                output_path=f'output_{selected_image.name}',
                show_result=True
            )
            
            print(f"\n✅ Processing completed!")
            print(f"   Vehicles detected: {len(result['detections'])}")
            print(f"   Processing time: {result['processing_time']:.3f} seconds")
            print(f"   Output saved to: output_{selected_image.name}")
            
        except (ValueError, IndexError) as e:
            print(f"❌ Invalid selection: {e}")
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    else:
        print("❌ No test images found in 'test/images' directory")
        print("Please add some images to test with")

def process_video():
    """Process a video file"""
    print("\n🎥 VIDEO PROCESSING")
    print("-" * 30)
    
    # Check for video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path('.').glob(f"*{ext}"))
    
    if video_files:
        print(f"Found {len(video_files)} video files:")
        for i, video in enumerate(video_files):
            print(f"{i+1}. {video.name}")
        
        choice = input("\nSelect video number (or press Enter for first video): ").strip()
        
        try:
            if choice == "":
                selected_video = video_files[0]
            else:
                selected_video = video_files[int(choice) - 1]
            
            print(f"\nProcessing: {selected_video.name}")
            print("Note: Video processing may take a while...")
            
            # Import and use the detector
            from simple_enhanced_detection import SimpleVehicleDetector
            
            detector = SimpleVehicleDetector(
                model_path='runs/detect/train6/weights/best.pt' if Path('runs/detect/train6/weights/best.pt').exists() else 'yolov8n.pt'
            )
            
            result = detector.process_video(
                video_path=str(selected_video),
                output_path=f'output_{selected_video.name}',
                show_result=True,
                max_frames=100  # Limit frames for testing
            )
            
            print(f"\n✅ Video processing completed!")
            print(f"   Frames processed: {result['total_frames']}")
            print(f"   Average FPS: {result['avg_fps']:.2f}")
            print(f"   Total time: {result['total_time']:.2f} seconds")
            print(f"   Output saved to: output_{selected_video.name}")
            
        except (ValueError, IndexError) as e:
            print(f"❌ Invalid selection: {e}")
        except Exception as e:
            print(f"❌ Error processing video: {e}")
    else:
        print("❌ No video files found in current directory")
        print("Please add a video file (.mp4, .avi, .mov, .mkv) to process")

def run_examples():
    """Run example usage scripts"""
    print("\n🚀 RUNNING EXAMPLES")
    print("-" * 30)
    
    try:
        from example_usage import example_image_processing, example_with_analytics, example_batch_processing
        
        print("Running image processing example...")
        example_image_processing()
        
        print("\nRunning analytics example...")
        example_with_analytics()
        
        print("\nRunning batch processing example...")
        example_batch_processing()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

def check_system_status():
    """Check the current system status"""
    print("\n🔍 SYSTEM STATUS")
    print("-" * 30)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check model files
    print("\n📁 Model Files:")
    model_paths = [
        'yolov8n.pt',
        'runs/detect/train6/weights/best.pt',
        'runs/detect/train6/weights/last.pt'
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            print(f"   ✅ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {model_path} (not found)")
    
    # Check dataset directories
    print("\n📊 Dataset Directories:")
    dataset_dirs = ['Dataset', 'train', 'test']
    for dir_name in dataset_dirs:
        if Path(dir_name).exists():
            img_count = len(list(Path(dir_name).glob("**/*.jpg")))
            label_count = len(list(Path(dir_name).glob("**/*.txt")))
            print(f"   ✅ {dir_name}/ (images: {img_count}, labels: {label_count})")
        else:
            print(f"   ❌ {dir_name}/ (not found)")
    
    # Check output directory
    print("\n📤 Output Directory:")
    if Path('runs').exists():
        print("   ✅ runs/ (training outputs exist)")
    else:
        print("   ❌ runs/ (no training outputs)")
    
    print(f"\n💻 Python: {sys.version}")
    print(f"📂 Working Directory: {os.getcwd()}")

def main():
    """Main function"""
    print("🚗 Welcome to the Vehicle Detection System!")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        print("Run: uv add -r requirements.txt")
        return
    
    # Main menu loop
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            process_image()
        elif choice == '2':
            process_video()
        elif choice == '3':
            run_examples()
        elif choice == '4':
            check_system_status()
        elif choice == '5':
            print("\n👋 Goodbye! Thank you for using the Vehicle Detection System.")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
