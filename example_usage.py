#!/usr/bin/env python
# coding: utf-8

"""
Example Usage of Enhanced Vehicle Detection System
==================================================

This script demonstrates how to use the enhanced vehicle detection system
for processing images and videos with various features.
"""

from vehicle_detection import VehicleDetector
import os

def example_image_processing():
    """Example of processing a single image"""
    print("=== Image Processing Example ===")
    
    # Initialize detector
    detector = VehicleDetector(
        model_path='yolov8n.pt',  # or use your trained model: 'runs/detect/train6/weights/best.pt'
        confidence_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Process a test image
    test_image = 'test/images/Highway_0_2020-07-30_jpg.rf.09e9d4467f17b2b870a5d1b94a38774a.jpg'
    
    if os.path.exists(test_image):
        result = detector.process_image(
            image_path=test_image,
            output_path='output_detection.jpg',
            show_result=True
        )
        
        print(f"Detection completed!")
        print(f"Found {len(result['detections'])} vehicles")
        print(f"Processing time: {result['processing_time']:.3f} seconds")
    else:
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path")

def example_video_processing():
    """Example of processing a video"""
    print("\n=== Video Processing Example ===")
    
    # Initialize detector
    detector = VehicleDetector(
        model_path='yolov8n.pt',  # or use your trained model
        confidence_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Process a video (you'll need to provide your own video file)
    video_path = 'input_video.mp4'  # Replace with your video path
    
    if os.path.exists(video_path):
        result = detector.process_video(
            video_path=video_path,
            output_path='output_video.mp4',
            show_result=True,
            save_frames=True  # Save individual frames for analysis
        )
        
        print(f"Video processing completed!")
        print(f"Total frames processed: {result['total_frames']}")
        print(f"Average FPS: {result['avg_fps']:.2f}")
        print(f"Total processing time: {result['total_time']:.2f} seconds")
        print(f"Vehicles detected: {result['vehicle_counts']}")
    else:
        print(f"Video file not found: {video_path}")
        print("Please provide a valid video path")

def example_with_analytics():
    """Example with comprehensive analytics and reporting"""
    print("\n=== Analytics and Reporting Example ===")
    
    # Initialize detector
    detector = VehicleDetector(
        model_path='yolov8n.pt',
        confidence_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Process multiple images for analytics
    test_dir = 'test/images'
    if os.path.exists(test_dir):
        image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:5]  # Process first 5 images
        
        for image_file in image_files:
            image_path = os.path.join(test_dir, image_file)
            result = detector.process_image(
                image_path=image_path,
                output_path=f'output_{image_file}',
                show_result=False
            )
            print(f"Processed: {image_file}")
        
        # Generate comprehensive report
        report = detector.generate_report('detailed_report.json')
        
        # Create analytics plots
        detector.plot_analytics('analytics_plots.png')
        
        print("\nAnalytics Summary:")
        print(f"Total vehicles detected: {report['summary']['total_vehicles_detected']}")
        print(f"Vehicle type distribution: {report['summary']['vehicle_type_distribution']}")
        print(f"Average confidence: {report['summary']['average_confidence']:.3f}")
    else:
        print(f"Test directory not found: {test_dir}")

def example_real_time_processing():
    """Example of real-time processing from webcam"""
    print("\n=== Real-time Processing Example ===")
    
    # Initialize detector
    detector = VehicleDetector(
        model_path='yolov8n.pt',
        confidence_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Press 'q' to quit real-time processing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = detector.model.predict(
            frame, 
            conf=detector.confidence_threshold, 
            iou=detector.iou_threshold,
            verbose=False
        )
        
        # Process detections
        detections = detector._process_detections(results[0], frame)
        
        # Draw results
        annotated_frame = detector._draw_detections(frame, detections)
        
        # Add analytics overlay
        annotated_frame = detector._add_analytics_overlay(annotated_frame, 0, 0)
        
        # Display
        cv2.imshow('Real-time Vehicle Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def example_batch_processing():
    """Example of batch processing multiple files"""
    print("\n=== Batch Processing Example ===")
    
    # Initialize detector
    detector = VehicleDetector(
        model_path='yolov8n.pt',
        confidence_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Process all images in a directory
    input_dir = 'test/images'
    output_dir = 'batch_output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(input_dir):
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f'processed_{image_file}')
            
            try:
                result = detector.process_image(
                    image_path=input_path,
                    output_path=output_path,
                    show_result=False
                )
                print(f"Processed {i+1}/{len(image_files)}: {image_file} - {len(result['detections'])} vehicles")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # Generate batch report
        report = detector.generate_report('batch_report.json')
        print(f"\nBatch processing completed!")
        print(f"Total vehicles detected: {report['summary']['total_vehicles_detected']}")
    else:
        print(f"Input directory not found: {input_dir}")

if __name__ == "__main__":
    import cv2
    
    print("Enhanced Vehicle Detection System - Example Usage")
    print("=" * 50)
    
    # Run examples
    try:
        # Example 1: Process a single image
        example_image_processing()
        
        # Example 2: Process a video (uncomment if you have a video file)
        # example_video_processing()
        
        # Example 3: Analytics and reporting
        example_with_analytics()
        
        # Example 4: Real-time processing (uncomment to use webcam)
        # example_real_time_processing()
        
        # Example 5: Batch processing
        example_batch_processing()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("uv add -r requirements.txt")
    
    print("\nExamples completed! Check the output files for results.") 