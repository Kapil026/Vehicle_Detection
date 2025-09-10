#!/usr/bin/env python
# coding: utf-8

"""
Simplified Enhanced Vehicle Detection System
============================================

This is a simplified version that works with minimal dependencies
and focuses on core functionality: image/video processing and classification.
"""

import os
import time
import json
from pathlib import Path

# Try to import optional dependencies gracefully
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: uv add opencv-python")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Install with: uv add numpy")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics not available. Install with: uv add ultralytics")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Install with: uv add matplotlib")

# Check for PyTorch compatibility
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = "Not installed"

class SimpleVehicleDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the simplified vehicle detector
        
        Args:
            model_path (str): Path to the YOLO model weights
            confidence_threshold (float): Minimum confidence for detections
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics is required. Install with: uv add ultralytics")
        
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required. Install with: uv add opencv-python")
        
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Vehicle classes (update based on your model)
        self.class_names = ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']
        
        # Analytics
        self.analytics = {
            'total_vehicles': 0,
            'vehicle_types': {},
            'confidence_scores': [],
            'processing_times': []
        }
    
    def process_image(self, image_path, output_path=None, show_result=True):
        """
        Process a single image for vehicle detection
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image
            show_result (bool): Whether to display the result
            
        Returns:
            dict: Detection results and analytics
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run detection
        start_time = time.time()
        results = self.model.predict(
            image, 
            conf=self.confidence_threshold,
            verbose=False
        )
        processing_time = time.time() - start_time
        
        # Process results
        detections = self._process_detections(results[0])
        
        # Draw results on image
        annotated_image = self._draw_detections(image, detections)
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Output saved to: {output_path}")
        
        # Display result
        if show_result and MATPLOTLIB_AVAILABLE:
            self._display_image(annotated_image, "Vehicle Detection Result")
        elif show_result:
            cv2.imshow('Vehicle Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Update analytics
        self._update_analytics(detections, processing_time)
        
        return {
            'detections': detections,
            'processing_time': processing_time,
            'image_path': image_path,
            'output_path': output_path
        }
    
    def process_video(self, video_path, output_path=None, show_result=True, max_frames=None):
        """
        Process a video for vehicle detection
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            show_result (bool): Whether to display the result
            max_frames (int): Maximum frames to process (for testing)
            
        Returns:
            dict: Video processing results and analytics
        """
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Limit frames for testing
            if max_frames and frame_count > max_frames:
                break
            
            # Process frame
            frame_start_time = time.time()
            results = self.model.predict(
                frame, 
                conf=self.confidence_threshold,
                verbose=False
            )
            frame_processing_time = time.time() - frame_start_time
            
            # Process detections
            detections = self._process_detections(results[0])
            
            # Draw results
            annotated_frame = self._draw_detections(frame, detections)
            
            # Add frame info overlay
            annotated_frame = self._add_frame_info(annotated_frame, frame_count, total_frames)
            
            # Write to output video
            if output_path:
                out.write(annotated_frame)
            
            # Display result
            if show_result:
                cv2.imshow('Vehicle Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Update analytics
            self._update_analytics(detections, frame_processing_time)
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_current = frame_count / elapsed_time
                print(f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} FPS)")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if show_result:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"Video processing completed:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average FPS: {avg_fps:.2f}")
        
        return {
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'analytics': self.analytics
        }
    
    def _process_detections(self, result):
        """Process YOLO detection results"""
        detections = []
        
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                class_id = int(box.cls.cpu().numpy())
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                }
                detections.append(detection)
        
        return detections
    
    def _draw_detections(self, image, detections):
        """Draw detection boxes and labels on image"""
        annotated_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            
            # Get color based on vehicle type
            color = self._get_vehicle_color(detection['class_name'])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
    
    def _get_vehicle_color(self, vehicle_type):
        """Get color for different vehicle types"""
        colors = {
            'car': (0, 255, 0),      # Green
            'bus': (255, 0, 0),      # Blue
            'truck': (0, 0, 255),    # Red
            'motorcycle': (255, 255, 0),  # Cyan
            'auto': (255, 0, 255),   # Magenta
            'lcv': (0, 255, 255),    # Yellow
            'tractor': (128, 0, 128), # Purple
            'multiaxle': (0, 128, 128) # Teal
        }
        return colors.get(vehicle_type, (128, 128, 128))  # Gray for unknown
    
    def _add_frame_info(self, frame, frame_count, total_frames):
        """Add frame information overlay"""
        overlay = frame.copy()
        
        # Frame info text
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Vehicles: {len(self.analytics.get('vehicle_types', {}))}",
            f"Total: {self.analytics.get('total_vehicles', 0)}"
        ]
        
        # Draw background rectangle
        cv2.rectangle(overlay, (10, 10), (250, 80), (0, 0, 0), -1)
        
        # Draw text
        for i, text in enumerate(info_text):
            cv2.putText(overlay, text, (20, 35 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def _update_analytics(self, detections, processing_time):
        """Update analytics data"""
        self.analytics['processing_times'].append(processing_time)
        
        for detection in detections:
            self.analytics['total_vehicles'] += 1
            vehicle_type = detection['class_name']
            self.analytics['vehicle_types'][vehicle_type] = self.analytics['vehicle_types'].get(vehicle_type, 0) + 1
            self.analytics['confidence_scores'].append(detection['confidence'])
    
    def _display_image(self, image, title="Image"):
        """Display image using matplotlib"""
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()
    
    def generate_report(self, output_path="vehicle_detection_report.json"):
        """Generate detection report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analytics': self.analytics,
            'summary': {
                'total_vehicles_detected': self.analytics['total_vehicles'],
                'vehicle_type_distribution': dict(self.analytics['vehicle_types']),
                'average_confidence': sum(self.analytics['confidence_scores']) / len(self.analytics['confidence_scores']) if self.analytics['confidence_scores'] else 0,
                'average_processing_time': sum(self.analytics['processing_times']) / len(self.analytics['processing_times']) if self.analytics['processing_times'] else 0
            }
        }
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {output_path}")
        return report
    
    def print_summary(self):
        """Print a summary of detections"""
        print("\n" + "="*50)
        print("VEHICLE DETECTION SUMMARY")
        print("="*50)
        print(f"Total vehicles detected: {self.analytics['total_vehicles']}")
        print(f"Vehicle types found: {len(self.analytics['vehicle_types'])}")
        
        if self.analytics['vehicle_types']:
            print("\nVehicle type distribution:")
            for vehicle_type, count in self.analytics['vehicle_types'].items():
                print(f"  {vehicle_type}: {count}")
        
        if self.analytics['confidence_scores']:
            avg_conf = sum(self.analytics['confidence_scores']) / len(self.analytics['confidence_scores'])
            print(f"\nAverage confidence: {avg_conf:.3f}")
        
        if self.analytics['processing_times']:
            avg_time = sum(self.analytics['processing_times']) / len(self.analytics['processing_times'])
            print(f"Average processing time: {avg_time:.3f} seconds")
        
        print("="*50)

def main():
    """Main function to run the simplified vehicle detection system"""
    print("Simplified Enhanced Vehicle Detection System")
    print("=" * 50)
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        print("❌ Ultralytics not available. Install with: uv add ultralytics")
        return
    
    if not OPENCV_AVAILABLE:
        print("❌ OpenCV not available. Install with: uv add opencv-python")
        return
    
    print("✅ Dependencies check passed!")
    
    # Initialize detector
    try:
        detector = SimpleVehicleDetector(
            model_path='yolov8n.pt',  # or use your trained model: 'runs/detect/train6/weights/best.pt'
            confidence_threshold=0.5
        )
        print("✅ Detector initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Example usage
    print("\n🚀 Ready to process images and videos!")
    print("\nExample usage:")
    print("detector = SimpleVehicleDetector()")
    print("result = detector.process_image('input.jpg', 'output.jpg')")
    print("result = detector.process_video('input.mp4', 'output.mp4', max_frames=100)")
    print("detector.generate_report()")
    print("detector.print_summary()")

if __name__ == "__main__":
    main()
