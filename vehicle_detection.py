#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Vehicle Detection and Classification System
===================================================

This script provides comprehensive vehicle detection and classification capabilities:
- Image and video processing
- Real-time vehicle tracking
- Vehicle counting and analytics
- Classification with confidence scores
- Output visualization and reporting
- Multiple output formats (video, JSON, CSV)

Author: Enhanced Vehicle Detection System
Date: 2024
"""

import os
import cv2
import numpy as np
import json
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some analytics features will be limited.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Some plotting features will be limited.")

# Check for PyTorch compatibility
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = "Not installed"

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the vehicle detector with YOLO model
        
        Args:
            model_path (str): Path to the YOLO model weights
            confidence_threshold (float): Minimum confidence for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Vehicle classes (update based on your model)
        self.class_names = ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']
        
        # Tracking variables
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.vehicle_count = defaultdict(int)
        self.frame_count = 0
        self.detection_history = []
        
        # Analytics
        self.analytics = {
            'total_vehicles': 0,
            'vehicle_types': defaultdict(int),
            'confidence_scores': [],
            'processing_times': [],
            'frame_rates': []
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
            iou=self.iou_threshold,
            verbose=False
        )
        processing_time = time.time() - start_time
        
        # Process results
        detections = self._process_detections(results[0], image)
        
        # Draw results on image
        annotated_image = self._draw_detections(image, detections)
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Output saved to: {output_path}")
        
        # Display result
        if show_result:
            self._display_image(annotated_image, "Vehicle Detection Result")
        
        # Update analytics
        self._update_analytics(detections, processing_time)
        
        return {
            'detections': detections,
            'processing_time': processing_time,
            'image_path': image_path,
            'output_path': output_path
        }
    
    def process_video(self, video_path, output_path=None, show_result=True, save_frames=False):
        """
        Process a video for vehicle detection and tracking
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            show_result (bool): Whether to display the result
            save_frames (bool): Whether to save individual frames
            
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
        
        # Setup frame saving
        if save_frames:
            frames_dir = Path(output_path).parent / "frames" if output_path else Path("output_frames")
            frames_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.frame_count += 1
            
            # Process frame
            frame_start_time = time.time()
            results = self.model.track(
                frame, 
                conf=self.confidence_threshold, 
                iou=self.iou_threshold,
                persist=True,
                verbose=False
            )
            frame_processing_time = time.time() - frame_start_time
            
            # Process detections
            detections = self._process_detections_with_tracking(results[0], frame)
            
            # Draw results
            annotated_frame = self._draw_detections_with_tracking(frame, detections, results[0])
            
            # Add analytics overlay
            annotated_frame = self._add_analytics_overlay(annotated_frame, frame_count, total_frames)
            
            # Save frame
            if save_frames and frame_count % 30 == 0:  # Save every 30th frame
                frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated_frame)
            
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
            if frame_count % 100 == 0:
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
            'vehicle_counts': dict(self.vehicle_count),
            'analytics': self.analytics
        }
    
    def _process_detections(self, result, image):
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
    
    def _process_detections_with_tracking(self, result, frame):
        """Process YOLO detection results with tracking"""
        detections = []
        
        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                class_id = int(box.cls.cpu().numpy())
                track_id = int(track_id.cpu().numpy())
                
                # Update track history
                center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                self.track_history[track_id].append(center)
                
                detection = {
                    'track_id': track_id,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                    'center': center
                }
                detections.append(detection)
                
                # Count vehicles (simple approach - can be enhanced)
                if track_id not in self.vehicle_count:
                    self.vehicle_count[track_id] = detection['class_name']
        
        return detections
    
    def _draw_detections(self, image, detections):
        """Draw detection boxes and labels on image"""
        annotated_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_image
    
    def _draw_detections_with_tracking(self, image, detections, result):
        """Draw detection boxes with tracking trails"""
        annotated_image = image.copy()
        
        # Draw tracking trails
        for track_id, trail in self.track_history.items():
            if len(trail) > 1:
                points = np.array(trail, dtype=np.int32)
                cv2.polylines(annotated_image, [points], False, (255, 0, 0), 2)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection.get('track_id', 'N/A')
            label = f"{detection['class_name']} (ID: {track_id}): {detection['confidence']:.2f}"
            
            # Color based on vehicle type
            color = self._get_vehicle_color(detection['class_name'])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
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
    
    def _add_analytics_overlay(self, frame, frame_count, total_frames):
        """Add analytics information overlay to frame"""
        # Create overlay
        overlay = frame.copy()
        
        # Analytics text
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Vehicles Detected: {len(self.vehicle_count)}",
            f"Processing Time: {self.analytics['processing_times'][-1]:.3f}s" if self.analytics['processing_times'] else "Processing Time: N/A"
        ]
        
        # Draw background rectangle
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        
        # Draw text
        for i, text in enumerate(info_text):
            cv2.putText(overlay, text, (20, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def _update_analytics(self, detections, processing_time):
        """Update analytics data"""
        self.analytics['processing_times'].append(processing_time)
        
        for detection in detections:
            self.analytics['total_vehicles'] += 1
            self.analytics['vehicle_types'][detection['class_name']] += 1
            self.analytics['confidence_scores'].append(detection['confidence'])
    
    def _display_image(self, image, title="Image"):
        """Display image using matplotlib"""
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def generate_report(self, output_path="vehicle_detection_report.json"):
        """Generate comprehensive detection report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'analytics': self.analytics,
            'vehicle_counts': dict(self.vehicle_count),
            'summary': {
                'total_vehicles_detected': self.analytics['total_vehicles'],
                'unique_vehicles_tracked': len(self.vehicle_count),
                'vehicle_type_distribution': dict(self.analytics['vehicle_types']),
                'average_confidence': np.mean(self.analytics['confidence_scores']) if self.analytics['confidence_scores'] else 0,
                'average_processing_time': np.mean(self.analytics['processing_times']) if self.analytics['processing_times'] else 0
            }
        }
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV report
        csv_path = output_path.replace('.json', '.csv')
        self._save_csv_report(csv_path)
        
        print(f"Report saved to: {output_path}")
        print(f"CSV report saved to: {csv_path}")
        
        return report
    
    def _save_csv_report(self, csv_path):
        """Save detection data to CSV"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Vehicle_Type', 'Count', 'Percentage'])
            
            total = sum(self.analytics['vehicle_types'].values())
            for vehicle_type, count in self.analytics['vehicle_types'].items():
                percentage = (count / total * 100) if total > 0 else 0
                writer.writerow([vehicle_type, count, f"{percentage:.2f}%"])
    
    def plot_analytics(self, save_path=None):
        """Create visualization plots for analytics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Vehicle type distribution
        vehicle_types = list(self.analytics['vehicle_types'].keys())
        vehicle_counts = list(self.analytics['vehicle_types'].values())
        
        axes[0, 0].bar(vehicle_types, vehicle_counts, color='skyblue')
        axes[0, 0].set_title('Vehicle Type Distribution')
        axes[0, 0].set_xlabel('Vehicle Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence score distribution
        if self.analytics['confidence_scores']:
            axes[0, 1].hist(self.analytics['confidence_scores'], bins=20, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Confidence Score Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # Processing time over frames
        if self.analytics['processing_times']:
            axes[1, 0].plot(self.analytics['processing_times'], color='orange')
            axes[1, 0].set_title('Processing Time Over Frames')
            axes[1, 0].set_xlabel('Frame Number')
            axes[1, 0].set_ylabel('Processing Time (seconds)')
        
        # Vehicle type pie chart
        if vehicle_counts:
            axes[1, 1].pie(vehicle_counts, labels=vehicle_types, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Vehicle Type Distribution (Pie Chart)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analytics plot saved to: {save_path}")
        
        plt.show()

def main():
    """Main function to run the vehicle detection system"""
    parser = argparse.ArgumentParser(description='Enhanced Vehicle Detection and Classification System')
    parser.add_argument('--input', '-i', required=True, help='Input image or video path')
    parser.add_argument('--output', '-o', help='Output path for processed file')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save-frames', action='store_true', help='Save individual frames (video only)')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VehicleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    # Determine if input is image or video
    input_path = args.input
    file_extension = Path(input_path).suffix.lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # Process image
        result = detector.process_image(
            image_path=input_path,
            output_path=args.output,
            show_result=args.show
        )
        print("Image processing completed!")
        
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        # Process video
        result = detector.process_video(
            video_path=input_path,
            output_path=args.output,
            show_result=args.show,
            save_frames=args.save_frames
        )
        print("Video processing completed!")
        
    else:
        print(f"Unsupported file format: {file_extension}")
        return
    
    # Generate report if requested
    if args.report:
        report = detector.generate_report()
        detector.plot_analytics()
        print("\nDetection Summary:")
        print(f"Total vehicles detected: {report['summary']['total_vehicles_detected']}")
        print(f"Unique vehicles tracked: {report['summary']['unique_vehicles_tracked']}")
        print(f"Average confidence: {report['summary']['average_confidence']:.3f}")
        print(f"Average processing time: {report['summary']['average_processing_time']:.3f} seconds")

if __name__ == "__main__":
    # Example usage without command line arguments
    print("Enhanced Vehicle Detection and Classification System")
    print("=" * 50)
    
    # Initialize detector
    detector = VehicleDetector()
    
    # Example: Process a test image
    # detector.process_image('test/images/sample.jpg', 'output_detection.jpg')
    
    # Example: Process a video
    # detector.process_video('input_video.mp4', 'output_video.mp4')
    
    # Example: Generate report
    # detector.generate_report()
    
    print("System ready! Use command line arguments or modify the script for your specific use case.")
    print("\nExample usage:")
    print("python vehicle_detection.py --input video.mp4 --output result.mp4 --show --report")
print("python vehicle_detection.py --input image.jpg --output result.jpg --show") 