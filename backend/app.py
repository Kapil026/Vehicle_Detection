#!/usr/bin/env python
# coding: utf-8

"""
Vehicle Detection System - Backend API
=======================================

Flask-based REST API for vehicle detection and classification.
Provides endpoints for image/video processing, status checking, and analytics.
"""

import os
import json
import time
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import io

# Import our vehicle detection system
from vehicle_detection import VehicleDetector
from simple_enhanced_detection import SimpleVehicleDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global detector instance
detector = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_detector():
    """Get or create detector instance"""
    global detector
    if detector is None:
        try:
            # Try to use trained model if available
            model_path = 'runs/detect/train6/weights/best.pt' if Path('runs/detect/train6/weights/best.pt').exists() else 'yolov8n.pt'
            detector = SimpleVehicleDetector(
                model_path=model_path,
                confidence_threshold=0.5
            )
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return None
    return detector

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Vehicle Detection API',
        'version': '1.0.0'
    })

@app.route('/api/status', methods=['GET'])
def system_status():
    """Get system status and configuration"""
    try:
        detector = get_detector()
        if detector is None:
            return jsonify({
                'status': 'error',
                'message': 'Detector not initialized',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Check model files
        model_paths = [
            'yolov8n.pt',
            'runs/detect/train6/weights/best.pt',
            'runs/detect/train6/weights/last.pt'
        ]
        
        models = {}
        for model_path in model_paths:
            if Path(model_path).exists():
                size_mb = Path(model_path).stat().st_size / (1024 * 1024)
                models[model_path] = {
                    'exists': True,
                    'size_mb': round(size_mb, 1)
                }
            else:
                models[model_path] = {
                    'exists': False,
                    'size_mb': 0
                }
        
        # Check dataset directories
        dataset_dirs = ['Dataset', 'train', 'test']
        datasets = {}
        for dir_name in dataset_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                img_count = len(list(dir_path.glob("**/*.jpg")))
                label_count = len(list(dir_path.glob("**/*.txt")))
                datasets[dir_name] = {
                    'exists': True,
                    'images': img_count,
                    'labels': label_count
                }
            else:
                datasets[dir_name] = {
                    'exists': False,
                    'images': 0,
                    'labels': 0
                }
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'detector': {
                'initialized': True,
                'model_path': detector.model_path if hasattr(detector, 'model_path') else 'Unknown',
                'confidence_threshold': detector.confidence_threshold
            },
            'models': models,
            'datasets': datasets,
            'system': {
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'working_directory': os.getcwd()
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/detect/image', methods=['POST'])
def detect_vehicles_image():
    """Process image for vehicle detection"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'File type not allowed'
            }), 400
        
        # Get detector
        detector = get_detector()
        if detector is None:
            return jsonify({
                'status': 'error',
                'message': 'Detector not initialized'
            }), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(upload_path)
        
        # Process image
        output_filename = f"output_{timestamp}_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        start_time = time.time()
        result = detector.process_image(
            image_path=upload_path,
            output_path=output_path,
            show_result=False
        )
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'processing_time': round(processing_time, 3),
            'detections': len(result['detections']),
            'vehicles_detected': [],
            'output_file': output_filename
        }
        
        # Add vehicle details
        for detection in result['detections']:
            response_data['vehicles_detected'].append({
                'class': detection['class_name'],
                'confidence': round(detection['confidence'], 3),
                'bbox': detection['bbox']
            })
        
        # Clean up uploaded file
        os.remove(upload_path)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_vehicles_video():
    """Process video for vehicle detection"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'File type not allowed'
            }), 400
        
        # Get detector
        detector = get_detector()
        if detector is None:
            return jsonify({
                'status': 'error',
                'message': 'Detector not initialized'
            }), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(upload_path)
        
        # Process video (limit frames for API)
        output_filename = f"output_{timestamp}_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        start_time = time.time()
        result = detector.process_video(
            video_path=upload_path,
            output_path=output_path,
            show_result=False,
            max_frames=100  # Limit frames for API
        )
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'processing_time': round(processing_time, 3),
            'total_frames': result['total_frames'],
            'avg_fps': round(result['avg_fps'], 2),
            'total_time': round(result['total_time'], 3),
            'output_file': output_filename
        }
        
        # Clean up uploaded file
        os.remove(upload_path)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/outputs/<filename>', methods=['GET'])
def get_output_file(filename):
    """Download processed output file"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get system analytics and statistics"""
    try:
        detector = get_detector()
        if detector is None:
            return jsonify({
                'status': 'error',
                'message': 'Detector not initialized'
            }), 500
        
        # Get analytics from detector
        analytics = detector.analytics if hasattr(detector, 'analytics') else {}
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'analytics': {
                'total_vehicles': analytics.get('total_vehicles', 0),
                'vehicle_types': dict(analytics.get('vehicle_types', {})),
                'confidence_scores': analytics.get('confidence_scores', []),
                'processing_times': analytics.get('processing_times', [])
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/clear-outputs', methods=['POST'])
def clear_outputs():
    """Clear all output files"""
    try:
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        files_removed = 0
        
        for file_path in output_dir.glob('*'):
            if file_path.is_file():
                file_path.unlink()
                files_removed += 1
        
        return jsonify({
            'status': 'success',
            'message': f'Cleared {files_removed} output files',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Serve frontend files
@app.route('/')
def serve_frontend():
    """Serve the frontend HTML"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory"""
    return send_from_directory('../frontend', path)

if __name__ == '__main__':
    print("🚗 Starting Vehicle Detection API...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"🌐 API will be available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
