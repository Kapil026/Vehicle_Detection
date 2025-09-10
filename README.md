# 🚗 Vehicle Detection System

## 📋 Overview

A comprehensive vehicle detection and classification system powered by YOLOv8, PyTorch, and OpenCV. This system can detect vehicles in images and videos, classify them by type, and provide detailed analytics. **Now includes a professional web interface!**

## 🌟 **New: Web Interface Available!**

The system now includes a modern web interface with:
- **🌐 Web API**: Flask backend with RESTful endpoints
- **🎨 Frontend**: Responsive HTML/JavaScript interface
- **📤 Drag & Drop**: Intuitive file uploads
- **📊 Real-time Results**: Live processing and visualization
- **📱 Mobile Ready**: Works on all devices

### **Quick Start Web Interface**
```bash
# Start the complete web interface
python start_web_interface.py
```

## 🏗️ **System Architecture**

### **Core Components**
- **YOLOv8 Models**: Pre-trained and custom-trained detection models
- **PyTorch Backend**: Deep learning inference engine
- **OpenCV Processing**: Computer vision and image manipulation
- **Web Interface**: Flask API + HTML/JavaScript frontend

### **File Structure**
```
Project/
├── main.py                          # Main entry point with interactive menu
├── vehicle_detection.py             # Full-featured detection system
├── simple_enhanced_detection.py     # Simplified version for basic use
├── example_usage.py                 # Example usage demonstrations
├── test_system.py                   # System testing and validation
├── backend/                         # Web API (Flask)
│   ├── app.py                      # API server
│   └── requirements.txt            # Backend dependencies
├── frontend/                       # Web interface
│   ├── index.html                  # Main web interface
│   └── script.js                   # Frontend functionality
├── start_web_interface.py          # Web interface startup script
├── Dataset/                        # Training dataset
├── train/                          # Training data
├── test/                           # Test data
├── runs/detect/                    # Training results and models
└── yolov8n.pt                      # Pre-trained YOLOv8 model
```

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
# Install all required packages
uv add -r requirements.txt

# Install web interface dependencies
uv add -r backend/requirements.txt
```

### **2. Choose Your Interface**

#### **Option A: Command Line Interface (Original)**
```bash
# Interactive menu system
python main.py

# Or run specific components
python simple_enhanced_detection.py
python vehicle_detection.py
python example_usage.py
```

#### **Option B: Web Interface (New!)**
```bash
# Start complete web interface
python start_web_interface.py

# Or start manually
cd backend
uv run python app.py
# Then open frontend/index.html in browser
```

### **3. Test the System**
```bash
# Run comprehensive system test
uv run python test_system.py
```

## 🌐 **Web Interface Features**

### **Backend API (Flask)**
- **Port**: 5000 (http://localhost:5000)
- **Endpoints**: 7 RESTful API endpoints
- **File Handling**: Secure uploads with validation
- **Integration**: Direct access to detection system

### **Frontend Interface**
- **Technology**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Tailwind CSS for responsive design
- **Features**: Drag-and-drop, real-time progress, results display

### **API Endpoints**
- `GET /api/health` - Health check
- `GET /api/status` - System status
- `GET /api/analytics` - Detection analytics
- `POST /api/detect/image` - Image processing
- `POST /api/detect/video` - Video processing
- `GET /api/outputs/<filename>` - Download results
- `POST /api/clear-outputs` - Clear outputs

### **Web Interface Usage**
1. **Start**: Run `python start_web_interface.py`
2. **Upload**: Drag & drop images/videos
3. **Process**: Real-time detection processing
4. **Results**: View detected vehicles with confidence scores
5. **Download**: Get processed files with detections

## 🎯 **Core Features**

### **Vehicle Detection**
- **Multi-class Detection**: Cars, trucks, buses, motorcycles
- **Real-time Processing**: Fast inference with YOLOv8
- **Confidence Scoring**: Reliable detection with configurable thresholds
- **Bounding Boxes**: Precise vehicle localization

### **Image Processing**
- **Multiple Formats**: JPG, PNG, JPEG support
- **Batch Processing**: Handle multiple images
- **Output Generation**: Processed images with detection overlays
- **Quality Preservation**: Maintain original image quality

### **Video Processing**
- **Video Formats**: MP4, AVI, MOV, MKV support
- **Frame Analysis**: Process video frames for vehicle detection
- **Performance Metrics**: FPS tracking and processing statistics
- **Output Videos**: Processed videos with detection annotations

### **Analytics & Reporting**
- **Detection Statistics**: Count and classify detected vehicles
- **Performance Metrics**: Processing time and accuracy
- **Confidence Analysis**: Distribution of detection confidence scores
- **Export Capabilities**: Generate detailed reports

## 🛠️ **Usage Examples**

### **Basic Image Detection**
```python
from simple_enhanced_detection import SimpleVehicleDetector

# Initialize detector
detector = SimpleVehicleDetector(
    model_path='yolov8n.pt',
    confidence_threshold=0.5
)

# Process image
result = detector.process_image(
    image_path='input.jpg',
    output_path='output.jpg',
    show_result=True
)

print(f"Detected {len(result['detections'])} vehicles")
```

### **Video Processing**
```python
from vehicle_detection import VehicleDetector

# Initialize detector
detector = VehicleDetector(
    model_path='runs/detect/train6/weights/best.pt',
    confidence_threshold=0.6
)

# Process video
result = detector.process_video(
    video_path='input.mp4',
    output_path='output.mp4',
    show_result=False
)

print(f"Processed {result['total_frames']} frames at {result['avg_fps']:.1f} FPS")
```

### **Web API Usage**
```bash
# Process image via API
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/detect/image

# Check system status
curl http://localhost:5000/api/status

# Get analytics
curl http://localhost:5000/api/analytics
```

## 🔧 **Configuration**

### **Model Configuration**
```python
# Confidence threshold (0.0 to 1.0)
confidence_threshold = 0.5

# IoU threshold for non-maximum suppression
iou_threshold = 0.45

# Model path (custom or pre-trained)
model_path = 'yolov8n.pt'  # or 'runs/detect/train6/weights/best.pt'
```

### **Web Interface Configuration**
```python
# Backend settings (backend/app.py)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Frontend settings (frontend/script.js)
this.apiBaseUrl = 'http://localhost:5000/api'
```

## 📊 **System Requirements**

### **Software Requirements**
- **Python**: 3.10 or higher
- **Package Manager**: `uv` (recommended) or `pip`
- **Operating System**: Windows, macOS, or Linux

### **Hardware Recommendations**
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and datasets
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)

### **Dependencies**
```bash
# Core ML/DL Libraries
torch==2.0.1
torchvision==0.15.2
ultralytics==8.3.182

# Computer Vision
opencv-python==4.11.0.86

# Data Science & Numerical Computing
numpy==1.26.4
pandas==2.3.1
scikit-learn==1.7.1

# Visualization
matplotlib==3.10.5
seaborn==0.13.2
Pillow==11.3.0

# Web Interface
Flask==2.3.3
Flask-CORS==4.0.0
Werkzeug==2.3.7
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Import Errors**
```bash
# Check if dependencies are installed
uv run python -c "import torch, cv2, ultralytics; print('✅ All imports successful')"

# Reinstall if needed
uv add -r requirements.txt
```

#### **2. Model Loading Issues**
```bash
# Check if model files exist
ls -la yolov8n.pt runs/detect/train6/weights/

# Download pre-trained model if missing
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

#### **3. Web Interface Issues**
```bash
# Check backend status
curl http://localhost:5000/api/health

# Verify dependencies
uv add -r backend/requirements.txt

# Check port availability
lsof -i :5000
```

#### **4. Performance Issues**
```bash
# Test system performance
uv run python test_system.py

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Debug Mode**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable Flask debug mode
app.run(debug=True)
```

## 📈 **Training Custom Models**

### **Dataset Preparation**
```bash
# Organize your dataset
Dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### **Training Command**
```bash
# Train custom model
uv run yolo train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=640

# Resume training
uv run yolo train resume runs/detect/train/weights/last.pt
```

### **Model Evaluation**
```bash
# Validate model performance
uv run yolo val model=runs/detect/train/weights/best.pt data=dataset.yaml

# Test on images
uv run yolo predict model=runs/detect/train/weights/best.pt source=test_images/
```

## 🔄 **Updates & Maintenance**

### **System Updates**
```bash
# Update dependencies
uv add --upgrade ultralytics opencv-python torch torchvision

# Update web interface
uv add --upgrade Flask Flask-CORS Werkzeug
```

### **Model Updates**
```bash
# Update YOLOv8
uv add --upgrade ultralytics

# Download latest pre-trained models
uv run yolo download yolov8n.pt
```

## 📞 **Support & Contributing**

### **Getting Help**
1. Check this documentation
2. Review troubleshooting section
3. Run system tests: `uv run python test_system.py`
4. Check web interface status

### **Contributing**
1. Fork the repository
2. Create feature branch
3. Make changes and test
4. Submit pull request

### **Reporting Issues**
- Include error messages
- Provide system information
- Describe steps to reproduce
- Attach relevant logs

## 🎉 **Getting Started Checklist**

- [ ] Install dependencies: `uv add -r requirements.txt`
- [ ] Install web interface: `uv add -r backend/requirements.txt`
- [ ] Test system: `uv run python test_system.py`
- [ ] Try command line: `python main.py`
- [ ] Try web interface: `python start_web_interface.py`
- [ ] Upload test images/videos
- [ ] Check system status and analytics
- [ ] Download processed results

## 🔮 **Future Enhancements**

### **Planned Features**
- **Real-time Streaming**: WebSocket support for live video
- **Batch Processing**: Multiple file uploads
- **User Authentication**: Login and user management
- **Result History**: Persistent storage of results
- **Advanced Analytics**: Charts and visualizations
- **API Keys**: Secure access control

### **Integration Possibilities**
- **Cloud Storage**: AWS S3, Google Cloud Storage
- **Message Queues**: RabbitMQ, Apache Kafka
- **Monitoring**: Prometheus, Grafana
- **Logging**: ELK Stack, Splunk

---

## 🎯 **Summary**

Your Vehicle Detection System is a **comprehensive solution** that combines:

- ✅ **Advanced ML**: YOLOv8 + PyTorch detection engine
- ✅ **Computer Vision**: OpenCV processing capabilities
- ✅ **Web Interface**: Professional Flask API + HTML/JS frontend
- ✅ **Multiple Interfaces**: Command line + web + API access
- ✅ **Production Ready**: Robust error handling and validation
- ✅ **Extensible**: Easy to customize and enhance

**Choose your interface: Command line for automation, Web interface for user experience! 🚗✨**

---

*For detailed web interface documentation, see the inline sections above. For complete system updates and changes, see `SUMMARY.md`.* 