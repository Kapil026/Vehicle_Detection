#!/usr/bin/env python
# coding: utf-8

"""
Download YOLOv8 model for deployment
==================================

This script downloads the YOLOv8n model if it doesn't exist.
Used during deployment to ensure the model is available.
"""

import os
from pathlib import Path
from ultralytics import YOLO

def download_model(model_name='yolov8n.pt'):
    """Download YOLOv8 model if it doesn't exist"""
    model_path = Path(model_name)
    
    if not model_path.exists():
        print(f"Downloading {model_name}...")
        try:
            # This will download the model
            model = YOLO(model_name)
            print(f"✅ Successfully downloaded {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    else:
        print(f"✅ Model {model_name} already exists")
        return True

if __name__ == "__main__":
    success = download_model()
    if not success:
        exit(1)  # Exit with error if download fails
