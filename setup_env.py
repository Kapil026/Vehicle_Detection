#!/usr/bin/env python
# coding: utf-8

"""
Setup environment variables for Vehicle Detection System
====================================================

This script sets up the required environment variables for the application.
"""

import os

# Set environment variables
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_APP'] = 'backend/app.py'
os.environ['FLASK_DEBUG'] = '1'
os.environ['SECRET_KEY'] = 'Z9wQNoRED0uNRP5k1ZXMFWYwOA33Hc6VNfnqas6VHx0='
os.environ['MAX_CONTENT_LENGTH'] = '104857600'
os.environ['UPLOAD_FOLDER'] = 'uploads'
os.environ['OUTPUT_FOLDER'] = 'outputs'
os.environ['CONFIDENCE_THRESHOLD'] = '0.5'
os.environ['MODEL_PATH'] = 'yolov8n.pt'

print("✅ Environment variables set successfully!")

# Import and run the web interface
import start_web_interface

if __name__ == "__main__":
    start_web_interface.main()
