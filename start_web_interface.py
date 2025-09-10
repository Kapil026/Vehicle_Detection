#!/usr/bin/env python
# coding: utf-8

"""
Vehicle Detection System - Web Interface Startup Script
=======================================================

This script starts the backend API server and opens the frontend in a web browser.
Provides a convenient way to launch the complete web interface.
"""

import os
import sys
import time
import webbrowser
import subprocess
import threading
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import flask_cors
        print("✅ Flask dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing Flask dependencies: {e}")
        print("Please install with: uv add -r backend/requirements.txt")
        return False

def start_backend():
    """Start the Flask backend server"""
    print("🚀 Starting Vehicle Detection API...")
    
    # Change to backend directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    os.chdir(backend_dir)
    
    try:
        # Start Flask server
        print("📡 API server starting on http://localhost:5000")
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False
    
    return True

def open_frontend():
    """Open the frontend in a web browser"""
    time.sleep(3)  # Wait for backend to start
    
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        print("🌐 Opening frontend in web browser...")
        webbrowser.open(f"file://{frontend_path.absolute()}")
    else:
        print("❌ Frontend not found at frontend/index.html")

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'uploads',
        'outputs',
        'backend/static',
        'backend/media',
        'frontend/build',
        'logs'
    ]
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"📁 Creating directory: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)
    return True

def main():
    """Main function to start the web interface"""
    print("🚗 Vehicle Detection System - Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
        
    # Ensure directories exist
    ensure_directories()
    
    # Check if backend and frontend exist
    if not Path("backend/app.py").exists():
        print("❌ Backend API not found at backend/app.py")
        return 1
    
    if not Path("frontend/index.html").exists():
        print("❌ Frontend not found at frontend/index.html")
        return 1
    
    print("✅ All components found")
    print("\n📋 Starting web interface...")
    print("1. Backend API server (Flask)")
    print("2. Frontend web interface (HTML/JS)")
    print("3. Opening in web browser")
    print("\nPress Ctrl+C to stop the server")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Open frontend in browser
    open_frontend()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down web interface...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
