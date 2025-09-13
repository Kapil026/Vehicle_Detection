// Vehicle Detection System - Frontend JavaScript
// Handles file uploads, API communication, and UI interactions

class VehicleDetectionFrontend {
    constructor() {
        // Use the current hostname for the API URL in production, localhost in development
        const isProduction = window.location.hostname !== 'localhost';
        this.apiBaseUrl = isProduction 
            ? `${window.location.protocol}//${window.location.host}/api`
            : 'http://localhost:10000/api';
        
        this.currentFileType = 'image';
        this.currentFile = null;
        this.currentResult = null;
        
        this.initializeEventListeners();
        this.checkApiHealth();
    }

    // ... rest of the file remains the same ...
}