// Vehicle Detection System - Frontend JavaScript
// Handles file uploads, API communication, and UI interactions

class VehicleDetectionFrontend {
    constructor() {
        // Use the current hostname for the API URL
        const isProduction = window.location.hostname.includes('render.com');
        this.apiBaseUrl = isProduction 
            ? '' // Empty string means use the same origin
            : 'http://localhost:10000';
        
        this.currentFileType = 'image';
        this.currentFile = null;
        this.currentResult = null;
        
        this.initializeEventListeners();
        this.checkApiHealth();
    }

    initializeEventListeners() {
        // File type selection
        document.getElementById('imageBtn').addEventListener('click', () => this.setFileType('image'));
        document.getElementById('videoBtn').addEventListener('click', () => this.setFileType('video'));
        
        // File upload
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        if (dropZone) {
            dropZone.addEventListener('click', () => fileInput && fileInput.click());
            dropZone.addEventListener('dragover', (e) => this.handleDragOver(e));
            dropZone.addEventListener('drop', (e) => this.handleFileDrop(e));
            dropZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
        
        // Buttons
        const statusBtn = document.getElementById('statusBtn');
        const analyticsBtn = document.getElementById('analyticsBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        
        if (statusBtn) {
            statusBtn.addEventListener('click', () => this.showSystemStatus());
        }
        if (analyticsBtn) {
            analyticsBtn.addEventListener('click', () => this.showAnalytics());
        }
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadResult());
        }
        
        // Modal close buttons
        const closeStatusModal = document.getElementById('closeStatusModal');
        const closeAnalyticsModal = document.getElementById('closeAnalyticsModal');
        
        if (closeStatusModal) {
            closeStatusModal.addEventListener('click', () => this.hideModal('statusModal'));
        }
        if (closeAnalyticsModal) {
            closeAnalyticsModal.addEventListener('click', () => this.hideModal('analyticsModal'));
        }
    }

    async makeApiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}/api${endpoint}`;
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    ...options.headers,
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${error.message}`);
            throw error;
        }
    }

    async checkApiHealth() {
        try {
            const data = await this.makeApiRequest('/health');
            if (data.status === 'healthy') {
                this.showStatus('success', 'Connected to Vehicle Detection API');
            } else {
                this.showStatus('error', 'API health check failed');
            }
        } catch (error) {
            console.error('Health check error:', error);
            this.showStatus('error', 'Cannot connect to API. Please check if the server is running.');
        }
    }

    // ... rest of your existing methods ...

    async uploadFile(file) {
        try {
            this.showUploadProgress();
            
            const formData = new FormData();
            formData.append('file', file);
            
            const endpoint = this.currentFileType === 'image' ? '/detect/image' : '/detect/video';
            const result = await this.makeApiRequest(endpoint, {
                method: 'POST',
                body: formData
            });
            
            this.handleDetectionResult(result);
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showStatus('error', `Upload failed: ${error.message}`);
            this.hideUploadProgress();
        }
    }

    async showSystemStatus() {
        try {
            const data = await this.makeApiRequest('/status');
            if (data.status === 'success') {
                this.displaySystemStatus(data);
            } else {
                this.showStatus('error', data.message || 'Failed to get system status');
            }
        } catch (error) {
            console.error('Status error:', error);
            this.showStatus('error', 'Failed to connect to API');
        }
        
        document.getElementById('statusModal').classList.remove('hidden');
    }

    async showAnalytics() {
        try {
            const data = await this.makeApiRequest('/analytics');
            if (data.status === 'success') {
                this.displayAnalytics(data.analytics);
            } else {
                this.showStatus('error', data.message || 'Failed to get analytics');
            }
        } catch (error) {
            console.error('Analytics error:', error);
            this.showStatus('error', 'Failed to connect to API');
        }
        
        document.getElementById('analyticsModal').classList.remove('hidden');
    }

    async downloadResult() {
        if (!this.currentResult || !this.currentResult.output_file) {
            this.showStatus('error', 'No result available for download');
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/outputs/${this.currentResult.output_file}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = this.currentResult.output_file;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            this.showStatus('success', 'Download started!');
            
        } catch (error) {
            console.error('Download error:', error);
            this.showStatus('error', `Download failed: ${error.message}`);
        }
    }

    // ... rest of your existing code ...
}

// Initialize the frontend when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VehicleDetectionFrontend();
});