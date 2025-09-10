// Vehicle Detection System - Frontend JavaScript
// Handles file uploads, API communication, and UI interactions

class VehicleDetectionFrontend {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000/api';
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
        document.getElementById('dropZone').addEventListener('click', () => document.getElementById('fileInput').click());
        document.getElementById('fileInput').addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop
        document.getElementById('dropZone').addEventListener('dragover', (e) => this.handleDragOver(e));
        document.getElementById('dropZone').addEventListener('drop', (e) => this.handleFileDrop(e));
        document.getElementById('dropZone').addEventListener('dragleave', (e) => this.handleDragLeave(e));
        
        // Buttons
        document.getElementById('statusBtn').addEventListener('click', () => this.showSystemStatus());
        document.getElementById('analyticsBtn').addEventListener('click', () => this.showAnalytics());
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadResult());
        
        // Modal close buttons
        document.getElementById('closeStatusModal').addEventListener('click', () => this.hideModal('statusModal'));
        document.getElementById('closeAnalyticsModal').addEventListener('click', () => this.hideModal('analyticsModal'));
        
        // Close modals when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.hideModal(e.target.id);
            }
        });
    }

    setFileType(type) {
        this.currentFileType = type;
        
        // Update button styles
        document.querySelectorAll('.file-type-btn').forEach(btn => {
            btn.classList.remove('bg-blue-600', 'text-white');
            btn.classList.add('bg-gray-200', 'text-gray-700');
        });
        
        const activeBtn = type === 'image' ? document.getElementById('imageBtn') : document.getElementById('videoBtn');
        activeBtn.classList.remove('bg-gray-200', 'text-gray-700');
        activeBtn.classList.add('bg-blue-600', 'text-white');
        
        // Update file input and text
        const fileInput = document.getElementById('fileInput');
        const fileTypesText = document.getElementById('fileTypesText');
        
        if (type === 'image') {
            fileInput.accept = 'image/*';
            fileTypesText.textContent = 'Supported formats: JPG, PNG, JPEG';
        } else {
            fileInput.accept = 'video/*';
            fileTypesText.textContent = 'Supported formats: MP4, AVI, MOV, MKV';
        }
        
        // Reset UI
        this.resetUI();
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        document.getElementById('dropZone').classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        document.getElementById('dropZone').classList.remove('dragover');
    }

    handleFileDrop(event) {
        event.preventDefault();
        document.getElementById('dropZone').classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        // Validate file type
        const allowedExtensions = this.currentFileType === 'image' 
            ? ['jpg', 'jpeg', 'png'] 
            : ['mp4', 'avi', 'mov', 'mkv'];
        
        const fileExtension = file.name.split('.').pop().toLowerCase();
        if (!allowedExtensions.includes(fileExtension)) {
            this.showStatus('error', `Invalid file type. Please upload a ${this.currentFileType} file.`);
            return;
        }
        
        this.currentFile = file;
        this.uploadFile(file);
    }

    async uploadFile(file) {
        try {
            this.showUploadProgress();
            
            const formData = new FormData();
            formData.append('file', file);
            
            const endpoint = this.currentFileType === 'image' ? '/detect/image' : '/detect/video';
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            this.handleDetectionResult(result);
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showStatus('error', `Upload failed: ${error.message}`);
            this.hideUploadProgress();
        }
    }

    showUploadProgress() {
        document.getElementById('uploadProgress').classList.remove('hidden');
        this.simulateProgress();
    }

    hideUploadProgress() {
        document.getElementById('uploadProgress').classList.add('hidden');
    }

    simulateProgress() {
        let progress = 0;
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 90) {
                progress = 90;
                clearInterval(interval);
            }
            
            progressBar.style.width = `${progress}%`;
            progressPercent.textContent = `${Math.round(progress)}%`;
        }, 200);
    }

    handleDetectionResult(result) {
        this.currentResult = result;
        this.hideUploadProgress();
        
        if (result.status === 'success') {
            this.showStatus('success', `Detection completed! Found ${result.detections || result.total_frames} items.`);
            this.displayResults(result);
        } else {
            this.showStatus('error', result.message || 'Detection failed');
        }
    }

    displayResults(result) {
        // Update summary cards
        if (result.detections !== undefined) {
            document.getElementById('vehicleCount').textContent = result.detections;
        } else if (result.total_frames !== undefined) {
            document.getElementById('vehicleCount').textContent = result.total_frames;
        }
        
        document.getElementById('processingTime').textContent = `${result.processing_time}s`;
        document.getElementById('outputFile').textContent = result.output_file;
        
        // Display vehicle details
        this.displayVehicleDetails(result);
        
        // Show results section
        document.getElementById('resultsSection').classList.remove('hidden');
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    displayVehicleDetails(result) {
        const vehicleDetails = document.getElementById('vehicleDetails');
        vehicleDetails.innerHTML = '';
        
        if (result.vehicles_detected && result.vehicles_detected.length > 0) {
            result.vehicles_detected.forEach((vehicle, index) => {
                const vehicleCard = this.createVehicleCard(vehicle, index);
                vehicleDetails.appendChild(vehicleCard);
            });
        } else {
            vehicleDetails.innerHTML = '<p class="text-gray-500 text-center py-4">No vehicle details available</p>';
        }
    }

    createVehicleCard(vehicle, index) {
        const card = document.createElement('div');
        card.className = 'vehicle-card bg-gray-50 p-4 rounded-lg border';
        
        const confidenceColor = vehicle.confidence > 0.8 ? 'text-green-600' : 
                               vehicle.confidence > 0.6 ? 'text-yellow-600' : 'text-red-600';
        
        card.innerHTML = `
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                        <i class="fas fa-car text-blue-600"></i>
                    </div>
                    <div>
                        <h4 class="font-medium text-gray-900">${vehicle.class}</h4>
                        <p class="text-sm text-gray-500">Vehicle #${index + 1}</p>
                    </div>
                </div>
                <div class="text-right">
                    <p class="font-medium ${confidenceColor}">${(vehicle.confidence * 100).toFixed(1)}%</p>
                    <p class="text-xs text-gray-500">Confidence</p>
                </div>
            </div>
        `;
        
        return card;
    }

    async showSystemStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/status`);
            const data = await response.json();
            
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

    displaySystemStatus(data) {
        const statusContent = document.getElementById('statusContent');
        
        let html = `
            <div class="space-y-4">
                <div class="bg-green-50 p-4 rounded-lg">
                    <h4 class="font-medium text-green-900">Detector Status</h4>
                    <p class="text-sm text-green-700">Model: ${data.detector.model_path}</p>
                    <p class="text-sm text-green-700">Confidence Threshold: ${data.detector.confidence_threshold}</p>
                </div>
                
                <div class="bg-blue-50 p-4 rounded-lg">
                    <h4 class="font-medium text-blue-900">Models</h4>
                    ${Object.entries(data.models).map(([path, info]) => `
                        <div class="flex justify-between text-sm">
                            <span class="text-blue-700">${path}</span>
                            <span class="text-blue-700">${info.exists ? `${info.size_mb} MB` : 'Not found'}</span>
                        </div>
                    `).join('')}
                </div>
                
                <div class="bg-purple-50 p-4 rounded-lg">
                    <h4 class="font-medium text-purple-900">Datasets</h4>
                    ${Object.entries(data.datasets).map(([name, info]) => `
                        <div class="flex justify-between text-sm">
                            <span class="text-purple-700">${name}/</span>
                            <span class="text-purple-700">${info.exists ? `${info.images} images, ${info.labels} labels` : 'Not found'}</span>
                        </div>
                    `).join('')}
                </div>
                
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h4 class="font-medium text-gray-900">System Info</h4>
                    <p class="text-sm text-gray-700">Python: ${data.system.python_version}</p>
                    <p class="text-sm text-gray-700">Working Directory: ${data.system.working_directory}</p>
                </div>
            </div>
        `;
        
        statusContent.innerHTML = html;
    }

    async showAnalytics() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/analytics`);
            const data = await response.json();
            
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

    displayAnalytics(analytics) {
        const analyticsContent = document.getElementById('analyticsContent');
        
        let html = `
            <div class="space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-blue-50 p-4 rounded-lg text-center">
                        <h4 class="font-medium text-blue-900">Total Vehicles</h4>
                        <p class="text-2xl font-bold text-blue-600">${analytics.total_vehicles}</p>
                    </div>
                    <div class="bg-green-50 p-4 rounded-lg text-center">
                        <h4 class="font-medium text-green-900">Vehicle Types</h4>
                        <p class="text-2xl font-bold text-green-600">${Object.keys(analytics.vehicle_types).length}</p>
                    </div>
                </div>
                
                <div class="bg-yellow-50 p-4 rounded-lg">
                    <h4 class="font-medium text-yellow-900">Vehicle Type Distribution</h4>
                    <div class="space-y-2 mt-2">
                        ${Object.entries(analytics.vehicle_types).map(([type, count]) => `
                            <div class="flex justify-between text-sm">
                                <span class="text-yellow-700">${type}</span>
                                <span class="text-yellow-700">${count}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="bg-purple-50 p-4 rounded-lg">
                    <h4 class="font-medium text-purple-900">Performance</h4>
                    <p class="text-sm text-purple-700">Average Confidence: ${analytics.confidence_scores.length > 0 ? (analytics.confidence_scores.reduce((a, b) => a + b, 0) / analytics.confidence_scores.length * 100).toFixed(1) + '%' : 'N/A'}</p>
                    <p class="text-sm text-purple-700">Processing Times: ${analytics.processing_times.length} samples</p>
                </div>
            </div>
        `;
        
        analyticsContent.innerHTML = html;
    }

    async downloadResult() {
        if (!this.currentResult || !this.currentResult.output_file) {
            this.showStatus('error', 'No result available for download');
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/outputs/${this.currentResult.output_file}`);
            
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

    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.showStatus('success', 'Connected to Vehicle Detection API');
            } else {
                this.showStatus('error', 'API health check failed');
            }
        } catch (error) {
            this.showStatus('error', 'Cannot connect to API. Please start the backend server.');
        }
    }

    showStatus(type, message) {
        const banner = document.getElementById('statusBanner');
        const icon = document.getElementById('statusIcon');
        const messageEl = document.getElementById('statusMessage');
        
        // Set icon and colors based on type
        if (type === 'success') {
            icon.className = 'fas fa-check-circle text-green-600';
            banner.className = 'bg-green-50 border border-green-200 text-green-800 p-4 rounded-lg mb-6';
        } else if (type === 'error') {
            icon.className = 'fas fa-exclamation-circle text-red-600';
            banner.className = 'bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg mb-6';
        } else {
            icon.className = 'fas fa-info-circle text-blue-600';
            banner.className = 'bg-blue-50 border border-blue-200 text-blue-800 p-4 rounded-lg mb-6';
        }
        
        messageEl.textContent = message;
        banner.classList.remove('hidden');
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            banner.classList.add('hidden');
        }, 5000);
    }

    hideModal(modalId) {
        document.getElementById(modalId).classList.add('hidden');
    }

    resetUI() {
        document.getElementById('resultsSection').classList.add('hidden');
        document.getElementById('uploadProgress').classList.add('hidden');
        this.currentFile = null;
        this.currentResult = null;
    }
}

// Initialize the frontend when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VehicleDetectionFrontend();
});
