// Vehicle Detection System - Frontend JavaScript
class VehicleDetectionFrontend {
    constructor() {
        // Use the same origin for API calls
        this.apiBaseUrl = '';
        this.currentFileType = 'image';
        this.currentFile = null;
        this.currentResult = null;
        this.progressInterval = null;
        
        this.initializeEventListeners();
        this.checkApiHealth();
    }

    initializeEventListeners() {
        console.log('Initializing event listeners...');
        
        // File type selection
        const imageBtn = document.getElementById('imageBtn');
        const videoBtn = document.getElementById('videoBtn');
        
        if (imageBtn) {
            imageBtn.onclick = () => this.setFileType('image');
        }
        if (videoBtn) {
            videoBtn.onclick = () => this.setFileType('video');
        }
        
        // File upload
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        if (dropZone) {
            dropZone.onclick = () => fileInput && fileInput.click();
            dropZone.ondragover = (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            };
            dropZone.ondrop = (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.processFile(files[0]);
                }
            };
            dropZone.ondragleave = () => dropZone.classList.remove('dragover');
        }
        
        if (fileInput) {
            fileInput.onchange = (e) => {
                const file = e.target.files[0];
                if (file) {
                    console.log('File selected:', file.name);
                    this.processFile(file);
                }
            };
        }
        
        // Button click handlers
        const statusBtn = document.getElementById('statusBtn');
        const analyticsBtn = document.getElementById('analyticsBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        
        if (statusBtn) {
            statusBtn.onclick = () => this.showSystemStatus();
        }
        if (analyticsBtn) {
            analyticsBtn.onclick = () => this.showAnalytics();
        }
        if (downloadBtn) {
            downloadBtn.onclick = () => this.downloadResult();
        }
        
        // Modal close handlers
        const closeStatusModal = document.getElementById('closeStatusModal');
        const closeAnalyticsModal = document.getElementById('closeAnalyticsModal');
        
        if (closeStatusModal) {
            closeStatusModal.onclick = () => this.hideModal('statusModal');
        }
        if (closeAnalyticsModal) {
            closeAnalyticsModal.onclick = () => this.hideModal('analyticsModal');
        }
    }

    setFileType(type) {
        console.log('Setting file type to:', type);
        this.currentFileType = type;
        
        const imageBtn = document.getElementById('imageBtn');
        const videoBtn = document.getElementById('videoBtn');
        const fileInput = document.getElementById('fileInput');
        const fileTypesText = document.getElementById('fileTypesText');
        
        if (imageBtn && videoBtn) {
            if (type === 'image') {
                imageBtn.classList.add('bg-blue-600', 'text-white');
                imageBtn.classList.remove('bg-gray-200', 'text-gray-700');
                videoBtn.classList.add('bg-gray-200', 'text-gray-700');
                videoBtn.classList.remove('bg-blue-600', 'text-white');
            } else {
                videoBtn.classList.add('bg-blue-600', 'text-white');
                videoBtn.classList.remove('bg-gray-200', 'text-gray-700');
                imageBtn.classList.add('bg-gray-200', 'text-gray-700');
                imageBtn.classList.remove('bg-blue-600', 'text-white');
            }
        }
        
        if (fileInput) {
            fileInput.accept = type === 'image' ? 'image/*' : 'video/*';
        }
        
        if (fileTypesText) {
            fileTypesText.textContent = type === 'image' 
                ? 'Supported formats: JPG, PNG, JPEG'
                : 'Supported formats: MP4, AVI, MOV, MKV';
        }
        
        this.resetUI();
    }

    async processFile(file) {
        console.log('Processing file:', file.name);
        
        const allowedExtensions = this.currentFileType === 'image' 
            ? ['jpg', 'jpeg', 'png'] 
            : ['mp4', 'avi', 'mov', 'mkv'];
        
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        if (!allowedExtensions.includes(fileExtension)) {
            this.showStatus('error', `Invalid file type. Please upload a ${this.currentFileType} file.`);
            return;
        }
        
        this.currentFile = file;
        await this.uploadFile(file);
    }

    async uploadFile(file) {
        try {
            console.log('Uploading file:', file.name);
            this.showUploadProgress();
            
            const formData = new FormData();
            formData.append('file', file);
            
            const endpoint = this.currentFileType === 'image' ? '/detect/image' : '/detect/video';
            const response = await fetch(`${this.apiBaseUrl}/api${endpoint}`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
            }
            
            const result = await response.json();
            console.log('Upload result:', result);
            
            this.handleDetectionResult(result);
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showStatus('error', `Upload failed: ${error.message}`);
            this.hideUploadProgress();
        }
    }

    handleDetectionResult(result) {
        console.log('Handling detection result:', result);
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
        console.log('Displaying results:', result);
        
        const vehicleCount = document.getElementById('vehicleCount');
        const processingTime = document.getElementById('processingTime');
        const outputFile = document.getElementById('outputFile');
        const vehicleDetails = document.getElementById('vehicleDetails');
        const resultsSection = document.getElementById('resultsSection');
        
        if (vehicleCount) {
            vehicleCount.textContent = result.detections !== undefined 
                ? result.detections 
                : result.total_frames || 0;
        }
        
        if (processingTime) {
            processingTime.textContent = `${result.processing_time || 0}s`;
        }
        
        if (outputFile) {
            outputFile.textContent = result.output_file || 'No file';
        }
        
        if (vehicleDetails) {
            vehicleDetails.innerHTML = '';
            
            if (result.vehicles_detected && result.vehicles_detected.length > 0) {
                result.vehicles_detected.forEach((vehicle, index) => {
                    const card = this.createVehicleCard(vehicle, index);
                    vehicleDetails.appendChild(card);
                });
            } else {
                vehicleDetails.innerHTML = '<p class="text-gray-500 text-center py-4">No vehicle details available</p>';
            }
        }
        
        if (resultsSection) {
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    createVehicleCard(vehicle, index) {
        const card = document.createElement('div');
        card.className = 'vehicle-card bg-gray-50 p-4 rounded-lg border mb-4';
        
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

    showUploadProgress() {
        const uploadProgress = document.getElementById('uploadProgress');
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        
        if (uploadProgress && progressBar && progressPercent) {
            uploadProgress.classList.remove('hidden');
            
            // Reset progress
            let progress = 0;
            progressBar.style.width = '0%';
            progressPercent.textContent = '0%';
            
            // Clear any existing interval
            if (this.progressInterval) {
                clearInterval(this.progressInterval);
            }
            
            // Start progress simulation
            this.progressInterval = setInterval(() => {
                if (progress < 90) {
                    progress += Math.random() * 10;
                    if (progress > 90) progress = 90;
                    
                    progressBar.style.width = `${progress}%`;
                    progressPercent.textContent = `${Math.round(progress)}%`;
                }
            }, 500);
        }
    }

    hideUploadProgress() {
        const uploadProgress = document.getElementById('uploadProgress');
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        
        // Clear the progress interval
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        // Complete the progress bar
        if (progressBar && progressPercent) {
            progressBar.style.width = '100%';
            progressPercent.textContent = '100%';
        }
        
        // Hide after a short delay
        setTimeout(() => {
            if (uploadProgress) {
                uploadProgress.classList.add('hidden');
            }
        }, 500);
    }

    showStatus(type, message) {
        console.log(`Status (${type}):`, message);
        
        const banner = document.getElementById('statusBanner');
        const icon = document.getElementById('statusIcon');
        const messageEl = document.getElementById('statusMessage');
        
        if (!banner || !icon || !messageEl) return;
        
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
        
        setTimeout(() => banner.classList.add('hidden'), 5000);
    }

    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    resetUI() {
        const resultsSection = document.getElementById('resultsSection');
        const uploadProgress = document.getElementById('uploadProgress');
        
        if (resultsSection) {
            resultsSection.classList.add('hidden');
        }
        if (uploadProgress) {
            uploadProgress.classList.add('hidden');
        }
        
        // Clear any ongoing progress simulation
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        this.currentFile = null;
        this.currentResult = null;
    }

    async showSystemStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/status`);
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

    async showAnalytics() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/analytics`);
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

    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/health`);
            const data = await response.json();
            
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

    displaySystemStatus(data) {
        const statusContent = document.getElementById('statusContent');
        if (!statusContent) return;
        
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

    displayAnalytics(analytics) {
        const analyticsContent = document.getElementById('analyticsContent');
        if (!analyticsContent) return;
        
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
}

// Initialize the frontend when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Vehicle Detection Frontend...');
    window.app = new VehicleDetectionFrontend();
});