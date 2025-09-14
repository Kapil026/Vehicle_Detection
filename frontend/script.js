// Vehicle Detection System - Frontend JavaScript
class VehicleDetectionFrontend {
    constructor() {
        // Use the same origin for API calls
        this.apiBaseUrl = '';
        this.currentFileType = 'image';
        this.currentFile = null;
        this.currentResult = null;
        
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

    // ... rest of your methods ...

    showUploadProgress() {
        const uploadProgress = document.getElementById('uploadProgress');
        if (uploadProgress) {
            uploadProgress.classList.remove('hidden');
            this.simulateProgress();
        }
    }

    hideUploadProgress() {
        const uploadProgress = document.getElementById('uploadProgress');
        if (uploadProgress) {
            uploadProgress.classList.add('hidden');
        }
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
        
        this.currentFile = null;
        this.currentResult = null;
    }
}

// Initialize the frontend when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Vehicle Detection Frontend...');
    window.app = new VehicleDetectionFrontend();
});