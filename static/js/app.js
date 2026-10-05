/**
 * YOLO Vision AI - Interactive Client-Side Controller
 * Handles image upload, webcam streaming, YOLO inference, interactive overlays,
 * compare split slider, and detection analytics.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ==========================================
    // State
    // ==========================================
    const state = {
        sourceType: 'sample', // 'sample' | 'upload' | 'webcam' | 'url'
        sampleName: 'dog.jpg',
        uploadedFile: null,
        urlImage: null,
        webcamImage: null,
        webcamStream: null,
        isLiveDetecting: false,
        liveDetectTimer: null,
        viewMode: 'annotated',
        confThreshold: 0.50,
        nmsThreshold: 0.40,
        resolution: 416,
        categoryFilter: 'all',
        lastResult: null,
        isProcessing: false
    };

    // ==========================================
    // DOM Elements
    // ==========================================
    const elements = {
        // Tabs
        tabs: document.querySelectorAll('.tab-btn'),
        tabPanels: document.querySelectorAll('.tab-content'),
        
        // Samples
        sampleCards: document.querySelectorAll('.sample-card'),
        
        // Upload
        dropzone: document.getElementById('dropzone'),
        fileInput: document.getElementById('file-input'),
        filePreviewCard: document.getElementById('file-preview-card'),
        filePreviewImg: document.getElementById('file-preview-img'),
        filePreviewName: document.getElementById('file-preview-name'),
        filePreviewSize: document.getElementById('file-preview-size'),
        btnRemoveFile: document.getElementById('btn-remove-file'),
        
        // Webcam
        webcamVideo: document.getElementById('webcam-video'),
        webcamCanvas: document.getElementById('webcam-canvas'),
        webcamPlaceholder: document.getElementById('webcam-placeholder'),
        webcamControls: document.getElementById('webcam-controls'),
        btnStartCamera: document.getElementById('btn-start-camera'),
        btnStopCamera: document.getElementById('btn-stop-camera'),
        btnWebcamSnap: document.getElementById('btn-webcam-snap'),
        btnToggleLive: document.getElementById('btn-toggle-live'),
        liveText: document.getElementById('live-text'),
        
        // URL
        imageUrlInput: document.getElementById('image-url-input'),
        btnFetchUrl: document.getElementById('btn-fetch-url'),
        
        // Sliders & Controls
        confSlider: document.getElementById('conf-slider'),
        confVal: document.getElementById('conf-val'),
        nmsSlider: document.getElementById('nms-slider'),
        nmsVal: document.getElementById('nms-val'),
        resButtons: document.querySelectorAll('.res-btn'),
        categoryFilter: document.getElementById('category-filter'),
        filterStatus: document.getElementById('filter-status'),
        btnDetect: document.getElementById('btn-detect'),
        
        // Toolbar & View Modes
        viewModeButtons: document.querySelectorAll('.view-mode-btn'),
        btnExportJson: document.getElementById('btn-export-json'),
        btnDownloadImg: document.getElementById('btn-download-img'),
        
        // Viewport & Layers
        viewportStage: document.getElementById('viewport-stage'),
        scanOverlay: document.getElementById('scan-overlay'),
        stageContainer: document.getElementById('stage-container'),
        imgAnnotated: document.getElementById('img-annotated'),
        imgOriginal: document.getElementById('img-original'),
        imgInteractiveBase: document.getElementById('img-interactive-base'),
        hudBoxesContainer: document.getElementById('hud-boxes-container'),
        
        // Compare Slider
        layerAnnotated: document.getElementById('layer-annotated'),
        layerInteractive: document.getElementById('layer-interactive'),
        layerCompare: document.getElementById('layer-compare'),
        layerOriginal: document.getElementById('layer-original'),
        compareContainer: document.getElementById('compare-container'),
        compareBeforeLayer: document.getElementById('compare-before-layer'),
        compareImgBefore: document.getElementById('compare-img-before'),
        compareImgAfter: document.getElementById('compare-img-after'),
        compareHandle: document.getElementById('compare-handle'),
        
        // Metadata
        metaFilename: document.getElementById('meta-filename'),
        metaDimensions: document.getElementById('meta-dimensions'),
        metaSummary: document.getElementById('meta-detected-summary'),
        
        // Metrics
        metricTime: document.getElementById('metric-time'),
        metricCount: document.getElementById('metric-count'),
        metricClasses: document.getElementById('metric-classes'),
        metricGrid: document.getElementById('metric-grid'),
        
        // Items List
        detectedItemsContainer: document.getElementById('detected-items-container'),
        categorySummaryTags: document.getElementById('category-summary-tags'),
        
        // Modal & Toast
        btnInfoModal: document.getElementById('btn-info-modal'),
        btnCloseModal: document.getElementById('btn-close-modal'),
        infoModal: document.getElementById('info-modal'),
        modalOpencvVer: document.getElementById('modal-opencv-ver'),
        toastContainer: document.getElementById('toast-container')
    };

    // ==========================================
    // Initialization
    // ==========================================
    function init() {
        checkHealth();
        setupTabs();
        setupSamples();
        setupDropzone();
        setupWebcam();
        setupSliders();
        setupViewModes();
        setupCompareSlider();
        setupModal();
        setupExportAndDownload();

        // Initial detection on bundled sample
        runDetection();
    }

    // Health check on backend
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (data.status === 'online') {
                if (elements.modalOpencvVer) {
                    elements.modalOpencvVer.textContent = `OpenCV ${data.opencv_version} (DNN Engine)`;
                }
            }
        } catch (e) {
            console.warn('Backend health check error:', e);
        }
    }

    // ==========================================
    // Navigation Tabs
    // ==========================================
    function setupTabs() {
        elements.tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                elements.tabs.forEach(t => t.classList.remove('active'));
                elements.tabPanels.forEach(p => p.classList.remove('active'));

                tab.classList.add('active');
                const targetPanel = document.getElementById(tab.dataset.target);
                if (targetPanel) targetPanel.classList.add('active');

                // Determine active source type
                if (tab.id === 'tab-samples') state.sourceType = 'sample';
                else if (tab.id === 'tab-upload') state.sourceType = 'upload';
                else if (tab.id === 'tab-webcam') state.sourceType = 'webcam';
                else if (tab.id === 'tab-url') state.sourceType = 'url';

                // Auto stop live stream if leaving webcam tab
                if (state.sourceType !== 'webcam' && state.isLiveDetecting) {
                    toggleLiveDetect(false);
                }
            });
        });
    }

    // ==========================================
    // Sample Images Gallery
    // ==========================================
    function setupSamples() {
        elements.sampleCards.forEach(card => {
            card.addEventListener('click', () => {
                elements.sampleCards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                state.sourceType = 'sample';
                state.sampleName = card.dataset.sample;
                state.uploadedFile = null;

                const thumb = card.querySelector('img').src;
                updateStagePreload(thumb, state.sampleName);
                runDetection();
            });
        });
    }

    // ==========================================
    // Drag & Drop / File Upload
    // ==========================================
    function setupDropzone() {
        const dropzone = elements.dropzone;
        const fileInput = elements.fileInput;

        dropzone.addEventListener('click', () => fileInput.click());

        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                handleUploadedFile(files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleUploadedFile(fileInput.files[0]);
            }
        });

        elements.btnRemoveFile.addEventListener('click', (e) => {
            e.stopPropagation();
            state.uploadedFile = null;
            elements.filePreviewCard.style.display = 'none';
            dropzone.style.display = 'block';
            fileInput.value = '';
        });
    }

    function handleUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload a valid image file (JPG, PNG, WEBP).', 'error');
            return;
        }

        state.sourceType = 'upload';
        state.uploadedFile = file;

        // Show preview card
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.filePreviewImg.src = e.target.result;
            elements.filePreviewName.textContent = file.name;
            elements.filePreviewSize.textContent = formatBytes(file.size);
            elements.dropzone.style.display = 'none';
            elements.filePreviewCard.style.display = 'flex';

            updateStagePreload(e.target.result, file.name);
            runDetection();
        };
        reader.readAsDataURL(file);
    }

    // ==========================================
    // URL Image Fetch
    // ==========================================
    if (elements.btnFetchUrl) {
        elements.btnFetchUrl.addEventListener('click', async () => {
            const url = elements.imageUrlInput.value.trim();
            if (!url) return;
            try {
                showToast('Fetching image...', 'info');
                const resp = await fetch(url);
                const blob = await resp.blob();
                const file = new File([blob], 'url_image.jpg', { type: blob.type });
                handleUploadedFile(file);
            } catch (err) {
                showToast('Failed to load image from URL. (CORS restricted)', 'error');
            }
        });
    }

    // ==========================================
    // Live Webcam Integration
    // ==========================================
    function setupWebcam() {
        elements.btnStartCamera.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
                });
                state.webcamStream = stream;
                elements.webcamVideo.srcObject = stream;
                elements.webcamPlaceholder.style.display = 'none';
                elements.webcamControls.style.display = 'flex';
                showToast('Webcam connected successfully!', 'success');
            } catch (err) {
                console.error(err);
                showToast('Camera permission denied or camera unavailable.', 'error');
            }
        });

        elements.btnStopCamera.addEventListener('click', () => {
            stopWebcam();
        });

        elements.btnWebcamSnap.addEventListener('click', () => {
            captureWebcamFrameAndDetect();
        });

        elements.btnToggleLive.addEventListener('click', () => {
            toggleLiveDetect(!state.isLiveDetecting);
        });
    }

    function stopWebcam() {
        if (state.isLiveDetecting) toggleLiveDetect(false);
        if (state.webcamStream) {
            state.webcamStream.getTracks().forEach(track => track.stop());
            state.webcamStream = null;
        }
        elements.webcamVideo.srcObject = null;
        elements.webcamPlaceholder.style.display = 'flex';
        elements.webcamControls.style.display = 'none';
    }

    function getWebcamFrameBase64() {
        const video = elements.webcamVideo;
        const canvas = elements.webcamCanvas;
        if (!video.videoWidth) return null;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        // Mirror horizontally to match mirrored preview
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.85);
    }

    function captureWebcamFrameAndDetect() {
        const b64 = getWebcamFrameBase64();
        if (!b64) return;
        state.sourceType = 'webcam';
        state.webcamImage = b64;
        updateStagePreload(b64, 'webcam_capture.jpg');
        runDetection();
    }

    function toggleLiveDetect(enable) {
        state.isLiveDetecting = enable;
        if (enable) {
            elements.btnToggleLive.classList.add('btn-primary');
            elements.liveText.textContent = 'Scanning Live...';
            state.liveDetectTimer = setInterval(() => {
                if (!state.isProcessing && state.webcamStream) {
                    captureWebcamFrameAndDetect();
                }
            }, 1400); // 1.4s throttle for smooth CPU inference
        } else {
            elements.btnToggleLive.classList.remove('btn-primary');
            elements.liveText.textContent = 'Continuous Detect';
            if (state.liveDetectTimer) {
                clearInterval(state.liveDetectTimer);
                state.liveDetectTimer = null;
            }
        }
    }

    // ==========================================
    // Sliders & Tuning Parameters
    // ==========================================
    function setupSliders() {
        // Confidence Slider
        elements.confSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            state.confThreshold = val / 100.0;
            elements.confVal.textContent = `${val}%`;
        });
        elements.confSlider.addEventListener('change', () => {
            runDetection();
        });

        // NMS Slider
        elements.nmsSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value, 10);
            state.nmsThreshold = val / 100.0;
            elements.nmsVal.textContent = state.nmsThreshold.toFixed(2);
        });
        elements.nmsSlider.addEventListener('change', () => {
            runDetection();
        });

        // Resolution Buttons
        elements.resButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.resButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.resolution = parseInt(btn.dataset.res, 10);
                elements.metricGrid.textContent = `${state.resolution} \u00d7 ${state.resolution}`;
                runDetection();
            });
        });

        // Category Filter
        elements.categoryFilter.addEventListener('change', (e) => {
            state.categoryFilter = e.target.value;
            elements.filterStatus.textContent = e.target.options[e.target.selectedIndex].text.split('(')[0].trim();
            runDetection();
        });

        // Detect Primary Button
        elements.btnDetect.addEventListener('click', () => {
            runDetection();
        });
    }

    // ==========================================
    // View Modes & Compare Split Slider
    // ==========================================
    function setupViewModes() {
        elements.viewModeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.viewModeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                switchViewMode(btn.dataset.mode);
            });
        });
    }

    function switchViewMode(mode) {
        state.viewMode = mode;
        elements.layerAnnotated.style.display = (mode === 'annotated') ? 'flex' : 'none';
        elements.layerInteractive.style.display = (mode === 'interactive') ? 'flex' : 'none';
        elements.layerCompare.style.display = (mode === 'compare') ? 'flex' : 'none';
        elements.layerOriginal.style.display = (mode === 'original') ? 'flex' : 'none';

        if (mode === 'compare') {
            syncCompareDimensions();
        }
    }

    function setupCompareSlider() {
        const container = elements.compareContainer;
        const handle = elements.compareHandle;
        const beforeLayer = elements.compareBeforeLayer;
        let isDragging = false;

        const updatePosition = (clientX) => {
            const rect = container.getBoundingClientRect();
            let x = clientX - rect.left;
            x = Math.max(0, Math.min(x, rect.width));
            const pct = (x / rect.width) * 100;
            handle.style.left = `${pct}%`;
            beforeLayer.style.width = `${pct}%`;
        };

        handle.addEventListener('mousedown', () => { isDragging = true; });
        window.addEventListener('mouseup', () => { isDragging = false; });
        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            updatePosition(e.clientX);
        });

        // Touch support
        handle.addEventListener('touchstart', () => { isDragging = true; }, { passive: true });
        window.addEventListener('touchend', () => { isDragging = false; });
        window.addEventListener('touchmove', (e) => {
            if (!isDragging || !e.touches[0]) return;
            updatePosition(e.touches[0].clientX);
        }, { passive: true });
    }

    function syncCompareDimensions() {
        const afterImg = elements.compareImgAfter;
        const beforeImg = elements.compareImgBefore;
        if (afterImg && beforeImg) {
            beforeImg.style.width = `${afterImg.clientWidth}px`;
            beforeImg.style.height = `${afterImg.clientHeight}px`;
        }
    }

    // ==========================================
    // Stage Preload & Image Update
    // ==========================================
    function updateStagePreload(src, filename) {
        elements.imgAnnotated.src = src;
        elements.imgOriginal.src = src;
        elements.imgInteractiveBase.src = src;
        elements.compareImgAfter.src = src;
        elements.compareImgBefore.src = src;
        elements.metaFilename.textContent = filename || 'image.jpg';
    }

    // ==========================================
    // Core YOLO Inference Execution
    // ==========================================
    async function runDetection() {
        if (state.isProcessing) return;
        state.isProcessing = true;

        if (!state.isLiveDetecting) {
            elements.scanOverlay.style.display = 'flex';
        }

        try {
            let res;

            if (state.sourceType === 'upload' && state.uploadedFile) {
                const formData = new FormData();
                formData.append('file', state.uploadedFile);
                formData.append('conf_threshold', state.confThreshold);
                formData.append('nms_threshold', state.nmsThreshold);
                formData.append('resolution', state.resolution);
                if (state.categoryFilter !== 'all') {
                    formData.append('allowed_classes', getCategoryClasses(state.categoryFilter));
                }

                res = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });
            } else if (state.sourceType === 'webcam' && state.webcamImage) {
                res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: state.webcamImage,
                        conf_threshold: state.confThreshold,
                        nms_threshold: state.nmsThreshold,
                        resolution: state.resolution,
                        allowed_classes: state.categoryFilter !== 'all' ? getCategoryClasses(state.categoryFilter) : null
                    })
                });
            } else {
                // Default: Sample image
                res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sample_name: state.sampleName,
                        conf_threshold: state.confThreshold,
                        nms_threshold: state.nmsThreshold,
                        resolution: state.resolution,
                        allowed_classes: state.categoryFilter !== 'all' ? getCategoryClasses(state.categoryFilter) : null
                    })
                });
            }

            const data = await res.json();
            if (!data.success) {
                throw new Error(data.error || 'Detection failed');
            }

            state.lastResult = data;
            renderDetectionResults(data);

        } catch (err) {
            console.error('Detection error:', err);
            showToast(err.message || 'Detection failed', 'error');
        } finally {
            state.isProcessing = false;
            elements.scanOverlay.style.display = 'none';
        }
    }

    // ==========================================
    // Render Results & HUD
    // ==========================================
    function renderDetectionResults(data) {
        const stats = data.stats;
        const detections = data.detections;

        // 1. Update Images
        elements.imgAnnotated.src = data.annotated_image;
        elements.imgOriginal.src = data.original_image;
        elements.imgInteractiveBase.src = data.original_image;
        elements.compareImgAfter.src = data.annotated_image;
        elements.compareImgBefore.src = data.original_image;

        // 2. Metrics Bar
        elements.metricTime.textContent = `${stats.inference_time_ms} ms`;
        elements.metricCount.textContent = stats.total_detections;
        elements.metricClasses.textContent = stats.unique_classes.length;
        elements.metricGrid.textContent = stats.input_resolution;

        // 3. Stage Metadata Footer
        elements.metaFilename.textContent = data.filename || 'image.jpg';
        elements.metaDimensions.textContent = `${stats.image_width} \u00d7 ${stats.image_height} px`;
        elements.metaSummary.textContent = detections.length > 0
            ? `Detected ${detections.length} objects (${stats.unique_classes.join(', ')})`
            : 'No objects detected above threshold';

        // 4. Render Category Summary Badges
        renderCategorySummary(stats.category_counts);

        // 5. Render Detected Items Cards
        renderItemsList(detections);

        // 6. Render Interactive HUD Overlay Boxes
        renderInteractiveHud(detections);

        syncCompareDimensions();
    }

    function renderCategorySummary(counts) {
        elements.categorySummaryTags.innerHTML = '';
        if (!counts || Object.keys(counts).length === 0) return;

        Object.entries(counts).forEach(([cat, count]) => {
            const badge = document.createElement('span');
            badge.className = 'category-tag';
            badge.textContent = `${cat}: ${count}`;
            elements.categorySummaryTags.appendChild(badge);
        });
    }

    function renderItemsList(detections) {
        elements.detectedItemsContainer.innerHTML = '';

        if (!detections || detections.length === 0) {
            elements.detectedItemsContainer.innerHTML = `
                <div class="empty-items-notice">
                    <span>No objects detected above ${Math.round(state.confThreshold * 100)}% confidence threshold.</span>
                </div>
            `;
            return;
        }

        detections.forEach(item => {
            const card = document.createElement('div');
            card.className = 'item-card';
            card.id = `item-card-${item.id}`;
            card.dataset.id = item.id;

            card.innerHTML = `
                <div class="item-header">
                    <div class="item-badge-group">
                        <span class="item-color-pill" style="background-color: ${item.color};"></span>
                        <span class="item-name">${item.label}</span>
                        <span class="item-category-label">${item.category}</span>
                    </div>
                    <span class="item-conf">${item.confidence_pct}%</span>
                </div>
                <div class="item-progress-track">
                    <div class="item-progress-fill" style="width: ${item.confidence_pct}%; background-color: ${item.color};"></div>
                </div>
                <div class="item-coords">
                    <span>Pos: [${item.box.x}, ${item.box.y}]</span>
                    <span>Dim: ${item.box.width} \u00d7 ${item.box.height} px</span>
                </div>
            `;

            // Hover sync with HUD boxes
            card.addEventListener('mouseenter', () => highlightDetection(item.id, true));
            card.addEventListener('mouseleave', () => highlightDetection(item.id, false));
            card.addEventListener('click', () => {
                switchViewMode('interactive');
                highlightDetection(item.id, true);
            });

            elements.detectedItemsContainer.appendChild(card);
        });
    }

    function renderInteractiveHud(detections) {
        elements.hudBoxesContainer.innerHTML = '';
        if (!detections || detections.length === 0) return;

        detections.forEach(item => {
            const box = document.createElement('div');
            box.className = 'hud-box';
            box.id = `hud-box-${item.id}`;
            box.style.borderColor = item.color;
            box.style.color = item.color;

            // Normalized coordinates (percentage of image)
            const nb = item.normalized_box;
            box.style.left = `${nb.x * 100}%`;
            box.style.top = `${nb.y * 100}%`;
            box.style.width = `${nb.width * 100}%`;
            box.style.height = `${nb.height * 100}%`;

            // Label Tag
            const tag = document.createElement('div');
            tag.className = 'hud-tag';
            tag.style.backgroundColor = item.color;
            tag.textContent = `${item.label} ${item.confidence_pct}%`;
            box.appendChild(tag);

            // Hover sync
            box.addEventListener('mouseenter', () => highlightDetection(item.id, true));
            box.addEventListener('mouseleave', () => highlightDetection(item.id, false));

            elements.hudBoxesContainer.appendChild(box);
        });
    }

    function highlightDetection(id, isHighlighted) {
        const hudBox = document.getElementById(`hud-box-${id}`);
        const itemCard = document.getElementById(`item-card-${id}`);

        if (hudBox) {
            if (isHighlighted) hudBox.classList.add('highlighted');
            else hudBox.classList.remove('highlighted');
        }

        if (itemCard) {
            if (isHighlighted) {
                itemCard.classList.add('highlighted');
                itemCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                itemCard.classList.remove('highlighted');
            }
        }
    }

    // ==========================================
    // Export & Download
    // ==========================================
    function setupExportAndDownload() {
        elements.btnDownloadImg.addEventListener('click', () => {
            if (!state.lastResult || !state.lastResult.annotated_image) {
                showToast('No detection output available to download.', 'error');
                return;
            }
            const link = document.createElement('a');
            link.download = `yolo-detection-${Date.now()}.jpg`;
            link.href = state.lastResult.annotated_image;
            link.click();
            showToast('Annotated image downloaded!', 'success');
        });

        elements.btnExportJson.addEventListener('click', () => {
            if (!state.lastResult) {
                showToast('No detection output available to export.', 'error');
                return;
            }
            const exportData = {
                metadata: {
                    model: 'YOLOv3',
                    timestamp: new Date().toISOString(),
                    filename: state.lastResult.filename,
                    stats: state.lastResult.stats
                },
                detections: state.lastResult.detections
            };

            const jsonStr = JSON.stringify(exportData, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const link = document.createElement('a');
            link.download = `yolo-detections-${Date.now()}.json`;
            link.href = URL.createObjectURL(blob);
            link.click();
            showToast('Detection results JSON exported!', 'success');
        });
    }

    // ==========================================
    // Modal & Toast Utilities
    // ==========================================
    function setupModal() {
        elements.btnInfoModal.addEventListener('click', () => {
            elements.infoModal.style.display = 'flex';
        });
        elements.btnCloseModal.addEventListener('click', () => {
            elements.infoModal.style.display = 'none';
        });
        elements.infoModal.addEventListener('click', (e) => {
            if (e.target === elements.infoModal) {
                elements.infoModal.style.display = 'none';
            }
        });
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span>${message}</span>
        `;
        elements.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 250);
        }, 3000);
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function getCategoryClasses(category) {
        const map = {
            'Animals': 'bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe',
            'Vehicles': 'bicycle,car,motorcycle,airplane,bus,train,truck,boat',
            'People': 'person',
            'Electronics': 'tv,laptop,mouse,remote,keyboard,cell phone',
            'Food': 'banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake',
            'Sports': 'frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket'
        };
        return map[category] || null;
    }

    // Initialize application
    init();
});
