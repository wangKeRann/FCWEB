// WebSocket connection and video streaming handler
// Version: 1.0.1 - Updated video type mapping
class VideoStreamHandler {
    constructor() {
        this.wsVisible = null;  // WebSocket for visible light video
        this.wsInfrared = null; // WebSocket for infrared video
        this.wsPoseDetection = null; // WebSocket for pose detection
        this.visibleLightVideo = document.getElementById('visible-light-video');
        this.infraredVideo = document.getElementById('infrared-video');
        this.currentMode = 'video_monitoring'; // Default mode
        this.currentDevice = 'robot'; // Default device
        this.statusIndicator = document.querySelector('.status-indicator');
        this.statusText = document.querySelector('.header-status-text');
        this.mediaSource = null;
        this.sourceBuffer = null;
        this.isPlaying = false;
        console.log('VideoStreamHandler initialized with default settings:');
        console.log(`- Device: ${this.currentDevice}`);
        console.log(`- Mode: ${this.currentMode}`);
        this.setupEventListeners();

        // Initialize with disconnected status
        this.updateStatus('disconnected', '设备状态: 未连接');

        // Connect to WebSocket servers
        this.connect();
    }

    setupEventListeners() {
        // Device control buttons
        const robotButton = document.getElementById('robot-btn');
        const droneButton = document.getElementById('drone-btn');

        robotButton.addEventListener('click', () => {
            this.currentDevice = 'robot';
            robotButton.classList.add('active');
            droneButton.classList.remove('active');
            console.log('Device changed to: robot');
            this.updateStream();
        });

        droneButton.addEventListener('click', () => {
            this.currentDevice = 'drone';
            droneButton.classList.add('active');
            robotButton.classList.remove('active');
            console.log('Device changed to: drone');
            this.updateStream();
        });

        // Control mode buttons
        const videoMonitoringBtn = document.getElementById('video-monitoring-btn');
        const smokeRemovalBtn = document.getElementById('smoke-removal-btn');
        const poseDetectionBtn = document.getElementById('pose-detection-btn');
        const multiModalBtn = document.getElementById('multi-modal-btn');
        const enhancementBtn = document.getElementById('enhancement-btn');

        videoMonitoringBtn.addEventListener('click', () => {
            this.currentMode = 'video_monitoring';
            this.updateActiveButton(videoMonitoringBtn);
            console.log('Mode changed to: video_monitoring');
            this.updateStream();
        });

        smokeRemovalBtn.addEventListener('click', () => {
            this.currentMode = 'smoke_removal';
            this.updateActiveButton(smokeRemovalBtn);
            console.log('Mode changed to: smoke_removal');
            this.updateStream();
        });

        poseDetectionBtn.addEventListener('click', () => {
            this.currentMode = 'pose_detection';
            this.updateActiveButton(poseDetectionBtn);
            console.log('[DEBUG] 切换到行为识别模式');
            this.updateStatus('connecting', '正在加载行为识别模型...');
            this.updateStream();
        });

        multiModalBtn.addEventListener('click', () => {
            this.currentMode = 'multi_modal';
            this.updateActiveButton(multiModalBtn);
            console.log('Mode changed to: multi_modal');
            this.updateStream();
        });

        enhancementBtn.addEventListener('click', () => {
            const enhancementControls = document.querySelector('.enhancement-controls');
            const isVisible = enhancementControls.style.display === 'block';

            if (!isVisible) {
                // 切换到增强模式
                this.currentMode = 'enhancement';
                this.updateActiveButton(enhancementBtn);
                console.log('[DEBUG] Switching to enhancement mode');
            } else {
                // 关闭控制面板时，移除active类
                enhancementBtn.classList.remove('active');
                // 恢复到默认的视频监测模式
                this.currentMode = 'video_monitoring';
                const videoMonitoringBtn = document.getElementById('video-monitoring-btn');
                this.updateActiveButton(videoMonitoringBtn);
            }

            enhancementControls.style.display = isVisible ? 'none' : 'block';
            console.log(`[DEBUG] Enhancement controls ${isVisible ? 'hidden' : 'shown'}`);
        });

        // Image enhancement controls
        const enhancementControls = document.querySelector('.enhancement-controls');
        const applyEnhancementBtn = document.getElementById('apply-enhancement');

        // Apply enhancement button
        applyEnhancementBtn.addEventListener('click', () => {
            const brightness = document.getElementById('brightness').value;
            const contrast = document.getElementById('contrast').value;
            const saturation = document.getElementById('saturation').value;

            console.log('[DEBUG] Enhancement button clicked with parameters:');
            console.log(`- Brightness: ${brightness}`);
            console.log(`- Contrast: ${contrast}`);
            console.log(`- Saturation: ${saturation}`);

            // 移除所有控制模式按钮的active类
            const controlModeButtons = document.querySelectorAll('.panel-content .apple-btn');
            controlModeButtons.forEach(btn => {
                if (btn.id !== 'enhancement-btn') {
                    btn.classList.remove('active');
                }
            });

            // 设置当前模式为enhancement
            this.currentMode = 'enhancement';

            // 清理现有的视频流
            this.cleanupResources().then(() => {
                // 应用新的增强参数
                this.applyImageEnhancement(brightness, contrast, saturation);
                // 重新连接并更新视频流
                this.connect();
            });
        });
    }

    updateActiveButton(activeButton) {
        // 移除所有按钮的active状态
        const buttons = activeButton.parentElement.querySelectorAll('.apple-btn');
        buttons.forEach(btn => {
            btn.classList.remove('active');
            // 如果是图像增强按钮，同时隐藏控制面板
            if (btn.id === 'enhancement-btn') {
                const controls = document.querySelector('.enhancement-controls');
                if (controls) {
                    controls.style.display = 'none';
                }
                // 同时移除视频监测按钮的active状态
                const videoMonitoringBtn = document.getElementById('video-monitoring-btn');
                if (videoMonitoringBtn) {
                    videoMonitoringBtn.classList.remove('active');
                }
            }
        });
        // 添加当前按钮的active状态
        activeButton.classList.add('active');
    }

    async updateStream() {
        console.log(`[DEBUG] Updating stream with device: ${this.currentDevice}, mode: ${this.currentMode}`);

        // Clean up existing resources and wait for cleanup to complete
        await this.cleanupResources();

        // Reset video elements
        this.visibleLightVideo.innerHTML = '';
        this.infraredVideo.innerHTML = '';

        // Show both video containers
        this.visibleLightVideo.style.display = 'block';
        this.infraredVideo.style.display = 'block';

        // Connect with new parameters
        this.connect();
    }

    connect() {
        console.log('[DEBUG] Starting WebSocket connection process...');
        this.updateStatus('connecting', '正在连接服务器...');

        // Connect to visible light video server
        console.log('[DEBUG] Creating new WebSocket connection for visible light video');
        this.wsVisible = new WebSocket('ws://localhost:8000/ws/client');
        this.setupWebSocketHandlers(this.wsVisible, 'visible');

        // Connect to infrared video server
        console.log('[DEBUG] Creating new WebSocket connection for infrared video');
        this.wsInfrared = new WebSocket('ws://localhost:8000/ws/client');
        this.setupWebSocketHandlers(this.wsInfrared, 'infrared');
    }

    setupWebSocketHandlers(ws, type) {
        ws.onopen = () => {
            console.log(`[DEBUG] ${type} WebSocket connection established successfully`);
            this.updateStatus('connected', '设备状态: 正常运行');
            console.log(`[DEBUG] Waiting for available video types from ${type} server...`);
        };

        ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                try {
                    const data = JSON.parse(event.data);

                    if (data.status === "info" && data.video_types) {
                        console.log(`[DEBUG] Received available video types from ${type}:`, data.video_types);
                        this.requestVideoStream(type);
                    } else if (data.status === "success") {
                        console.log(`[DEBUG] ${type} server success message:`, data.message);
                    } else if (data.status === "warning") {
                        console.warn(`[WARNING] ${type} server warning:`, data.message);
                        this.updateStatus('connected', '设备状态: 正常运行');
                        this.showNoStreamWarning(type);
                    } else if (data.status === "error") {
                        console.error(`[ERROR] ${type} server error:`, data.message);
                        this.updateStatus('error', `设备状态: ${data.message}`);
                        this.showError(type === 'visible' ? 'visible-light-container' : 'infrared-container');
                    }
                } catch (error) {
                    console.error(`[ERROR] Failed to parse message from ${type}:`, error);
                }
            } else {
                if (event.data instanceof Blob) {
                    const reader = new FileReader();
                    reader.onload = () => {
                        this.handleVideoData(reader.result, type);
                    };
                    reader.readAsArrayBuffer(event.data);
                } else if (event.data instanceof ArrayBuffer) {
                    this.handleVideoData(event.data, type);
                } else {
                    console.error(`[ERROR] Received data from ${type} is neither Blob nor ArrayBuffer:`, event.data);
                }
            }
        };

        ws.onerror = (error) => {
            console.error(`[ERROR] ${type} WebSocket error occurred:`, error);
            this.updateStatus('error', '设备状态: 连接错误');
            this.showError(type === 'visible' ? 'visible-light-container' : 'infrared-container');
            setTimeout(() => {
                console.log(`[DEBUG] Attempting to reconnect ${type}...`);
                if (type === 'visible') {
                    this.wsVisible = new WebSocket('ws://localhost:8000/ws/client');
                    this.setupWebSocketHandlers(this.wsVisible, 'visible');
                } else {
                    this.wsInfrared = new WebSocket('ws://localhost:8000/ws/client');
                    this.setupWebSocketHandlers(this.wsInfrared, 'infrared');
                }
            }, 5000);
        };

        ws.onclose = (event) => {
            console.log(`[DEBUG] ${type} WebSocket connection closed`);
            this.updateStatus('disconnected', '设备状态: 已断开连接');
            setTimeout(() => {
                console.log(`[DEBUG] Attempting to reconnect ${type}...`);
                if (type === 'visible') {
                    this.wsVisible = new WebSocket('ws://localhost:8000/ws/client');
                    this.setupWebSocketHandlers(this.wsVisible, 'visible');
                } else {
                    this.wsInfrared = new WebSocket('ws://localhost:8000/ws/client');
                    this.setupWebSocketHandlers(this.wsInfrared, 'infrared');
                }
            }, 5000);
        };
    }

    requestVideoStream(type) {
        const ws = type === 'visible' ? this.wsVisible : this.wsInfrared;
        let videoType;

        // 根据type参数决定请求的视频类型
        videoType = type === 'visible' ? 'visible' : 'thermal';

        // 确定处理类型
        let process_type = "original";
        if (this.currentMode === 'smoke_removal') {
            process_type = "smoke_removal";
        } else if (this.currentMode === 'pose_detection') {
            process_type = "pose_detection";
        } else if (this.currentMode === 'enhancement') {
            process_type = "enhancement";
        } else if (this.currentMode === 'multi_modal') {
            process_type = "multimodal_detection";
        }

        const request = {
            device_type: this.currentDevice,
            video_type: videoType,
            process_type: process_type,
            frame_rate: 30,
            continuous: true
        };

        console.log(`[DEBUG] Preparing to send ${videoType} video stream request:`, JSON.stringify(request, null, 2));

        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log(`[DEBUG] ${videoType} WebSocket connection is open, sending request...`);
            ws.send(JSON.stringify(request));
            console.log(`[DEBUG] ${videoType} request sent successfully`);
        } else {
            console.error(`[ERROR] ${videoType} WebSocket is not connected. Current state:`, ws ? ws.readyState : 'no connection');
            this.updateStatus('error', '设备状态: 连接未就绪');
        }
    }

    handleVideoData(data, videoType) {
        try {
            // Create a Blob from the received JPEG data
            const blob = new Blob([data], { type: 'image/jpeg' });
            const blobUrl = URL.createObjectURL(blob);

            // Get the appropriate video container
            const videoContainer = videoType === 'visible' ? this.visibleLightVideo : this.infraredVideo;

            // Clean up previous resources
            if (videoContainer._currentBlobUrl) {
                URL.revokeObjectURL(videoContainer._currentBlobUrl);
            }
            videoContainer.innerHTML = '';

            // Create and append new image
            const img = document.createElement('img');
            img.src = blobUrl;
            videoContainer.appendChild(img);

            // Store the new Blob URL for future cleanup
            videoContainer._currentBlobUrl = blobUrl;

            // Hide error and warning messages for this specific video type
            this.hideError(videoType === 'visible' ? 'visible-light-container' : 'infrared-container');
            this.hideNoStreamWarning(videoType);
        } catch (error) {
            console.error(`[ERROR] 处理${videoType}视频数据时出错:`, error);
            this.showError(videoType === 'visible' ? 'visible-light-container' : 'infrared-container');
        }
    }

    handleMetadata(metadata) {
        // Handle different types of metadata
        switch (metadata.type) {
            case 'video_info':
                this.updateVideoInfo(metadata);
                break;
            case 'error':
                console.error('Server error:', metadata.message);
                break;
            default:
                // 处理检测结果
                if (this.currentMode === 'pose_detection' && metadata.detections) {
                    this.updatePoseDetections(metadata.detections);
                } else if (this.currentMode === 'multi_modal' && (metadata.rgb_detections || metadata.ir_detections)) {
                    this.updateMultimodalDetections(metadata.rgb_detections, metadata.ir_detections);
                }
                console.log('Received metadata:', metadata);
        }
    }

    updateVideoInfo(info) {
        // Update video information in the UI
        console.log('Video info updated:', info);
    }

    updatePoseDetections(detections) {
        // 更新姿势检测结果
        const detectionInfo = document.getElementById('pose-detection-info');
        if (detectionInfo) {
            let infoText = '检测结果: ';
            if (detections && detections.length > 0) {
                infoText += detections.map(det => `${det.label} (${(det.confidence * 100).toFixed(1)}%)`).join(', ');
            } else {
                infoText += '未检测到目标';
            }
            detectionInfo.textContent = infoText;
        }
    }

    updateMultimodalDetections(rgbDetections, irDetections) {
        // 更新多模态检测结果
        const detectionInfo = document.getElementById('pose-detection-info');
        if (detectionInfo) {
            let infoText = '检测结果: ';
            let allDetections = [];

            // 合并RGB和红外检测结果
            if (rgbDetections && rgbDetections.length > 0) {
                allDetections = allDetections.concat(rgbDetections.map(det => `RGB: ${det.label} (${(det.confidence * 100).toFixed(1)}%)`));
            }
            if (irDetections && irDetections.length > 0) {
                allDetections = allDetections.concat(irDetections.map(det => `IR: ${det.label} (${(det.confidence * 100).toFixed(1)}%)`));
            }

            if (allDetections.length > 0) {
                infoText += allDetections.join(', ');
            } else {
                infoText += '未检测到目标';
            }
            detectionInfo.textContent = infoText;
        }
    }

    applyImageEnhancement(brightness, contrast, saturation) {
        console.log(`[DEBUG] Applying image enhancement: brightness=${brightness}, contrast=${contrast}, saturation=${saturation}`);

        // 首先确保我们处于增强模式
        this.currentMode = 'enhancement';

        // 如果WebSocket连接不存在，先建立连接
        if (!this.wsVisible || !this.wsInfrared) {
            console.log('[DEBUG] WebSocket connections not found, establishing new connections...');
            this.connect();
            return;
        }

        // 调整参数范围以匹配后端实现
        const adjustedParams = {
            brightness: parseInt(brightness) - 50, // 将 [0,100] 转换为 [-50,50]
            contrast: parseInt(contrast) / 100.0,  // 将 [0,200] 转换为 [0,2]
            saturation: parseInt(saturation) / 100.0 // 将 [0,200] 转换为 [0,2]
        };

        // 构建增强参数请求
        const enhancementRequest = {
            type: 'enhancement',
            parameters: adjustedParams
        };

        // 发送增强参数到两个视频流
        if (this.wsVisible && this.wsVisible.readyState === WebSocket.OPEN) {
            console.log('[DEBUG] Sending enhancement parameters to visible stream');
            this.wsVisible.send(JSON.stringify(enhancementRequest));
        }

        if (this.wsInfrared && this.wsInfrared.readyState === WebSocket.OPEN) {
            console.log('[DEBUG] Sending enhancement parameters to infrared stream');
            this.wsInfrared.send(JSON.stringify(enhancementRequest));
        }

        // 更新调试信息
        const debugInfo = document.getElementById('enhancement-params');
        if (debugInfo) {
            debugInfo.textContent = `亮度: ${brightness}, 对比度: ${contrast}, 饱和度: ${saturation}`;
        }

        // 重新请求视频流
        this.requestVideoStream('visible');
        this.requestVideoStream('infrared');
    }

    showError(containerId) {
        const container = document.getElementById(containerId);
        const errorElement = container.querySelector('.video-error');
        if (errorElement) {
            errorElement.style.display = 'flex';
        }
    }

    hideError(containerId) {
        const container = document.getElementById(containerId);
        const errorElement = container.querySelector('.video-error');
        if (errorElement) {
            errorElement.style.display = 'none';
        }
    }

    updateStatus(status, message) {
        // Remove all status classes
        this.statusIndicator.classList.remove('connected', 'connecting', 'disconnected', 'error');
        // Add the new status class
        this.statusIndicator.classList.add(status);
        // Update status text
        this.statusText.textContent = message;
    }

    showNoStreamWarning(type) {
        const containerId = type === 'visible' ? 'visible-light-container' : 'infrared-container';
        const container = document.getElementById(containerId);
        const errorElement = container.querySelector('.video-error');
        if (errorElement) {
            errorElement.textContent = '无视频流';
            errorElement.style.display = 'flex';
        }
    }

    hideNoStreamWarning(type) {
        const containerId = type === 'visible' ? 'visible-light-container' : 'infrared-container';
        const container = document.getElementById(containerId);
        const errorElement = container.querySelector('.video-error');
        if (errorElement) {
            errorElement.style.display = 'none';
        }
    }

    togglePlayPause() {
        const playPauseBtn = document.getElementById('play-pause-btn');
        const icon = playPauseBtn.querySelector('i');

        if (this.isPlaying) {
            // Pause video
            if (this.currentDevice === 'drone') {
                this.visibleLightVideo.pause();
            } else {
                this.infraredVideo.pause();
            }
            icon.className = 'fas fa-play';
            this.isPlaying = false;
            console.log('Video paused');
        } else {
            // Play video
            if (this.currentDevice === 'drone') {
                this.visibleLightVideo.play();
            } else {
                this.infraredVideo.play();
            }
            icon.className = 'fas fa-pause';
            this.isPlaying = true;
            console.log('Video playing');
        }
    }

    cleanupResources() {
        // Close and nullify WebSocket connections
        if (this.wsVisible) {
            console.log('[DEBUG] Closing existing visible light WebSocket connection');
            // Remove all event listeners before closing
            this.wsVisible.onopen = null;
            this.wsVisible.onmessage = null;
            this.wsVisible.onerror = null;
            this.wsVisible.onclose = null;
            this.wsVisible.close();
            this.wsVisible = null;
        }
        if (this.wsInfrared) {
            console.log('[DEBUG] Closing existing infrared WebSocket connection');
            // Remove all event listeners before closing
            this.wsInfrared.onopen = null;
            this.wsInfrared.onmessage = null;
            this.wsInfrared.onerror = null;
            this.wsInfrared.onclose = null;
            this.wsInfrared.close();
            this.wsInfrared = null;
        }

        // Revoke all Blob URLs
        if (this.visibleLightVideo.lastUrl) {
            URL.revokeObjectURL(this.visibleLightVideo.lastUrl);
            this.visibleLightVideo.lastUrl = null;
        }
        if (this.infraredVideo.lastUrl) {
            URL.revokeObjectURL(this.infraredVideo.lastUrl);
            this.infraredVideo.lastUrl = null;
        }

        // Clear any pending timeouts
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        // Reset video elements
        this.visibleLightVideo.innerHTML = '';
        this.infraredVideo.innerHTML = '';

        // Reset status
        this.updateStatus('disconnected', '设备状态: 切换中...');

        // Add a small delay to ensure all resources are cleaned up
        return new Promise(resolve => setTimeout(resolve, 100));
    }
}

// Initialize the video stream handler when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.videoStreamHandler = new VideoStreamHandler();
}); 