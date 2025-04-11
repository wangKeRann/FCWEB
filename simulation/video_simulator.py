import asyncio
import websockets
import cv2
import json
import base64
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoSimulator:
    def __init__(self, video_path: str, device_type: str, video_type: str, server_url: str = "ws://localhost:8000/ws/video"):
        """
        Initialize the video simulator.
        
        Args:
            video_path: Path to the video file
            device_type: Type of device ('drone' or 'robot')
            video_type: Type of video ('regular' or 'infrared')
            server_url: WebSocket server URL
        """
        self.video_path = video_path
        self.device_type = device_type
        self.video_type = video_type
        self.server_url = server_url
        self.cap = None
        self.is_running = False
        self.max_retries = 5  # 最大重试次数
        self.retry_delay = 5  # 重试间隔（秒）
        
    async def connect(self):
        """Establish WebSocket connection with the server with retry mechanism."""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                self.ws = await websockets.connect(self.server_url)
                logger.info(f"Connected to {self.server_url}")
                return True
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(f"Connection attempt {retry_count} failed: {e}")
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to connect after {self.max_retries} attempts: {e}")
                    return False
        
    async def reconnect(self):
        """Attempt to reconnect to the server."""
        logger.info("Attempting to reconnect...")
        if await self.connect():
            logger.info("Reconnected successfully")
            return True
        return False
        
    def start_video_capture(self):
        """Start video capture from the file."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        self.is_running = True
        
    def stop_video_capture(self):
        """Stop video capture and release resources."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def get_frame(self) -> Optional[tuple]:
        """Get a frame from the video capture."""
        if not self.is_running or not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            # Video ended, restart from beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        return ret, frame
        
    def encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode frame to JPEG bytes."""
        try:
            # 设置JPEG编码参数
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # JPEG质量设置为80
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not success:
                logger.error("Failed to encode frame to JPEG")
                return None
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return None
        
    async def send_frame(self, frame: np.ndarray):
        """Send frame data through WebSocket."""
        try:
            # 编码帧为JPEG字节
            logger.debug("Encoding frame to JPEG...")
            frame_bytes = self.encode_frame(frame)
            if frame_bytes is None:
                logger.error("Failed to encode frame, skipping...")
                return
                
            frame_size = len(frame_bytes)
            logger.debug(f"Frame encoded successfully, size: {frame_size} bytes")
            
            # 准备元数据
            metadata = {
                "device_type": self.device_type,
                "video_type": self.video_type,
                "frame_size": frame_size
            }
            
            # 发送元数据
            logger.debug(f"Sending metadata: {metadata}")
            await self.ws.send(json.dumps({
                "type": "text",
                "text": json.dumps(metadata)
            }))
            
            # 发送帧数据
            logger.debug("Sending frame data...")
            await self.ws.send(json.dumps({
                "type": "bytes",
                "bytes": base64.b64encode(frame_bytes).decode('utf-8')
            }))
            logger.debug("Frame data sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            # 如果发送失败，尝试重新连接
            if await self.reconnect():
                logger.info("Reconnected successfully, retrying frame send...")
                try:
                    # 重新发送元数据
                    await self.ws.send(json.dumps({
                        "type": "text",
                        "text": json.dumps(metadata)
                    }))
                    # 重新发送帧数据
                    await self.ws.send(json.dumps({
                        "type": "bytes",
                        "bytes": base64.b64encode(frame_bytes).decode('utf-8')
                    }))
                    logger.info("Frame sent successfully after reconnection")
                except Exception as e:
                    logger.error(f"Failed to send frame after reconnection: {e}")
            
    async def run(self):
        """Main loop for video streaming."""
        while True:  # 外层循环确保程序持续运行
            if not await self.connect():
                logger.warning("Initial connection failed, waiting before retry...")
                await asyncio.sleep(self.retry_delay)
                continue
                
            try:
                self.start_video_capture()
                logger.info(f"Starting {self.video_type} video stream for {self.device_type}")
                
                frame_count = 0
                start_time = time.time()
                
                while self.is_running:
                    ret, frame = self.get_frame()
                    if ret:
                        frame_count += 1
                        await self.send_frame(frame)
                        
                        # 计算并显示FPS
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 1.0:  # 每秒更新一次FPS
                            fps = frame_count / elapsed_time
                            logger.info(f"Current FPS: {fps:.2f}")
                            frame_count = 0
                            start_time = time.time()
                            
                        # Control frame rate (adjust as needed)
                        await asyncio.sleep(0.033)  # ~30 FPS
                        
            except Exception as e:
                logger.error(f"Error in video streaming: {e}")
            finally:
                self.stop_video_capture()
                try:
                    await self.ws.close()
                except:
                    pass
                
            # 如果发生错误，等待一段时间后重试
            logger.info("Stream ended, attempting to restart...")
            await asyncio.sleep(self.retry_delay)
            
async def main():
    # 定义所有可用的视频流配置
    all_video_configs = [
        {
            "device_type": "drone",
            "video_type": "visible",
            "video_path": "video/drone_video.mp4",
            "enabled": True  # 默认启用无人机普通视频
        },
        {
            "device_type": "drone",
            "video_type": "thermal",
            "video_path": "video/drone_infrared.mp4",
            "enabled": True  # 默认启用无人机热成像视频
        },
        {
            "device_type": "robot",
            "video_type": "visible",
            "video_path": "video/dog_video.mp4",
            "enabled": True  # 默认启用机械狗普通视频
        },
        {
            "device_type": "robot",
            "video_type": "thermal",
            "video_path": "video/dog_infrared.mp4",
            "enabled": True  # 默认启用机械狗热成像视频
        }
    ]
    
    # 只选择启用的视频流
    enabled_configs = [config for config in all_video_configs if config["enabled"]]
    
    if not enabled_configs:
        logger.warning("No video streams enabled. Please enable at least one video stream in the configuration.")
        return
    
    # Create and run simulators for enabled video streams
    tasks = []
    for config in enabled_configs:
        if Path(config["video_path"]).exists():
            simulator = VideoSimulator(
                video_path=config["video_path"],
                device_type=config["device_type"],
                video_type=config["video_type"]
            )
            tasks.append(asyncio.create_task(simulator.run()))
            logger.info(f"Starting {config['video_type']} video stream for {config['device_type']}")
        else:
            logger.warning(f"Video file not found: {config['video_path']}")
            
    if tasks:
        await asyncio.gather(*tasks)
    else:
        logger.error("No valid video files found to simulate")

if __name__ == "__main__":
    asyncio.run(main()) 