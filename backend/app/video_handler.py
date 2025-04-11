import os
import json
import logging
import asyncio
import traceback
from datetime import datetime
from typing import Set, Dict, Optional, List, Tuple
import time

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
import base64

from app.smoke_removal import SmokeRemoval
from app.video_processor import VideoProcessor
from app.pose_processor import PoseProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoHandler:
    def __init__(self):
        # 存储活跃的WebSocket连接
        self.active_connections: Set[WebSocket] = set()
        
        # 存储前端请求类型
        self.client_requests: Dict[WebSocket, Dict[str, str]] = {}
        
        # 创建视频保存目录
        self.VIDEO_SAVE_DIR = "videos"
        os.makedirs(self.VIDEO_SAVE_DIR, exist_ok=True)
        
        # 视频类型映射
        self.VIDEO_TYPE_MAPPING = {
            "visible": "visible",
            "thermal": "thermal"
        }
        
        # 为每种视频类型创建子目录
        self.VIDEO_TYPES = {
            "drone_visible": "drone_visible",
            "drone_thermal": "drone_thermal",
            "robot_visible": "robot_visible",
            "robot_thermal": "robot_thermal"
        }
        
        for video_type in self.VIDEO_TYPES.values():
            os.makedirs(os.path.join(self.VIDEO_SAVE_DIR, video_type), exist_ok=True)
        
        # 存储每个连接的视频写入器
        self.video_writers: Dict[WebSocket, Dict[str, cv2.VideoWriter]] = {}
        
        # 视频片段时长（秒）
        self.SEGMENT_DURATION = 60  # 每60秒创建一个新的视频文件
        
        # 存储每个连接的开始时间
        self.start_times: Dict[WebSocket, Dict[str, datetime]] = {}
        
        # 存储每个连接的帧计数
        self.frame_counts: Dict[WebSocket, Dict[str, int]] = {}
        
        # 日志记录间隔（帧数）
        self.LOG_INTERVAL = 30  # 每30帧记录一次日志
        
        # 存储视频帧缓冲区
        self.frame_buffer: Dict[str, List[Tuple[datetime, np.ndarray]]] = {
            "drone_visible": [],
            "drone_thermal": [],
            "robot_visible": [],
            "robot_thermal": []
        }
        
        # 缓冲区最大大小
        self.MAX_BUFFER_SIZE = 300  # 存储10秒的30fps视频
        
        # 连接类型标识
        self.connection_types: Dict[WebSocket, str] = {}  # "source" 或 "client"
        
        # 消息接收锁
        self.receive_locks: Dict[WebSocket, asyncio.Lock] = {}
        
        # 连接状态
        self.connection_states: Dict[WebSocket, bool] = {}  # True 表示连接活跃
        
        self.enhancement_params = {
            'brightness': 100,
            'contrast': 100,
            'saturation': 100
        }
        self.video_processor = VideoProcessor()
        
        # 添加姿态处理器
        self.pose_processor = PoseProcessor(is_server=True)
    
    async def handle_websocket(self, websocket: WebSocket):
        """处理WebSocket连接和视频流"""
        try:
            await websocket.accept()
            self.active_connections.add(websocket)
            self.receive_locks[websocket] = asyncio.Lock()
            self.connection_states[websocket] = True
            logger.info(f"New WebSocket connection established from {websocket.client.host}")
            
            # 初始化该连接的视频写入器和开始时间
            self.video_writers[websocket] = {}
            self.start_times[websocket] = {}
            self.frame_counts[websocket] = {}
            self.client_requests[websocket] = {}
            
            # 发送欢迎消息
            await websocket.send_json({
                "status": "connected",
                "message": "Connected to video streaming server"
            })
            
            # 等待一小段时间，让客户端准备好
            await asyncio.sleep(0.5)
            
            # 检查连接类型
            connection_type = await self._determine_connection_type(websocket)
            self.connection_types[websocket] = connection_type
            logger.info(f"Connection type determined: {connection_type}")
            
            if connection_type == "source":
                # 处理视频源连接
                await self._handle_source_connection(websocket)
            else:
                # 处理客户端连接
                await self._handle_client_connection(websocket)
                
        except WebSocketDisconnect:
            logger.info(f"Client disconnected: {websocket.client.host}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # 标记连接为不活跃
            self.connection_states[websocket] = False
            
            # 关闭所有视频写入器
            self._close_video_writers(websocket)
            self.active_connections.remove(websocket)
            if websocket in self.client_requests:
                del self.client_requests[websocket]
            if websocket in self.connection_types:
                del self.connection_types[websocket]
            if websocket in self.receive_locks:
                del self.receive_locks[websocket]
            if websocket in self.connection_states:
                del self.connection_states[websocket]
            logger.info(f"WebSocket connection closed: {websocket.client.host}")
    
    async def _safe_receive(self, websocket: WebSocket) -> Optional[dict]:
        """安全地接收WebSocket消息"""
        if not self.connection_states.get(websocket, False):
            return None
            
        try:
            async with self.receive_locks[websocket]:
                try:
                    data = await websocket.receive()
                    return data
                except WebSocketDisconnect:
                    self.connection_states[websocket] = False
                    logger.info(f"WebSocket disconnected in _safe_receive: {websocket.client.host}")
                    return None
        except Exception as e:
            logger.error(f"Error receiving message: {str(e)}")
            return None
    
    async def _determine_connection_type(self, websocket: WebSocket) -> str:
        """确定连接类型（视频源或客户端）"""
        try:
            # 等待一小段时间，看是否收到视频数据
            try:
                # 设置超时
                data = await asyncio.wait_for(websocket.receive(), timeout=2.0)
                logger.info(f"Received initial data: {data}")
                
                if isinstance(data, dict) and "type" in data and data["type"] == "websocket.receive":
                    if "text" in data:
                        try:
                            # 解析第一层JSON
                            inner_data = json.loads(data["text"])
                            logger.info(f"Parsed inner data: {inner_data}")
                            
                            if isinstance(inner_data, dict) and "type" in inner_data and inner_data["type"] == "text" and "text" in inner_data:
                                # 解析第二层JSON
                                metadata = json.loads(inner_data["text"])
                                logger.info(f"Parsed metadata: {metadata}")
                                
                                if "device_type" in metadata and "video_type" in metadata:
                                    # 这是一个视频源连接
                                    logger.info("Identified as video source connection")
                                    # 将解析后的数据保存，供后续使用
                                    self.last_received_data = {
                                        "metadata": metadata,
                                        "raw_text": data["text"]
                                    }
                                    return "source"
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            pass
            except asyncio.TimeoutError:
                logger.info("Timeout waiting for initial data, assuming client connection")
                pass
            except Exception as e:
                logger.error(f"Error in connection type determination: {str(e)}")
                logger.error(traceback.format_exc())
            
            # 默认认为是客户端连接
            logger.info("Defaulting to client connection")
            return "client"
        except Exception as e:
            logger.error(f"Error determining connection type: {str(e)}")
            logger.error(traceback.format_exc())
            return "client"  # 默认认为是客户端连接
    
    def _map_video_type(self, device_type: str, video_type: str) -> str:
        """将配置中的视频类型映射到实际的目录名称"""
        mapped_type = self.VIDEO_TYPE_MAPPING.get(video_type, video_type)
        return f"{device_type}_{mapped_type}"
    
    async def _handle_source_connection(self, websocket: WebSocket):
        """处理视频源连接"""
        #logger.info(f"Handling video source connection from {websocket.client.host}")
        
        # 发送确认消息给视频源
        try:
            await websocket.send_json({
                "status": "connected",
                "message": "Video source connection established"
            })
            #logger.info("Sent confirmation message to video source")
            
            # 如果有之前接收到的数据，先处理它
            if hasattr(self, 'last_received_data'):
                metadata = self.last_received_data["metadata"]
                device_type = metadata.get("device_type")
                video_type = metadata.get("video_type")
                frame_size = metadata.get("frame_size")
                
                # 映射视频类型到实际的目录名称
                mapped_video_type = self._map_video_type(device_type, video_type)
                #logger.info(f"Processing initial data: device_type={device_type}, video_type={mapped_video_type}, frame_size={frame_size}")
                delattr(self, 'last_received_data')  # 使用后删除
        except Exception as e:
            logger.error(f"Error sending confirmation message: {str(e)}")
        
        # 等待视频源发送数据
        while self.connection_states.get(websocket, False):
            try:
                # 接收视频帧数据
                #logger.info("Waiting for video frame data...")
                frame_data = await websocket.receive()
                #logger.info(f"Received frame data type: {frame_data.get('type')}")
                
                if not isinstance(frame_data, dict) or "type" not in frame_data or frame_data["type"] != "websocket.receive":
                    #logger.error("Invalid frame data format")
                    continue
                    
                inner_frame_data = json.loads(frame_data["text"])
                if not isinstance(inner_frame_data, dict) or "type" not in inner_frame_data or inner_frame_data["type"] != "bytes":
                    #logger.error("Invalid inner frame data format")
                    continue
                    
                frame_bytes = inner_frame_data["bytes"]
                if isinstance(frame_bytes, str):
                    # 如果帧数据是字符串，尝试将其转换为字节
                    try:
                        # 使用 base64 解码
                        frame_bytes = base64.b64decode(frame_bytes)
                    except Exception as e:
                        logger.error(f"Error converting frame data to bytes: {str(e)}")
                        continue
                        
                actual_size = len(frame_bytes)
                #logger.info(f"Received video frame data: {actual_size} bytes")
                
                # 解码视频帧
                try:
                    # 将字节转换为numpy数组
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    #logger.info(f"Frame array shape: {frame_array.shape}, size: {frame_array.size}")
                    
                    if frame_array.size == 0:
                        logger.error("Empty frame array")
                        continue
                        
                    # 尝试不同的解码标志
                    decode_flags = [
                        cv2.IMREAD_COLOR,
                        cv2.IMREAD_UNCHANGED,
                        cv2.IMREAD_GRAYSCALE
                    ]
                    
                    frame = None
                    for flag in decode_flags:
                        frame = cv2.imdecode(frame_array, flag)
                        if frame is not None:
                            #logger.info(f"Successfully decoded frame with flag {flag}")
                            break
                            
                    if frame is None:
                        logger.error("Failed to decode video frame with all flags")
                        # 保存原始数据用于调试
                        debug_path = os.path.join(self.VIDEO_SAVE_DIR, "debug_frame.bin")
                        with open(debug_path, "wb") as f:
                            f.write(frame_bytes)
                        logger.info(f"Saved debug frame data to {debug_path}")
                        continue
                        
                    #logger.info(f"Successfully decoded frame with shape: {frame.shape}")
                    
                    # 获取或创建视频写入器
                    video_writer = self._get_or_create_video_writer(websocket, mapped_video_type, frame.shape)
                    
                    # 写入视频帧
                    if video_writer.isOpened():
                        video_writer.write(frame)
                        frame_count = self.frame_counts[websocket].get(mapped_video_type, 0) + 1
                        self.frame_counts[websocket][mapped_video_type] = frame_count
                        #logger.info(f"Successfully wrote frame {frame_count} to video file for {mapped_video_type}")
                        
                        # 将帧添加到缓冲区
                        self._add_frame_to_buffer(mapped_video_type, frame)
                        #logger.info(f"Added frame to buffer for {mapped_video_type}")
                    else:
                        logger.error(f"Video writer is not opened for {mapped_video_type}")
                        
                except Exception as e:
                    logger.error(f"Error processing video frame: {str(e)}")
                    logger.error(traceback.format_exc())
                    
            except WebSocketDisconnect:
                logger.info(f"Video source disconnected: {websocket.client.host}")
                # 通知所有正在观看该视频流的客户端
                await self._notify_clients_source_disconnected(mapped_video_type)
                break
            except Exception as e:
                logger.error(f"Error in video source connection: {str(e)}")
                logger.error(traceback.format_exc())
                # 通知所有正在观看该视频流的客户端
                await self._notify_clients_source_disconnected(mapped_video_type)
                break
    
    async def _notify_clients_source_disconnected(self, video_type: str):
        """通知所有正在观看指定视频流的客户端视频源已断开"""
        logger.info(f"Notifying clients about source disconnection for {video_type}")
        disconnected_message = {
            "status": "error",
            "message": f"Video source disconnected for {video_type}",
            "video_type": video_type
        }
        
        # 遍历所有活跃的客户端连接
        for client in self.active_connections.copy():
            if not self.connection_states.get(client, False):
                continue
                
            try:
                # 检查客户端是否正在观看该视频流
                if client in self.client_requests and video_type in self.client_requests[client]:
                    await client.send_json(disconnected_message)
                    logger.info(f"Notified client {client.client.host} about source disconnection")
            except Exception as e:
                logger.error(f"Error notifying client about source disconnection: {str(e)}")
                # 如果通知失败，可能是客户端已断开，移除连接
                self.connection_states[client] = False
                self.active_connections.remove(client)
                if client in self.client_requests:
                    del self.client_requests[client]
    
    async def _handle_client_connection(self, websocket: WebSocket):
        """处理客户端连接"""
        logger.info(f"Handling client connection from {websocket.client.host}")
        
        # 发送可用的视频类型列表
        try:
            available_types = list(self.VIDEO_TYPES.keys())
            logger.info(f"Sending available video types to client: {available_types}")
            await websocket.send_json({
                "status": "info",
                "message": "Available video types",
                "video_types": available_types
            })
        except Exception as e:
            logger.error(f"Error sending video types: {str(e)}")
            return
        
        # 设置初始通知时间
        last_notification_time = time.time()
        notification_interval = 5  # 通知间隔（秒）
        
        # 等待客户端请求
        while self.connection_states.get(websocket, False):
            try:
                # 接收客户端请求，设置超时
                try:
                    data = await asyncio.wait_for(self._safe_receive(websocket), timeout=10.0)
                except asyncio.TimeoutError:
                    # 如果超时，发送心跳消息保持连接
                    try:
                        await websocket.send_json({
                            "status": "heartbeat",
                            "message": "Connection alive"
                        })
                        continue
                    except Exception as e:
                        logger.error(f"Error sending heartbeat: {str(e)}")
                        break
                
                if data is None:  # 连接已断开
                    logger.warning("Client connection lost")
                    break
                    
                # 检查消息格式
                if not isinstance(data, dict):
                    logger.warning(f"Received invalid data type: {type(data)}")
                    continue
                    
                # 处理WebSocket消息
                if data.get("type") == "websocket.receive" and "text" in data:
                    try:
                        request = json.loads(data["text"])
                        device_type = request.get("device_type")
                        video_type = request.get("video_type")
                        process_type = request.get("process_type", "original")  # 默认为原始视频
                        continuous = request.get("continuous", True)  # 默认为持续发送
                        frame_rate = request.get("frame_rate", 30)  # 默认帧率
                        
                        logger.info(f"Received client request: device_type={device_type}, video_type={video_type}, process_type={process_type}, continuous={continuous}, frame_rate={frame_rate}")
                        
                        # 验证请求参数
                        if not device_type or not video_type:
                            error_msg = "Missing required parameters: device_type and video_type are required"
                            logger.error(error_msg)
                            await websocket.send_json({
                                "status": "error",
                                "message": error_msg
                            })
                            continue
                        
                        # 映射视频类型到实际的目录名称
                        mapped_video_type = self._map_video_type(device_type, video_type)
                        
                        # 发送成功响应
                        await websocket.send_json({
                            "status": "success",
                            "message": f"Request processed: {mapped_video_type} - {process_type}"
                        })
                        
                        # 开始发送视频流
                        while continuous and self.connection_states.get(websocket, False):
                            try:
                                # 检查是否有缓冲的帧
                                if not self.frame_buffer[mapped_video_type]:
                                    current_time = time.time()
                                    if current_time - last_notification_time >= notification_interval:
                                        warning_msg = f"No video stream available for {mapped_video_type}"
                                        logger.warning(warning_msg)
                                        await websocket.send_json({
                                            "status": "warning",
                                            "message": warning_msg,
                                            "video_type": mapped_video_type,
                                            "timestamp": current_time
                                        })
                                        last_notification_time = current_time
                                    await asyncio.sleep(1)  # 等待1秒后重试
                                    continue
                                
                                # 获取最新的帧
                                latest_frame = self.frame_buffer[mapped_video_type][-1][1]
                                
                                # 根据处理类型处理帧
                                if process_type == "smoke_removal":
                                    processed_frame = SmokeRemoval.process_frame(latest_frame)
                                elif process_type == "enhancement":
                                    processed_frame = self.apply_image_enhancement(latest_frame)
                                elif process_type == "pose_detection":
                                    try:
                                        logger.info("开始行为识别处理")
                                        result = self.pose_processor.process_frame(latest_frame)
                                        if 'error' not in result:
                                            logger.info(f"行为识别处理成功，检测到 {len(result['detections'])} 个目标")
                                            for det in result['detections']:
                                                logger.info(f"检测到行为: {det['label']}, 置信度: {det['confidence']:.2f}")
                                            img_data = base64.b64decode(result['draw_frame'])
                                            nparr = np.frombuffer(img_data, np.uint8)
                                            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                            if processed_frame is None:
                                                logger.error("行为识别结果解码失败")
                                                processed_frame = latest_frame
                                        else:
                                            logger.error(f"行为识别处理失败: {result['error']}")
                                            processed_frame = latest_frame
                                    except Exception as e:
                                        logger.error(f"行为识别处理异常: {str(e)}")
                                        logger.error(traceback.format_exc())
                                        processed_frame = latest_frame
                                else:
                                    processed_frame = latest_frame
                                
                                # 编码处理后的帧为JPEG格式
                                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                
                                # 发送帧数据
                                await websocket.send_bytes(buffer.tobytes())
                                
                                # 控制帧率
                                await asyncio.sleep(1.0 / frame_rate)
                                
                            except Exception as e:
                                error_msg = f"Error processing frame: {str(e)}"
                                logger.error(error_msg)
                                logger.error(traceback.format_exc())
                                await websocket.send_json({
                                    "status": "error",
                                    "message": error_msg
                                })
                                break
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)}")
                        continue
                    
                elif data.get("type") == "enhancement":
                    # 更新增强参数
                    params = data.get("parameters", {})
                    self.enhancement_params.update(params)
                    logger.info(f"Updated enhancement parameters: {self.enhancement_params}")
                    # 发送确认消息
                    await websocket.send_json({
                        "status": "success",
                        "message": f"Enhancement parameters updated: {self.enhancement_params}"
                    })
                    continue
                    
            except Exception as e:
                error_msg = f"Error handling client connection: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                await websocket.send_json({
                    "status": "error",
                    "message": error_msg
                })
                break
    
    async def _process_and_broadcast_frame(self, websocket: WebSocket, video_type_dir: str, 
                                          frame: np.ndarray, metadata_str: str, frame_data: bytes):
        """根据客户端请求处理并广播帧"""
        # 检查是否有客户端请求此视频类型
        for client in self.active_connections.copy():
            if client == websocket or not self.connection_states.get(client, False):  # 跳过发送者和不活跃的连接
                continue
                
            if client in self.client_requests and video_type_dir in self.client_requests[client]:
                process_type = self.client_requests[client][video_type_dir]
                
                try:
                    if process_type == "original":
                        # 直接发送原始帧
                        await client.send_text(metadata_str)
                        await client.send_bytes(frame_data)
                    elif process_type == "smoke_removal":
                        # 应用去雾处理
                        processed_frame = SmokeRemoval.process_frame(frame)
                        
                        # 编码处理后的帧
                        _, buffer = cv2.imencode('.jpg', processed_frame)
                        processed_data = buffer.tobytes()
                        
                        # 更新元数据中的帧大小
                        metadata = json.loads(metadata_str)
                        metadata["frame_size"] = len(processed_data)
                        updated_metadata = json.dumps(metadata)
                        
                        # 发送处理后的帧
                        await client.send_text(updated_metadata)
                        await client.send_bytes(processed_data)
                    elif process_type == "enhancement":
                        # 应用图像增强
                        processed_frame = self.apply_image_enhancement(frame)
                        
                        # 编码处理后的帧
                        _, buffer = cv2.imencode('.jpg', processed_frame)
                        processed_data = buffer.tobytes()
                        
                        # 更新元数据中的帧大小
                        metadata = json.loads(metadata_str)
                        metadata["frame_size"] = len(processed_data)
                        updated_metadata = json.dumps(metadata)
                        
                        # 发送处理后的帧
                        await client.send_text(updated_metadata)
                        await client.send_bytes(processed_data)
                    else:
                        # 未知的处理类型，发送原始帧
                        await client.send_text(metadata_str)
                        await client.send_bytes(frame_data)
                        
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {str(e)}")
                    self.connection_states[client] = False
                    self.active_connections.remove(client)
    
    def _add_frame_to_buffer(self, video_type_dir: str, frame: np.ndarray):
        """将帧添加到缓冲区"""
        # 添加时间戳和帧
        self.frame_buffer[video_type_dir].append((datetime.now(), frame))
        
        # 如果缓冲区超过最大大小，移除最旧的帧
        if len(self.frame_buffer[video_type_dir]) > self.MAX_BUFFER_SIZE:
            self.frame_buffer[video_type_dir].pop(0)
    
    def _get_or_create_video_writer(self, websocket: WebSocket, video_type_dir: str, frame_shape) -> cv2.VideoWriter:
        """获取或创建视频写入器"""
        if video_type_dir not in self.video_writers[websocket]:
            try:
                # 创建新的视频写入器
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 确保目录存在
                save_dir = os.path.join(self.VIDEO_SAVE_DIR, video_type_dir)
                os.makedirs(save_dir, exist_ok=True)
                logger.info(f"Ensuring directory exists: {save_dir}")
                
                # 获取帧的高度和宽度
                height, width = frame_shape[:2]
                logger.info(f"Frame dimensions: {width}x{height}")
                
                # 创建视频写入器
                save_path = os.path.join(save_dir, f"{timestamp}.avi")
                logger.info(f"Creating video writer at: {save_path}")
                
                # 使用MJPG编码器，这是最通用的选择
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
                
                if not video_writer.isOpened():
                    logger.error("Failed to create video writer")
                    raise Exception("Failed to create video writer")
                
                logger.info("Video writer created successfully")
                self.video_writers[websocket][video_type_dir] = video_writer
                self.start_times[websocket][video_type_dir] = datetime.now()
                
                # 验证文件是否被创建
                if not os.path.exists(save_path):
                    logger.error(f"Video file was not created at {save_path}")
                    raise Exception("Video file was not created")
                    
                logger.info(f"Video file created successfully at {save_path}")
                
            except Exception as e:
                logger.error(f"Error creating video writer: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        return self.video_writers[websocket][video_type_dir]
    
    def _check_segment_rotation(self, websocket: WebSocket, video_type_dir: str):
        """检查是否需要创建新的视频片段"""
        if video_type_dir in self.start_times[websocket]:
            start_time = self.start_times[websocket][video_type_dir]
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            
            if elapsed_seconds >= self.SEGMENT_DURATION:
                try:
                    # 关闭当前视频写入器
                    if video_type_dir in self.video_writers[websocket]:
                        writer = self.video_writers[websocket][video_type_dir]
                        if writer.isOpened():
                            writer.release()
                            logger.info(f"Closed video segment for {video_type_dir}")
                            
                            # 记录总帧数
                            total_frames = self.frame_counts[websocket].get(video_type_dir, 0)
                            logger.info(f"Total frames processed for {video_type_dir}: {total_frames}")
                        
                        # 删除写入器引用
                        del self.video_writers[websocket][video_type_dir]
                    
                    # 重置开始时间
                    self.start_times[websocket][video_type_dir] = current_time
                    
                except Exception as e:
                    logger.error(f"Error rotating video segment: {str(e)}")
                    logger.error(traceback.format_exc())
    
    def _close_video_writers(self, websocket: WebSocket):
        """关闭所有视频写入器"""
        if websocket in self.video_writers:
            try:
                for video_type_dir, writer in self.video_writers[websocket].items():
                    if writer.isOpened():
                        writer.release()
                        logger.info(f"Closed video writer for {video_type_dir}")
                        
                        # 记录总帧数
                        total_frames = self.frame_counts[websocket].get(video_type_dir, 0)
                        logger.info(f"Total frames processed for {video_type_dir}: {total_frames}")
                
                # 清理资源
                del self.video_writers[websocket]
                if websocket in self.start_times:
                    del self.start_times[websocket]
                if websocket in self.frame_counts:
                    del self.frame_counts[websocket]
                    
            except Exception as e:
                logger.error(f"Error closing video writers: {str(e)}")
                logger.error(traceback.format_exc())
    
    def apply_image_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """应用图像增强效果"""
        try:
            return self.video_processor.adjust_frame(
                frame,
                brightness=self.enhancement_params['brightness'],
                contrast=self.enhancement_params['contrast'] / 100.0,
                saturation=self.enhancement_params['saturation'] / 100.0
            )
        except Exception as e:
            logger.error(f"Error applying image enhancement: {str(e)}")
            return frame 