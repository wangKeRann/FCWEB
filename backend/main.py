from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import numpy as np
from pydantic import BaseModel
from typing import Optional, List
import base64
import cv2

from app.video_handler import VideoHandler
from app.video_processor import VideoProcessor
from app.multimodal_processor import MultiModalProcessor
from app.pose_processor import PoseProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建处理器实例
video_handler = VideoHandler()
multimodal_processor = MultiModalProcessor(is_server=True)
pose_processor = PoseProcessor()  # 使用默认参数

class VideoAdjustRequest(BaseModel):
    visible_path: str
    infrared_path: str
    output_path: str
    brightness: Optional[int] = 0
    contrast: Optional[float] = 1.0
    saturation: Optional[float] = 1.0

class MultimodalDetectionRequest(BaseModel):
    rgb_path: str
    ir_path: str
    file_type: str = "image"  # 默认为图片处理

class PoseFrameRequest(BaseModel):
    rgb_frame: str  # base64编码的RGB图像
    ir_frame: str   # base64编码的红外图像

def base64_to_cv2(base64_string: str) -> np.ndarray:
    """将base64编码的图像转换为OpenCV格式"""
    try:
        # 解码base64
        img_data = base64.b64decode(base64_string)
        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        # 解码图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"图像解码失败: {str(e)}")
        raise HTTPException(status_code=400, detail="图像解码失败")

@app.post("/api/adjust_video")
async def api_adjust_video(request: VideoAdjustRequest):
    """
    视频增强API（支持可见光和红外视频）
    """
    if not os.path.exists(request.visible_path):
        raise HTTPException(status_code=400, detail="可见光视频文件不存在")
    if not os.path.exists(request.infrared_path):
        raise HTTPException(status_code=400, detail="红外视频文件不存在")
    
    try:
        result = VideoProcessor.process_videos(
            request.visible_path,
            request.infrared_path,
            request.output_path,
            request.brightness,
            request.contrast,
            request.saturation
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/multimodal_detection")
async def api_multimodal_detection(request: MultimodalDetectionRequest):
    """
    多模态检测API
    
    处理RGB和红外图像/视频的目标检测请求
    """
    if not os.path.exists(request.rgb_path):
        raise HTTPException(status_code=400, detail="RGB图像/视频文件不存在")
    if not os.path.exists(request.ir_path):
        raise HTTPException(status_code=400, detail="红外图像/视频文件不存在")
    
    try:
        if request.file_type == "image":
            result = multimodal_processor.process_image(request.rgb_path, request.ir_path)
        else:
            output_path = os.path.join(multimodal_processor.output_dir, "output.mp4")
            result = multimodal_processor.process_video(request.rgb_path, request.ir_path, output_path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "code": 1,
            "message": "处理成功",
            "data": result
        }
    except Exception as e:
        logger.error(f"多模态检测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pose_detection")
async def api_pose_detection(request: PoseFrameRequest):
    """
    姿态检测API
    
    处理RGB和红外图像帧的目标检测请求
    """
    try:
        # 解码图像
        rgb_frame = base64_to_cv2(request.rgb_frame)
        ir_frame = base64_to_cv2(request.ir_frame)
        
        # 处理图像帧
        result = pose_processor.process_frames(rgb_frame, ir_frame)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
            
        return {
            'code': 1,
            'message': '处理成功',
            'data': result
        }
        
    except Exception as e:
        logger.error(f"姿态检测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Video streaming server is running"}

@app.websocket("/ws/video")
async def video_source_endpoint(websocket: WebSocket):
    """处理来自视频源（无人机/机械狗）的WebSocket连接"""
    logger.info(f"New video source connection from {websocket.client.host}")
    try:
        await video_handler.handle_websocket(websocket)
    except WebSocketDisconnect:
        logger.info(f"Video source disconnected: {websocket.client.host}")
    except Exception as e:
        logger.error(f"Error in video source connection: {str(e)}")

@app.websocket("/ws/client")
async def client_endpoint(websocket: WebSocket):
    """处理来自前端的WebSocket连接"""
    logger.info(f"New client connection from {websocket.client.host}")
    try:
        await video_handler.handle_websocket(websocket)
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client.host}")
    except Exception as e:
        logger.error(f"Error in client connection: {str(e)}")

@app.websocket("/ws/pose_detection")
async def pose_detection_endpoint(websocket: WebSocket):
    """处理姿势识别请求的WebSocket连接"""
    logger.info(f"New pose detection connection from {websocket.client.host}")
    try:
        await websocket.accept()
        while True:
            # 接收前端发送的数据
            data = await websocket.receive_json()
            
            # 处理姿势识别请求
            if data.get("type") == "pose_detection":
                try:
                    # 获取RGB和红外图像数据
                    rgb_frame = base64_to_cv2(data.get("rgb_frame"))
                    ir_frame = base64_to_cv2(data.get("ir_frame"))
                    
                    # 调用姿势处理器
                    result = pose_processor.process_frames(rgb_frame, ir_frame)
                    
                    # 发送处理结果回前端
                    await websocket.send_json({
                        "status": "success",
                        "data": result
                    })
                except Exception as e:
                    logger.error(f"姿势识别处理失败: {str(e)}")
                    await websocket.send_json({
                        "status": "error",
                        "message": str(e)
                    })
    except WebSocketDisconnect:
        logger.info(f"Pose detection client disconnected: {websocket.client.host}")
    except Exception as e:
        logger.error(f"Error in pose detection connection: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting video streaming server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 