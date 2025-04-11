import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch

logger = logging.getLogger(__name__)

class MultiModalProcessor:
    """多模态处理器类，用于处理RGB和红外图像/视频的目标检测"""
    
    def __init__(self, model_path: str = '../multi.pt',
                 output_dir: str = None):
        """
        初始化多模态处理器
        
        Args:
            model_path: YOLO模型路径
            output_dir: 输出目录
        """
        try:
            self.model = YOLO(model_path)
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise
        
    @staticmethod
    def convert_to_jpg(input_path: str, quality: int = 90) -> str:
        """
        将图片转换为JPG格式
        
        Args:
            input_path: 输入图片路径
            quality: JPG质量，范围0-100
            
        Returns:
            转换后的图片路径
        """
        try:
            endname = os.path.splitext(input_path)[-1].lower()
            if endname in ('.png', '.bmp', '.webp'):
                output_path = os.path.splitext(input_path)[0] + '.jpg'
                with Image.open(input_path) as img:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(output_path, 'JPEG', quality=quality)
                return output_path
            return input_path
        except Exception as e:
            logger.error(f"图片转换失败: {str(e)}")
            return input_path

    def process_image(self, rgb_path: str, ir_path: str) -> dict:
        """
        处理图片文件
        
        Args:
            rgb_path: RGB图片路径
            ir_path: 红外图片路径
            
        Returns:
            处理结果字典
        """
        try:
            # 转换图片格式
            rgb_path = self.convert_to_jpg(rgb_path)
            ir_path = self.convert_to_jpg(ir_path)

            # 进行目标检测
            results = self.model(source=[rgb_path, ir_path], save=True, 
                              project=self.output_dir, name="image_detection")
            
            # 处理检测结果
            detections = self._process_detections(results)
            
            # 获取输出图片路径
            output_rgb = os.path.join(self.output_dir, "image_detection", os.path.basename(rgb_path))
            output_ir = os.path.join(self.output_dir, "image_detection", os.path.basename(ir_path))

            return {
                "message": "图片处理完成",
                "detections": detections,
                "output_rgb": output_rgb,
                "output_ir": output_ir
            }

        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            return {"error": str(e)}

    def process_video(self, rgb_path: str, ir_path: str, output_path: str) -> dict:
        """
        处理视频文件
        
        Args:
            rgb_path: RGB视频路径
            ir_path: 红外视频路径
            output_path: 输出视频路径
            
        Returns:
            处理结果字典
        """
        try:
            # 打开输入视频
            rgb_cap = cv2.VideoCapture(rgb_path)
            ir_cap = cv2.VideoCapture(ir_path)
            
            if not rgb_cap.isOpened() or not ir_cap.isOpened():
                return {"error": "无法打开视频文件"}

            # 获取视频参数
            fps = rgb_cap.get(cv2.CAP_PROP_FPS)
            width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

            frame_count = 0
            detection_results = []

            while True:
                ret1, rgb_frame = rgb_cap.read()
                ret2, ir_frame = ir_cap.read()
                
                if not ret1 or not ret2:
                    break

                # 处理每一帧
                results = self.model(source=[rgb_frame, ir_frame], save=True, 
                                  project=self.output_dir, name="video_detection")
                
                # 获取检测结果
                detections = self._process_detections(results)
                detection_results.append(detections)

                # 将两帧图像水平拼接
                combined_frame = np.hstack((rgb_frame, ir_frame))
                out.write(combined_frame)
                
                frame_count += 1

            # 释放资源
            rgb_cap.release()
            ir_cap.release()
            out.release()

            return {
                "message": "视频处理完成",
                "frame_count": frame_count,
                "detections": detection_results,
                "output_path": output_path
            }

        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}")
            return {"error": str(e)}

    def _process_detections(self, results) -> dict:
        """
        处理检测结果
        
        Args:
            results: YOLO检测结果
            
        Returns:
            检测统计字典
        """
        try:
            counts = {'p1': 0}  # 根据你的类别修改
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    counts['p1'] += 1  # 根据你的类别修改
            return counts
        except Exception as e:
            logger.error(f"检测结果处理失败: {str(e)}")
            return {'p1': 0} 