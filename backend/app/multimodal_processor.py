import os
import logging
import cv2
import numpy as np
from PIL import Image
import base64
from typing import Dict, List, Tuple, Optional
import time

logger = logging.getLogger(__name__)

class MultiModalProcessor:
    """多模态处理器类，用于处理RGB和红外图像的目标检测"""
    
    def __init__(self, is_server: bool = False):
        """
        初始化多模态处理器
        
        Args:
            is_server: 是否在服务器环境运行
        """
        self.is_server = is_server
        self.model = None
        self.last_process_time = 0
        self.frame_cache: Dict[str, np.ndarray] = {}
        self.cache_size = 5  # 缓存最近5帧
        
        if is_server:
            try:
                from ultralytics import YOLO
                self.model = YOLO('/home/anonym4/wkr/FCWEB2/backend/multi.pt')
                self.model.conf = 0.25  # 降低置信度阈值
                logger.info("YOLOv8模型加载成功")
            except ImportError:
                logger.warning("服务器环境未安装 ultralytics 包")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise

    def preprocess_frame(self, frame: np.ndarray, is_ir: bool = False) -> np.ndarray:
        """
        预处理图像帧
        
        Args:
            frame: 输入图像帧
            is_ir: 是否为红外图像
            
        Returns:
            预处理后的图像帧
        """
        try:
            # 图像归一化
            frame = frame.astype(np.float32) / 255.0
            
            if is_ir:
                # 红外图像增强
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
                # 确保图像是单通道的8位无符号整数类型
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                else:
                    frame = frame.astype(np.uint8)
                frame = cv2.equalizeHist(frame)
            else:
                # RGB图像增强
                frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
            
            return frame
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            return frame

    def fuse_detections(self, rgb_detections: List[Dict], ir_detections: List[Dict], 
                       iou_threshold: float = 0.5) -> List[Dict]:
        """
        融合RGB和红外检测结果
        
        Args:
            rgb_detections: RGB检测结果
            ir_detections: 红外检测结果
            iou_threshold: IOU阈值
            
        Returns:
            融合后的检测结果
        """
        fused_detections = []
        
        # 计算IOU
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            return intersection / float(box1_area + box2_area - intersection)
        
        # 融合检测结果
        for rgb_det in rgb_detections:
            best_iou = 0
            best_ir_det = None
            
            for ir_det in ir_detections:
                iou = calculate_iou(rgb_det['bbox'], ir_det['bbox'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_ir_det = ir_det
            
            if best_ir_det:
                # 融合置信度
                fused_conf = (rgb_det['confidence'] + best_ir_det['confidence']) / 2
                fused_detections.append({
                    'bbox': rgb_det['bbox'],
                    'confidence': fused_conf,
                    'class_id': rgb_det['class_id'],
                    'rgb_confidence': rgb_det['confidence'],
                    'ir_confidence': best_ir_det['confidence']
                })
            else:
                fused_detections.append(rgb_det)
        
        return fused_detections

    def process_frame(self, frame: np.ndarray, is_ir: bool = False) -> dict:
        """
        处理单帧图像，只识别人物类别
        
        Args:
            frame: 输入图像帧 (BGR格式)
            is_ir: 是否为红外图像
            
        Returns:
            处理结果字典，包含检测框信息
        """
        try:
            # 确保图像是BGR格式
            if len(frame.shape) == 2:  # 如果是灰度图
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # 创建原始图像的副本用于绘制
            draw_frame = frame.copy()
            
            # 图像预处理
            processed_frame = self.preprocess_frame(frame, is_ir)
            
            if self.is_server and self.model is not None:
                logger.info("YOLO模型已加载")
                logger.info(f"输入图像尺寸: {processed_frame.shape}")
                logger.info(f"输入图像类型: {'红外' if is_ir else 'RGB'}")
                
                # 如果是红外图像，进行额外的预处理
                if is_ir:
                    # 转换为灰度图
                    if len(processed_frame.shape) == 3:
                        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    # 直方图均衡化
                    processed_frame = cv2.equalizeHist(processed_frame)
                    # 转换为3通道
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                
                # 临时降低置信度阈值
                original_conf = self.model.conf
                self.model.conf = 0.15  # 降低置信度阈值以提高检测率
                
                # 服务器环境使用YOLO模型
                results = self.model(processed_frame)
                logger.info(f"YOLO处理完成，结果类型: {type(results)}")
                
                # 恢复原始置信度阈值
                self.model.conf = original_conf
                
                # 处理检测结果
                detections = []
                for result in results:
                    boxes = result.boxes
                    logger.info(f"检测框数量: {len(boxes)}")
                    logger.info(f"置信度阈值: {self.model.conf}")
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 只处理人物类别（类别ID为0）
                        if cls == 0:
                            logger.info(f"检测到人物: 置信度={conf}")
                            
                            # 在图像上绘制检测框
                            color = (0, 255, 0) if not is_ir else (0, 0, 255)  # RGB用绿色，红外用红色
                            thickness = 2
                            
                            # 确保坐标是整数
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 绘制矩形
                            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # 添加类别和置信度标签
                            label_text = f"Person {conf:.2f}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            font_thickness = 2
                            
                            # 获取文本大小
                            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                            
                            # 绘制文本背景
                            cv2.rectangle(draw_frame, 
                                        (x1, y1 - text_height - 10),
                                        (x1 + text_width, y1),
                                        color, -1)
                            
                            # 绘制文本
                            cv2.putText(draw_frame, label_text,
                                      (x1, y1 - 5),
                                      font, font_scale,
                                      (255, 255, 255),  # 白色文本
                                      font_thickness)
                            
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class_id': cls,
                                'class_name': 'person'
                            })
            else:
                logger.warning("YOLO模型未加载")
                return {'error': 'YOLO模型未加载'}
            
            # 将绘制后的图像转换为base64
            _, buffer = cv2.imencode('.jpg', draw_frame)
            draw_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'detections': detections,
                'frame_shape': frame.shape,
                'draw_frame': draw_frame_base64
            }
            
        except Exception as e:
            logger.error(f"帧处理失败: {str(e)}")
            return {'error': str(e)}

    def process_frames(self, rgb_frame: np.ndarray, ir_frame: np.ndarray) -> dict:
        """
        处理RGB和红外图像帧，支持流式处理
        
        Args:
            rgb_frame: RGB图像帧 (BGR格式)
            ir_frame: 红外图像帧 (BGR格式)
            
        Returns:
            处理结果字典，包含检测结果图片的base64编码和检测框信息
        """
        try:
            # 检查处理间隔，调整为约5fps
            current_time = time.time()
            if current_time - self.last_process_time < 0.2:  # 约5fps
                return {'code': 2, 'message': '处理频率过高，跳过当前帧'}
            self.last_process_time = current_time
            
            if self.is_server and self.model is not None:
                # 确保输入图像是3通道的BGR格式
                if len(rgb_frame.shape) == 2:  # 如果是灰度图
                    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
                if len(ir_frame.shape) == 2:  # 如果是灰度图
                    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                elif len(ir_frame.shape) == 3 and ir_frame.shape[2] == 1:  # 如果是单通道彩色图
                    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                
                # 创建结果保存目录
                result_dir = os.path.join("/tmp/fcweb2", "detection")
                os.makedirs(result_dir, exist_ok=True)
                
                # 将图像保存为临时文件
                temp_dir = "/tmp/fcweb2"
                os.makedirs(temp_dir, exist_ok=True)
                
                rgb_path = os.path.join(temp_dir, "temp_rgb.jpg")
                ir_path = os.path.join(temp_dir, "temp_ir.jpg")
                
                cv2.imwrite(rgb_path, rgb_frame)
                cv2.imwrite(ir_path, ir_frame)
                
                # 使用与原始API相同的方式调用模型
                # 分别处理RGB和红外图像
                rgb_results = self.model(rgb_path, save=True, save_txt=True, project=result_dir, name="detection", exist_ok=True)
                ir_results = self.model(ir_path, save=True, save_txt=True, project=result_dir, name="detection", exist_ok=True)
                
                # 处理检测结果
                detections = []
                rgb_detections = []
                ir_detections = []
                rgb_draw_frame = rgb_frame.copy()
                ir_draw_frame = ir_frame.copy()
                
                # 处理RGB检测结果
                for result in rgb_results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls == 0:  # 只处理人物类别
                            rgb_detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class_id': cls,
                                'class_name': 'person'
                            })
                
                # 处理红外检测结果
                for result in ir_results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls == 0:  # 只处理人物类别
                            ir_detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class_id': cls,
                                'class_name': 'person'
                            })
                
                # 融合检测结果
                detections = self.fuse_detections(rgb_detections, ir_detections)
                
                # 在图像上绘制检测框
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    color = (0, 255, 0)  # 绿色
                    thickness = 2
                    
                    # 绘制矩形
                    cv2.rectangle(rgb_draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    cv2.rectangle(ir_draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # 添加置信度标签
                    label = f"Person {conf:.2f}"
                    cv2.putText(rgb_draw_frame, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                    cv2.putText(ir_draw_frame, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                
                # 将处理后的图像转换为base64
                _, rgb_buffer = cv2.imencode('.jpg', rgb_draw_frame)
                _, ir_buffer = cv2.imencode('.jpg', ir_draw_frame)
                rgb_base64 = base64.b64encode(rgb_buffer).decode('utf-8')
                ir_base64 = base64.b64encode(ir_buffer).decode('utf-8')
                
                # 清理临时文件
                os.remove(rgb_path)
                os.remove(ir_path)
                
                return {
                    'code': 1,
                    'message': '识别成功！',
                    'rgb_draw_frame': rgb_base64,
                    'ir_draw_frame': ir_base64,
                    'detections': detections
                }
            else:
                logger.warning("YOLO模型未加载")
                return {'code': 0, 'message': '识别失败！'}
            
        except Exception as e:
            logger.error(f"双帧处理失败: {str(e)}")
            return {'code': 0, 'message': f'识别失败：{str(e)}'}

    @staticmethod
    def convert_to_jpg(input_path: str, output_folder: str, quality: int = 90) -> str:
        """
        将图片转换为JPG格式
        
        Args:
            input_path: 输入图片路径
            output_folder: 输出文件夹
            quality: JPG质量，范围0-100
            
        Returns:
            转换后的图片路径
        """
        try:
            ext = os.path.splitext(input_path)[-1].lower()
            if ext in ('.png', '.bmp', '.webp'):
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
                with Image.open(input_path) as img:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(output_path, 'JPEG', quality=quality)
                logger.info(f"转换成功: {input_path} -> {output_path}")
                return output_path
            return input_path
        except Exception as e:
            logger.error(f"图像转换失败: {str(e)}")
            return input_path