import os
import cv2
import numpy as np
import logging
from PIL import Image
import base64

logger = logging.getLogger(__name__)

class PoseProcessor:
    """姿态检测处理器类，用于处理RGB和红外图像的目标检测"""
    
    def __init__(self, is_server: bool = False):
        """
        初始化姿态检测处理器
        
        Args:
            is_server: 是否在服务器环境运行
        """
        self.is_server = is_server
        self.model = None
        
        if is_server:
            try:
                from ultralytics import YOLO
                self.model = YOLO('../zishi.pt')
                logger.info("YOLOv8模型加载成功")
            except ImportError:
                logger.warning("服务器环境未安装 ultralytics 包")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise

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

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧 (BGR格式)
            
        Returns:
            处理结果字典，包含检测框信息
        """
        try:
            # 创建原始图像的副本用于绘制
            draw_frame = frame.copy()
            
            # 定义行为类别的颜色映射
            color_map = {
                0: (0, 255, 255),  # 黄色 - walk
                1: (0, 0, 255),    # 红色 - stand
                2: (255, 0, 0),    # 蓝色 - squat
                3: (255, 255, 0),  # 青色 - lie
                4: (255, 0, 255),  # 紫色 - bend
                5: (0, 255, 0),    # 绿色 - wave
                6: (128, 0, 128)   # 深紫色 - sit
            }
            
            # 定义行为类别的标签映射
            label_map = {
                0: "walk",
                1: "stand",
                2: "squat",
                3: "lie",
                4: "bend",
                5: "wave",
                6: "sit"
            }
            
            if self.is_server and self.model is not None:
                logger.info("YOLO模型已加载")
                # 服务器环境使用YOLO模型
                results = self.model(frame)
                
                # 处理检测结果
                detections = []
                for result in results:
                    boxes = result.boxes
                    logger.info(f"检测框数量: {len(boxes)}")
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        logger.info(f"检测到目标: 类别={cls}, 置信度={conf}")
                        
                        # 获取对应的颜色和标签
                        color = color_map.get(cls, (0, 255, 0))  # 默认使用绿色
                        label = label_map.get(cls, f"class_{cls}")
                        
                        # 在图像上绘制检测框
                        cv2.rectangle(draw_frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    color, 2)
                        
                        # 添加类别和置信度标签
                        label_text = f"{label} {conf:.1f}"
                        # 使用支持中文的字体
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(draw_frame, label_text, 
                                  (int(x1), int(y1) - 10), 
                                  font, 0.5, 
                                  color, 2)
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': cls,
                            'label': label
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
        处理RGB和红外图像帧
        
        Args:
            rgb_frame: RGB图像帧 (BGR格式)
            ir_frame: 红外图像帧 (BGR格式)
            
        Returns:
            处理结果字典
        """
        try:
            # 处理RGB帧
            rgb_result = self.process_frame(rgb_frame)
            if 'error' in rgb_result:
                return rgb_result
                
            # 处理红外帧
            ir_result = self.process_frame(ir_frame)
            if 'error' in ir_result:
                return ir_result
            
            return {
                'rgb_detections': rgb_result['detections'],
                'ir_detections': ir_result['detections'],
                'rgb_shape': rgb_result['frame_shape'],
                'ir_shape': ir_result['frame_shape'],
                'rgb_draw_frame': rgb_result['draw_frame'],  # 添加绘制后的RGB图像
                'ir_draw_frame': ir_result['draw_frame']     # 添加绘制后的红外图像
            }
            
        except Exception as e:
            logger.error(f"双帧处理失败: {str(e)}")
            return {'error': str(e)}

    def process_folder(self, input_folder: str, output_folder: str) -> list:
        """
        处理文件夹中的所有图像
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            
        Returns:
            处理结果列表
        """
        results = []
        if not os.path.exists(input_folder):
            return results

        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                input_path = os.path.join(input_folder, filename)

                # 转换格式
                converted_path = self.convert_to_jpg(input_path, output_folder)

                # 模型推理
                try:
                    if self.is_server and self.model is not None:
                        self.model.predict(
                            source=converted_path,
                            save=True,
                            save_txt=True,
                            project=output_folder,
                            name="",
                            exist_ok=True,
                        )

                    # 检查结果文件
                    exp_folder = os.path.join(output_folder, "predict")
                    result_path = os.path.join(exp_folder, os.path.basename(converted_path))

                    if os.path.exists(result_path):
                        results.append({
                            'input_path': input_path,
                            'output_path': result_path,
                            'status': 'success'
                        })
                    else:
                        results.append({
                            'input_path': input_path,
                            'output_path': None,
                            'status': 'failed',
                            'message': 'Result file not generated'
                        })
                except Exception as e:
                    results.append({
                        'input_path': input_path,
                        'output_path': None,
                        'status': 'error',
                        'message': str(e)
                    })

        return results

    def process_folders(self, rgb_folder: str, ir_folder: str) -> dict:
        """
        处理RGB和IR文件夹
        
        Args:
            rgb_folder: RGB图像文件夹路径
            ir_folder: IR图像文件夹路径
            
        Returns:
            处理结果字典
        """
        try:
            # 准备输出目录
            rgb_output = os.path.join(self.output_base, "RGB")
            ir_output = os.path.join(self.output_base, "IR")

            # 处理RGB文件夹
            rgb_results = self.process_folder(rgb_folder, rgb_output)

            # 处理IR文件夹
            ir_results = self.process_folder(ir_folder, ir_output)

            # 统计结果
            total = len(rgb_results) + len(ir_results)
            success = sum(1 for r in rgb_results + ir_results if r['status'] == 'success')

            response = OrderedDict([
                ('code', 1),
                ('rgb_results', rgb_results),
                ('ir_results', ir_results),
                ('timestamp', datetime.now().isoformat()),
                ('rgb_input_folder', rgb_folder),
                ('rgb_output_folder', os.path.join(rgb_output, "predict")),
                ('ir_input_folder', ir_folder),
                ('ir_output_folder', os.path.join(ir_output, "predict")),
                ('message', f'处理完成，共处理{total}个文件，成功{success}个')
            ])

            return response

        except Exception as e:
            logger.error(f"处理错误: {str(e)}")
            return {
                'code': 0,
                'message': f'处理错误: {str(e)}'
            } 