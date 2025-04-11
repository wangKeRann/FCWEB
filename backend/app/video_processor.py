import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    @staticmethod
    def adjust_frame(frame: np.ndarray, brightness: int = 0, contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        """
        调整单帧图像的亮度、对比度和饱和度
        
        Args:
            frame: 输入图像帧
            brightness: 亮度调整值，范围为 [-100, 100]
            contrast: 对比度调整值，大于 1 增加对比度，小于 1 降低对比度
            saturation: 饱和度调整值，大于 1 增加饱和度，小于 1 降低饱和度
            
        Returns:
            调整后的图像帧
        """
        try:
            # 将 BGR 图像转换为 HLS 颜色空间
            hls_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

            # 调整亮度
            hls_img[:, :, 1] = np.clip(hls_img[:, :, 1] + brightness, 0, 255)

            # 调整饱和度
            hls_img[:, :, 2] = np.clip(hls_img[:, :, 2] * saturation, 0, 255)

            # 将 HLS 图像转换回 BGR 颜色空间
            adjusted_frame = cv2.cvtColor(hls_img, cv2.COLOR_HLS2BGR)

            # 调整对比度
            adjusted_frame = cv2.convertScaleAbs(adjusted_frame, alpha=contrast, beta=0)
            
            return adjusted_frame
            
        except Exception as e:
            logger.error(f"Error in frame adjustment: {str(e)}")
            return frame

    @staticmethod
    def process_videos(
        visible_path: str,
        infrared_path: str,
        output_path: str,
        brightness: int = 0,
        contrast: float = 1.0,
        saturation: float = 1.0
    ) -> dict:
        """
        同时处理可见光和红外视频
        
        Args:
            visible_path: 可见光视频路径
            infrared_path: 红外视频路径
            output_path: 输出视频路径
            brightness: 亮度调整值，范围为 [-100, 100]
            contrast: 对比度调整值，大于 1 增加对比度，小于 1 降低对比度
            saturation: 饱和度调整值，大于 1 增加饱和度，小于 1 降低饱和度
            
        Returns:
            处理结果信息
        """
        try:
            # 打开输入视频
            visible_cap = cv2.VideoCapture(visible_path)
            infrared_cap = cv2.VideoCapture(infrared_path)
            
            if not visible_cap.isOpened():
                return {"error": "无法打开可见光视频"}
            if not infrared_cap.isOpened():
                return {"error": "无法打开红外视频"}

            # 获取视频的帧率和尺寸
            fps = visible_cap.get(cv2.CAP_PROP_FPS)
            width = int(visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # 宽度翻倍以并排显示

            while True:
                # 读取两路视频的帧
                ret1, visible_frame = visible_cap.read()
                ret2, infrared_frame = infrared_cap.read()
                
                if not ret1 or not ret2:
                    break

                # 处理可见光帧
                adjusted_visible = VideoProcessor.adjust_frame(visible_frame, brightness, contrast, saturation)
                
                # 将红外图像转换为彩色（如果原始是灰度图）
                if len(infrared_frame.shape) == 2:
                    infrared_frame = cv2.cvtColor(infrared_frame, cv2.COLOR_GRAY2BGR)
                
                # 调整红外图像大小以匹配可见光图像
                infrared_frame = cv2.resize(infrared_frame, (width, height))
                
                # 将两帧图像水平拼接
                combined_frame = np.hstack((adjusted_visible, infrared_frame))

                # 写入输出视频
                out.write(combined_frame)

            # 释放资源
            visible_cap.release()
            infrared_cap.release()
            out.release()
            return {"message": f"视频处理完成，已保存到 {output_path}"}
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            return {"error": str(e)} 