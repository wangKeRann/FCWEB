import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SmokeRemoval:
    """去雾处理类，实现各种去雾算法"""
    
    @staticmethod
    def dark_channel_prior(image, patch_size=15):
        """
        暗通道先验去雾算法
        
        Args:
            image: 输入图像 (BGR格式)
            patch_size: 局部区域大小
            
        Returns:
            去雾后的图像
        """
        try:
            # 转换为RGB格式
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 计算暗通道
            dark_channel = np.min(rgb, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
            dark_channel = cv2.erode(dark_channel, kernel)
            
            # 估计大气光值
            flat_dark = dark_channel.flatten()
            flat_image = rgb.reshape(-1, 3)
            
            # 选择暗通道前0.1%的像素
            top_pixels = int(0.001 * len(flat_dark))
            indices = np.argsort(flat_dark)[-top_pixels:]
            
            # 计算这些像素的平均值作为大气光值
            atmospheric = np.mean(flat_image[indices], axis=0)
            
            # 估计透射率
            transmission = 1 - 0.95 * dark_channel / np.max(atmospheric)
            
            # 细化透射率
            transmission = cv2.medianBlur(transmission.astype(np.float32), 3)
            
            # 限制透射率的最小值
            transmission = np.maximum(transmission, 0.1)
            
            # 恢复图像
            result = np.empty_like(rgb, dtype=np.float32)
            for i in range(3):
                result[:, :, i] = (rgb[:, :, i].astype(np.float32) - atmospheric[i]) / transmission + atmospheric[i]
            
            # 裁剪到有效范围
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # 转换回BGR格式
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Error in dark channel prior: {str(e)}")
            return image  # 出错时返回原始图像
    
    @staticmethod
    def process_frame(frame, method="dark_channel"):
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            method: 去雾方法，默认为"dark_channel"
            
        Returns:
            处理后的图像帧
        """
        if method == "dark_channel":
            return SmokeRemoval.dark_channel_prior(frame)
        else:
            logger.warning(f"Unknown smoke removal method: {method}")
            return frame 