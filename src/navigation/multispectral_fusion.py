# 多光谱-视觉融合定位（报告 3.1.1）
# 根据光照置信度动态调整 RGB / NIR 权重，融合图像用于后续定位与垄线检测

import cv2
import numpy as np


def multispectral_fusion(rgb_image, nir_image, lightness=None):
    """
    多光谱-视觉融合定位
    :param rgb_image: RGB 图像 (H, W, 3)
    :param nir_image: 近红外图像 (H, W) 单通道或 (H, W, 1)
    :param lightness: RGB 光照强度，None 则从灰度方差推导置信度
    :return: fused_image (H, W, 3), nir_weight (float)
    """
    if rgb_image is None or nir_image is None:
        return rgb_image, 0.5

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    light_variance = np.var(gray)
    # 光照置信度：方差大表示光照信息可靠
    light_confidence = np.clip(light_variance / 1000.0, 0.2, 0.8)
    nir_weight = 1.0 - light_confidence
    rgb_weight = light_confidence

    # 归一化 NIR 到 0-255 并扩展为 3 通道
    if nir_image.ndim == 2:
        nir_normalized = cv2.normalize(nir_image, None, 0, 255, cv2.NORM_MINMAX)
        nir_3ch = cv2.cvtColor(nir_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        nir_normalized = cv2.normalize(nir_image, None, 0, 255, cv2.NORM_MINMAX)
        if nir_normalized.shape[2] == 1:
            nir_3ch = cv2.cvtColor(nir_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            nir_3ch = nir_normalized.astype(np.uint8)

    # 尺寸一致
    if nir_3ch.shape[:2] != rgb_image.shape[:2]:
        nir_3ch = cv2.resize(nir_3ch, (rgb_image.shape[1], rgb_image.shape[0]))

    fused_image = cv2.addWeighted(rgb_image, rgb_weight, nir_3ch, nir_weight, 0)
    return fused_image, nir_weight
