# 自适应垄线特征提取算法（报告 3.1.1）
# 根据光照动态选择 Canny + 形态学 / Lab a 通道 / 标准 Canny，并结合垄距先验过滤直线

import cv2
import numpy as np


def adaptive_ridge_detection(image, lightness=None, angle_threshold_deg=15, min_line_length=100, max_line_gap=20):
    """
    自适应垄线特征提取算法
    :param image: 输入 RGB 图像 (H, W, 3) BGR
    :param lightness: 光照强度 0-255，若为 None 则从图像灰度均值计算
    :param angle_threshold_deg: 与水平方向夹角阈值，大于此角度的直线被过滤
    :param min_line_length: 霍夫直线最小长度
    :param max_line_gap: 霍夫直线最大间隙
    :return: 检测到的垄线数组 (N, 1, 4) 或 None
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if lightness is None:
        lightness = float(np.mean(gray))

    # 根据光照选择策略
    if lightness > 200:  # 强光区域
        edges = cv2.Canny(gray, 150, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    elif lightness < 50:  # 阴影区域：Lab a 通道
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        edges = cv2.Canny(lab[:, :, 1], 30, 80)
    else:  # 正常光照
        edges = cv2.Canny(gray, 50, 150)

    # 霍夫直线检测
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return None

    valid_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        # 垄线近似水平：与水平夹角 < 15° 或 > 165°
        if angle < angle_threshold_deg or angle > (180 - angle_threshold_deg):
            valid_lines.append(line)

    return np.array(valid_lines) if valid_lines else None
