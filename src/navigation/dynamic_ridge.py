# 作物生长自适应垄线动态规划（报告 3.1.2）
# 垄线偏移量 delta_w = k1*LAI + k2*h - k3*d，用于实时修正路径

import numpy as np


def dynamic_ridge_model(lai, plant_height, planting_density, k1=0.02, k2=0.015, k3=0.008, max_delta=0.05):
    """
    作物生长导致的垄线偏移量 (m)
    :param lai: 叶面积指数
    :param plant_height: 作物株高 (m)
    :param planting_density: 种植密度 (株/m²)
    :param k1, k2, k3: 田间拟合系数
    :param max_delta: 最大偏移量 ±m
    :return: delta_w (m)
    """
    delta_w = k1 * lai + k2 * plant_height - k3 * planting_density
    return float(np.clip(delta_w, -max_delta, max_delta))


def generate_adaptive_path(ridge_width, path_length=50, step=0.5, base_x=2.0):
    """
    基于动态垄距生成往返作业路径点
    :param ridge_width: 当前垄距 (m)
    :param path_length: 路径纵向长度 (m)
    :param step: 纵向步长 (m)
    :param base_x: 基准 x 坐标 (m)
    :return: list of (x, y, yaw), yaw 简化用 0
    """
    path = []
    for i, y in enumerate(np.arange(0, path_length, step)):
        # 垄间切换：奇偶行 x 不同
        x = base_x if (i // int(5 / step)) % 2 == 0 else base_x + ridge_width
        path.append((float(x), float(y), 0.0))
    return path
