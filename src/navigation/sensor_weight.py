# 多因素传感器融合权重自适应（报告 3.1.2）
# 随机森林：输入 [光照, 湿度, 作物密度, 遮挡率] -> 输出 [rgb, nir, lidar, uwb] 权重

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os


class SensorWeightPredictor:
    """根据环境特征预测各传感器权重，用于多传感器融合."""

    def __init__(self, model_path=None):
        self.model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self._trained = False
        if model_path and os.path.isfile(model_path):
            try:
                import joblib
                self.model = joblib.load(model_path)
                self._trained = True
            except Exception:
                self._init_fallback_weights()

    def _init_fallback_weights(self):
        """无预训练模型时使用简单规则作为 fallback."""
        self._trained = False

    def predict_weights(self, lightness, humidity, crop_density, occlusion_rate):
        """
        预测各传感器权重
        :param lightness: 光照强度 0-255
        :param humidity: 环境湿度 (%)
        :param crop_density: 作物密度 (株/m²)
        :param occlusion_rate: 遮挡率 0-1
        :return: [rgb_weight, nir_weight, lidar_weight, uwb_weight]
        """
        features = np.array([[lightness, humidity, crop_density, occlusion_rate]], dtype=np.float64)
        if self._trained:
            weights = self.model.predict(features)[0]
        else:
            # 规则 fallback：遮挡高时提高 lidar/uwb
            rgb = 0.5 - 0.3 * occlusion_rate
            nir = 0.2 + 0.2 * (1 - lightness / 255.0)
            lidar = 0.2 + 0.4 * occlusion_rate
            uwb = 0.1 + 0.1 * occlusion_rate
            weights = np.array([rgb, nir, lidar, uwb])
        weights = np.clip(weights, 0.05, 0.8)
        weights = weights / np.sum(weights)
        return list(weights)

    def fit(self, X, Y):
        """X: (N, 4) [lightness, humidity, crop_density, occlusion], Y: (N, 4) 权重."""
        self.model.fit(X, Y)
        self._trained = True
