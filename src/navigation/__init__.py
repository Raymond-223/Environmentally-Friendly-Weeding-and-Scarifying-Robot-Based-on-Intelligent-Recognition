# 导航模块：自适应垄线、多光谱融合、避障、轨迹预测

from .ridge_detection import adaptive_ridge_detection
from .multispectral_fusion import multispectral_fusion
from .sensor_weight import SensorWeightPredictor
from .trajectory_predictor import TrajformerTiny, predict_and_avoid
from .dynamic_ridge import dynamic_ridge_model, generate_adaptive_path
from .navigator import WeedRobotNavigator

__all__ = [
    "adaptive_ridge_detection",
    "multispectral_fusion",
    "SensorWeightPredictor",
    "TrajformerTiny",
    "predict_and_avoid",
    "dynamic_ridge_model",
    "generate_adaptive_path",
    "WeedRobotNavigator",
]
