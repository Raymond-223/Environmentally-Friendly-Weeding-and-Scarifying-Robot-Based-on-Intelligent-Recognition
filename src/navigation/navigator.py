# 导航核心整合：垄线检测 + 多光谱融合 + 动态垄距 + 轨迹预测 + 传感器权重
# 独立运行版（无 ROS），便于在 PC 上演示；实际部署可替换为 ROS2 话题

import numpy as np
import cv2

from .ridge_detection import adaptive_ridge_detection
from .multispectral_fusion import multispectral_fusion
from .dynamic_ridge import dynamic_ridge_model, generate_adaptive_path
from .sensor_weight import SensorWeightPredictor
from .trajectory_predictor import TrajformerTiny, predict_and_avoid


class WeedRobotNavigator:
    """除草机器人导航逻辑（无 ROS 版）：输入图像与传感器数据，输出速度与路径索引."""

    def __init__(self, ridge_prior=0.8, safe_distance=1.0):
        self.ridge_prior = ridge_prior
        self.safe_distance = safe_distance
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.target_path = generate_adaptive_path(ridge_prior)
        self.path_index = 0
        self.nir_img = None
        self.humidity = 50.0
        self.planting_density = 20.0
        self.occlusion_rate = 0.0
        self.lai = 0.0
        self.plant_height = 0.0
        self.obstacle_history = []

        self.sensor_weight_predictor = SensorWeightPredictor()
        try:
            import torch
            self.traj_model = TrajformerTiny()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.traj_model.to(self.device)
        except Exception:
            self.traj_model = None
            self.device = "cpu"

    def update_pose(self, x, y, yaw):
        self.current_pose = np.array([x, y, yaw], dtype=float)

    def set_nir(self, nir_image):
        self.nir_img = nir_image

    def set_env(self, humidity=None, crop_density=None, occlusion_rate=None, lai=None, plant_height=None):
        if humidity is not None:
            self.humidity = humidity
        if crop_density is not None:
            self.planting_density = crop_density
        if occlusion_rate is not None:
            self.occlusion_rate = occlusion_rate
        if lai is not None:
            self.lai = lai
        if plant_height is not None:
            self.plant_height = plant_height

    def step(self, rgb_image):
        """
        单步导航：输入一帧 RGB，输出 (linear_x, angular_z, path_index)。
        若存在 NIR 则先融合再垄线检测；若有轨迹模型则做主动避障判断。
        """
        if rgb_image is None:
            return 0.0, 0.0, self.path_index

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        lightness = float(np.mean(gray))

        # 多光谱融合（若有 NIR）
        if self.nir_img is not None:
            weights = self.sensor_weight_predictor.predict_weights(
                lightness, self.humidity, self.planting_density, self.occlusion_rate
            )
            rgb_w, nir_w = weights[0], weights[1]
            fused, _ = multispectral_fusion(rgb_image, self.nir_img, lightness)
            work_img = fused
        else:
            work_img = rgb_image

        # 动态垄距
        delta_w = dynamic_ridge_model(self.lai, self.plant_height, self.planting_density)
        current_ridge = self.ridge_prior + delta_w
        self.target_path = generate_adaptive_path(current_ridge)

        # 垄线检测与横向纠偏
        lines = adaptive_ridge_detection(work_img, lightness)
        if lines is not None and len(lines) > 0:
            centers = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                centers.append((x1 + x2) / 2)
            target_pixel_x = np.mean(centers) - rgb_image.shape[1] / 2
            angular_z = -0.002 * target_pixel_x
        else:
            angular_z = 0.0

        # 纯跟踪路径
        if self.path_index < len(self.target_path):
            tx, ty, _ = self.target_path[self.path_index]
            dist = np.hypot(tx - self.current_pose[0], ty - self.current_pose[1])
            if dist < 0.5:
                self.path_index = min(self.path_index + 1, len(self.target_path) - 1)
            yaw_err = np.arctan2(
                np.sin(0 - self.current_pose[2]),
                np.cos(0 - self.current_pose[2])
            )
            linear_x = 0.3
            angular_z = angular_z + 0.5 * yaw_err
        else:
            linear_x = 0.0

        # 主动避障（轨迹预测）：这里简化不调用语义分割，仅演示轨迹预测接口
        need_avoid, avoid_info = predict_and_avoid(
            self.obstacle_history, self.traj_model, self.target_path,
            safe_distance=self.safe_distance, device=self.device
        )
        if need_avoid and avoid_info:
            linear_x = 0.1
            angular_z = avoid_info.get("turn", angular_z)

        return linear_x, angular_z, self.path_index

    def add_obstacle_position(self, x, y):
        """记录障碍物位置用于轨迹预测."""
        self.obstacle_history.append((x, y))
        if len(self.obstacle_history) > 30:
            self.obstacle_history = self.obstacle_history[-30:]
