# navigation — 导航模块

实现大棚环境下的**视觉/多光谱导航**逻辑：垄线检测、多光谱融合、传感器权重、轨迹预测、动态垄距与路径跟踪。当前为无 ROS 独立实现，便于在 PC 上演示；实机可改为订阅/发布 ROS2 话题。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| **ridge_detection.py** | 自适应垄线特征提取：根据光照选择 Canny+形态学 / Lab a 通道 / 标准 Canny，结合垄距与角度先验过滤直线。 |
| **multispectral_fusion.py** | 多光谱-视觉融合：按光照置信度加权 RGB 与 NIR，输出融合图像供定位与垄线检测使用。 |
| **sensor_weight.py** | 多因素传感器权重预测：基于随机森林或规则，根据光照、湿度、作物密度、遮挡率输出 RGB/NIR/Lidar/UWB 权重。 |
| **trajectory_predictor.py** | 动态障碍物轨迹预测：TrajformerTiny（3 帧位置 → 未来 2 帧），及 `predict_and_avoid` 避障逻辑接口。 |
| **dynamic_ridge.py** | 作物生长自适应垄线：垄线偏移量模型、基于动态垄距的往返路径生成。 |
| **navigator.py** | 导航整合类 `WeedRobotNavigator`：融合上述模块，输入 RGB（可选 NIR）与环境参数，输出线速度、角速度与路径索引。 |

---

## 主要接口

- `adaptive_ridge_detection(image, lightness=None, ...)` → 垄线数组或 None  
- `multispectral_fusion(rgb_image, nir_image, lightness=None)` → 融合图像, nir_weight  
- `SensorWeightPredictor.predict_weights(lightness, humidity, crop_density, occlusion_rate)` → [rgb, nir, lidar, uwb]  
- `TrajformerTiny`：输入 `(B, 3, 2)`，输出 `(B, 4)`  
- `dynamic_ridge_model(lai, plant_height, planting_density)` → delta_w  
- `generate_adaptive_path(ridge_width, ...)` → 路径点列表  
- `WeedRobotNavigator.step(rgb_image)` → linear_x, angular_z, path_index  
