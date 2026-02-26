# src — 算法与业务逻辑

本目录为项目**核心代码**，按功能分为导航、检测、控制三大模块，均可在安装项目后通过 `import` 使用。

---

## 目录结构

| 目录 | 说明 |
|------|------|
| **navigation/** | 导航：垄线检测、多光谱融合、传感器权重、轨迹预测、动态垄距、导航器整合。详见 [navigation/README.md](navigation/README.md) |
| **detection/** | 目标检测：DGL-DETR 及进阶模型、推理管道。详见 [detection/README.md](detection/README.md) |
| **control/** | 除草策略与执行联动（功率/时间/深度）。详见 [control/README.md](control/README.md) |

---

## 使用示例

```python
from navigation import adaptive_ridge_detection, WeedRobotNavigator
from detection import DGL_DETR
from detection.inference import WeedDetector
from control import weed_control_strategy, load_weed_control_config
```

安装方式：在项目根目录执行 `pip install -e .`。
