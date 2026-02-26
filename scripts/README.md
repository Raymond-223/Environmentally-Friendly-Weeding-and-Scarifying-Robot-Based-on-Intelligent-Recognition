# scripts — 可执行脚本目录

本目录包含可直接运行的**演示与训练脚本**，用于体验导航/检测效果或跑通训练流程。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| **run_navigation_demo.py** | 导航演示：从摄像头或视频读帧，做自适应垄线检测、多光谱融合，可选完整路径跟踪（无 ROS）。 |
| **run_detection_demo.py** | 检测演示：对图像或摄像头进行杂草检测，可视化框与类别，并输出除草策略示例。 |
| **train_detection.py** | 训练示例：使用占位数据集跑通 DGL-DETR / FineGrained 训练与保存流程，便于替换为真实数据。 |

---

## 运行方式

在**项目根目录**下执行（或先 `pip install -r requirements.txt` / `pip install -e .`）：

```bash
# 导航演示
python scripts/run_navigation_demo.py [--source 0|video.mp4] [--no-nav]

# 检测演示
python scripts/run_detection_demo.py [--source 0|image.jpg] [--checkpoint path] [--conf 0.5]

# 训练示例
python scripts/train_detection.py --model dgl_detr|fine_grained [--epochs 2] [--out checkpoints]
```

各脚本支持 `--help` 查看完整参数。
