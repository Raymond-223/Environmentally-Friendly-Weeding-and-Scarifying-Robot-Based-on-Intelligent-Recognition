<p align="center">
  <img src="https://img.shields.io/badge/🌱-环保除草-2d5016?style=for-the-badge" alt="环保除草" />
  <img src="https://img.shields.io/badge/🤖-智能识别-0d7377?style=for-the-badge" alt="智能识别" />
  <img src="https://img.shields.io/badge/🏠-大棚机器人-994e2e?style=for-the-badge" alt="大棚机器人" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5c3ee8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" />
</p>

<h1 align="center">基于智能识别的环保除草及松土机器人</h1>

<p align="center">
  大棚环境下具备 <strong>自主导航</strong>、<strong>智能杂草识别</strong> 与 <strong>精准除草/松土</strong> 的机器人系统
</p>

<p align="center">
  <sub>算法实现参考《导航及目标检测板块报告》</sub>
</p>

---

## ✨ 功能概览

| 模块 | 功能 |
|------|------|
| 🧭 **导航** | 自适应垄线检测、多光谱-视觉融合、作物生长动态垄距、轨迹预测避障、多因素传感器权重 |
| 🔍 **检测** | DGL-DETR 杂草检测、细粒度多任务（种类/根深/生长阶段）、增量学习、四模态融合、遮挡补全、轻量化 |
| ⚙️ **控制** | 根据检测结果输出除草策略（功率/时间/深度） |

---

## 📁 项目结构

```
├── src/           → 算法与业务逻辑（导航 / 检测 / 控制）
├── config/        → 配置文件
├── scripts/       → 可执行脚本（演示、训练）
├── requirements.txt
└── pyproject.toml
```

各目录说明见：[src/](src/README.md) · [config/](config/README.md) · [scripts/](scripts/README.md)

---

## 🛠 环境要求

- **Python** ≥ 3.8  
- **PyTorch** ≥ 2.0（建议带 CUDA）  
- OpenCV、NumPy、scikit-learn、PyYAML  

---

## 📦 安装

```bash
# 进入项目目录
cd 基于智能识别的环保除草及松土机器人

# 仅安装依赖
pip install -r requirements.txt

# 或可编辑安装（任意位置 import navigation / detection / control）
pip install -e .
```

---

## 🚀 使用

### 导航演示（摄像头/视频）

```bash
python scripts/run_navigation_demo.py
python scripts/run_navigation_demo.py --source video.mp4
python scripts/run_navigation_demo.py --no-nav   # 仅垄线检测与融合
```

### 检测演示（图像/摄像头）

```bash
python scripts/run_detection_demo.py
python scripts/run_detection_demo.py --source image.jpg
python scripts/run_detection_demo.py --checkpoint checkpoints/dgl_detr.pth --conf 0.6
```

### 训练示例

```bash
python scripts/train_detection.py --model dgl_detr --epochs 2 --batch 4
python scripts/train_detection.py --model fine_grained --epochs 2 --out checkpoints
```

实际应用请使用真实大棚杂草数据集。

### 配置

编辑 `config/default.yaml` 可调整导航、检测、除草策略等参数。

---

## 📖 核心算法

- **自适应垄线**：按光照选用 Canny+形态学 / Lab a 通道 / 标准 Canny，结合垄距与角度先验过滤。
- **多光谱融合**：按光照置信度加权 RGB 与 NIR，支持多因素传感器权重。
- **DGL-DETR**：可变形卷积 + GCNet + BiFPN + 解耦头；可选细粒度多任务、增量学习、四模态 CMAF、遮挡补全 LCINet、轻量化剪枝/蒸馏。

---

## 📌 说明

- 导航为**无 ROS** 独立实现，实机可改为 ROS2 话题。
- 检测需自行训练或使用公开杂草数据集，项目不附带预训练权重。

---

## 📄 许可证

MIT
