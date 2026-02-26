# detection — 目标检测模块

实现**杂草/作物目标检测**及相关进阶功能：DGL-DETR 基础模型、细粒度多任务、增量学习、四模态融合、遮挡补全、轻量化工具，以及完整推理管道。

---

## 目录与文件说明

### models/ — 模型定义

| 文件 | 说明 |
|------|------|
| **dgl_detr.py** | DGL-DETR 基础模型：可变形卷积、GCNet、C2f_Deform、BiFPN、解耦检测头（回归 + 分类）。 |
| **fine_grained.py** | 细粒度多任务：杂草种类、根深等级、作物生长阶段等多头输出，用于与除草策略联动。 |
| **incremental.py** | 在线增量学习：MemoryReplay 历史缓存、IncrementalDGL_DETR（冻结大部分 backbone，新增分类头）。 |
| **multimodal.py** | 四模态融合：CMAF（SelfAttention + CrossAttention）融合 RGB/NIR/Depth/Thermal；FourModalDGL_DETR。 |
| **lcinet.py** | 遮挡补全：LCINetGenerator（可见区域+掩码+上下文→补全图像）、LCINetDiscriminator。 |
| **lightweight.py** | 轻量化工具：结构化剪枝、知识蒸馏、参数统计等，便于部署到低成本硬件。 |

### 根目录

| 文件 | 说明 |
|------|------|
| **inference.py** | 推理管道：预处理、DGL-DETR/FineGrained 调用、解码、NMS，以及 `WeedDetector` 封装类。 |

---

## 主要接口

- **模型**：`DGL_DETR`, `FineGrainedDGL_DETR`, `IncrementalDGL_DETR`, `FourModalDGL_DETR`, `LCINetGenerator`, `LCINetDiscriminator`, `CMAF` 等（见 `models/__init__.py`）。  
- **推理**：`WeedDetector(model_type, num_classes, checkpoint_path).run(image)` → boxes, scores, labels。  
- **工具**：`preprocess`, `decode_detection`, `nms_boxes`。  

训练需使用大棚杂草数据集；项目不附带预训练权重。
