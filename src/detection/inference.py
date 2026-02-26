# 检测推理管道：预处理 -> 模型 -> 后处理（NMS）-> 除草策略

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .models.dgl_detr import DGL_DETR
    from .models.fine_grained import FineGrainedDGL_DETR
except Exception:
    from models.dgl_detr import DGL_DETR
    from models.fine_grained import FineGrainedDGL_DETR


def preprocess(image, input_size=(640, 640), device="cpu"):
    """图像预处理：BGR HWC -> tensor NCHW 归一化."""
    import cv2
    if isinstance(image, np.ndarray):
        img = cv2.resize(image, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    else:
        img = image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img = (img.to(device) - mean) / std
    return img


def decode_detection(reg_out, cls_out, conf_threshold=0.5, input_size=(640, 640)):
    """
    将 head 输出解码为框与类别（简化：特征图网格点 + reg 偏移）。
    reg_out: (1, 4, H, W), cls_out: (1, C, H, W)
    """
    B, C, H, W = cls_out.shape
    probs = F.softmax(cls_out, dim=1)
    max_prob, labels = probs.max(dim=1)
    reg = reg_out[0].permute(1, 2, 0).reshape(-1, min(4, reg_out.size(1)))
    prob_flat = max_prob.reshape(-1)
    label_flat = labels.reshape(-1)
    keep = prob_flat >= conf_threshold
    if keep.sum() == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    reg = reg[keep]
    prob_flat = prob_flat[keep]
    label_flat = label_flat[keep]
    grid_y = torch.arange(H, device=cls_out.device, dtype=torch.float32)
    grid_x = torch.arange(W, device=cls_out.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    gx_flat = (gx.reshape(-1)[keep] + 0.5) / W * input_size[1]
    gy_flat = (gy.reshape(-1)[keep] + 0.5) / H * input_size[0]
    scale_x, scale_y = input_size[1] / max(W, 1), input_size[0] / max(H, 1)
    dx = (reg[:, 0] * scale_x).cpu().numpy()
    dy = (reg[:, 1] * scale_y).cpu().numpy()
    w = (reg[:, 2].clamp(0.01, 2) * scale_x * 16).cpu().numpy()
    h = (reg[:, 3].clamp(0.01, 2) * scale_y * 16).cpu().numpy()
    cx = gx_flat.cpu().numpy() + dx
    cy = gy_flat.cpu().numpy() + dy
    x1 = (cx - w / 2).clip(0, input_size[1])
    y1 = (cy - h / 2).clip(0, input_size[0])
    x2 = (cx + w / 2).clip(0, input_size[1])
    y2 = (cy + h / 2).clip(0, input_size[0])
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, prob_flat.cpu().numpy(), label_flat.cpu().numpy().astype(np.int64)


def nms_boxes(boxes, scores, labels, iou_threshold=0.45):
    """按类别做 NMS."""
    if len(boxes) == 0:
        return boxes, scores, labels
    import cv2
    keep = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), 0.0, iou_threshold
    )
    if isinstance(keep, np.ndarray):
        keep = keep.flatten()
    else:
        keep = list(keep) if hasattr(keep, "__iter__") else []
    return boxes[keep], scores[keep], labels[keep]


class WeedDetector:
    """杂草检测推理封装."""

    def __init__(self, model_type="dgl_detr", num_classes=2, checkpoint_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_type == "fine_grained":
            self.model = FineGrainedDGL_DETR(num_weeds=num_classes - 1, num_depth=3, num_growth=2)
        else:
            self.model = DGL_DETR(num_classes=num_classes)
        self.model.to(self.device)
        self.model.eval()
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.device)
            if "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
        self.input_size = (640, 640)
        self.conf_threshold = 0.5
        self.nms_threshold = 0.45

    def run(self, image):
        """返回 (boxes, scores, labels) 均为 numpy."""
        x = preprocess(image, self.input_size, self.device)
        with torch.no_grad():
            out = self.model(x)
        reg_out, cls_out = None, None
        if isinstance(out, tuple):
            if len(out) == 2 and out[0].dim() >= 3 and out[1].dim() >= 3:
                reg_out, cls_out = out[0], out[1]
            else:
                # FineGrained 等返回 4 元组 (weed_cls, depth_cls, growth_cls, reg)，无空间图，返回空
                return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        else:
            reg_out, cls_out = out, None
        if cls_out is None:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        boxes, scores, labels = decode_detection(
            reg_out, cls_out, self.conf_threshold, self.input_size
        )
        boxes, scores, labels = nms_boxes(boxes, scores, labels, self.nms_threshold)
        return boxes, scores, labels
