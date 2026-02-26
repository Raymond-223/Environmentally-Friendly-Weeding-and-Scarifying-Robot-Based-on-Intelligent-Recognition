# 动态障碍物轨迹预测：TrajformerTiny（报告 3.1.2）
# 输入过去 3 帧位置 (x,y)，输出未来 2 帧位置

import numpy as np
import torch
import torch.nn as nn


class TrajformerTiny(nn.Module):
    """轻量化轨迹预测：3 帧输入 -> 2 帧输出 (x1,y1, x2,y2)."""

    def __init__(self, input_dim=2, output_dim=4, hidden_dim=64, nhead=2, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim * 3, output_dim)  # 3 帧展平 -> 4 维

    def forward(self, x):
        """
        x: (batch, 3, 2) 最近 3 帧的 (x,y)
        return: (batch, 4) 未来 2 帧 (x1,y1, x2,y2)
        """
        # (B, 3, 2) -> (B, 3, 2) 保持
        enc = self.encoder(x)
        enc_flat = enc.reshape(x.size(0), -1)
        return self.decoder(enc_flat)


def predict_and_avoid(obstacle_history, traj_model, current_path, safe_distance=1.5, device="cpu"):
    """
    根据障碍物历史预测轨迹，若与路径交点距离 < safe_distance 则返回需避障及圆弧路径参数。
    简化：仅返回是否需要避障与建议转向。
    """
    if len(obstacle_history) < 3 or traj_model is None:
        return False, None

    traj_model.eval()
    recent = np.array(obstacle_history[-3:], dtype=np.float32).reshape(1, 3, 2)
    t = torch.from_numpy(recent).to(device)
    with torch.no_grad():
        pred = traj_model(t)
    pred_pos = pred.cpu().numpy().reshape(2, 2)

    # 简化：若预测点与原点距离过近则需避障
    for (px, py) in pred_pos:
        dist = np.hypot(px, py)
        if dist < safe_distance:
            return True, {"pred_pos": pred_pos, "turn": 0.3 if px < 0 else -0.3}
    return False, None
