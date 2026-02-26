# 极端轻量化：剪枝 + 知识蒸馏 + 量化（报告 3.2.2）

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgl_detr import DGL_DETR


def structured_pruning(model, importance_threshold=0.1):
    """基于 L1 范数的通道重要性剪枝（简化：仅做权重 mask，不改变结构）."""
    pruned = copy.deepcopy(model)
    for name, module in list(pruned.named_modules()):
        if isinstance(module, nn.Conv2d) and "conv" in name:
            imp = torch.norm(module.weight.data, p=1, dim=(1, 2, 3))
            keep = imp > (imp.max() * importance_threshold)
            if keep.sum() < 1:
                keep = torch.ones_like(keep, dtype=torch.bool)
            # 简化：仅记录 mask，实际推理可用 index_select
            setattr(module, "_channel_mask", keep)
    return pruned


def knowledge_distillation(teacher_model, student_model, dataloader, epochs=10, device="cpu", lr=1e-4):
    """知识蒸馏：教师 -> 学生."""
    teacher_model.eval()
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    alpha = 0.3
    T = 3.0

    for epoch in range(epochs):
        total_loss = 0.0
        n_batch = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                imgs = batch
                targets = None
            imgs = imgs.to(device)
            with torch.no_grad():
                if hasattr(teacher_model, "forward_backbone"):
                    t_out = teacher_model(imgs)
                else:
                    t_out = teacher_model(imgs)
                if isinstance(t_out, tuple):
                    t_logits = t_out[0]
                else:
                    t_logits = t_out
                teacher_prob = F.softmax(t_logits / T, dim=1)

            s_out = student_model(imgs)
            if isinstance(s_out, tuple):
                s_logits = s_out[0]
            else:
                s_logits = s_out
            student_log_prob = F.log_softmax(s_logits / T, dim=1)
            kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)
            if targets is not None and "weed_label" in targets:
                ce_loss = F.cross_entropy(s_logits, targets["weed_label"].to(device))
                loss = alpha * ce_loss + (1 - alpha) * kl_loss
            else:
                loss = kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        if n_batch:
            print(f"Distill Epoch {epoch+1}, Loss: {total_loss/n_batch:.4f}")
    return student_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
