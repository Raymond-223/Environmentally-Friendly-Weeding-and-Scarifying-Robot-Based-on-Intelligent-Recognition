# 在线增量学习：记忆重放 + 轻量增量训练（报告 3.2.2）

import numpy as np
import torch
import torch.nn as nn
from .dgl_detr import DGL_DETR
from .fine_grained import FineGrainedDGL_DETR


class MemoryReplay:
    """历史样本缓存，用于增量训练时重放."""

    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, samples):
        if isinstance(samples, torch.Tensor):
            samples = [samples[i].cpu().numpy() for i in range(samples.size(0))]
        self.buffer.extend(samples)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size :]

    def sample(self, num_samples):
        if len(self.buffer) == 0:
            return None
        n = min(num_samples, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]


class IncrementalDGL_DETR(nn.Module):
    """增量学习版：冻结大部分 backbone，新增分类头 + 记忆重放."""

    def __init__(self, pretrained_model, num_new_weeds=2):
        super().__init__()
        if isinstance(pretrained_model, FineGrainedDGL_DETR):
            self.backbone = pretrained_model.backbone
            self.neck = pretrained_model.neck
            self.feat_align = pretrained_model.feat_align
            self.num_levels = pretrained_model.num_levels
            self.old_weed_cls = pretrained_model.weed_cls_head
            self.depth_cls_head = pretrained_model.depth_cls_head
            self.growth_cls_head = pretrained_model.growth_cls_head
            self.reg_head = pretrained_model.reg_head
            self.old_num_weeds = pretrained_model.weed_cls_head.out_features
        else:
            raise TypeError("pretrained_model 应为 FineGrainedDGL_DETR")

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.old_weed_cls.parameters():
            p.requires_grad = False

        self.new_weed_cls = nn.Linear(512, num_new_weeds)
        self.memory_replay = MemoryReplay(buffer_size=1000)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.SmoothL1Loss()

    def _get_features(self, x):
        feat = self.backbone.forward_backbone(x)
        feats = [feat[:, :, :: (2 ** i), :: (2 ** i)].contiguous() for i in range(self.num_levels)]
        multi = self.neck(feats)
        return self.feat_align(multi[-1])

    def forward(self, x, new_samples=None, new_targets=None):
        feat = self._get_features(x)
        old_cls = self.old_weed_cls(feat)
        new_cls = self.new_weed_cls(feat)
        total_cls = torch.cat([old_cls, new_cls], dim=1)
        depth_cls = self.depth_cls_head(feat)
        growth_cls = self.growth_cls_head(feat)
        reg = self.reg_head(feat)

        if self.training and new_samples is not None and new_targets is not None:
            new_feat = self._get_features(new_samples)
            new_cls_loss = self.cls_criterion(self.new_weed_cls(new_feat), new_targets["weed_label"])
            new_reg_loss = self.reg_criterion(self.reg_head(new_feat), new_targets["bbox"])
            history = self.memory_replay.sample(100)
            if history is not None and len(history) > 0:
                hist_t = torch.tensor(np.stack(history), dtype=torch.float32, device=x.device)
                if hist_t.dim() == 3:
                    hist_t = hist_t.unsqueeze(1).permute(0, 1, 3, 2) if hist_t.shape[-1] == 3 else hist_t.unsqueeze(1)
                hist_feat = self._get_features(hist_t)
                hist_loss = self.cls_criterion(self.old_weed_cls(hist_feat), torch.zeros(hist_t.size(0), dtype=torch.long, device=x.device))
            else:
                hist_loss = torch.tensor(0.0, device=x.device)
            total_loss = new_cls_loss + new_reg_loss + 0.5 * hist_loss
            return total_loss, (total_cls, depth_cls, growth_cls, reg)
        return total_cls, depth_cls, growth_cls, reg
