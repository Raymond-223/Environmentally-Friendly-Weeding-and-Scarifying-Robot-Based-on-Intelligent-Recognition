# 杂草-作物细粒度多任务检测 + 除草策略联动（报告 3.2.2）

import torch
import torch.nn as nn
from .dgl_detr import DGL_DETR


class FineGrainedDGL_DETR(nn.Module):
    """细粒度多任务 DGL-DETR：杂草种类 + 根深等级 + 作物生长阶段."""

    def __init__(self, num_weeds=8, num_depth=3, num_growth=2, base_model=None):
        super().__init__()
        if base_model is None:
            base_model = DGL_DETR(num_classes=num_weeds + 1)
        self.backbone = base_model
        self.neck = base_model.neck
        self.num_levels = getattr(base_model, "num_levels", 4)

        self.feat_align = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.weed_cls_head = nn.Linear(512, num_weeds)
        self.depth_cls_head = nn.Linear(512, num_depth)
        self.growth_cls_head = nn.Linear(512, num_growth)
        self.reg_head = nn.Linear(512, 4)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.SmoothL1Loss()
        self.alpha = 0.3
        self.beta = 0.2

    def _get_features(self, x):
        feat = self.backbone.forward_backbone(x)
        feats = [feat[:, :, :: (2 ** i), :: (2 ** i)].contiguous() for i in range(self.num_levels)]
        multi = self.neck(feats)
        fine_grained_feat = self.feat_align(multi[-1])
        return fine_grained_feat

    def forward(self, x, targets=None):
        fine_grained_feat = self._get_features(x)

        weed_cls = self.weed_cls_head(fine_grained_feat)
        depth_cls = self.depth_cls_head(fine_grained_feat)
        growth_cls = self.growth_cls_head(fine_grained_feat)
        reg = self.reg_head(fine_grained_feat)

        if self.training and targets is not None:
            cls_loss = self.cls_criterion(weed_cls, targets["weed_label"])
            depth_loss = self.cls_criterion(depth_cls, targets["depth_label"])
            growth_loss = self.cls_criterion(growth_cls, targets["growth_label"])
            reg_loss = self.reg_criterion(reg, targets["bbox"])
            total_loss = cls_loss + self.alpha * depth_loss + self.beta * growth_loss + reg_loss
            return total_loss, (weed_cls, depth_cls, growth_cls, reg)
        return weed_cls, depth_cls, growth_cls, reg
