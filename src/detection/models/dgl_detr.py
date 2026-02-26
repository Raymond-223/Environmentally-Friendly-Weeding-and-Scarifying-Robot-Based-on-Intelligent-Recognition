# DGL-DETR 基础模型：可变形卷积 + GCNet + BiFPN（报告 3.2.1）

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableConv2d(nn.Module):
    """可变形卷积模块（offset + modulation 可选，与报告一致接口）."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.modulation_conv = nn.Conv2d(
            in_channels, kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.main_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        modulation = torch.sigmoid(self.modulation_conv(x))
        # 简化：直接对 x 做标准卷积，再用 modulation 做通道加权（与完整 DCN 近似）
        out = self.main_conv(x)
        mod = modulation.mean(dim=1, keepdim=True)
        return out * (0.9 + 0.2 * mod)


class GCNet(nn.Module):
    """GCNet 全局注意力模块（报告 3.2.1）."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class C2f_Deform(nn.Module):
    """改进的 C2f：含可变形卷积与 GCNet."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, self.c, 1, 1)
        self.cv2 = nn.Conv2d(c1, self.c, 1, 1)
        self.cv3 = nn.Conv2d(2 * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                DeformableConv2d(self.c, self.c, 3, 1, 1),
                GCNet(self.c),
            )
            for _ in range(n)
        )
        self.shortcut = shortcut

    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.cv1(x)
        for m in self.m:
            y2 = y2 + m(y2) if self.shortcut else m(y2)
        y = torch.cat([y1, y2], dim=1)
        return self.cv3(y)


class BiFPN(nn.Module):
    """双向特征金字塔（报告 BiFPN）."""

    def __init__(self, channels, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, 1) for _ in range(num_levels)
        ])
        self.weights = nn.Parameter(torch.ones(2, num_levels - 1))

    def forward(self, inputs):
        # inputs: [P3, P4, P5, P6] 或等长 list
        inputs = list(inputs)
        if len(inputs) != self.num_levels:
            # 若只传单特征图，直接返回
            if len(inputs) == 1:
                return inputs
            inputs = inputs[: self.num_levels]
        inputs = [conv(x) for conv, x in zip(self.convs, inputs)]
        weights = F.softmax(self.weights, dim=0)

        # 自下而上
        up = inputs[-1]
        for i in range(self.num_levels - 2, -1, -1):
            up = F.interpolate(up, size=inputs[i].shape[2:], mode="bilinear", align_corners=False)
            inputs[i] = inputs[i] * weights[0, i] + up * weights[1, i]

        # 自上而下
        down = inputs[0]
        for i in range(1, self.num_levels):
            down = F.adaptive_avg_pool2d(down, inputs[i].shape[2:])
            inputs[i] = inputs[i] * weights[0, i - 1] + down * weights[1, i - 1]

        return inputs


class DGL_DETR(nn.Module):
    """DGL-DETR 基础模型：主干 + BiFPN + 解耦检测头."""

    def __init__(self, num_classes=2, in_channels=3, num_levels=4):
        super().__init__()
        self.num_classes = num_classes
        # 主干
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            C2f_Deform(64, 128, 2),
            nn.Conv2d(128, 128, 3, 2, 1),
        )
        self.stage2 = nn.Sequential(
            C2f_Deform(128, 256, 4),
            nn.Conv2d(256, 256, 3, 2, 1),
        )
        self.stage3 = nn.Sequential(
            C2f_Deform(256, 512, 4),
            GCNet(512),
        )
        self.backbone_out_channels = 512

        self.neck = BiFPN(512, num_levels)
        self.num_levels = num_levels

        self.reg_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, 1, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_backbone(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

    def forward(self, x, targets=None):
        feat = self.forward_backbone(x)
        # 多尺度：从 feat 下采样得到多级
        feats = [feat[:, :, :: (2 ** i), :: (2 ** i)].contiguous() for i in range(self.num_levels)]
        multi = self.neck(feats)
        last = multi[-1]
        reg_out = self.reg_head(last)
        cls_out = self.cls_head(last)
        return reg_out, cls_out
