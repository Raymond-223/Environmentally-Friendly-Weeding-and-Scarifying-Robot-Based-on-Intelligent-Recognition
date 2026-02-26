# 遮挡杂草生成式补全 LCINet（报告 3.2.2）

import torch
import torch.nn as nn


class LCINetGenerator(nn.Module):
    """轻量化遮挡补全生成器：可见区域 + 遮挡掩码 + 上下文 -> 补全图像."""

    def __init__(self, in_channels=3 + 1 + 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.context_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x_vis, mask, context):
        # x_vis (b,3,h,w), mask (b,1,h,w), context (b,3,H,W)
        ctx_feat = self.context_encoder(context).unsqueeze(-1).unsqueeze(-1)
        ctx_feat = ctx_feat.repeat(1, 1, x_vis.size(2), x_vis.size(3))
        inp = torch.cat([x_vis, mask, ctx_feat], dim=1)
        enc = self.encoder(inp)
        dec = self.decoder(enc)
        return x_vis * (1 - mask) + dec * mask


class LCINetDiscriminator(nn.Module):
    """判别器：真实性 + 上下文一致性."""

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0),
        )

    def forward(self, x, context):
        realness = self.main(x)
        ctx_in = torch.cat([x, context], dim=1)
        context_consist = self.context_branch(ctx_in)
        return realness, context_consist
