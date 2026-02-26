# RGB-D + 多光谱四模态融合（CMAF）（报告 3.2.2）

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgl_detr import C2f_Deform, GCNet


class SelfAttention(nn.Module):
    """自注意力."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = max(dim // heads, 1)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class CrossAttention(nn.Module):
    """交叉注意力."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = max(dim // heads, 1)
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(b, context.size(1), 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class CMAF(nn.Module):
    """跨模态注意力融合：RGB / NIR / Depth / Thermal."""

    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.rgb_proj = nn.Conv2d(3, dim, 1, 1)
        self.nir_proj = nn.Conv2d(6, dim, 1, 1)
        self.depth_proj = nn.Conv2d(1, dim, 1, 1)
        self.thermal_proj = nn.Conv2d(1, dim, 1, 1)

        self.rgb_attn = SelfAttention(dim, heads)
        self.nir_attn = SelfAttention(dim, heads)
        self.depth_attn = SelfAttention(dim, heads)
        self.thermal_attn = SelfAttention(dim, heads)

        self.cross_attn1 = CrossAttention(dim, heads)
        self.cross_attn2 = CrossAttention(dim, heads)
        self.cross_attn3 = CrossAttention(dim, heads)

        self.fusion_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, rgb, nir, depth, thermal):
        b, _, h, w = rgb.shape
        rgb_feat = self.rgb_proj(rgb).flatten(2).transpose(1, 2)
        nir_feat = self.nir_proj(nir).flatten(2).transpose(1, 2)
        depth_feat = self.depth_proj(depth).flatten(2).transpose(1, 2)
        thermal_feat = self.thermal_proj(thermal).flatten(2).transpose(1, 2)

        rgb_att = self.rgb_attn(rgb_feat)
        nir_att = self.nir_attn(nir_feat)
        depth_att = self.depth_attn(depth_feat)
        thermal_att = self.thermal_attn(thermal_feat)

        fusion1 = self.cross_attn1(rgb_att, nir_att)
        fusion2 = self.cross_attn2(depth_att, thermal_att)
        final = self.cross_attn3(fusion1, fusion2).transpose(1, 2).view(b, -1, h, w)
        out = self.act(self.norm(self.fusion_conv(final)))
        return out


class FourModalDGL_DETR(nn.Module):
    """四模态融合 + 主干 + 检测头（简化：复用单模态 backbone 结构）."""

    def __init__(self, num_classes=10, dim=512):
        super().__init__()
        self.cmaf = CMAF(dim=dim)
        self.backbone = nn.Sequential(
            C2f_Deform(dim, dim, 4),
            GCNet(dim),
        )
        self.reg_head = nn.Sequential(nn.Conv2d(dim, 256, 3, 1, 1), nn.ReLU(), nn.Conv2d(256, 4, 1))
        self.cls_head = nn.Sequential(nn.Conv2d(dim, 256, 3, 1, 1), nn.ReLU(), nn.Conv2d(256, num_classes, 1))
        self.crop_height_threshold = 0.07

    def forward(self, rgb, nir, depth, thermal):
        fused = self.cmaf(rgb, nir, depth, thermal)
        feat = self.backbone(fused)
        reg_out = self.reg_head(feat)
        cls_out = self.cls_head(feat)
        return reg_out, cls_out
