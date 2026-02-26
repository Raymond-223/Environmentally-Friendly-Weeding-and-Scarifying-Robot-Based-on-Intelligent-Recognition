#!/usr/bin/env python3
# 检测模型训练脚本示例：DGL-DETR / FineGrained（需自备数据集）

import sys
import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DummyWeedDataset(Dataset):
    """占位数据集：随机张量，用于验证训练流程."""

    def __init__(self, num_samples=100, size=640, num_classes=3):
        self.num_samples = num_samples
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.rand(3, self.size, self.size)
        weed_label = torch.randint(0, self.num_classes, (1,)).item()
        depth_label = torch.randint(0, 3, (1,)).item()
        growth_label = torch.randint(0, 2, (1,)).item()
        bbox = torch.tensor([0.25, 0.25, 0.5, 0.5])
        return img, {
            "weed_label": weed_label,
            "depth_label": depth_label,
            "growth_label": growth_label,
            "bbox": bbox,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dgl_detr", choices=["dgl_detr", "fine_grained"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "fine_grained":
        from detection.models.fine_grained import FineGrainedDGL_DETR
        model = FineGrainedDGL_DETR(num_weeds=8, num_depth=3, num_growth=2)
        dataset = DummyWeedDataset(num_samples=80, num_classes=8)
        def collate(batch):
            imgs = torch.stack([b[0] for b in batch])
            targets = {
                "weed_label": torch.tensor([b[1]["weed_label"] for b in batch]),
                "depth_label": torch.tensor([b[1]["depth_label"] for b in batch]),
                "growth_label": torch.tensor([b[1]["growth_label"] for b in batch]),
                "bbox": torch.stack([b[1]["bbox"] for b in batch]),
            }
            return imgs, targets
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate)
    else:
        from detection.models.dgl_detr import DGL_DETR
        model = DGL_DETR(num_classes=3)
        dataset = DummyWeedDataset(num_classes=3)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
        collate = None

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                imgs = batch[0].to(device)
                targets = batch[1]
                if isinstance(targets, dict):
                    targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
            else:
                imgs = batch.to(device)
                targets = None

            out = model(imgs, targets)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], torch.Tensor) and out[0].dim() == 0:
                loss, _ = out
            elif args.model == "dgl_detr":
                reg, cls = out[0], out[1]
                # 占位：无标注时用分类头 L2 正则
                loss = cls.pow(2).mean()
            else:
                loss, _ = out
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(n,1):.4f}")

    path = os.path.join(args.out, f"{args.model}.pth")
    torch.save(model.state_dict(), path)
    print("Saved:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
