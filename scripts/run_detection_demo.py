#!/usr/bin/env python3
# 检测演示：图像/摄像头 -> DGL-DETR 杂草检测 -> 可视化 + 除草策略

import sys
import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np

# 可选：从 config 读
try:
    import yaml
    cfg_path = os.path.join(ROOT, "config", "default.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    else:
        CFG = {}
except Exception:
    CFG = {}


def main():
    parser = argparse.ArgumentParser(description="杂草检测演示")
    parser.add_argument("--source", type=str, default="0", help="摄像头(0)或图像/视频路径")
    parser.add_argument("--checkpoint", type=str, default="", help="模型权重路径(可选)")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    args = parser.parse_args()

    from detection.inference import WeedDetector
    from control.weed_control import weed_control_strategy

    num_classes = (CFG.get("detection") or {}).get("num_weed_classes", 8)
    # DGL-DETR 分类数：背景 + 杂草类
    num_classes = max(2, num_classes + 1) if num_classes else 3
    detector = WeedDetector(
        model_type="dgl_detr",
        num_classes=num_classes,
        checkpoint_path=args.checkpoint or None,
    )
    detector.conf_threshold = args.conf

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, int) or (isinstance(source, str) and len(source) <= 2):
        cap = cv2.VideoCapture(int(source) if isinstance(source, str) else source)
        if not cap.isOpened():
            print("无法打开摄像头:", args.source)
            return 1
        single_image = False
    else:
        if not os.path.isfile(source):
            print("文件不存在:", source)
            return 1
        ext = os.path.splitext(source)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp"):
            frame = cv2.imread(source)
            single_image = True
            cap = None
        else:
            cap = cv2.VideoCapture(source)
            single_image = False

    def run_one(img):
        boxes, scores, labels = detector.run(img)
        vis = img.copy()
        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis, f"#{lb} {sc:.2f}", (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 除草策略示例（细粒度时可用 depth_level, growth_stage）
            strat = weed_control_strategy(int(lb), 1, 0)
            cv2.putText(vis, f"P{strat['power']} T{strat['duration']}ms",
                       (int(x1), int(y2) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        return vis

    if single_image:
        out = run_one(frame)
        cv2.imshow("detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    print("按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = run_one(frame)
        cv2.imshow("detection", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
