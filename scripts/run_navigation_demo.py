#!/usr/bin/env python3
# 导航演示：摄像头/视频 -> 自适应垄线检测 + 融合 + 路径跟踪（无 ROS）

import sys
import os
import argparse

# 将项目根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np
from navigation.ridge_detection import adaptive_ridge_detection
from navigation.multispectral_fusion import multispectral_fusion
from navigation.navigator import WeedRobotNavigator


def main():
    parser = argparse.ArgumentParser(description="导航演示：垄线检测与路径跟踪")
    parser.add_argument("--source", type=str, default="0", help="摄像头索引(0)或视频路径")
    parser.add_argument("--no-nav", action="store_true", help="仅做垄线检测与融合，不运行完整导航")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("无法打开视频源:", args.source)
        return 1

    nav = WeedRobotNavigator(ridge_prior=0.8, safe_distance=1.0) if not args.no_nav else None
    # 模拟 NIR：用灰度图代替
    use_fusion = True

    print("按 q 退出。若 --no-nav：仅显示垄线检测与融合结果。")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lightness = float(np.mean(gray))
        # 用灰度图模拟 NIR
        nir_sim = gray
        if use_fusion:
            fused, nir_w = multispectral_fusion(frame, nir_sim, lightness)
            work = fused
        else:
            work = frame

        lines = adaptive_ridge_detection(work, lightness)
        vis = work.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0].astype(int)
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if nav is not None:
            nav.set_nir(nir_sim)
            linear_x, angular_z, idx = nav.step(frame)
            cv2.putText(
                vis, f"v={linear_x:.2f} w={angular_z:.2f} idx={idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        cv2.imshow("navigation", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
