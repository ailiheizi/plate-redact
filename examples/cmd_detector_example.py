#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
外部检测器桥的示例实现（--detector cmd:python3 examples/cmd_detector_example.py）

这个脚本演示 adapters.CommandDetector 的 NDJSON 协议：把任意检测器（包括本仓库
不打包的第三方 GPL/AGPL 实现，或你自己的模型）接到打码流程里，而不需要修改本仓库。

协议（每行一个 JSON，UTF-8）:
  父 -> 子: {"frame": 0, "width": 1920, "height": 1080, "image_png_b64": "..."}
  子 -> 父: {"detections": [{"rect": [x1, y1, x2, y2], "score": 0.9, "text": "AB123"}]}

注意:
  - stdout 只能输出协议 JSON；日志一律写 stderr，否则会污染协议。
  - 子进程退出/超时/返回非法 JSON 会被视为致命错误，fail-closed 下会中止整段渲染。

依赖: 仅需 numpy 与 opencv-python。
"""
import base64
import json
import sys

import cv2
import numpy as np


def detect_plates(frame_bgr):
    """在这里接入你自己的检测器，返回 [(x1, y1, x2, y2, score, text), ...]

    下面是"车牌颜色反色"的玩具示例，仅用于演示协议，不具备实用性：
    真实场景请在此调用你的模型（可自由使用任何许可证的代码）。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (180, 60, 255))  # 亮且低饱和的块
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 12 or w / max(h, 1) < 1.5:
            continue
        out.append((float(x), float(y), float(x + w), float(y + h), 0.5, ''))
    return out


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f'非法请求: {exc}', file=sys.stderr)
            return 1
        raw = base64.b64decode(req.get('image_png_b64', ''))
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f'帧 {req.get("frame")} 解码失败', file=sys.stderr)
            return 1
        detections = [
            {'rect': [b[0], b[1], b[2], b[3]], 'score': b[4], 'text': b[5]}
            for b in detect_plates(frame)
        ]
        sys.stdout.write(json.dumps({'detections': detections}) + '\n')
        sys.stdout.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
