# -*- coding: UTF-8 -*-
"""
轻量单引擎版: 车牌检测 + 纯色填充打码(参考实现)

与 plate_redact.py 的分工:
  plate_redact.py     完整引擎 —— 多引擎并集 / 双遍渲染 / 回溯提前打码 / 审计发布
  plate_redact_eu.py  轻量参考实现 —— 单引擎(fast-alpr) + 轻量帧间保持(BoxTracker)

两条路径共用同一套打码原语(纯色填充)、帧间保持思路与审计发布门禁, 只是这里的
跟踪器更简单(单类、速度平滑 + 文本投票), 适合先跑通再逐步加引擎。

面向车牌为白底深字的场景(如欧洲), 因此额外做了两条几何约束:
  - 高置信度框放宽宽高比, 低置信度框只拒极端形状(倾斜/透视车牌比例会到 0.5~1.5)
  - 框区域平均亮度过低视为深色车身误检(白底车牌不可能整块全黑)

用法:
  图片目录:  python plate_redact_eu.py --source photos/ --output out/
  视频文件:  python plate_redact_eu.py --source dashcam.mp4 --output out.mp4
  并集:      python plate_redact_eu.py --source in.mp4 --output out.mp4 \
                 --detector fast-alpr --detector hyperlpr3
"""
import argparse
import os
import time
from collections import Counter

import cv2

from adapters import DetectorError, build_detectors, build_vehicle_detector
from plate_redact import (audit_render, clamp_box, expand_box, redact_roi,
                          scene_switch_score, _remove)


# --------------------------------------------------------------------------- #
# 几何约束与填充
# --------------------------------------------------------------------------- #
def eu_filter(dets, frame_shape=None, pad_ratio=0.06, min_bright=40, frame=None):
    """按车牌几何/亮度特征过滤误检, 并处理被画面裁掉的车牌

    - 高置信度框(>=0.55)只拒极端宽高比, 兼容竖排/透视视角的大车车牌
    - 低置信度框大幅放宽: 单帧几何不可靠, 误检主要靠跟踪器多帧一致性过滤
    - 贴边的框向外扩展: 车牌被画面裁掉一部分时, 不扩展就会漏掉外露的一角
    - 区域过暗(深色车身)直接丢弃
    """
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 8:
            continue
        ratio = w / max(h, 1e-6)
        if d.score >= 0.55:
            if ratio < 0.4 or ratio > 10.0:
                continue
        else:
            if ratio < 0.5 or ratio > 8.0:
                continue
        box = [x1, y1, x2, y2]
        if frame_shape is not None:
            box = expand_edge_box(box, frame_shape, pad_ratio)
        if frame is not None and frame_shape is not None:
            if box_region_too_dark(frame, box, min_bright):
                continue
        d.rect = box
        out.append(d)
    return out


def expand_edge_box(box, frame_shape, pad_ratio):
    """检测框贴近画面边缘时向外扩展, 覆盖被裁掉的车牌部分"""
    x1, y1, x2, y2 = [float(v) for v in box]
    h_img, w_img = frame_shape[:2]
    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    margin = 12  # 距边缘多少像素内视为被裁剪
    if x1 <= margin:
        x1 = max(0, x1 - pw)
    if y1 <= margin:
        y1 = max(0, y1 - ph)
    if x2 >= w_img - margin:
        x2 = min(w_img, x2 + pw)
    if y2 >= h_img - margin:
        y2 = min(h_img, y2 + ph)
    return [x1, y1, x2, y2]


def box_region_too_dark(frame, box, min_bright=40):
    """区域平均亮度过低 → 夜间深色车身误检(白底车牌应较亮)"""
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return True
    g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(g.mean()) < min_bright


def pad_box_roi(box, frame_shape, pad_ratio):
    """打码框外扩: 覆盖检测框抖动 + 最小可感知尺寸保证"""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    pw, ph = w * pad_ratio, h * pad_ratio
    min_w, min_h = 44.0, 20.0
    if w < min_w:
        pw = max(pw, (min_w - w) / 2)
    if h < min_h:
        ph = max(ph, (min_h - h) / 2)
    return clamp_box([x1 - pw, y1 - ph, x2 + pw, y2 + ph], frame_shape)


def iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


# --------------------------------------------------------------------------- #
# 轻量帧间保持
# --------------------------------------------------------------------------- #
class BoxTracker:
    """单类帧间保持: 检测失败/短暂漏检时沿用上一帧框, 匀速外推 + 平滑, 消除闪烁

    比 plate_tracker.PlateTracker 简单: 没有两级关联与多目标 ID, 只维护
    "还在画面里的打码框"。框中心驶出画面或退化时立即丢弃, 避免色块残留在边缘。
    """

    def __init__(self, max_age=12, iou_thr=0.3, smooth=0.5):
        self.max_age = max_age
        self.iou_thr = iou_thr
        self.smooth = smooth
        self.locked = []      # [{'box','vel','text','texts','age'}]
        self.frame_shape = None

    def reset(self):
        self.locked = []

    def set_frame_shape(self, shape):
        self.frame_shape = shape

    def _out_of_view(self, box, ratio=0.2):
        """已驶出画面: 可见部分占比低于 ratio 或中心点出画面"""
        if self.frame_shape is None:
            return False
        img_h, img_w = self.frame_shape[:2]
        x1, y1, x2, y2 = box
        vis_w = max(0, min(x2, img_w) - max(x1, 0))
        vis_h = max(0, min(y2, img_h) - max(y1, 0))
        if vis_w / max(x2 - x1, 1e-6) < ratio or vis_h / max(y2 - y1, 1e-6) < ratio:
            return True
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return cx < 0 or cx > img_w or cy < 0 or cy > img_h

    @staticmethod
    def _too_small(box):
        return box[2] - box[0] < 10 or box[3] - box[1] < 6

    @staticmethod
    def _vote_text(texts):
        """多帧文本投票: 取出现最多的, 减少识别抖动(仅用于日志/标注)"""
        if not texts:
            return ''
        cnt = Counter(texts)
        best, n = cnt.most_common(1)[0]
        return best if n >= 2 else texts[-1]

    def _smoothed(self, prev_box, vel, new_box):
        sm = self.smooth
        pred = [prev_box[k] + vel[k] for k in range(4)]
        return [pred[k] * (1 - sm) + new_box[k] * sm for k in range(4)]

    def update(self, dets):
        """dets: [(box, text, conf)] -> 返回匹配/平滑后的 [(box, text, conf)]"""
        matched = set()
        out = []
        for box, text, conf in dets:
            best_i, best_s = -1, 0
            for i, t in enumerate(self.locked):
                if i in matched:
                    continue
                s = iou(box, t['box'])
                if s > best_s:
                    best_s, best_i = s, i
            if best_i >= 0 and best_s >= self.iou_thr:
                t = self.locked[best_i]
                new_box = self._smoothed(t['box'], t['vel'], box)
                vel = [new_box[k] - t['box'][k] for k in range(4)]
                vel = [max(-12.0, min(12.0, v)) for v in vel]  # 限幅防跳变
                t['box'], t['vel'], t['age'] = new_box, vel, 0
                if text:
                    t['texts'] = (t['texts'] + [text])[-8:]
                    t['text'] = self._vote_text(t['texts'])
                matched.add(best_i)
                out.append((new_box, t['text'], conf))
            else:
                out.append((box, text, conf))
        for box, text, conf in dets:
            if not any(j in matched and iou(box, t['box']) >= self.iou_thr
                       for j, t in enumerate(self.locked)):
                self.locked.append({'box': list(box), 'vel': [0.0] * 4,
                                    'text': text, 'texts': [text] if text else [],
                                    'age': 0})
        fresh = []
        for i, t in enumerate(self.locked):
            if i in matched:
                fresh.append(t)
            elif t['age'] < self.max_age:
                t['age'] += 1
                t['box'] = [t['box'][k] + t['vel'][k] for k in range(4)]  # 未匹配: 按速度外推
                if self._out_of_view(t['box']) or self._too_small(t['box']):
                    continue
                fresh.append(t)
        self.locked = fresh
        return out

    def carry_over(self, force_clear=False):
        """检测为空时沿用仍在有效期内的框(带运动预测, 框连续不跳变)"""
        if force_clear:
            self.locked = []
            return []
        out, fresh = [], []
        for t in self.locked:
            t['box'] = [t['box'][k] + t['vel'][k] for k in range(4)]
            if self._out_of_view(t['box']) or self._too_small(t['box']):
                continue
            if t['age'] >= self.max_age:
                continue
            t['age'] += 1
            fresh.append(t)
            out.append((t['box'], t['text'], 0.0))
        self.locked = fresh
        return out


# --------------------------------------------------------------------------- #
# 图片 / 视频
# --------------------------------------------------------------------------- #
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def process_images(source, output, detector, args):
    os.makedirs(output, exist_ok=True)
    files = sorted(os.path.join(source, f) for f in os.listdir(source)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXT)
    total = 0.0
    for i, path in enumerate(files):
        img = cv2.imread(path)
        if img is None:
            print(f'[skip] {path}')
            continue
        t0 = time.time()
        dets = eu_filter(detector.detect(img) or [], img.shape, frame=img)
        cost = time.time() - t0
        total += cost
        for d in dets:
            redact_roi(img, pad_box_roi(d.box, img.shape, args.pad), args.fill)
            if args.annotate:
                x1, y1, x2, y2 = clamp_box(d.box, img.shape)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output, os.path.basename(path)), img)
        print(f'{i} {os.path.basename(path)} [{cost*1000:.0f}ms] 车牌 {len(dets)} 个')
    if files:
        print(f'图片处理完成: {len(files)} 张, 平均 {total/len(files)*1000:.0f} ms/张')


def process_video(source, output, detector, args, vehicle_detector=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f'无法打开视频: {source}')
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    draft = output + '.draft.mp4'
    writer = cv2.VideoWriter(draft, fourcc, fps_in, (w, h))
    if not writer.isOpened():
        raise SystemExit(f'无法创建草稿视频: {draft}')

    tracker = BoxTracker(max_age=args.track_age, iou_thr=args.track_iou,
                         smooth=args.smooth)
    tracker.set_frame_shape((h, w))
    planned, prev_gray, prev_frame = {}, None, None
    still_count, miss_count, frame_idx = 0, 0, 0
    t_all = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break
            t0 = time.time()
            # 画面静止(暗场/车位空/镜头不动)时清空保持框, 避免色块挂在不动的画面上
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(cv2.absdiff(gray, prev_gray).mean())
                still_count = still_count + 1 if diff < 6.0 else 0
                # 镜头切换: 直方图相关性骤降(画面完全更换), 立即清空保持框
                if args.scene_thresh > 0 and \
                        scene_switch_score(prev_frame, frame) < args.scene_thresh:
                    tracker.reset()
                    still_count = 0
                    print(f'[镜头切换] 帧 {frame_idx}')
            prev_gray, prev_frame = gray, frame
            if still_count >= args.still_clear:
                tracker.reset()

            dets = eu_filter(detector.detect(frame) or [], frame.shape, args.pad, frame=frame)
            track_dets, miss_count = _track_step(tracker, dets, miss_count)
            for box, text, conf in track_dets:
                pad_box = pad_box_roi(box, frame.shape, args.pad)
                redact_roi(frame, pad_box, args.fill)
                planned.setdefault(frame_idx, []).append(pad_box)
                if args.annotate:
                    x1, y1, x2, y2 = clamp_box(pad_box, frame.shape)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, text, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 车辆区域兜底(两轮车尾部窄条), 只在车牌检测/保持都为空时触发
            if args.vehicle_fallback and vehicle_detector is not None and not track_dets:
                for vb in vehicle_detector.detect(frame, args.vehicle_conf) or []:
                    x1, y1, x2, y2 = [float(v) for v in vb[:4]]
                    if x2 - x1 < 30 or y2 - y1 < 20:
                        continue
                    cls = int(vb[5]) if len(vb) > 5 else -1
                    if cls not in (1, 3, -1):
                        continue
                    tail = [x1, y1 + (y2 - y1) * 0.5, x2, y2]
                    pad_box = pad_box_roi(tail, frame.shape, args.pad)
                    redact_roi(frame, pad_box, args.fill)
                    planned.setdefault(frame_idx, []).append(pad_box)
            writer.write(frame)
            if frame_idx % 30 == 0:
                print(f'帧 {frame_idx}: 打码 {len(track_dets)} 块, '
                      f'{1.0 / max(1e-6, time.time() - t0):.1f} fps')
    except DetectorError as exc:
        cap.release()
        writer.release()
        _remove(draft)
        raise SystemExit(f'检测器失败, 已中止且未产出任何视频: {exc}')
    finally:
        if cap.isOpened():
            cap.release()
        if writer.isOpened():
            writer.release()

    blocking, warnings = audit_render(source, draft, planned,
                                      flat_tol=args.audit_flat_tol)
    for msg in warnings[:20]:
        print(f'[审计][warn] {msg}')
    if blocking:
        for msg in blocking[:20]:
            print(f'[审计][阻断] {msg}')
        if args.fail_closed:
            _remove(draft)
            _remove(output)
            raise SystemExit(
                f'审计未通过({len(blocking)} 项), fail-closed 已丢弃草稿, 未产出成品')
        print('[warn] --fail-closed 0: 审计未通过仍然发布, 该文件不可作为合规成品')
    os.replace(draft, output)
    print(f'视频处理完成: {frame_idx} 帧, 用时 {time.time()-t_all:.1f}s, 输出: {output}')


def _track_step(tracker, dets, miss_count):
    """一帧的跟踪推进: 有检出就关联, 没有就沿用保持框"""
    if dets:
        return tracker.update([(tuple(d.box), d.text, d.score) for d in dets]), 0
    return tracker.carry_over(), miss_count + 1


def build_parser():
    p = argparse.ArgumentParser(description='车牌检测 + 纯色填充打码(轻量单引擎版)')
    p.add_argument('--source', required=True, help='图片目录/图片文件/视频文件')
    p.add_argument('--output', default='out', help='输出目录(图片)或视频文件路径')
    p.add_argument('--detector', action='append', default=None,
                   help='检测器规格, 可重复取并集: fast-alpr[:检测模型[:OCR模型[:设备]]]'
                        ' | hyperlpr3 | vlm:<权重目录> | cmd:<命令>')
    p.add_argument('--detector-conf', type=float, default=0.3)
    p.add_argument('--hyper-conf', type=float, default=0.35)
    p.add_argument('--vehicle-detector', default='', help='ultralytics:<权重> | cmd:<命令>')
    p.add_argument('--vehicle-conf', type=float, default=0.2)
    p.add_argument('--vehicle-fallback', type=int, default=0,
                   help='车辆区域兜底: 无车牌候选时对两轮车尾部窄条加打码')
    p.add_argument('--fill', default='mean', choices=['mean', 'black'], help='填充方式')
    p.add_argument('--pad', type=float, default=0.35, help='打码框外扩比例')
    p.add_argument('--track-age', type=int, default=90, help='保持框最大帧数')
    p.add_argument('--track-iou', type=float, default=0.3, help='帧间关联 IoU 阈值')
    p.add_argument('--smooth', type=float, default=0.55, help='框平滑系数(0-1)')
    p.add_argument('--still-clear', type=int, default=30, help='连续静止多少帧后清空保持框')
    p.add_argument('--scene-thresh', type=float, default=0.70, help='镜头切换阈值(0=关闭)')
    p.add_argument('--fail-closed', type=int, default=1, help='审计不通过时丢弃草稿')
    p.add_argument('--audit-flat-tol', type=float, default=6.0, help='纯色判定容差')
    p.add_argument('--fourcc', default='mp4v')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--annotate', type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.detector is None:
        args.detector = ['fast-alpr']
    src = args.source
    if src.strip().lstrip('+-').isdigit():
        raise SystemExit('摄像头源无法做双遍校验与审计, 请先录制成视频文件再处理')
    if not os.path.exists(src):
        raise SystemExit(f'输入不存在: {src}')

    detector = build_detectors(args.detector, conf=args.detector_conf,
                               hyper_conf=args.hyper_conf)
    vehicle_detector = build_vehicle_detector(args.vehicle_detector) \
        if args.vehicle_fallback else None
    try:
        if os.path.isdir(src):
            process_images(src, args.output, detector, args)
            return 0
        ext = os.path.splitext(src)[1].lower()
        if ext in IMAGE_EXT:
            img = cv2.imread(src)
            for d in eu_filter(detector.detect(img) or [], img.shape, frame=img):
                redact_roi(img, pad_box_roi(d.box, img.shape, args.pad), args.fill)
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            cv2.imwrite(args.output, img)
            print(f'-> {args.output}')
            return 0
        if os.path.exists(args.output):
            _remove(args.output)
            print('[fail-closed] 已移除同名旧输出, 避免误认为本次结果')
        process_video(src, args.output, detector, args, vehicle_detector)
        return 0
    finally:
        detector.close()
        if vehicle_detector is not None:
            vehicle_detector.close()


if __name__ == '__main__':
    raise SystemExit(main())
