# -*- coding: UTF-8 -*-
"""
车牌检测 + 纯色填充打码引擎（可插拔检测器 / 时间冗余跟踪 / 双遍渲染 / fail-closed 发布）

设计目标: 输出视频里任何"人眼可读的车牌"都必须被遮挡, 而不是"模型在测试集上准确率高"。
因此本工具不追求实时, 而是把算力花在召回与可校验性上:

  1. 多引擎并集      --detector 可重复传入, 取并集(任何单引擎都有盲区)
  2. 时间冗余        tracker 多帧确认 + 丢检时沿用已确认框(运动预测外推)
  3. 提前打码        --pre-frames: 用轨迹最早速度把框反推到车牌"刚进入画面/还模糊"的帧
  4. 纯色填充        区域平均色填充, 不留任何字符结构(马赛克/模糊在边缘仍可辨字符)
  5. fail-closed     先写 .draft, 通过审计才原子发布; 审计不通过就丢弃草稿, 不留未经遮挡的成品

第三方检测引擎一律通过 adapters.py 的可插拔接口接入, 本仓库不打包其代码与权重。
详见 README.md。

用法:
  图片:  python plate_redact.py --source in_dir --output out_dir
  视频:  python plate_redact.py --source in.mp4 --output out.mp4 --pre-frames 15
  并集:  python plate_redact.py --source in.mp4 --output out.mp4 \
             --detector fast-alpr --detector hyperlpr3 --detector cmd:python3 my_det.py
"""
import argparse
import os
import time

import cv2
import numpy as np

from adapters import (Detection, DetectorError, build_detectors,
                      build_vehicle_detector)
from plate_tracker import PlateTracker


# --------------------------------------------------------------------------- #
# 基础图像工具
# --------------------------------------------------------------------------- #
def to_uint8_bgr(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def clamp_box(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    return [x1, y1, x2, y2]


def redact_roi(img, box, fill='mean'):
    """纯色填充: 用区域平均色(或纯黑)整块覆盖车牌。

    为什么不用马赛克/高斯模糊: 模糊只是降低高频, 在边缘与低分辨率下仍可能
    残留可辨的字符轮廓; 纯色填充在信息论意义上直接抹掉整块区域, 无字符结构可还原。
    """
    x1, y1, x2, y2 = clamp_box(box, img.shape)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = (0, 0, 0) if fill == 'black' else roi.mean(axis=(0, 1))


def expand_box(box, frame_shape, ratio=1.6):
    """以框中心为基准按比例放大(覆盖检测框抖动), 并裁剪到画面内"""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = w * ratio, h * ratio
    return clamp_box([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], frame_shape)


def nms_detections(dets, iou_thr=0.35):
    """按置信度降序的 NMS, 去掉同一块牌的重叠候选"""
    if len(dets) <= 1:
        return dets
    boxes = np.array([list(d.box) for d in dets], dtype=float)
    confs = np.array([d.score for d in dets])
    order = np.argsort(-confs)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        b = boxes[i]
        o = boxes[rest]
        ix1 = np.maximum(b[0], o[:, 0]); iy1 = np.maximum(b[1], o[:, 1])
        ix2 = np.minimum(b[2], o[:, 2]); iy2 = np.minimum(b[3], o[:, 3])
        iw = np.maximum(0, ix2 - ix1); ih = np.maximum(0, iy2 - iy1)
        inter = iw * ih
        union = ((b[2] - b[0]) * (b[3] - b[1]) + (o[:, 2] - o[:, 0]) * (o[:, 3] - o[:, 1])
                 - inter)
        iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0)
        order = rest[iou < iou_thr]
    return [dets[k] for k in keep]


def scene_switch_score(f1, f2):
    """帧间场景切换评分: 归一化灰度直方图相关系数, 越小差异越大

    镜头切换时上一场景的跟踪框必须立即清空, 否则误检框会"残留"到新场景,
    在完全无关的画面上打出一块色块(既是隐私误报也是画质事故)。
    """
    small1 = cv2.resize(f1, (64, 36))
    small2 = cv2.resize(f2, (64, 36))
    g1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [32], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [32], [0, 256])
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


# 可选的车牌号格式校验(默认关闭, 见 README「隐私优先的取舍」):
# 过滤误检会同时过滤掉真实车牌 -> 漏打码, 属于不可接受的方向, 因此默认只用于日志。
PLATE_RE = None  # 由 --plate-regex 注入, 默认使用下面这个常见格式
DEFAULT_PLATE_REGEX = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]{1}[A-Z0-9]{4,7}$'


def is_valid_plate(plate_no, regex=None):
    import re
    if not plate_no:
        return False
    return bool(re.match(regex or DEFAULT_PLATE_REGEX, plate_no.strip()))


# --------------------------------------------------------------------------- #
# 检测: 局部重检 / 放大兜底 (全部走适配器, 不 import 任何第三方引擎)
# --------------------------------------------------------------------------- #
def detect_in_roi(detector, frame, roi_rect, frame_shape, source='roi'):
    """在指定区域(裁剪放大)内做检测, 返回映射回原图坐标的 Detection 列表"""
    x1, y1, x2, y2 = clamp_box(roi_rect, frame_shape)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return []
    roi = frame[y1:y2, x1:x2]
    out = []
    for d in detector.detect(roi) or []:
        bx1, by1, bx2, by2 = d.box
        out.append(Detection(rect=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                             score=d.score, text=d.text, source=source))
    return out


def zoom_fallback_detect(detector, frame, img_size=None):
    """放大兜底: 全图 2x, 不行再 2x2 分块各 2x

    远距/低对比度车牌在原分辨率下检测器会漏, 放大后小目标变成中目标即可召回。
    非实时流程, 这个代价换召回是划算的。
    """
    results = []
    h, w = frame.shape[:2]
    f2 = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    for d in detector.detect(f2) or []:
        bx1, by1, bx2, by2 = d.box
        results.append(Detection(rect=(bx1 / 2, by1 / 2, bx2 / 2, by2 / 2),
                                 score=d.score, text=d.text, source='zoom'))
    if results:
        return results
    for gy in range(2):
        for gx in range(2):
            x1, y1 = gx * w // 2, gy * h // 2
            x2, y2 = (gx + 1) * w // 2, (gy + 1) * h // 2
            tile = frame[y1:y2, x1:x2]
            tile2 = cv2.resize(tile, (tile.shape[1] * 2, tile.shape[0] * 2),
                               interpolation=cv2.INTER_CUBIC)
            for d in detector.detect(tile2) or []:
                bx1, by1, bx2, by2 = d.box
                results.append(Detection(rect=(bx1 / 2 + x1, by1 / 2 + y1,
                                               bx2 / 2 + x1, by2 / 2 + y1),
                                         score=d.score, text=d.text, source='zoom'))
    return results


# --------------------------------------------------------------------------- #
# 兜底: 车辆区域加打码(只加不减)
# --------------------------------------------------------------------------- #
def vehicle_fallback_boxes(vehicle_detector, frame, conf=0.2, two_wheeler_only=True):
    """返回需要"兜底加打码"的车辆尾部区域 [(x1,y1,x2,y2), ...]

    注意方向性: 车辆检测只用来"多打一块", 不用来"过滤候选"。车牌检测器漏检
    时, 车辆还在画面里就说明牌也还在; 与其输出未遮挡帧, 不如在车尾下部打一条窄条。
    两轮车(自行车/摩托车)只打下部窄条, 不糊车身。
    """
    if vehicle_detector is None:
        return []
    out = []
    h_img, w_img = frame.shape[:2]
    for vb in vehicle_detector.detect(frame, conf=conf) or []:
        x1, y1, x2, y2 = [float(v) for v in vb[:4]]
        conf_v = float(vb[4]) if len(vb) > 4 else 0.0
        cls = int(vb[5]) if len(vb) > 5 else -1
        w, h = x2 - x1, y2 - y1
        if w < 30 or h < 20 or conf_v < conf:
            continue
        if two_wheeler_only and cls not in (1, 3, -1):
            continue  # 只兜底两轮车; 轿车/卡车靠车牌检测器精确打码
        if h > h_img * 0.5 or w > w_img * 0.5:
            out.append([x1, y1, x2, y2])          # 占画面过半的近距离大车: 整车保底
        else:
            out.append([x1, y1 + h * 0.5, x2, y1 + h * 0.9])  # 车尾下部窄条
    return out


# --------------------------------------------------------------------------- #
# 审计: 双遍渲染后逐帧校验, 不通过就不发布
# --------------------------------------------------------------------------- #
def audit_render(source_path, rendered_path, planned, flat_tol=6.0, verbose=True):
    """校验渲染结果: 每一处"计划打码"的位置在成品里必须真的是纯色块。

    检查项:
      - 帧数一致(渲染过程不能丢帧/多帧)
      - 每个计划框在成品中近似纯色(逐通道标准差 <= flat_tol) -> 字符结构已消失
      - 计划框相对源帧确有变化(否则说明框错位, 记为 warning)

    返回 (blocking_issues, warnings)。blocking 非空时调用方必须丢弃草稿。
    """
    blocking, warnings = [], []
    cap_s = cv2.VideoCapture(source_path)
    cap_r = cv2.VideoCapture(rendered_path)
    if not cap_s.isOpened() or not cap_r.isOpened():
        return ['无法打开源或渲染结果进行校验'], warnings
    idx = 0
    checked = 0
    while True:
        ok_s, fr_s = cap_s.read()
        ok_r, fr_r = cap_r.read()
        if not ok_s or not ok_r:
            if ok_s != ok_r:
                blocking.append(
                    f'帧数不一致: 源在帧 {idx} 结束, 渲染结果 ok={ok_r}; 反之亦然')
            break
        idx += 1
        boxes = planned.get(idx)
        if not boxes:
            continue
        if fr_s.shape != fr_r.shape:
            blocking.append(f'帧 {idx} 尺寸不一致: {fr_s.shape} vs {fr_r.shape}')
            continue
        for box in boxes:
            x1, y1, x2, y2 = clamp_box(box, fr_r.shape)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            region = fr_r[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
            std = float(region.std(axis=0).max())
            checked += 1
            if std > flat_tol:
                blocking.append(
                    f'帧 {idx} 框 {[x1, y1, x2, y2]} 未成为纯色块 (std={std:.1f} > {flat_tol})')
            mad = float(np.abs(fr_s[y1:y2, x1:x2].astype(np.float32)
                               - fr_r[y1:y2, x1:x2].astype(np.float32)).mean())
            if mad < 1.0:
                warnings.append(f'帧 {idx} 框 {[x1, y1, x2, y2]} 与源几乎一致 (mad={mad:.2f})')
    cap_s.release()
    cap_r.release()
    if verbose:
        print(f'[审计] 校验 {checked} 处打码区域, {idx} 帧; '
              f'阻断问题 {len(blocking)} 项, 警告 {len(warnings)} 项')
    return blocking, warnings


def _remove(path, verbose=True):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            if verbose:
                print(f'[清理] 已删除 {path}')
    except OSError as exc:
        print(f'[warn] 删除 {path} 失败: {exc}')


# --------------------------------------------------------------------------- #
# 图片
# --------------------------------------------------------------------------- #
def process_images(source, output, detector, args):
    os.makedirs(output, exist_ok=True)
    ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    files = sorted(os.path.join(source, f) for f in os.listdir(source)
                   if os.path.splitext(f)[1].lower() in ext)
    total = 0.0
    for i, path in enumerate(files):
        img = cv2.imread(path)
        if img is None:
            print(f'[skip] {path}')
            continue
        t0 = time.time()
        dets = nms_detections(detector.detect(img) or [])
        for d in dets:
            box = expand_box(d.box, img.shape, 1.0 + args.pad * 2)
            redact_roi(img, box, args.fill)
            if args.annotate:
                x1, y1, x2, y2 = clamp_box(box, img.shape)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, d.text, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cost = time.time() - t0
        total += cost
        plates = ' '.join(d.text for d in dets if d.text) or '(未检出文字)'
        cv2.imwrite(os.path.join(output, os.path.basename(path)), img)
        print(f'{i} {os.path.basename(path)} [{cost*1000:.0f}ms] 候选 {len(dets)} 个: {plates}')
    if files:
        print(f'图片处理完成: {len(files)} 张, 平均 {total/len(files)*1000:.0f} ms/张')


# --------------------------------------------------------------------------- #
# 视频: 第一遍渲染 + 往前推第二遍 + 审计发布
# --------------------------------------------------------------------------- #
def _run_detection(frame, detector, args, tracker, vehicle_detector, scene_cut):
    """单帧检测: 主检测 -> 局部重检 -> 放大兜底 -> 车辆兜底。返回 (dets, vehicle_boxes)"""
    dets = list(detector.detect(frame) or [])
    vboxes = []
    if args.vehicle_fallback and vehicle_detector is not None:
        vboxes = vehicle_fallback_boxes(vehicle_detector, frame, args.vehicle_conf)
    # 主检测为空: 先在已确认轨迹附近低阈值重检(ROI 精化), 再全图低阈值重检
    if not dets:
        active = [t for t in tracker.active_tracks() if t[2]]
        if active and not scene_cut:
            for box, _, _ in active[:args.max_tracks]:
                dets += detect_in_roi(detector, frame,
                                      expand_box(box, frame.shape, args.roi_expand),
                                      frame.shape)
            if not dets:
                for box, _, _ in active[:args.max_tracks]:
                    dets += detect_in_roi(detector, frame,
                                          expand_box(box, frame.shape, args.roi_expand * 2),
                                          frame.shape)
        else:
            dets = detect_in_roi(detector, frame, [0, 0, frame.shape[1], frame.shape[0]],
                                 frame.shape)
    # 放大兜底(非实时): 找回小目标/低对比度车牌
    if not dets and args.zoom_fallback:
        dets = zoom_fallback_detect(detector, frame)
    dets = nms_detections(dets)
    # 隐私优先: 默认不按文本格式删候选(删候选 = 漏打码), 只在 --plate-filter 1 时过滤
    if args.plate_filter:
        dets = [d for d in dets if is_valid_plate(d.text, args.plate_regex)]
    return dets, vboxes


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

    tracker = PlateTracker(max_age=args.track_max_age,
                           iou_high=args.track_iou, iou_low=0.1,
                           conf_high=args.conf, conf_low=args.low_conf,
                           dist_high=60, dist_low=120,
                           min_hits=1, max_tracks=max(args.max_tracks, 20))
    planned = {}            # frame_idx -> [打码框, ...] 供审计与第二遍使用
    prev_frame = None
    last_cache = []         # 最近确认过的打码框(丢检时沿用 -> 时间冗余)
    frame_idx = 0
    t_all = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                print(f'[调试] 达到 --max-frames {args.max_frames}, 提前停止')
                break
            t0 = time.time()
            scene_cut = False
            if prev_frame is not None and args.scene_thresh > 0:
                if scene_switch_score(prev_frame, frame) < args.scene_thresh:
                    scene_cut = True
                    tracker.reset()
                    last_cache = []
            prev_frame = frame
            tracker.predict()

            if (frame_idx - 1) % max(1, args.interval) == 0:
                dets, vboxes = _run_detection(frame, detector, args, tracker,
                                              vehicle_detector, scene_cut)
                dets = sorted(dets, key=lambda d: -d.score)
                tracker.update([(d.box[0], d.box[1], d.box[2], d.box[3], d.score,
                                 is_valid_plate(d.text, args.plate_regex)
                                 if args.plate_filter else True)
                                for d in dets], frame_idx)
                # 车辆兜底: 直接打码, 不进 tracker(窄条框被预测外推会漂移)
                for vb in vboxes:
                    pad_box = expand_box(vb, frame.shape, 1.0 + args.pad * 2)
                    redact_roi(frame, pad_box, args.fill)
                    planned.setdefault(frame_idx, []).append(pad_box)
                if frame_idx % 30 == 1 or scene_cut:
                    print(f'帧 {frame_idx}: 候选 {len(dets)} 个, 跟踪目标 '
                          f'{len(tracker.active_tracks())} 个'
                          f'{" [镜头切换]" if scene_cut else ""}')

            # 只打码近期确认过的有效轨迹(多帧一致性 -> 单帧误检不会变成色块)
            active = tracker.active_tracks(max_stale=args.track_stale)[:args.max_tracks]
            if active:
                last_cache = [(b, tid) for b, tid, v in active if v]
            boxes_to_redact = active
            if not active and last_cache and not scene_cut:
                # 检测失败但最近确认过: 继续沿用(运动预测外推), 覆盖模糊/遮挡间隙
                boxes_to_redact = [(b, tid, True) for b, tid in last_cache]
            for box, tid, valid in boxes_to_redact:
                if not valid:
                    continue
                bw, bh = box[2] - box[0], box[3] - box[1]
                if bw < args.min_box_w or bh < args.min_box_h:
                    continue  # 碎片框(水印/文字/UI 误检)
                if args.edge_filter and bw < 25 and bh < 40 and (
                        box[0] <= 8 or box[2] >= frame.shape[1] - 8):
                    continue  # 贴边的极小竖条: 水印/UI 误检
                pad_box = expand_box(box, frame.shape, 1.0 + args.pad * 2)
                redact_roi(frame, pad_box, args.fill)
                planned.setdefault(frame_idx, []).append(pad_box)
                if args.annotate:
                    x1, y1, x2, y2 = clamp_box(pad_box, frame.shape)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'ID{tid}', (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            writer.write(frame)
            if frame_idx % 50 == 0:
                print(f'处理到第 {frame_idx} 帧, '
                      f'{1.0 / max(1e-6, time.time() - t0):.1f} fps')
    except DetectorError as exc:
        # fail-closed: 检测链路不可信时立即中止, 不留任何成品
        cap.release()
        writer.release()
        _remove(draft)
        raise SystemExit(f'检测器失败, 已中止且未产出任何视频: {exc}')
    finally:
        if cap.isOpened():
            cap.release()
        if writer.isOpened():
            writer.release()

    print(f'第一遍完成: {frame_idx} 帧, 用时 {time.time() - t_all:.1f}s')

    # ---- 第二遍: 往前推补打(利用轨迹最早速度反推车牌刚进入画面/还模糊的帧) ----
    final_draft = draft
    if args.pre_frames > 0:
        back_boxes = back_project(tracker, args.pre_frames)
        if back_boxes:
            final_draft = output + '.draft2.mp4'
            _render_back_boxes(draft, final_draft, back_boxes, fourcc, fps_in,
                               (w, h), planned, args.pad)
            _remove(draft)
            print(f'往前推补打完成: 覆盖 {len(back_boxes)} 帧')

    # ---- 审计 -> 原子发布 / 丢弃 ----
    blocking, warnings = audit_render(source, final_draft, planned,
                                      flat_tol=args.audit_flat_tol)
    for wmsg in warnings[:20]:
        print(f'[审计][warn] {wmsg}')
    if blocking:
        for msg in blocking[:20]:
            print(f'[审计][阻断] {msg}')
        if args.fail_closed:
            _remove(final_draft)
            _remove(output)
            raise SystemExit(
                f'审计未通过({len(blocking)} 项), fail-closed 已丢弃草稿, 未产出成品')
        print('[warn] --fail-closed 0: 审计未通过仍然发布, 该文件不可作为合规成品')
    os.replace(final_draft, output)
    print(f'视频处理完成: {frame_idx} 帧, 用时 {time.time()-t_all:.1f}s, 输出: {output}')


def back_project(tracker, pre_frames, max_back=8):
    """用轨迹最早两帧的速度, 把框反推到"首次检出之前"的若干帧

    只用最早两帧的速度: 车辆加速/减速时, 末尾速度不代表刚进入画面时的运动。
    反推距离限制在 max_back 帧内, 避免长时间外推的累积误差。
    返回 {frame_idx: [box, ...]}
    """
    out = {}
    histories = list(getattr(tracker, 'all_history', []))
    for t in getattr(tracker, 'tracks', []):
        if len(getattr(t, 'history', [])) >= 2:
            histories.append(t.history)
    for hist in histories:
        if len(hist) < 2:
            continue
        (f1, b1), (f2, b2) = hist[0], hist[1]
        dt = max(1, f2 - f1)
        vel = [(b2[k] - b1[k]) / dt for k in range(4)]
        f0, b0 = hist[0]
        for k in range(1, min(pre_frames, f0 - 1, max_back) + 1):
            out.setdefault(f0 - k, []).append([b0[i] - vel[i] * k for i in range(4)])
    return out


def _render_back_boxes(src_video, dst_video, back_boxes, fourcc, fps, size, planned,
                       pad=0.08):
    """第二遍渲染: 在第一遍草稿上补打回溯帧, 并把补打的框并入审计计划

    回溯框和第一遍一样做外扩: 它是靠匀速假设外推出来的, 存在预测误差,
    不外扩就可能留下一条没盖住的车牌边缘。
    """
    cap = cv2.VideoCapture(src_video)
    out = cv2.VideoWriter(dst_video, fourcc, fps, size)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        for box in back_boxes.get(idx, []):
            x1, y1, x2, y2 = clamp_box(box, frame.shape)
            if x2 - x1 < 10 or y2 - y1 < 4:
                continue
            pad_box = expand_box([x1, y1, x2, y2], frame.shape, 1.0 + pad * 2)
            redact_roi(frame, pad_box)
            planned.setdefault(idx, []).append(pad_box)
        out.write(frame)
    cap.release()
    out.release()


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def build_parser():
    p = argparse.ArgumentParser(
        description='车牌检测 + 纯色填充打码(可插拔检测器 / 双遍渲染 / fail-closed)')
    p.add_argument('--source', required=True,
                   help='输入: 图片文件/图片目录/视频文件(摄像头不支持, 请先录制为文件)')
    p.add_argument('--output', default='out', help='输出目录(图片)或视频文件路径')
    p.add_argument('--detector', action='append', default=None,
                   help='检测器规格, 可重复传入取并集: fast-alpr[:检测模型[:OCR模型[:设备]]]'
                        ' | hyperlpr3 | vlm:<权重目录> | cmd:<命令>')
    p.add_argument('--detector-conf', type=float, default=0.3, help='检测置信度阈值')
    p.add_argument('--hyper-conf', type=float, default=0.35, help='HyperLPR3 置信度阈值')
    p.add_argument('--vlm-split', type=int, default=2, help='VLM 兜底时画面分块数')
    p.add_argument('--vehicle-detector', default='',
                   help='车辆兜底检测器: ultralytics:<权重> | cmd:<命令> (需自行安装)')
    p.add_argument('--vehicle-conf', type=float, default=0.2, help='车辆检测置信度阈值')
    p.add_argument('--vehicle-fallback', type=int, default=1,
                   help='车辆区域兜底: 检测不到车牌时对两轮车尾部窄条加打码')
    p.add_argument('--fill', default='mean', choices=['mean', 'black'],
                   help='打码填充: mean=区域平均色(默认), black=纯黑')
    p.add_argument('--pad', type=float, default=0.08, help='打码框额外外扩比例(每边)')
    p.add_argument('--min-box-w', type=int, default=25, help='打码框最小宽度')
    p.add_argument('--min-box-h', type=int, default=12, help='打码框最小高度')
    p.add_argument('--edge-filter', type=int, default=1,
                   help='过滤贴边极小竖条(水印/UI 误检)')
    p.add_argument('--conf', type=float, default=0.3, help='主检测置信度阈值')
    p.add_argument('--low-conf', type=float, default=0.15, help='重检时的低置信度阈值')
    p.add_argument('--interval', type=int, default=1,
                   help='抽帧检测间隔(1=逐帧), 中间帧由跟踪器预测框打码')
    p.add_argument('--track-max-age', type=int, default=30,
                   help='轨迹连续丢失多少帧后删除')
    p.add_argument('--track-iou', type=float, default=0.4, help='轨迹关联 IoU 阈值')
    p.add_argument('--track-stale', type=int, default=8,
                   help='轨迹超过多少帧未更新则不再打码(防僵尸框残留)')
    p.add_argument('--max-tracks', type=int, default=12,
                   help='同时打码的轨迹上限(车多场景建议调大, 截断会漏打)')
    p.add_argument('--roi-expand', type=float, default=1.8, help='局部重检区域放大倍数')
    p.add_argument('--zoom-fallback', type=int, default=0,
                   help='放大兜底: 检测失败时全图/分块 2x 重检')
    p.add_argument('--plate-filter', type=int, default=0,
                   help='按车牌号格式过滤候选(默认关: 过滤会漏打码, 仅建议用于图片)')
    p.add_argument('--plate-regex', default=DEFAULT_PLATE_REGEX,
                   help='--plate-filter 使用的正则')
    p.add_argument('--scene-thresh', type=float, default=0.70,
                   help='镜头切换阈值(直方图相关系数, 低于视为切换, 0=关闭)')
    p.add_argument('--pre-frames', type=int, default=15,
                   help='往前推打码帧数(0=关闭): 车牌刚进入画面/还模糊时就补打')
    p.add_argument('--fail-closed', type=int, default=1,
                   help='审计不通过时丢弃草稿而不是发布(默认 1, 强烈建议保持)')
    p.add_argument('--audit-flat-tol', type=float, default=6.0,
                   help='判定"已是纯色块"的逐通道标准差容差')
    p.add_argument('--fourcc', default='mp4v', help='视频编码 fourcc')
    p.add_argument('--max-frames', type=int, default=0, help='最多处理帧数(调试, 0=全部)')
    p.add_argument('--annotate', type=int, default=0, help='绘制检测框(调试用)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.detector is None:
        args.detector = ['fast-alpr']

    src = args.source
    if src.strip().lstrip('+-').isdigit():
        raise SystemExit(
            '摄像头源无法做双遍校验与审计, 请先录制成视频文件再处理')
    if not os.path.exists(src):
        raise SystemExit(f'输入不存在: {src}')

    try:
        detector = build_detectors(args.detector, conf=args.detector_conf,
                                   hyper_conf=args.hyper_conf, vlm_split=args.vlm_split)
        vehicle_detector = build_vehicle_detector(args.vehicle_detector) \
            if args.vehicle_fallback else None
    except DetectorError as exc:
        raise SystemExit(f'检测器不可用: {exc}')
    try:
        if os.path.isdir(src):
            process_images(src, args.output, detector, args)
            return 0
        ext = os.path.splitext(src)[1].lower()
        if ext in IMAGE_EXT:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            img = cv2.imread(src)
            dets = nms_detections(detector.detect(img) or [])
            for d in dets:
                redact_roi(img, expand_box(d.box, img.shape, 1.0 + args.pad * 2), args.fill)
            cv2.imwrite(args.output, img)
            print(f'车牌候选 {len(dets)} 个 -> {args.output}')
            return 0
        # 视频: 先清掉可能存在的旧成品, 保证"文件存在 == 本次审计通过"
        if os.path.exists(args.output):
            _remove(args.output)
            print('[fail-closed] 已移除同名旧输出, 避免误认为本次结果')
        process_video(src, args.output, detector, args, vehicle_detector)
        return 0
    except DetectorError as exc:
        # 图片/单帧路径的引擎失败: 同样不产出任何结果
        raise SystemExit(f'检测器运行失败, 未产出结果: {exc}')
    finally:
        detector.close()
        if vehicle_detector is not None:
            vehicle_detector.close()


if __name__ == '__main__':
    raise SystemExit(main())
