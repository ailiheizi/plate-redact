#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
合成素材的 fail-closed 演示与自测夹具（不需要任何真实视频、权重或第三方引擎）

这个脚本自己生成一段**合成帧序列**：几块深色"车身"在渐变背景上移动，其中一辆
车尾有一块亮底带深色条纹的"车牌"。检测用的是纯 numpy/opencv 的玩具实现
（亮块/深块阈值 + 轮廓），只用来产生候选；安全性完全由 ``fail_closed.py`` 的
输出侧审计独立判定。

它同时演示四种结局：
  pass                 正常路径：两遍渲染一致、遮挡到位 → 原子发布最终成品
  missing_mask         渲染时漏掉某一帧的遮挡（模拟渲染缺陷）→ 阻断，只留 draft
  residual_plate       遮挡都做了，但车身里还有一块没被计划覆盖的牌 → 记 UNKNOWN
  unexplained_vehicle  画面里有一辆车从未有过任何遮挡计划 → 该轨迹 UNKNOWN

用法::

    python examples/fail_closed_synthetic.py --outdir /tmp/fc-demo --mode pass
    python examples/fail_closed_synthetic.py --outdir /tmp/fc-demo --mode residual_plate

依赖: numpy + opencv-python（与仓库其它模块一致）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fail_closed as fc  # noqa: E402

BACKGROUND_TOP = 8          # 合成背景：深色渐变（刻意不落在任何检测阈值里）
BACKGROUND_BOTTOM = 30
BODY_GRAY = 70              # 车身灰度
PLATE_GRAY = 235            # 车牌底灰度
GLYPH_GRAY = 25             # 车牌字符灰度
VEHICLE_GRAY_RANGE = (40, 90)
PLATE_BRIGHT_MIN = 200


@dataclass
class VehicleSpec:
    """一辆合成车辆：车身框 + 可选的"车牌"（亮底 + 深色条纹 = 有字符纹理）。"""
    x: float
    y: float
    w: int
    h: int
    vx: float = 0.0
    vy: float = 0.0
    body_gray: int = BODY_GRAY
    plate: bool = True
    plate_offset: tuple = (20, 24)
    plate_size: tuple = (30, 10)
    glyphs: int = 4


@dataclass
class Scenario:
    """一个合成场景。``vehicles`` 里的每辆车都会进入车辆轨迹。"""
    width: int = 256
    height: int = 144
    fps: float = 12.0
    frames: int = 30
    vehicles: tuple = (
        VehicleSpec(x=20, y=80, w=70, h=40, vx=3.0),          # 有牌的车
    )
    extra_unexplained_vehicle: bool = False

    def resolved_vehicles(self) -> tuple:
        rows = list(self.vehicles)
        if self.extra_unexplained_vehicle:
            # 一辆永远没有牌的车：轨迹级审计必须把它记成 UNKNOWN。
            rows.append(VehicleSpec(x=180, y=24, w=56, h=30, vx=-2.0,
                                    body_gray=62, plate=False))
        return tuple(rows)


def scenario_for_mode(mode: str) -> Scenario:
    if mode == "unexplained_vehicle":
        return Scenario(extra_unexplained_vehicle=True)
    return Scenario()


# --------------------------------------------------------------------------- #
# 合成帧与合成视频
# --------------------------------------------------------------------------- #
def _background(width: int, height: int) -> np.ndarray:
    column = np.linspace(BACKGROUND_TOP, BACKGROUND_BOTTOM, height, dtype=np.float32)
    gray = np.repeat(column[:, None], width, axis=1)
    return np.dstack([gray, gray, gray]).astype(np.uint8)


def _draw_plate(frame: np.ndarray, x: int, y: int, size: tuple,
                glyphs: int) -> None:
    """画一块"车牌"：亮底 + 若干深色竖条（竖条就是"字符纹理"的来源）。"""
    w, h = int(size[0]), int(size[1])
    frame[y:y + h, x:x + w] = PLATE_GRAY
    if glyphs <= 0:
        return
    step = max(3, w // (glyphs + 1))
    bar_w = max(1, step // 3)
    for index in range(glyphs):
        bx = x + step * (index + 1) - bar_w // 2
        frame[y + 2:y + h - 2, bx:bx + bar_w] = GLYPH_GRAY


def render_frame(scenario: Scenario, index: int, *, extra_plate: bool = False) -> np.ndarray:
    """渲染第 index 帧；``extra_plate`` 用于"篡改"演示。"""
    frame = _background(scenario.width, scenario.height)
    for vehicle in scenario.resolved_vehicles():
        x = int(round(vehicle.x + vehicle.vx * index))
        y = int(round(vehicle.y + vehicle.vy * index))
        frame[y:y + vehicle.h, x:x + vehicle.w] = vehicle.body_gray
        if vehicle.plate:
            px = x + int(vehicle.plate_offset[0])
            py = y + int(vehicle.plate_offset[1])
            _draw_plate(frame, px, py, vehicle.plate_size, vehicle.glyphs)
            if extra_plate:
                # 车身里另画一块没被计划覆盖的牌：模拟"漏打 / 篡改"。
                _draw_plate(frame, x + 4, y + 6, (24, 8), 3)
    return frame


def write_source(path: Path, scenario: Scenario) -> dict:
    """把合成帧序列写成视频（这就是全部"素材"，运行完可随时删掉）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(scenario.fps), (scenario.width, scenario.height))
    if not writer.isOpened():
        raise RuntimeError("无法创建合成源视频")
    try:
        for index in range(scenario.frames):
            writer.write(render_frame(scenario, index))
    finally:
        writer.release()
    meta = fc.probe_video(path)
    if meta["frames"] != scenario.frames:
        raise RuntimeError(f"合成源帧数异常: {meta['frames']} != {scenario.frames}")
    return meta


# --------------------------------------------------------------------------- #
# 玩具检测器（只产生候选，不承担任何"安全"结论）
# --------------------------------------------------------------------------- #
def toy_plate_boxes(frame: np.ndarray) -> list[list[float]]:
    """亮底 + 深色条纹的块 → 车牌候选（形态学闭运算把字符间隙连起来）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray >= PLATE_BRIGHT_MIN).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 5 or not 1.2 <= w / max(h, 1) <= 8.0:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def toy_vehicle_boxes(frame: np.ndarray) -> list[list[float]]:
    """深色大块 → 车辆候选（车牌是车身里的亮洞，不影响车身外框）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low, high = VEHICLE_GRAY_RANGE
    mask = ((gray >= low) & (gray <= high)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 25 or h < 15 or w / max(h, 1) < 1.2:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def make_plan(*, pad: float = 0.10) -> fc.PlanFn:
    """候选生成：逐帧玩具检测 + 仓库既有的 PlateTracker 做轨迹保持。"""
    from plate_tracker import PlateTracker

    def plan(frozen_source: Path) -> fc.PlanBundle:
        cap = cv2.VideoCapture(str(frozen_source))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开冻结输入: {frozen_source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        plate_tracker = PlateTracker(max_age=30)
        vehicle_tracker = PlateTracker(max_age=30)
        plans: list[list[list[float]]] = []
        observations: dict[int, dict[int, list[float]]] = {}
        confidences: dict[int, dict[int, float]] = {}
        frame_no = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                plate_dets = sorted(
                    [(box, 0.9) for box in toy_plate_boxes(frame)],
                    key=lambda row: row[1], reverse=True)
                plate_tracker.predict()
                active = plate_tracker.update(
                    [(b[0], b[1], b[2], b[3], score, True)
                     for b, score in plate_dets], frame_no)
                plans.append([fc.expand_box(box, frame.shape, pad)
                              for box, _tid, _valid in active])
                vehicle_dets = sorted(
                    [(box, 0.8) for box in toy_vehicle_boxes(frame)],
                    key=lambda row: row[1], reverse=True)
                vehicle_tracker.predict()
                vehicle_active = vehicle_tracker.update(
                    [(b[0], b[1], b[2], b[3], score, True)
                     for b, score in vehicle_dets], frame_no)
                for box, track_id, _valid in vehicle_active:
                    observations.setdefault(track_id, {})[frame_no] = list(box)
                    confidences.setdefault(track_id, {})[frame_no] = 0.8
                frame_no += 1
        finally:
            cap.release()
        tracks = [fc.VehicleTrack(id=track_id, observations=observations[track_id],
                                  confidences=confidences.get(track_id, {}))
                  for track_id in sorted(observations)]
        return fc.PlanBundle(plans=plans, vehicle_tracks=tracks, width=width,
                             height=height, fps=fps, meta={"frames": frame_no})
    return plan


# --------------------------------------------------------------------------- #
# 故意有缺陷的渲染器（用于证明阻断真的会发生）
# --------------------------------------------------------------------------- #
def defective_render(*, skip_mask_frames: tuple = (),
                     on_frame=None) -> fc.RenderFn:
    """返回一个渲染遍函数：与默认渲染器同签名，但会漏打或叠加未打码的牌。

    ``skip_mask_frames`` 里的帧不做任何遮挡（模拟遮挡缺失）；
    ``on_frame(frame, index)`` 用于在遮挡之后往画面里补一块"漏打的牌"。
    它仍然完整跑完两遍并给出帧链，所以问题只能由**输出侧审计**发现——
    这正是要演示的点：审计不信任渲染器。
    """
    skip = {int(v) for v in skip_mask_frames}

    def render(source, destination, plans, *, width, height, fps,
               fourcc: str = "mp4v", progress: bool = False,
               full_audit: bool = True) -> dict:
        destination = Path(destination)
        if destination.exists():
            destination.unlink()
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开渲染源: {source}")
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*fourcc),
                                 float(fps) if fps and fps > 0 else 25.0,
                                 (int(width), int(height)))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"无法创建视频: {destination}")
        source_chain = fc.FrameChain()
        masked_chain = fc.FrameChain()
        frame_no = 0
        try:
            while frame_no < len(plans):
                ok, frame = cap.read()
                if not ok:
                    break
                source_chain.update(frame_no, frame)
                if frame_no not in skip:
                    for box in plans[frame_no]:
                        fc.solid_fill(frame, box)
                if on_frame is not None:
                    on_frame(frame, frame_no)
                masked_chain.update(frame_no, frame)
                writer.write(frame)
                frame_no += 1
            if frame_no != len(plans):
                raise RuntimeError(f"源视频提前结束: {frame_no} != {len(plans)}")
            if full_audit:
                extra, _ = cap.read()
                if extra:
                    raise RuntimeError(f"源视频在计划 {len(plans)} 帧之后仍有额外帧")
        finally:
            writer.release()
            cap.release()
        return {"submitted_frames": frame_no,
                "source_frame_chain_sha256": source_chain.hexdigest(),
                "masked_input_frame_chain_sha256": masked_chain.hexdigest()}
    return render


# --------------------------------------------------------------------------- #
# 跑一个场景
# --------------------------------------------------------------------------- #
def run_scenario(outdir: Path, mode: str = "pass", *, keep: bool = True,
                 max_frames: int = 0) -> fc.RunReport:
    """生成合成源并按指定模式跑完整链路，返回报告。"""
    outdir = Path(outdir)
    if outdir.exists() and not keep:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scenario = scenario_for_mode(mode)
    source = outdir / f"synthetic_{mode}_source.mp4"
    write_source(source, scenario)

    render = None
    if mode == "missing_mask":
        render = defective_render(skip_mask_frames=(len(range(scenario.frames)) // 2,))
    elif mode == "residual_plate":
        # 在**车身内部、计划框之外**补一块未打码的牌：计划遮挡都到位，
        # 只有输出侧的"残留纹理"检查能发现它。
        lead = scenario.resolved_vehicles()[0]

        def _inject_extra_plate(frame: np.ndarray, index: int) -> None:
            if index != scenario.frames // 2:
                return
            x = int(round(lead.x + lead.vx * index)) + 6
            y = int(round(lead.y + lead.vy * index)) + 4
            _draw_plate(frame, x, y, (24, 8), 3)

        render = defective_render(on_frame=_inject_extra_plate)

    return fc.run_fail_closed(source, outdir / f"synthetic_{mode}_out.mp4",
                              plan=make_plan(), render=render,
                              max_frames=max_frames, progress=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="合成素材 fail-closed 演示（不读取任何真实素材）")
    parser.add_argument('--outdir', default=None,
                        help='输出目录（合成源视频也写在这里，可随时删除；'
                             '默认用系统临时目录下的新目录）')
    parser.add_argument('--mode', default='pass',
                        choices=['pass', 'missing_mask', 'residual_plate',
                                 'unexplained_vehicle'])
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir) if args.outdir else Path(
        tempfile.mkdtemp(prefix="fail-closed-demo-"))
    report = run_scenario(outdir, args.mode, max_frames=args.max_frames)
    print(f"[演示] {report.summary()}")
    for issue in report.issues:
        print(f"  - {issue.get('id')}: {issue.get('type')}")
    for warning in report.warnings:
        print(f"  ~ {warning.get('id')}: {warning.get('type')}")
    print(f"[演示] 产物: 最终={report.output} draft={report.draft} "
          f"报告={report.report_path}")
    return 0 if report.status == "PASS" else 1


if __name__ == '__main__':
    raise SystemExit(main())
