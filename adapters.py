# -*- coding: UTF-8 -*-
"""
可插拔车牌检测适配器层

本模块自身只依赖标准库 + numpy/opencv。所有第三方检测引擎都通过「按需导入」
的适配器接入：仓库不打包任何第三方代码与权重，也不在默认路径上 import 它们。

内置适配器（均需使用者自行安装并遵守其许可证）:
    fast-alpr     MIT         车牌检测 + OCR（YOLOv9 检测 + MobileViT OCR）
    hyperlpr3     Apache-2.0  倾斜/多视角车牌补充检测
    vlm           Apache-2.0  Qwen2.5-VL 视觉大模型兜底（见 plate_llm.py）
    ultralytics   AGPL-3.0    车辆检测兜底（可选）
    cmd           -           外部子进程桥：接入任意自研/第三方引擎（含 GPL/AGPL）

适配器契约:
    detect(frame_bgr) -> [Detection(rect=(x1,y1,x2,y2), score, text, source), ...]
    坐标是「输入帧」的像素坐标；引擎不可用或运行失败时抛 DetectorError。

注册自定义适配器:
    from adapters import register_detector, PlateDetector

    @register_detector('my-engine')
    def _make(arg: str, kw: dict) -> PlateDetector:
        return MyDetector(arg, **kw)

命令行用法（可重复传入实现并集）:
    --detector fast-alpr
    --detector fast-alpr --detector hyperlpr3
    --detector cmd:python3 my_detector.py
"""
from __future__ import annotations

import base64
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import cv2


@dataclass
class Detection:
    """一个车牌候选框（像素坐标，x1<=x2, y1<=y2）"""
    rect: Sequence[float]
    score: float = 0.0
    text: str = ''
    source: str = ''

    @property
    def box(self) -> tuple:
        x1, y1, x2, y2 = [float(v) for v in self.rect]
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


class DetectorError(RuntimeError):
    """检测引擎不可用或运行失败。

    调用方在 fail-closed 模式下必须把它当成致命错误（中止渲染并丢弃草稿），
    绝不能吞掉异常继续输出未经遮挡的帧。
    """


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlateDetector:
    """检测器基类。子类只需实现 detect()。"""
    name = 'base'

    def detect(self, frame) -> List[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class UnionDetector(PlateDetector):
    """多引擎并集：合并多个检测器结果，按 IoU 去重。

    这是本项目的核心思路之一——任何单引擎都有盲区（车型、视角、分辨率、
    字符集），并集比调参更能提升召回；重复框用 NMS 式去重避免重复打码。
    """
    name = 'union'

    def __init__(self, detectors: Sequence[PlateDetector], dedup_iou: float = 0.3):
        self.detectors = list(detectors)
        self.dedup_iou = dedup_iou

    def detect(self, frame) -> List[Detection]:
        merged: List[Detection] = []
        for det in self.detectors:
            try:
                found = det.detect(frame) or []
            except DetectorError:
                raise
            except Exception as exc:  # 引擎内部异常一律上抛, 由 fail-closed 决定去留
                raise DetectorError(f'检测器 {det.name} 运行失败: {exc}') from exc
            for item in found:
                if any(iou_xyxy(item.box, kept.box) >= self.dedup_iou for kept in merged):
                    continue
                merged.append(item)
        return merged

    def close(self) -> None:
        for det in self.detectors:
            det.close()


# --------------------------------------------------------------------------- #
# fast-alpr（MIT）：默认的通用检测引擎
# --------------------------------------------------------------------------- #
def _sync_model_cache() -> None:
    """离线运行支持：脚本旁若有 model_cache/ 目录，同步到 ~/.cache/

    fast-alpr 从 ~/.cache/open-image-models 与 ~/.cache/fast-plate-ocr 加载权重；
    在无外网环境里把权重目录放在仓库外并预置到这两个路径即可离线运行。
    """
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_cache')
    if not os.path.isdir(local):
        return
    for sub in ('open-image-models', 'fast-plate-ocr'):
        src = os.path.join(local, sub)
        dst = os.path.join(os.path.expanduser('~'), '.cache', sub)
        if os.path.isdir(src) and not os.path.isdir(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
            print(f'模型缓存已同步: {src} -> {dst}')


class FastAlprDetector(PlateDetector):
    """fast-alpr 适配器（MIT）。需自行 pip install fast-alpr 并遵守其许可证。"""
    name = 'fast-alpr'

    def __init__(self, detector_model: str = 'yolo-v9-t-640-license-plate-end2end',
                 ocr_model: str = 'european-plates-mobile-vit-v2-model',
                 conf: float = 0.3, device: str = 'cpu'):
        try:
            from fast_alpr.alpr import ALPR
        except ImportError as exc:
            raise DetectorError(
                'fast-alpr 未安装（pip install fast-alpr，MIT 许可证）') from exc
        _sync_model_cache()
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        self.alpr = ALPR(detector_model=detector_model,
                         ocr_model=ocr_model,
                         detector_conf_thresh=conf,
                         ocr_device=device,
                         detector_providers=providers)

    def detect(self, frame) -> List[Detection]:
        out = []
        for r in self.alpr.predict(frame):
            bb = r.detection.bounding_box
            text = r.ocr.text if getattr(r, 'ocr', None) else ''
            out.append(Detection(rect=(bb.x1, bb.y1, bb.x2, bb.y2),
                                 score=float(r.detection.confidence),
                                 text=text or '',
                                 source=self.name))
        return out


# --------------------------------------------------------------------------- #
# HyperLPR3（Apache-2.0）：倾斜 / 多视角补充检测
# --------------------------------------------------------------------------- #
class HyperLprDetector(PlateDetector):
    """HyperLPR3 适配器（Apache-2.0）。需自行 pip install hyperlpr3 并遵守其许可证。"""
    name = 'hyperlpr3'

    def __init__(self, conf: float = 0.35):
        try:
            import hyperlpr3 as lpr3
        except ImportError as exc:
            raise DetectorError(
                'hyperlpr3 未安装（pip install hyperlpr3，Apache-2.0 许可证）') from exc
        self.catcher = lpr3.LicensePlateCatcher()
        self.conf = conf

    def detect(self, frame) -> List[Detection]:
        out = []
        for r in self.catcher(frame):
            if not r or len(r) < 2:
                continue
            plate, conf = r[0], float(r[1])
            box = r[3] if len(r) > 3 else None
            if conf < self.conf or not box:
                continue
            out.append(Detection(rect=(box[0], box[1], box[2], box[3]),
                                 score=conf, text=str(plate), source=self.name))
        return out


# --------------------------------------------------------------------------- #
# 视觉大模型兜底（Qwen2.5-VL，Apache-2.0）
# --------------------------------------------------------------------------- #
class VlmDetector(PlateDetector):
    """Qwen2.5-VL 兜底适配器：只判断「像不像车牌」，不识别文字。

    慢，但能覆盖专用检测器的盲区（货车车身、远距小目标、非拉丁字符集）。
    需要使用者自行准备权重与 torch/transformers 环境。
    """
    name = 'vlm'

    def __init__(self, model_path: str, split_blocks: int = 2, device: str = None):
        if not model_path or not os.path.isdir(model_path):
            raise DetectorError(f'VLM 权重目录不存在: {model_path!r}')
        from plate_llm import VLPlateDetector
        self.vlm = VLPlateDetector.get_instance(model_path, device)
        self.split_blocks = split_blocks

    def detect(self, frame) -> List[Detection]:
        boxes = self.vlm.detect(frame, split_blocks=self.split_blocks, verbose=False)
        return [Detection(rect=b, score=0.5, text='', source=self.name) for b in boxes]


# --------------------------------------------------------------------------- #
# 外部进程桥：不打包第三方代码也能接入任意引擎
# --------------------------------------------------------------------------- #
class CommandDetector(PlateDetector):
    """外部检测器桥（子进程 NDJSON 协议）。

    用于接入本仓库不打包的第三方/自研引擎（含 GPL/AGPL 实现）：由使用者
    自行安装、自行遵守其许可证，仓库只定义协议，不携带其代码与权重。

    协议（每行一个 JSON，UTF-8）:
      父 -> 子: {"frame": <int>, "width": <int>, "height": <int>, "image_png_b64": "..."}
      子 -> 父: {"detections": [{"rect": [x1,y1,x2,y2], "score": 0.9, "text": "AB123"}]}
      子进程 stderr 原样转发到本进程 stderr，便于观察引擎日志。

    子进程退出、超时或返回非法 JSON 都视为致命错误（fail-closed 下中止渲染）。
    """
    name = 'cmd'

    def __init__(self, command: str, timeout: float = 120.0):
        if not command.strip():
            raise DetectorError('cmd 适配器缺少命令，用法 --detector cmd:<命令>')
        self.command = command
        self.timeout = timeout
        try:
            self.proc = subprocess.Popen(
                shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, bufsize=1)
        except OSError as exc:
            raise DetectorError(f'无法启动外部检测器 {command!r}: {exc}') from exc
        self.frame_idx = 0

    def detect(self, frame) -> List[Detection]:
        if self.proc.poll() is not None:
            raise DetectorError(
                f'外部检测器已退出（returncode={self.proc.returncode}）: {self.command}')
        ok, buf = cv2.imencode('.png', frame)
        if not ok:
            raise DetectorError('帧编码 PNG 失败，无法送往外部检测器')
        payload = {
            'frame': self.frame_idx,
            'width': int(frame.shape[1]),
            'height': int(frame.shape[0]),
            'image_png_b64': base64.b64encode(buf.tobytes()).decode('ascii'),
        }
        self.frame_idx += 1
        try:
            self.proc.stdin.write(json.dumps(payload) + '\n')
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise DetectorError(f'与外部检测器通信失败: {exc}') from exc
        if not line:
            raise DetectorError(
                f'外部检测器未返回结果（returncode={self.proc.poll()}）: {self.command}')
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DetectorError(f'外部检测器返回非法 JSON: {line[:200]!r}') from exc
        out = []
        for item in data.get('detections', []) or []:
            rect = item.get('rect')
            if not rect or len(rect) < 4:
                continue
            out.append(Detection(rect=[float(v) for v in rect[:4]],
                                 score=float(item.get('score', 0.0) or 0.0),
                                 text=str(item.get('text', '') or ''),
                                 source=self.name))
        return out

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# 车辆检测（兜底打码用，可选）
# --------------------------------------------------------------------------- #
class VehicleDetector:
    """车辆检测基类：detect(frame) -> [(x1,y1,x2,y2,conf,cls), ...]"""
    name = 'vehicle-base'

    def detect(self, frame, conf: float = 0.2) -> list:
        raise NotImplementedError

    def close(self) -> None:
        pass


class UltralyticsVehicleDetector(VehicleDetector):
    """ultralytics YOLO 车辆检测（AGPL-3.0，需自行安装并遵守其许可证）。

    只做「车辆区域兜底」：车牌检测器完全失效时，对车辆尾部窄条打码，
    避免整车涂抹。属于可选项，默认不启用。
    """
    name = 'ultralytics'
    COCO_VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

    def __init__(self, weights: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                'ultralytics 未安装（AGPL-3.0，需自行安装并遵守其许可证）') from exc
        self.model = YOLO(weights)

    def detect(self, frame, conf: float = 0.2) -> list:
        res = self.model(frame, conf=conf, classes=self.COCO_VEHICLE_CLASSES, verbose=False)
        out = []
        for b in res[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            out.append((x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
        return out


class CommandVehicleDetector(VehicleDetector):
    """把车辆检测也交给外部进程（协议同 CommandDetector）。"""
    name = 'vehicle-cmd'

    def __init__(self, command: str):
        self.inner = CommandDetector(command)
        self.inner.name = self.name

    def detect(self, frame, conf: float = 0.2) -> list:
        out = []
        for det in self.inner.detect(frame):
            x1, y1, x2, y2 = det.box
            out.append((x1, y1, x2, y2, det.score, -1))
        return out

    def close(self) -> None:
        self.inner.close()


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
_FACTORIES: Dict[str, Callable[[str, dict], PlateDetector]] = {}
_VEHICLE_FACTORIES: Dict[str, Callable[[str, dict], VehicleDetector]] = {}


def register_detector(name: str):
    """注册自定义车牌检测适配器。工厂函数签名为 (arg: str, kw: dict) -> PlateDetector"""
    def deco(fn):
        _FACTORIES[name] = fn
        return fn
    return deco


def register_vehicle_detector(name: str):
    """注册自定义车辆检测适配器。"""
    def deco(fn):
        _VEHICLE_FACTORIES[name] = fn
        return fn
    return deco


def _fast_alpr_factory(arg: str, kw: dict) -> PlateDetector:
    parts = [p for p in arg.split(':') if p] if arg else []
    return FastAlprDetector(
        detector_model=parts[0] if len(parts) > 0 else 'yolo-v9-t-640-license-plate-end2end',
        ocr_model=parts[1] if len(parts) > 1 else 'european-plates-mobile-vit-v2-model',
        conf=kw.get('conf', 0.3),
        device=parts[2] if len(parts) > 2 else 'cpu')


_FACTORIES.update({
    'fast-alpr': _fast_alpr_factory,
    'hyperlpr3': lambda arg, kw: HyperLprDetector(conf=kw.get('hyper_conf', 0.35)),
    'vlm': lambda arg, kw: VlmDetector(arg, split_blocks=kw.get('vlm_split', 2)),
    'cmd': lambda arg, kw: CommandDetector(arg),
})

_VEHICLE_FACTORIES.update({
    'ultralytics': lambda arg, kw: UltralyticsVehicleDetector(arg),
    'cmd': lambda arg, kw: CommandVehicleDetector(arg),
})


def available_detectors() -> List[str]:
    return sorted(_FACTORIES)


def build_detector(spec: str, **kw) -> PlateDetector:
    """按 spec 构建检测器。spec 形如 name[:arg1[:arg2]]"""
    name, _, arg = spec.partition(':')
    factory = _FACTORIES.get(name)
    if factory is None:
        raise DetectorError(
            f'未知检测器 {name!r}；已注册: {", ".join(available_detectors())}')
    return factory(arg, kw)


def build_detectors(specs: Sequence[str], **kw) -> PlateDetector:
    """构建一个或多个检测器，多个时返回并集（双引擎并集）。"""
    if not specs:
        raise DetectorError('至少需要一个 --detector')
    built = [build_detector(s, **kw) for s in specs]
    return built[0] if len(built) == 1 else UnionDetector(built)


def build_vehicle_detector(spec: str, **kw) -> VehicleDetector:
    if not spec:
        return None
    name, _, arg = spec.partition(':')
    factory = _VEHICLE_FACTORIES.get(name)
    if factory is None:
        raise DetectorError(
            f'未知车辆检测器 {name!r}；已注册: {", ".join(sorted(_VEHICLE_FACTORIES))}')
    return factory(arg, kw)
