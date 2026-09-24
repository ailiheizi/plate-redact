# -*- coding: UTF-8 -*-
"""
fail-closed 审计与发布链路（独立可运行实现）

把「模型会不会漏检」这种不可证的命题，换成一条可执行的门禁：

    输入冻结副本 → 排他 lease → 生成 canonical draft → 三类 failure marker → 原子发布

对应 DESIGN.md 第 2 / 3 / 5 节，这里只保留机制本身，不包含任何素材、权重与
第三方引擎代码：

1. **输入冻结副本 + 排他 lease**
   把输入按 O_NOFOLLOW 单描述符拷进私有目录并记录实际拷贝字节的摘要；
   「先哈希路径、再重开路径」不构成绑定，两次打开之间叶子可能被替换。
   lease 用 O_EXCL 建立，永不抢占；并发下同一输出 stem 只有一个执行者。

2. **canonical draft + 三类 failure marker + 原子发布**
   未证明安全时产物只叫 ``<out>.draft.<ext>``；通过后才 ``os.replace`` 成最终名。
   运行开始先移除同名旧成品，保证「文件存在 == 本次通过」。
   三类生命周期 failure marker（冻结副本清理 / lease 释放 / 草稿清理）任一出现，
   都表示本轮不可信：隔离最终名并写 ``status=INCOMPLETE``。

3. **输出侧独立审计：按车辆轨迹要求「可解释牌区」**
   审计器不信任检测、跟踪与渲染：编码完成后**重新解码**，逐车辆轨迹检查
   「这一帧这辆车是否有一块属于它的遮挡」以及「车辆框内是否还有未被计划覆盖的、
   亮且有字符纹理的牌区」。一条轨迹找不到解释就记 UNKNOWN 并阻断；车辆框 ≠ 牌框，
   两者不能互相顶替。

4. **双遍一致性**
   源像素帧链 / 遮挡后帧链 / 编码再解码的帧链三者必须自洽：两遍独立渲染必须
   得到相同的源链、相同的遮挡链与相同的解码链。

5. **清理失败也算阻断**
   冻结副本清理、lease 释放、参考遍临时产物清理，任一失败都不允许把中间态
   留成成功；失败类别写成 marker 并把最终名隔离。

依赖只有 numpy 与 opencv-python；第三方检测引擎一律通过调用方注入的 ``plan``
可调用对象（或 ``--detector`` 适配器规格）接入，本模块不 import 任何第三方代码，
也不在本文件中写死任何素材名与绝对路径。

所有判定阈值都是**保守默认值**，可用参数覆盖；它们不是性能指标。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = 1
FRAME_CHAIN_DOMAIN = b"plate-redact/fail-closed/frame-chain/v1\0"
DRAFT_INFIX = ".draft"
REPORT_SUFFIX = ".audit.json"
#: 三类生命周期 failure marker。任一出现都代表本轮不可信。
MARKER_KINDS = ("frozen_input_cleanup", "output_lock_release", "draft_cleanup")
INFRASTRUCTURE_ISSUE_PREFIXES = (
    "engine:", "decode:", "render:", "encoding:", "output:",
    "lease:", "cleanup:", "coverage:", "audit:", "review:",
)
# 以下阈值均为可配置的保守默认值，不是标定过的性能指标：
#   flat_tol    判定「已是纯色块」的逐通道标准差上限（留视频编码噪声余量）
#   texture_tol 判定「仍有字符纹理」的局部对比度下限
#   bright_min  参与牌区纹理判定的最低亮度
DEFAULT_FLAT_TOL = 6.0
DEFAULT_TEXTURE_TOL = 12.0
DEFAULT_BRIGHT_MIN = 140.0
TEXTURE_WINDOW = (16, 6)
TEXTURE_MIN_PIXELS = 24
COVERAGE_MIN_FRACTION = 0.90


# --------------------------------------------------------------------------- #
# 路径与文件原语：一律按词法路径操作，不跟随最后一段符号链接
# --------------------------------------------------------------------------- #
def lexical_path(value: str | os.PathLike[str]) -> Path:
    """返回绝对路径，但不解析最后一段符号链接。"""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _assert_directory(path: Path, *, create: bool = False) -> Path:
    """确认路径是普通目录（不是符号链接/特殊文件）；必要时创建。"""
    target = lexical_path(path)
    if create:
        os.makedirs(target, mode=0o700, exist_ok=True)
    info = os.lstat(target)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"不是普通目录: {target}")
    return target


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _regular_leaf(path: Path) -> os.stat_result | None:
    info = _lstat_or_none(path)
    if info is None or not stat.S_ISREG(info.st_mode):
        return None
    return info


def _same_regular_leaf(path: Path, before: os.stat_result) -> bool:
    info = _lstat_or_none(path)
    return (info is not None and stat.S_ISREG(info.st_mode)
            and (info.st_dev, info.st_ino) == (before.st_dev, before.st_ino))


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """对普通文件求 SHA-256（O_NOFOLLOW 打开，拒绝符号链接与特殊文件）。"""
    target = lexical_path(path)
    fd = os.open(str(target), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"不是普通文件: {target}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _write_all_fd(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("短写: 无法完整写入")
        offset += written


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _assert_replaceable_leaf(path: Path) -> None:
    """只允许替换普通文件；目录/特殊文件/符号链接一律拒绝。"""
    info = _lstat_or_none(path)
    if info is None:
        return
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"拒绝写入符号链接: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"拒绝覆盖非普通文件: {path}")


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes,
                       *, mode: int = 0o600) -> None:
    """同目录暂存 + fsync + os.replace：读者只会看到旧内容或完整新内容。"""
    target = lexical_path(path)
    _assert_directory(target.parent, create=True)
    _assert_replaceable_leaf(target)
    staged = target.with_name(f".{target.name}.stage-{secrets.token_hex(8)}")
    fd: int | None = None
    try:
        fd = os.open(str(staged), os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), mode)
        _write_all_fd(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(staged, target)
        _fsync_directory(target.parent)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if _lstat_or_none(staged) is not None:
            try:
                os.unlink(staged)
            except OSError:
                pass


def atomic_write_json(path: str | os.PathLike[str], payload: object) -> None:
    """原子写 JSON；allow_nan=False 保证不会落出非有限数值。"""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, encoded)


def load_strict_json(path: str | os.PathLike[str]) -> object:
    """严格读取 JSON：拒绝重复键与非有限数值。"""
    def _no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复的 JSON 键: {key}")
            result[key] = value
        return result

    def _no_nonfinite(value: str):
        raise ValueError(f"非有限 JSON 数值: {value}")

    with open(lexical_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_no_duplicates,
                         parse_constant=_no_nonfinite)


def canonical_draft_path(output: str | os.PathLike[str]) -> Path:
    """唯一能代表「未通过」的产物名：<out>.draft.<ext>。"""
    target = lexical_path(output)
    return target.with_name(target.stem + DRAFT_INFIX + target.suffix)


def report_path_for(output: str | os.PathLike[str]) -> Path:
    target = lexical_path(output)
    return target.with_name(target.stem + REPORT_SUFFIX)


def failure_marker_path(output: str | os.PathLike[str], kind: str) -> Path:
    if kind not in MARKER_KINDS:
        raise ValueError(f"未知的 failure marker 类别: {kind!r}")
    target = lexical_path(output)
    return target.with_name(f"{target.stem}.{kind}_failure.json")


def remove_stale_final_output(output: str | os.PathLike[str]) -> bool:
    """移除同名旧成品，保证「文件存在 == 本次通过」。

    只删普通文件/符号链接；目录或特殊文件保持原样并报错（fail-closed）。
    """
    target = lexical_path(output)
    info = _lstat_or_none(target)
    if info is None:
        return False
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise RuntimeError(f"拒绝删除非普通文件: {target}")
    os.unlink(target)
    _fsync_directory(target.parent)
    return True


def quarantine_final_output(output: str | os.PathLike[str]) -> Path | None:
    """把意外出现的最终名移开（保留取证价值），返回隔离后的路径。"""
    target = lexical_path(output)
    info = _lstat_or_none(target)
    if info is None:
        return None
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise RuntimeError(f"拒绝隔离非普通文件: {target}")
    quarantined = target.with_name(f"{target.name}.quarantine-{secrets.token_hex(6)}")
    os.replace(target, quarantined)
    _fsync_directory(target.parent)
    return quarantined


def write_lifecycle_failure_marker(output: str | os.PathLike[str], kind: str,
                                   errors: Sequence[str],
                                   error: BaseException | None = None) -> Path | None:
    """写持久化的硬阻断 marker；写失败时返回 None（调用方仍按阻断处理）。"""
    marker = failure_marker_path(output, kind)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "INCOMPLETE",
        "kind": kind,
        "output": str(lexical_path(output)),
        "errors": list(errors),
        "error": repr(error) if error is not None else None,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "meaning": "本轮中间态未清理干净，产物不可信，必须人工处理后重跑。",
    }
    try:
        atomic_write_json(marker, payload)
        return marker
    except Exception as exc:  # marker 写不进去也不能让调用方误判为通过
        print(f"[fail-closed] {kind} failure marker 写入失败: {exc}",
              file=sys.stderr, flush=True)
        return None


def _remove_temp_leaf(path: Path) -> None:
    """清理参考遍等临时叶子；目录/特殊文件不静默处理，直接报错。"""
    info = _lstat_or_none(path)
    if info is None:
        return
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise RuntimeError(f"拒绝删除非普通临时文件: {path}")
    os.unlink(path)
    _fsync_directory(path.parent)


# --------------------------------------------------------------------------- #
# 输入冻结副本
# --------------------------------------------------------------------------- #
class FrozenInputSet:
    """把输入拷进私有目录，并记录**实际拷贝字节**的摘要。

    冻结副本是拷贝而不是硬链接：硬链接的路径稳定，但仍与原 inode 共享就地写入。
    下游只允许打开冻结路径；``cleanup()`` 返回非空即代表本轮必须阻断。
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = lexical_path(root) if root else None
        self._entries: dict[str, tuple[Path, str, int]] = {}

    def freeze(self, role: str, path: str | os.PathLike[str]) -> Path:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role or ""):
            raise ValueError(f"非法冻结角色名: {role!r}")
        if role in self._entries:
            raise RuntimeError(f"重复冻结同一角色: {role}")
        source = lexical_path(path)
        if self.root is None:
            self.root = Path(tempfile.mkdtemp(prefix="fail-closed-inputs-"))
            os.chmod(self.root, 0o700)
        fd = os.open(str(source), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"冻结输入不是普通文件: {source}")
            destination = self.root / f"{role}{source.suffix}"
            out_fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT
                             | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            digest = hashlib.sha256()
            size = 0
            try:
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    _write_all_fd(out_fd, chunk)
                os.fsync(out_fd)
            finally:
                os.close(out_fd)
        finally:
            os.close(fd)
        if size <= 0:
            raise RuntimeError(f"冻结输入为空文件: {source}")
        self._entries[role] = (destination, digest.hexdigest(), size)
        _fsync_directory(self.root)
        return destination

    def path(self, role: str) -> Path:
        return self._entries[role][0]

    def digest(self, role: str) -> str:
        return self._entries[role][1]

    def entries(self) -> dict[str, dict[str, object]]:
        return {role: {"sha256": digest, "bytes": size}
                for role, (_path, digest, size) in self._entries.items()}

    def verify(self) -> list[str]:
        """重新哈希冻结副本，确认下游读到的就是冻结那一刻的字节。"""
        errors: list[str] = []
        for role, (path, digest, size) in sorted(self._entries.items()):
            try:
                actual = sha256_file(path)
            except Exception as exc:
                errors.append(f"冻结副本不可读 {role}: {exc!r}")
                continue
            if not hmac.compare_digest(actual, digest):
                errors.append(f"冻结副本内容被改动 {role}: {actual} != {digest}")
                continue
            if os.lstat(path).st_size != size:
                errors.append(f"冻结副本长度变化 {role}")
        return errors

    def cleanup(self) -> list[str]:
        """先校验再删除；任何一步失败都作为阻断错误返回。"""
        errors = self.verify()
        if self.root is not None and self.root.exists():
            try:
                shutil.rmtree(self.root)
            except Exception as exc:
                errors.append(f"冻结副本目录删除失败 {self.root}: {exc!r}")
        return errors


# --------------------------------------------------------------------------- #
# 排他 lease：同一个输出 stem 只允许一个执行者
# --------------------------------------------------------------------------- #
class OutputLockError(RuntimeError):
    """输出 stem 已被另一轮占用（或存在未归档锁）。"""


def output_lock_path(output: str | os.PathLike[str]) -> Path:
    target = lexical_path(output)
    return target.with_suffix(".lock")


class OutputLease:
    """O_EXCL 排他 lease；永不抢占已存在的锁，释放时校验 inode 身份。"""

    def __init__(self, output: str | os.PathLike[str]) -> None:
        self.output = lexical_path(output)
        self.path = output_lock_path(self.output)
        self.token = secrets.token_hex(24)
        self._identity: os.stat_result | None = None
        self._held = False

    def acquire(self) -> "OutputLease":
        _assert_directory(self.path.parent, create=True)
        payload = json.dumps({
            "schema_version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "started_at": dt.datetime.now().astimezone().isoformat(),
            "token": self.token,
            "output": str(self.output),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd: int | None = None
        try:
            fd = os.open(str(self.path), flags, 0o600)
            _write_all_fd(fd, payload)
            os.fsync(fd)
            self._identity = os.fstat(fd)
            self._held = True
            _fsync_directory(self.path.parent)
            return self
        except FileExistsError as exc:
            raise OutputLockError(
                f"输出正在被其他运行占用或存在未归档锁: {self.path}") from exc
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def release(self) -> None:
        if not self._held:
            return
        info = self.path.lstat()
        identity = self._identity
        if (identity is None or not stat.S_ISREG(info.st_mode)
                or info.st_dev != identity.st_dev or info.st_ino != identity.st_ino):
            raise RuntimeError(f"输出锁 inode 已被替换，拒绝删除: {self.path}")
        self.path.unlink()
        _fsync_directory(self.path.parent)
        self._held = False

    def __enter__(self) -> "OutputLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def acquire_output_lock(output: str | os.PathLike[str]) -> OutputLease:
    """获取 fail-closed lease；已存在的锁不会被自动偷走。"""
    return OutputLease(output).acquire()


# --------------------------------------------------------------------------- #
# 帧链：域分离 + 形状/序号的顺序敏感摘要
# --------------------------------------------------------------------------- #
class FrameChain:
    """解码帧序列的 SHA-256 累加器。

    直接拼接像素缓冲在尺寸可变时是有歧义的，因此每帧都按
    「帧号 / 维数 / 形状 / dtype / 字节」分帧写入，并加版本化域分隔符，
    避免别处算出的 SHA-256 被误当成帧序证据。
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(FRAME_CHAIN_DOMAIN)

    def update(self, frame_no: int, frame: np.ndarray) -> None:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.dtype != np.uint8 or frame_no < 0):
            raise RuntimeError(f"非法帧，无法建立帧序证据: frame={frame_no}")
        contiguous = np.ascontiguousarray(frame)
        dtype_name = contiguous.dtype.str.encode("ascii")
        self._digest.update(int(frame_no).to_bytes(8, "big"))
        self._digest.update(len(contiguous.shape).to_bytes(1, "big"))
        for value in contiguous.shape:
            self._digest.update(int(value).to_bytes(8, "big"))
        self._digest.update(len(dtype_name).to_bytes(2, "big"))
        self._digest.update(dtype_name)
        self._digest.update(memoryview(contiguous).cast("B"))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def decoded_frame_chain(path: str | os.PathLike[str]) -> dict:
    """完整解码一个视频，返回其呈现顺序的精确帧链。"""
    target = lexical_path(path)
    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频进行帧序验证: {target}")
    digest = FrameChain()
    count = 0
    dimensions: list[int] | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            shape = [int(frame.shape[1]), int(frame.shape[0])]
            if dimensions is None:
                dimensions = shape
            elif shape != dimensions:
                raise RuntimeError(
                    f"帧序验证视频尺寸在第 {count} 帧变化: {shape} != {dimensions}")
            digest.update(count, frame)
            count += 1
    finally:
        cap.release()
    return {
        "decoded_frames": count,
        "dimensions": dimensions,
        "frame_chain_sha256": digest.hexdigest(),
    }


# --------------------------------------------------------------------------- #
# 几何与遮挡原语
# --------------------------------------------------------------------------- #
def clamp_box(box: Sequence[float], shape: Sequence[int]) -> list[int]:
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [max(0, min(int(round(x1)), width)), max(0, min(int(round(y1)), height)),
            max(0, min(int(round(x2)), width)), max(0, min(int(round(y2)), height))]


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def expand_box(box: Sequence[float], shape: Sequence[int], ratio: float = 0.0) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    dx = (x2 - x1) * float(ratio)
    dy = (y2 - y1) * float(ratio)
    return clamp_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], shape)


def solid_fill(frame: np.ndarray, box: Sequence[float], *, fill: str = "mean") -> bool:
    """区域平均色纯色填充：信息被整体抹除，不存在可被锐化恢复的字符结构。"""
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return False
    region = frame[y1:y2, x1:x2]
    if fill == "black":
        color = np.zeros(3, dtype=np.uint8)
    else:
        color = np.clip(np.rint(region.reshape(-1, 3).mean(axis=0)), 0, 255).astype(np.uint8)
    region[:] = color
    return True


def region_covered_fraction(region: Sequence[int],
                            boxes: Iterable[Sequence[float]],
                            *, step: int = 2) -> float:
    """区域内被计划框覆盖的像素比例（按网格采样近似，区域都很小）。"""
    rx1, ry1, rx2, ry2 = [int(v) for v in region]
    if rx2 <= rx1 or ry2 <= ry1:
        return 1.0
    normalized = [clamp_box(b, (1 << 30, 1 << 30)) for b in boxes]
    total = 0
    covered = 0
    for y in range(ry1, ry2, max(1, int(step))):
        for x in range(rx1, rx2, max(1, int(step))):
            total += 1
            if any(b[0] <= x < b[2] and b[1] <= y < b[3] for b in normalized):
                covered += 1
    return covered / total if total else 1.0


def plate_texture_regions(frame: np.ndarray, roi_box: Sequence[float], *,
                          bright_min: float = DEFAULT_BRIGHT_MIN,
                          texture_tol: float = DEFAULT_TEXTURE_TOL,
                          window: Sequence[int] = TEXTURE_WINDOW,
                          min_pixels: int = TEXTURE_MIN_PIXELS) -> list[list[int]]:
    """在 ROI 内找「亮且局部对比度高」的区域，即仍可能被人眼读出字符的牌区。

    用滑窗均值/方差判定：纯色遮挡块的方差≈0 不会命中；只有仍保留字符明暗
    结构的亮块才会命中。这些阈值全部可配置，默认值只是保守起点。
    """
    x1, y1, x2, y2 = clamp_box(roi_box, frame.shape)
    if x2 - x1 < 8 or y2 - y1 < 6:
        return []
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win_w = max(4, min(int(window[0]), crop.shape[1]))
    win_h = max(3, min(int(window[1]), crop.shape[0]))
    mean = cv2.boxFilter(gray, -1, (win_w, win_h), normalize=True,
                         borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(gray * gray, -1, (win_w, win_h), normalize=True,
                            borderType=cv2.BORDER_REPLICATE)
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    hot = ((variance >= float(texture_tol) ** 2)
           & (mean >= float(bright_min))).astype(np.uint8)
    if not hot.any():
        return []
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(hot, connectivity=8)
    regions: list[list[int]] = []
    for index in range(1, count):
        rx, ry, rw, rh, area = stats[index]
        if int(area) < int(min_pixels):
            continue
        regions.append([int(x1 + rx), int(y1 + ry), int(x1 + rx + rw), int(y1 + ry + rh)])
    return regions


# --------------------------------------------------------------------------- #
# 候选生成（检测 + 跟踪）的输出契约
# --------------------------------------------------------------------------- #
@dataclass
class VehicleTrack:
    """一条车辆轨迹：``observations`` 是「这辆车在哪些帧被观察到」的框。

    审计只把车辆框当作「必须被解释的对象」，绝不把它当成牌区证据。
    """
    id: int
    cls: str = "vehicle"
    observations: dict[int, list[float]] = field(default_factory=dict)
    confidences: dict[int, float] = field(default_factory=dict)


@dataclass
class PlanBundle:
    """``plan`` 可调用对象的返回值；审计器不信任它的正确性，只把它当输入。"""
    plans: list[list[list[float]]]
    vehicle_tracks: list[VehicleTrack]
    width: int
    height: int
    fps: float
    meta: dict = field(default_factory=dict)


PlanFn = Callable[[Path], PlanBundle]
#: 渲染一遍的签名；默认实现是 ``render_masked_draft``，可被调用方替换
#: （例如接上 plate_redact.py 的双遍渲染器，或注入故意有缺陷的渲染器做回归）。
RenderFn = Callable[..., dict]


def validate_plan_bundle(bundle: object, *, frames: int | None = None) -> list[dict]:
    """校验候选输出的形状；任何畸形都不抛异常而是记阻断问题（fail-closed）。"""
    issues: list[dict] = []
    if not isinstance(bundle, PlanBundle):
        issues.append({"id": "engine:plan", "type": "engine_plan_invalid",
                       "non_dismissible": True, "decision": "UNKNOWN",
                       "error": f"plan 返回类型非法: {type(bundle).__name__}"})
        return issues
    if not isinstance(bundle.plans, list) or not bundle.plans:
        issues.append({"id": "engine:plan_empty", "type": "engine_plan_invalid",
                       "non_dismissible": True, "decision": "UNKNOWN",
                       "error": "plan 未给出任何帧的计划框"})
    if frames is not None and len(bundle.plans) != frames:
        issues.append({"id": "engine:plan_frames", "type": "engine_plan_invalid",
                       "non_dismissible": True, "decision": "UNKNOWN",
                       "expected_frames": frames, "planned_frames": len(bundle.plans)})
    if not bundle.vehicle_tracks:
        # 「没有车辆」与「车辆引擎没跑」是两件事，后者不配当作阴性结论。
        issues.append({"id": "coverage:no_vehicle_tracks",
                       "type": "coverage_no_vehicle_evidence",
                       "non_dismissible": True, "decision": "UNKNOWN",
                       "error": "没有任何车辆轨迹证据，无法做轨迹级解释性审计"})
    for track in bundle.vehicle_tracks:
        if not isinstance(track, VehicleTrack) or not track.observations:
            issues.append({"id": f"coverage:vehicle_track:{getattr(track, 'id', '?')}",
                           "type": "coverage_track_invalid",
                           "non_dismissible": True, "decision": "UNKNOWN"})
    return issues


# --------------------------------------------------------------------------- #
# 问题分类：人工只裁「视觉候选」，基础设施阻断永不可豁免
# --------------------------------------------------------------------------- #
def issue_is_non_dismissible(issue: object) -> bool:
    if not isinstance(issue, Mapping):
        return True  # 畸形问题行无法被安全解释
    if issue.get("non_dismissible"):
        return True
    issue_id = str(issue.get("id", ""))
    issue_type = str(issue.get("type", ""))
    if (issue_id.startswith(INFRASTRUCTURE_ISSUE_PREFIXES)
            or issue_type.startswith(("engine_", "decode_", "output_", "render_",
                                      "encoding_", "lease_", "cleanup_", "review_"))):
        return True
    if issue.get("reviewable") is True:
        return False
    return bool(issue.get("blocking"))


def issue_digest(issue: Mapping[str, object]) -> str:
    """把问题内容压成摘要（排除易变的 id 与错误文本），供人工复核绑定。"""
    evidence = {key: issue[key] for key in
                ("type", "decision", "frames", "first_frame", "last_frame",
                 "box", "reasons", "class")
                if key in issue}
    payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_safe_issue_ids(review: object) -> dict[str, str | None]:
    """把复核输入规范成 ``{issue_id: digest|None}``。

    接受 ``{"safe_issue_ids": {...}}`` / ``{"safe_issue_ids": [...]}``
    或直接的映射/列表。摘要缺失时只按 id 匹配，并在报告里标注。
    """
    if review is None:
        return {}
    rows = review
    if isinstance(review, Mapping) and "safe_issue_ids" in review:
        rows = review["safe_issue_ids"]
    if rows is None:
        return {}
    if isinstance(rows, Mapping):
        return {str(key): (None if value is None else str(value))
                for key, value in rows.items()}
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return {str(item): None for item in rows}
    raise ValueError("safe_issue_ids 必须是列表或映射")


def validate_safe_issue_ids(safe_ids: Mapping[str, str | None],
                            issues: Sequence[Mapping[str, object]]) -> list[dict]:
    """复核不得豁免基础设施阻断，也不得出现未知 id 或摘要不符。"""
    errors: list[dict] = []
    by_id = {str(issue.get("id")): issue for issue in issues
             if isinstance(issue, Mapping)}
    for safe_id, digest in sorted(safe_ids.items()):
        issue = by_id.get(safe_id)
        if issue is None:
            errors.append({"id": f"review:unknown:{safe_id}",
                           "type": "review_unknown_safe_id",
                           "non_dismissible": True, "decision": "UNKNOWN",
                           "safe_issue_id": safe_id})
            continue
        if issue_is_non_dismissible(issue):
            errors.append({"id": f"review:non_dismissible:{safe_id}",
                           "type": "review_non_dismissible_safe_id",
                           "non_dismissible": True, "decision": "UNKNOWN",
                           "safe_issue_id": safe_id})
            continue
        if digest is not None and not hmac.compare_digest(
                digest, issue_digest(issue)):
            errors.append({"id": f"review:binding:{safe_id}",
                           "type": "review_binding_mismatch",
                           "non_dismissible": True, "decision": "UNKNOWN",
                           "safe_issue_id": safe_id,
                           "expected_digest": issue_digest(issue)})
    return errors


def apply_review_dismissals(issues: Sequence[dict],
                            safe_ids: Mapping[str, str | None]) -> list[dict]:
    """应用真人安全裁决，但保留每一条基础设施阻断。"""
    return [issue for issue in issues
            if issue_is_non_dismissible(issue)
            or str(issue.get("id")) not in safe_ids]


# --------------------------------------------------------------------------- #
# 轨迹级「可解释牌区」
# --------------------------------------------------------------------------- #
def plan_box_belongs_to_vehicle(box: Sequence[float], vehicle_box: Sequence[float],
                                *, pad: float = 0.10) -> bool:
    """计划框中心是否落在该车辆框（外扩 pad）内。车辆框 ≠ 牌框，这里只回答归属。"""
    cx, cy = box_center(box)
    vx1, vy1, vx2, vy2 = [float(v) for v in vehicle_box]
    width = vx2 - vx1
    height = vy2 - vy1
    return ((vx1 - pad * width) <= cx <= (vx2 + pad * width)
            and (vy1 - pad * height) <= cy <= (vy2 + pad * height))


def explainable_plate_region(frame: np.ndarray, vehicle_box: Sequence[float],
                             owned_boxes: Sequence[Sequence[float]], *,
                             texture_tol: float = DEFAULT_TEXTURE_TOL,
                             bright_min: float = DEFAULT_BRIGHT_MIN
                             ) -> tuple[bool, str | None, list[list[int]]]:
    """这一帧、这辆车是否有「可解释牌区」。

    两条独立要求，缺一即 UNKNOWN：

    1. 存在属于该车辆的遮挡框（正向解释：遮挡是有计划的，不是碰巧涂了一块）；
    2. 该车辆框内不存在**未被计划框覆盖**的亮且带字符纹理区域
       （反向残留检查：编码后仍然读得出的牌区，说明漏打）。

    返回 ``(是否可解释, 原因, 残留区域)``。
    """
    if not owned_boxes:
        return False, "no_planned_mask_for_vehicle", []
    residual: list[list[int]] = []
    for region in plate_texture_regions(frame, vehicle_box,
                                        bright_min=bright_min,
                                        texture_tol=texture_tol):
        if region_covered_fraction(region, owned_boxes) >= COVERAGE_MIN_FRACTION:
            continue
        residual.append(region)
    if residual:
        return False, "residual_plate_texture", residual
    return True, None, []


def unresolved_vehicle_issues(plans: Sequence[Sequence[Sequence[float]]],
                              vehicle_tracks: Sequence[VehicleTrack]) -> list[dict]:
    """只按计划框（不解码成品）汇总没有可解释牌区的车辆轨迹。

    用途是给调用方一个「不打开视频也能先看一遍」的预览；判定结论仍以
    :func:`audit_decoded_output` 在**编码后重新解码**的结果为准。
    """
    issues: list[dict] = []
    for track in vehicle_tracks:
        uncovered: list[int] = []
        for frame_no, vehicle_box in sorted(track.observations.items()):
            frame_plans = plans[frame_no] if 0 <= frame_no < len(plans) else []
            owned = [box for box in frame_plans
                     if plan_box_belongs_to_vehicle(box, vehicle_box)]
            if not owned:
                uncovered.append(int(frame_no))
        if not uncovered:
            continue
        representative = max(uncovered, key=lambda f: track.confidences.get(f, 0.0))
        issues.append({
            "id": f"vehicle_track:{track.id}",
            "type": "vehicle_without_plate_mask",
            "decision": "UNKNOWN",
            "reviewable": True,
            "class": track.cls,
            "frame": int(representative),
            "frames": [int(f) for f in uncovered],
            "first_frame": int(min(uncovered)),
            "last_frame": int(max(uncovered)),
            "confidence": round(float(track.confidences.get(representative, 0.0)), 4),
            "box": [round(float(v), 2) for v in track.observations[representative]],
        })
    return issues


# --------------------------------------------------------------------------- #
# 渲染与双遍一致性
# --------------------------------------------------------------------------- #
def probe_video(path: str | os.PathLike[str]) -> dict:
    """读取源视频的帧数/尺寸/帧率；读不到就报错（不允许静默降级）。"""
    target = lexical_path(path)
    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {target}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if [int(frame.shape[1]), int(frame.shape[0])] != [width, height]:
                raise RuntimeError(f"源视频在第 {decoded} 帧尺寸变化")
            decoded += 1
    finally:
        cap.release()
    if decoded <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"源视频没有可解码帧: {target}")
    return {"frames": decoded, "width": width, "height": height,
            "fps": fps if fps > 0 else 0.0, "declared_frames": declared}


def _prepare_render_leaf(path: Path) -> None:
    _assert_directory(path.parent, create=True)
    info = _lstat_or_none(path)
    if info is None:
        return
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise RuntimeError(f"拒绝覆盖非普通文件: {path}")
    os.unlink(path)


def render_masked_draft(source: str | os.PathLike[str], destination: Path,
                        plans: Sequence[Sequence[Sequence[float]]], *,
                        width: int, height: int, fps: float,
                        fourcc: str = "mp4v", progress: bool = False,
                        full_audit: bool = True) -> dict:
    """独立解码源、按计划遮挡、编码成一个渲染遍。

    调用方会用它跑**两遍**（写两个不同的叶子）：共享内存帧序列只能证明
    「同一份内存被写了两遍」，不能证明源可以被同序复现。

    ``full_audit=False`` 表示诊断性质的局部运行：此时不做「源是否还有额外帧」
    的完整扫描，但这类运行在编排层永远只能产出 draft。
    """
    source = str(lexical_path(source))
    _prepare_render_leaf(destination)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开渲染源: {source}")
    render_fps = float(fps) if fps and fps > 0 else 25.0
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*fourcc),
                             render_fps, (int(width), int(height)))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建视频: {destination}")
    source_chain = FrameChain()
    masked_chain = FrameChain()
    frame_no = 0
    try:
        while frame_no < len(plans):
            ok, frame = cap.read()
            if not ok:
                break
            shape = [int(frame.shape[1]), int(frame.shape[0])]
            if shape != [int(width), int(height)]:
                raise RuntimeError(
                    f"渲染源尺寸在第 {frame_no} 帧变化: {shape} != {[width, height]}")
            source_chain.update(frame_no, frame)
            for box in plans[frame_no]:
                solid_fill(frame, box)
            masked_chain.update(frame_no, frame)
            writer.write(frame)
            frame_no += 1
            if progress and frame_no % 50 == 0:
                print(f"渲染 {frame_no}/{len(plans)} 帧", flush=True)
        if frame_no != len(plans):
            raise RuntimeError(
                f"源视频提前结束: 计划 {len(plans)} 帧，实际 {frame_no} 帧")
        if full_audit:
            extra, _ = cap.read()
            if extra:
                raise RuntimeError(f"源视频在计划 {len(plans)} 帧之后仍有额外帧")
    finally:
        writer.release()
        cap.release()
    if _regular_leaf(destination) is None:
        raise RuntimeError(f"渲染器未生成普通文件: {destination}")
    return {
        "submitted_frames": frame_no,
        "source_frame_chain_sha256": source_chain.hexdigest(),
        "masked_input_frame_chain_sha256": masked_chain.hexdigest(),
    }


def two_pass_issues(primary: Mapping[str, object], reference: Mapping[str, object],
                    decoded_primary: Mapping[str, object],
                    decoded_reference: Mapping[str, object], *,
                    expected_frames: int,
                    expected_dimensions: Sequence[int]) -> list[dict]:
    """源像素 / 遮挡后 / 解码帧链 三条链必须自洽。"""
    issues: list[dict] = []

    def mismatch(issue_id: str, detail: dict) -> None:
        issues.append({"id": issue_id, "type": "render_pass_mismatch",
                       "non_dismissible": True, "decision": "UNKNOWN", **detail})

    for key, issue_id in (("source_frame_chain_sha256", "render:two_pass_source"),
                          ("masked_input_frame_chain_sha256", "render:two_pass_masked")):
        if primary.get(key) != reference.get(key):
            mismatch(issue_id, {"chain": key, "primary": primary.get(key),
                                "reference": reference.get(key)})
    if primary.get("submitted_frames") != expected_frames:
        mismatch("render:primary_frames", {"primary": primary.get("submitted_frames"),
                                           "expected_frames": int(expected_frames)})
    if reference.get("submitted_frames") != expected_frames:
        mismatch("render:reference_frames", {"reference": reference.get("submitted_frames"),
                                             "expected_frames": int(expected_frames)})
    if decoded_primary.get("frame_chain_sha256") != decoded_reference.get("frame_chain_sha256"):
        mismatch("render:decoded_chain",
                 {"primary": decoded_primary.get("frame_chain_sha256"),
                  "reference": decoded_reference.get("frame_chain_sha256")})
    for label, decoded in (("primary", decoded_primary), ("reference", decoded_reference)):
        if int(decoded.get("decoded_frames") or 0) != int(expected_frames):
            mismatch(f"render:decoded_frames_{label}",
                     {"decoded_frames": decoded.get("decoded_frames"),
                      "expected_frames": int(expected_frames)})
        if list(decoded.get("dimensions") or []) != [int(v) for v in expected_dimensions]:
            mismatch(f"render:decoded_dimensions_{label}",
                     {"decoded_dimensions": decoded.get("dimensions"),
                      "expected_dimensions": [int(v) for v in expected_dimensions]})
    return issues


# --------------------------------------------------------------------------- #
# 输出侧独立审计：只看「文件里到底有什么」
# --------------------------------------------------------------------------- #
def audit_decoded_output(draft: str | os.PathLike[str], *,
                         plans: Sequence[Sequence[Sequence[float]]],
                         vehicle_tracks: Sequence[VehicleTrack],
                         expected_frames: int,
                         expected_dimensions: Sequence[int],
                         flat_tol: float = DEFAULT_FLAT_TOL,
                         texture_tol: float = DEFAULT_TEXTURE_TOL,
                         bright_min: float = DEFAULT_BRIGHT_MIN) -> dict:
    """编码完成后重新解码，逐车辆轨迹要求「可解释牌区」；无解释记 UNKNOWN 并阻断。

    这一步不信任检测、跟踪和渲染：它只回答两件事——
    计划要打码的地方是不是真的成了纯色块；车辆框里是不是还有未被解释的可读牌区。
    这里始终报出全部候选问题，人工安全裁决由 ``validate_safe_issue_ids`` 与
    ``apply_review_dismissals`` 统一处理（这样摘要有误时报的是摘要不符，而不是「未知问题」）。
    """
    target = lexical_path(draft)
    issues: list[dict] = []
    warnings: list[dict] = []
    binding: dict[str, object] = {
        "decoded_frames": 0,
        "decoded_dimensions": None,
        "decoded_frame_chain_sha256": None,
        "flat_checked": 0,
        "residual_regions": 0,
    }
    if _regular_leaf(target) is None:
        issues.append({"id": "output:leaf", "type": "output_leaf_error",
                       "non_dismissible": True, "decision": "UNKNOWN"})
        return {"issues": issues, "warnings": warnings, "binding": binding}

    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        issues.append({"id": "output:decode", "type": "output_decode_error",
                       "non_dismissible": True, "decision": "UNKNOWN"})
        return {"issues": issues, "warnings": warnings, "binding": binding}
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if [width, height] != [int(v) for v in expected_dimensions]:
        issues.append({"id": "output:dimensions", "type": "output_dimensions_mismatch",
                       "non_dismissible": True, "decision": "UNKNOWN",
                       "expected_dimensions": [int(v) for v in expected_dimensions],
                       "decoded_dimensions": [width, height]})
    chain = FrameChain()
    uncovered: dict[int, list[int]] = {}
    residual: dict[int, list[dict]] = {}
    decoded_frames = 0
    flat_checked = 0
    residual_regions = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no = decoded_frames
            chain.update(frame_no, frame)
            decoded_frames += 1
            frame_plans = plans[frame_no] if 0 <= frame_no < len(plans) else []
            for index, box in enumerate(frame_plans):
                x1, y1, x2, y2 = clamp_box(box, frame.shape)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                region = frame[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
                flat_checked += 1
                std = float(region.std(axis=0).max())
                if std > float(flat_tol):
                    issues.append({
                        "id": f"render:flat:{frame_no}:{index}",
                        "type": "render_not_flat",
                        "non_dismissible": True, "decision": "UNKNOWN",
                        "frame": frame_no, "box": [x1, y1, x2, y2],
                        "std": round(std, 2), "flat_tol": float(flat_tol),
                        "meaning": "计划的遮挡没有真正落到成品画面上。",
                    })
            for track in vehicle_tracks:
                vehicle_box = track.observations.get(frame_no)
                if vehicle_box is None:
                    continue
                owned = [box for box in frame_plans
                         if plan_box_belongs_to_vehicle(box, vehicle_box)]
                explainable, reason, regions = explainable_plate_region(
                    frame, vehicle_box, owned,
                    texture_tol=texture_tol, bright_min=bright_min)
                if explainable:
                    continue
                if reason == "no_planned_mask_for_vehicle":
                    uncovered.setdefault(track.id, []).append(frame_no)
                else:
                    residual_regions += len(regions)
                    residual.setdefault(track.id, []).append(
                        {"frame": frame_no, "regions": regions,
                         "box": [round(float(v), 2) for v in vehicle_box]})
    finally:
        cap.release()

    binding["decoded_frames"] = decoded_frames
    binding["decoded_dimensions"] = [width, height]
    binding["decoded_frame_chain_sha256"] = chain.hexdigest()
    binding["flat_checked"] = flat_checked
    binding["residual_regions"] = residual_regions
    if decoded_frames != int(expected_frames):
        issues.append({"id": "output:frames", "type": "output_frame_count_mismatch",
                       "non_dismissible": True, "decision": "UNKNOWN",
                       "decoded_frames": decoded_frames,
                       "expected_frames": int(expected_frames)})
    if len(plans) > decoded_frames:
        warnings.append({"id": "output:plans_beyond_eof", "type": "plan_beyond_eof",
                         "planned_frames": len(plans), "decoded_frames": decoded_frames})

    by_id = {track.id: track for track in vehicle_tracks}
    for track_id in sorted(set(uncovered) | set(residual)):
        track = by_id.get(track_id)
        issue_id = f"vehicle_track:{track_id}"
        reasons: dict[str, object] = {}
        if uncovered.get(track_id):
            frames = sorted(uncovered[track_id])
            reasons["no_planned_mask_for_vehicle"] = {
                "frames": frames, "first_frame": frames[0], "last_frame": frames[-1]}
        if residual.get(track_id):
            reasons["residual_plate_texture"] = residual[track_id]
        frames = sorted(set(uncovered.get(track_id, []))
                        | {row["frame"] for row in residual.get(track_id, [])})
        representative = frames[0]
        issues.append({
            "id": issue_id,
            "type": "vehicle_without_explainable_plate_region",
            "decision": "UNKNOWN",
            "reviewable": True,
            "class": getattr(track, "cls", "vehicle") if track else "vehicle",
            "frame": int(representative),
            "frames": [int(f) for f in frames],
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "box": [round(float(v), 2) for v in track.observations[representative]]
            if track and representative in track.observations else None,
            "reasons": reasons,
            "meaning": ("该车辆轨迹在成品里仍缺少可解释的牌区证据；"
                        "必须由真人补齐精确框或确认确实无牌，否则永久阻断。"),
        })
    return {"issues": issues, "warnings": warnings, "binding": binding}


# --------------------------------------------------------------------------- #
# 发布
# --------------------------------------------------------------------------- #
def publish_canonical_draft(output: str | os.PathLike[str],
                            draft: str | os.PathLike[str]) -> None:
    """把已通过全部审计的 canonical draft 原子发布为最终产物。"""
    output = lexical_path(output)
    draft = lexical_path(draft)
    draft_info = _regular_leaf(draft)
    if draft_info is None:
        raise RuntimeError(f"canonical draft 不是普通文件: {draft}")
    if _lstat_or_none(output) is not None:
        raise RuntimeError(f"最终输出在发布前已存在: {output}")
    if not _same_regular_leaf(draft, draft_info):
        raise RuntimeError(f"canonical draft 在发布前已被替换: {draft}")
    _assert_replaceable_leaf(output)
    moved = False
    try:
        os.replace(draft, output)
        moved = True
        _fsync_directory(output.parent)
        if _regular_leaf(output) is None:
            raise RuntimeError(f"发布后最终输出不是普通文件: {output}")
    except Exception:
        if moved:
            try:
                if _regular_leaf(output) is not None and _lstat_or_none(draft) is None:
                    os.replace(output, draft)
                    _fsync_directory(draft.parent)
            except Exception as rollback_exc:
                raise RuntimeError(f"发布后回滚 draft 失败: {rollback_exc!r}") from rollback_exc
        raise


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
@dataclass
class RunReport:
    status: str
    output: Path
    draft: Path | None
    report_path: Path | None
    issues: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def summary(self) -> str:
        return (f"status={self.status} issues={len(self.issues)} "
                f"warnings={len(self.warnings)} output={self.output}")


STATUS_MEANING = {
    "PASS": "审计通过，最终产物就是本次运行的输出；文件存在 == 本次通过。",
    "BLOCKED": "存在未解释的阻断项，只保留 canonical draft，不产出最终文件。",
    "SMOKE_TEST": "诊断运行（例如限制了帧数），永远不构成通过证明。",
    "INCOMPLETE": "生命周期清理失败或流程中断，中间态不可信，必须人工处理后重跑。",
}


def _append_issue_once(issues: list[dict], issue: dict) -> None:
    issue_id = issue.get("id")
    if issue_id is not None and any(row.get("id") == issue_id for row in issues):
        return
    issues.append(issue)


def _reference_temp_path(draft: Path) -> Path:
    # 保留扩展名：编码器要靠后缀选择封装格式。
    return draft.with_name(f".{draft.stem}.reference-{secrets.token_hex(6)}{draft.suffix}")


class _CleanupLedger:
    """按 marker 类别收集清理失败信息；任一类别非空都代表本轮必须阻断。"""

    def __init__(self) -> None:
        self._errors: dict[str, list[str]] = {kind: [] for kind in MARKER_KINDS}

    def add(self, kind: str, message: str) -> None:
        if kind not in self._errors:
            raise ValueError(f"未知的 failure marker 类别: {kind!r}")
        self._errors[kind].append(message)

    def failed_kinds(self) -> list[str]:
        return [kind for kind in MARKER_KINDS if self._errors[kind]]

    def errors_for(self, kind: str) -> list[str]:
        return list(self._errors[kind])

    def all_messages(self) -> list[str]:
        return [f"{kind}: {message}"
                for kind in MARKER_KINDS for message in self._errors[kind]]

    def __bool__(self) -> bool:
        return bool(self.failed_kinds())


def _cleanup_lifecycle(frozen: FrozenInputSet, lease: OutputLease | None,
                       reference: Path | None) -> _CleanupLedger:
    """清理冻结副本 / 参考遍临时产物 / lease；逐项记录失败，绝不抛出。"""
    ledger = _CleanupLedger()
    try:
        for message in frozen.cleanup():
            ledger.add("frozen_input_cleanup", message)
    except Exception as exc:
        ledger.add("frozen_input_cleanup", f"{type(exc).__name__}: {exc}")
    if reference is not None:
        try:
            _remove_temp_leaf(reference)
        except Exception as exc:
            ledger.add("draft_cleanup", f"参考遍临时产物: {type(exc).__name__}: {exc}")
    if lease is not None:
        try:
            lease.release()
        except Exception as exc:
            ledger.add("output_lock_release", f"{type(exc).__name__}: {exc}")
    return ledger


def run_fail_closed(source: str | os.PathLike[str],
                    output: str | os.PathLike[str], *,
                    plan: PlanFn,
                    render: RenderFn | None = None,
                    safe_issue_ids: object = None,
                    flat_tol: float = DEFAULT_FLAT_TOL,
                    texture_tol: float = DEFAULT_TEXTURE_TOL,
                    bright_min: float = DEFAULT_BRIGHT_MIN,
                    max_frames: int = 0,
                    fourcc: str = "mp4v",
                    progress: bool = False,
                    freeze_root: str | os.PathLike[str] | None = None) -> RunReport:
    """跑完整条 fail-closed 链路并返回报告。

    ``plan`` 负责候选生成（检测 + 跟踪），签名 ``plan(frozen_source) -> PlanBundle``；
    它拿到的永远是冻结副本路径。``max_frames`` 非零表示诊断运行，永远只产 draft。
    """
    output_path = lexical_path(output)
    _assert_directory(output_path.parent, create=True)
    draft = canonical_draft_path(output_path)
    report_path = report_path_for(output_path)
    issues: list[dict] = []
    warnings: list[dict] = []
    artifacts: dict[str, object] = {}
    frozen = FrozenInputSet(root=freeze_root)
    lease: OutputLease | None = None
    reference: Path | None = None
    status = "INCOMPLETE"
    published = False
    smoke = int(max_frames or 0) > 0
    frozen_entries: dict[str, object] = {}
    two_pass: dict[str, object] = {}
    audit_binding: dict[str, object] = {}

    try:
        # ---- 阶段 1：冻结输入 + 排他 lease + 清掉同名旧成品 ----
        frozen_source = frozen.freeze("source", source)
        frozen_entries = frozen.entries()
        lease = acquire_output_lock(output_path)
        if remove_stale_final_output(output_path):
            warnings.append({"id": "output:stale_final",
                             "type": "stale_final_output_removed",
                             "meaning": "运行前已移除同名旧成品，避免旧文件冒充本次结果。"})

        probe = probe_video(frozen_source)
        frames = int(probe["frames"])
        dimensions = [int(probe["width"]), int(probe["height"])]

        # ---- 阶段 2：候选生成 ----
        bundle: PlanBundle | None = None
        try:
            bundle = plan(frozen_source)
        except Exception as exc:
            _append_issue_once(issues, {"id": "engine:plan", "type": "engine_plan_error",
                                        "non_dismissible": True, "decision": "UNKNOWN",
                                        "error": repr(exc)})
        if bundle is not None:
            for issue in validate_plan_bundle(bundle, frames=frames if not smoke else None):
                _append_issue_once(issues, issue)

        plans: list[list[list[float]]] = []
        vehicle_tracks: list[VehicleTrack] = []
        if bundle is not None and isinstance(bundle.plans, list) and bundle.plans:
            plans = bundle.plans[:max_frames] if smoke else bundle.plans
            vehicle_tracks = list(bundle.vehicle_tracks or [])
            if not smoke and len(plans) != frames:
                _append_issue_once(issues, {
                    "id": "engine:plan_frames", "type": "engine_plan_frames_mismatch",
                    "non_dismissible": True, "decision": "UNKNOWN",
                    "planned_frames": len(plans), "source_frames": frames})

        # ---- 阶段 3：双遍渲染 + 一致性 ----
        render_fn = render or render_masked_draft
        if plans:
            primary_pass = render_fn(
                frozen_source, draft, plans, width=dimensions[0],
                height=dimensions[1], fps=float(bundle.fps) if bundle else 0.0,
                fourcc=fourcc, progress=progress, full_audit=not smoke)
            reference = _reference_temp_path(draft)
            reference_pass = render_fn(
                frozen_source, reference, plans, width=dimensions[0],
                height=dimensions[1], fps=float(bundle.fps) if bundle else 0.0,
                fourcc=fourcc, progress=False, full_audit=not smoke)
            decoded_primary = decoded_frame_chain(draft)
            decoded_reference = decoded_frame_chain(reference)
            for issue in two_pass_issues(primary_pass, reference_pass,
                                         decoded_primary, decoded_reference,
                                         expected_frames=len(plans),
                                         expected_dimensions=dimensions):
                _append_issue_once(issues, issue)
            two_pass = {
                "planned_frames": len(plans),
                "source_frame_chain_sha256": primary_pass["source_frame_chain_sha256"],
                "masked_input_frame_chain_sha256":
                    primary_pass["masked_input_frame_chain_sha256"],
                "decoded_frame_chain_sha256": decoded_primary["frame_chain_sha256"],
            }
            artifacts["draft_sha256"] = sha256_file(draft)

            # ---- 阶段 4：输出侧独立审计（对 draft 解码；发布只是同一 inode 改名）----
            audit = audit_decoded_output(
                draft, plans=plans, vehicle_tracks=vehicle_tracks,
                expected_frames=len(plans), expected_dimensions=dimensions,
                flat_tol=flat_tol, texture_tol=texture_tol,
                bright_min=bright_min)
            for issue in audit["issues"]:
                _append_issue_once(issues, issue)
            warnings.extend(audit["warnings"])
            audit_binding = audit["binding"]

        # ---- 阶段 5：人工安全裁决（只对视觉候选生效）----
        try:
            safe_map = normalize_safe_issue_ids(safe_issue_ids)
        except Exception as exc:
            safe_map = {}
            _append_issue_once(issues, {"id": "review:input", "type": "review_input_error",
                                        "non_dismissible": True, "decision": "UNKNOWN",
                                        "error": repr(exc)})
        for issue in validate_safe_issue_ids(safe_map, issues):
            _append_issue_once(issues, issue)
        remaining = apply_review_dismissals(issues, safe_map)

        # ---- 阶段 6：决定发布还是保持 draft ----
        if remaining:
            # 什么都没渲染出来（候选生成就失败）属于流程中断，而不是「跑完但被阻断」。
            status = "BLOCKED" if plans else "INCOMPLETE"
            if _lstat_or_none(output_path) is not None:
                quarantined = quarantine_final_output(output_path)
                warnings.append({"id": "output:quarantine",
                                 "type": "final_output_quarantined",
                                 "quarantined": str(quarantined)})
        elif smoke:
            status = "SMOKE_TEST"
        elif _regular_leaf(draft) is None:
            status = "BLOCKED"
            _append_issue_once(issues, {"id": "output:draft_missing",
                                        "type": "output_leaf_error",
                                        "non_dismissible": True, "decision": "UNKNOWN",
                                        "error": "审计通过但 canonical draft 不存在"})
        else:
            audited_sha256 = str(artifacts.get("draft_sha256") or "")
            publish_canonical_draft(output_path, draft)
            published = True
            published_sha256 = sha256_file(output_path)
            if not audited_sha256 or not hmac.compare_digest(
                    audited_sha256, published_sha256):
                # 发布前后必须是同一个字节流：否则本轮审计结论不适用于成品。
                raise RuntimeError("发布后的产物与已审计的 draft 内容不一致")
            status = "PASS"
            artifacts["output_sha256"] = published_sha256

        if status != "PASS" and _regular_leaf(draft) is not None:
            artifacts["draft"] = str(draft)
    except Exception as exc:
        status = "INCOMPLETE"
        _append_issue_once(issues, {"id": "engine:pipeline", "type": "engine_pipeline_error",
                                    "non_dismissible": True, "decision": "UNKNOWN",
                                    "error": repr(exc)})
        try:
            quarantined = quarantine_final_output(output_path)
            if quarantined is not None:
                issues.append({"id": "output:quarantine_on_error",
                               "type": "output_quarantined_on_error",
                               "non_dismissible": True, "decision": "UNKNOWN",
                               "quarantined": str(quarantined)})
        except Exception as remove_exc:
            _append_issue_once(issues, {"id": "output:final_isolation",
                                        "type": "output_cleanup_error",
                                        "non_dismissible": True, "decision": "UNKNOWN",
                                        "error": repr(remove_exc)})
    finally:
        # ---- 阶段 7：生命周期清理；任何失败都算阻断 ----
        ledger = _cleanup_lifecycle(frozen, lease, reference)
        if ledger:
            status = "INCOMPLETE"
            for kind in ledger.failed_kinds():
                marker = write_lifecycle_failure_marker(
                    output_path, kind, ledger.errors_for(kind))
                _append_issue_once(issues, {
                    "id": f"cleanup:{kind}", "type": f"cleanup_{kind}_failure",
                    "non_dismissible": True, "decision": "UNKNOWN",
                    "errors": ledger.errors_for(kind),
                    "marker": str(marker) if marker else None,
                    "meaning": "清理失败也算阻断：中间态不允许留下当成功。",
                })
            try:
                quarantined = quarantine_final_output(output_path)
                if quarantined is not None:
                    artifacts["quarantined"] = str(quarantined)
            except Exception as exc:
                _append_issue_once(issues, {"id": "cleanup:final_isolation",
                                            "type": "cleanup_output_isolation_failure",
                                            "non_dismissible": True, "decision": "UNKNOWN",
                                            "error": repr(exc)})
        if status == "INCOMPLETE" and published and _lstat_or_none(output_path) is None:
            artifacts.pop("output_sha256", None)

    # ---- 报告：PASS 报告写不出来就必须撤掉产物（没有证据就没有结论）----
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "meaning": STATUS_MEANING[status],
        "source": {"path": os.fspath(source), "frozen_inputs": frozen_entries},
        "output": {"path": str(output_path), "exists": _lstat_or_none(output_path) is not None},
        "draft": {"path": str(draft), "exists": _regular_leaf(draft) is not None},
        "two_pass": two_pass,
        "decoded_output_audit": audit_binding,
        "issues": issues,
        "warnings": warnings,
        "review_issue_bindings": {str(issue.get("id")): issue_digest(issue)
                                  for issue in issues
                                  if isinstance(issue, Mapping) and issue.get("reviewable")},
        "config": {"flat_tol": float(flat_tol), "texture_tol": float(texture_tol),
                   "bright_min": float(bright_min), "max_frames": int(max_frames),
                   "fourcc": fourcc},
        "artifacts": artifacts,
        "created_at": dt.datetime.now().astimezone().isoformat(),
    }
    report_written = True
    try:
        atomic_write_json(report_path, report)
    except Exception as exc:
        report_written = False
        issues.append({"id": "output:report", "type": "output_report_error",
                       "non_dismissible": True, "decision": "UNKNOWN",
                       "error": repr(exc)})
        if status == "PASS":
            status = "INCOMPLETE"
            report["status"] = status
            report["meaning"] = STATUS_MEANING[status]
            report["issues"] = issues
            try:
                quarantined = quarantine_final_output(output_path)
                if quarantined is not None:
                    artifacts["quarantined"] = str(quarantined)
            except Exception as isolate_exc:
                issues.append({"id": "output:report_isolation",
                               "type": "output_cleanup_error",
                               "non_dismissible": True, "decision": "UNKNOWN",
                               "error": repr(isolate_exc)})

    if status == "PASS":
        artifacts["output_sha256"] = artifacts.get("output_sha256") or sha256_file(output_path)
    return RunReport(status=status, output=output_path,
                     draft=draft if _regular_leaf(draft) is not None else None,
                     report_path=report_path if report_written else None,
                     issues=issues, warnings=warnings, artifacts=artifacts,
                     extra={"two_pass": two_pass, "decoded_output_audit": audit_binding,
                            "frozen_inputs": frozen_entries, "published": published})


# --------------------------------------------------------------------------- #
# 适配器接入：用仓库既有的可插拔检测器做候选生成（第三方依赖惰性 import）
# --------------------------------------------------------------------------- #
def make_adapter_plan(detectors, vehicle_detector=None, *, pad: float = 0.08,
                      vehicle_conf: float = 0.2, max_stale: int = 6) -> PlanFn:
    """把 adapters.py 的检测器包装成 ``plan`` 可调用对象。

    这只负责「生成候选」，不负责「证明安全」：候选的正确性由输出侧审计独立复核。
    没有车辆检测器时不会伪造轨迹——``validate_plan_bundle`` 会因缺少车辆证据阻断。
    """
    def plan(frozen_source: Path) -> PlanBundle:
        from plate_tracker import PlateTracker  # 惰性 import：纯 numpy 实现

        cap = cv2.VideoCapture(str(frozen_source))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开冻结输入: {frozen_source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        plate_tracker = PlateTracker()
        vehicle_tracker = PlateTracker(max_age=30) if vehicle_detector is not None else None
        plans: list[list[list[float]]] = []
        observations: dict[int, dict[int, list[float]]] = {}
        confidences: dict[int, dict[int, float]] = {}
        frame_no = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                shape = (int(frame.shape[0]), int(frame.shape[1]))
                detections = []
                for det in detectors.detect(frame) or []:
                    x1, y1, x2, y2 = det.box
                    detections.append(([x1, y1, x2, y2], float(det.score)))
                detections.sort(key=lambda row: row[1], reverse=True)
                plate_tracker.predict()
                active = plate_tracker.update(
                    [(box[0], box[1], box[2], box[3], score, True)
                     for box, score in detections], frame_no)
                plans.append([expand_box(box, shape, pad) for box, _tid, _valid in active])
                if vehicle_tracker is not None:
                    vehicle_detections = []
                    for box in vehicle_detector.detect(frame, conf=vehicle_conf) or []:
                        x1, y1, x2, y2 = [float(v) for v in box[:4]]
                        score = float(box[4]) if len(box) > 4 else 0.0
                        vehicle_detections.append(([x1, y1, x2, y2], score))
                    vehicle_detections.sort(key=lambda row: row[1], reverse=True)
                    vehicle_tracker.predict()
                    vehicle_active = vehicle_tracker.update(
                        [(box[0], box[1], box[2], box[3], score, True)
                         for box, score in vehicle_detections], frame_no)
                    for box, track_id, _valid in vehicle_active:
                        observations.setdefault(track_id, {})[frame_no] = list(box)
                        confidences.setdefault(track_id, {})[frame_no] = 0.0
                frame_no += 1
        finally:
            cap.release()
        if frame_no <= 0:
            raise RuntimeError(f"冻结输入没有可解码帧: {frozen_source}")
        tracks = [VehicleTrack(id=track_id, observations=observations[track_id],
                               confidences=confidences.get(track_id, {}))
                  for track_id in sorted(observations)]
        return PlanBundle(plans=plans, vehicle_tracks=tracks, width=width,
                          height=height, fps=fps, meta={"frames": frame_no})
    return plan


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="fail-closed 审计与发布：冻结输入 → 排他 lease → draft → 原子发布")
    parser.add_argument('--source', required=True,
                        help='输入视频文件（摄像头索引会被拒绝：实时流没有可复核帧序）')
    parser.add_argument('--output', required=True, help='最终产物路径')
    parser.add_argument('--detector', action='append', default=None,
                        help='车牌检测引擎规格，可重复传入取并集（默认 fast-alpr）')
    parser.add_argument('--detector-conf', type=float, default=0.3)
    parser.add_argument('--hyper-conf', type=float, default=0.35)
    parser.add_argument('--vehicle-detector', default='',
                        help='车辆检测引擎规格；轨迹级审计需要它，否则只会产 draft')
    parser.add_argument('--vehicle-conf', type=float, default=0.2)
    parser.add_argument('--review', default='', help='复核 JSON：{"safe_issue_ids": {...}}')
    parser.add_argument('--pad', type=float, default=0.08, help='打码框外扩比例')
    parser.add_argument('--flat-tol', type=float, default=DEFAULT_FLAT_TOL,
                        help='判定纯色块的逐通道标准差上限（可配置）')
    parser.add_argument('--texture-tol', type=float, default=DEFAULT_TEXTURE_TOL,
                        help='判定残留字符纹理的局部对比度下限（可配置）')
    parser.add_argument('--bright-min', type=float, default=DEFAULT_BRIGHT_MIN,
                        help='参与牌区纹理判定的最低亮度（可配置）')
    parser.add_argument('--fourcc', default='mp4v', help='视频编码 fourcc')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='诊断用：非零时永远只产 draft，不构成任何通过证明')
    parser.add_argument('--progress', type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source
    if source.strip().lstrip('+-').isdigit() or re.fullmatch(r"\d+", source.strip()):
        raise SystemExit('摄像头源无法做双遍校验与输出侧审计，请先录制成文件')
    if not os.path.exists(source):
        raise SystemExit(f'输入不存在: {source}')

    # 第三方引擎在这里才被 import（惰性）：仓库不打包也不默认加载它们。
    from adapters import DetectorError, build_detectors, build_vehicle_detector

    specs = args.detector or ['fast-alpr']
    try:
        detectors = build_detectors(specs, conf=args.detector_conf,
                                    hyper_conf=args.hyper_conf)
        vehicle_detector = (build_vehicle_detector(args.vehicle_detector)
                            if args.vehicle_detector else None)
    except DetectorError as exc:
        raise SystemExit(f'检测器不可用: {exc}')

    review = load_strict_json(args.review) if args.review else None
    plan = make_adapter_plan(detectors, vehicle_detector, pad=args.pad,
                             vehicle_conf=args.vehicle_conf)
    try:
        report = run_fail_closed(
            source, args.output, plan=plan, safe_issue_ids=review,
            flat_tol=args.flat_tol, texture_tol=args.texture_tol,
            bright_min=args.bright_min, max_frames=args.max_frames,
            fourcc=args.fourcc, progress=bool(args.progress))
    finally:
        detectors.close()
        if vehicle_detector is not None:
            vehicle_detector.close()

    print(f"[fail-closed] {report.summary()}")
    for issue in report.issues:
        print(f"  - {issue.get('id')}: {issue.get('type')}")
    if report.status != "PASS":
        print(f"[fail-closed] 未产出最终成品；可复核 artifacts: {report.artifacts}")
        return 1
    print(f"[fail-closed] 已原子发布: {report.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
