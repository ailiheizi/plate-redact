# plate-redact

行车记录仪 / 监控视频里的**车牌自动打码**工具。目标不是"模型在测试集上准确率高"，
而是**输出的每一帧里、任何人眼可读的车牌都已经被抹掉**。

为此它放弃了实时性：可以离线跑、可以两遍渲染、可以任意回溯。算力全部花在
**召回**和**可校验性**上——因为漏打一块牌照是隐私事故，慢一点不是。

> 本仓库**不包含任何图片、视频素材与模型权重**，也不打包第三方代码。
> 检测引擎全部通过可插拔适配器接入，请自行安装并遵守各自的许可证。

---

## 一、设计思路

### 1. 双引擎并集，而不是把单引擎调到最好

任何单一检测器都有系统性盲区：训练分布外的车型、极端视角、运动模糊、低分辨率、
不认识的字符集。与其反复调阈值，不如**让多个引擎各自独立地找，然后取并集**。

```
--detector fast-alpr --detector hyperlpr3 --detector cmd:python3 my_detector.py
```

并集在隐私场景下是明显划算的：多检出的代价是一块多余色块，漏检的代价是车牌外泄。
重叠的候选由 NMS 去重，不会重复涂。`adapters.UnionDetector` 是这条思路的实现。

### 2. 时间冗余 + 运动预测：让打码不依赖单帧检测成败

逐帧独立检测必然有抖动：这一帧检出、下一帧漏检，车牌就会**闪一下**。
项目把"车牌是否存在"从**单帧问题**变成**轨迹问题**：

- `plate_tracker.PlateTracker` 两级关联（高置信 IoU + 低置信距离关联），
  检测失败时用**匀速运动预测**外推框位置，短暂遮挡/模糊期间照常打码；
- 已确认轨迹被缓存，检测器整段失灵时沿用最近确认的框继续打码；
- 新轨迹要**多帧一致**才输出打码（单帧误检不会变成画面上的色块）；
- 框中心驶出画面、框体退化、镜头切换（直方图相关性骤降）时立即清除，避免色块残留。

**提前打码（回溯）**：车牌"刚进入画面/还很模糊"时检测器往往还没认出它，
但那几帧其实已经能看见号码。第二遍渲染用轨迹**最早两帧**的速度把框反推到首次检出之前
（`--pre-frames`，默认 15 帧，单次反推上限 8 帧以限制误差）。
用最早速度而不是末尾速度，是因为车辆加速/减速时末尾速度并不代表刚入画时的运动。

### 3. 纯色填充，而不是马赛克或高斯模糊

马赛克和模糊只是**衰减高频**。在低分辨率、强边缘、或车牌占像素很少时，
模糊后仍可能残留可辨的字符轮廓——对"打码"来说这是失败。

本项目用**区域平均色整块填充**：区域内的像素被替换为一个常数，
信息被整体抹除，不存在可被锐化/超分恢复的字符结构。`redact_roi()` 只有三行，
这是刻意的——打码原语越简单越不可能出错。副作用是会产生一块明显的色块，
这正是隐私场景想要的：**看得出这里被遮挡了**。

### 4. 双遍渲染 + 一致性校验

渲染分两遍，第二遍专门补回溯帧。两遍之间不是"写完就算"，而是**回到成品里逐帧验证**：

- 某一帧计划打码的每一个框，在成品里必须**真的是近似纯色**
  （逐通道标准差 ≤ `--audit-flat-tol`，默认 6.0，留了视频编码噪声余量）；
  不是纯色就说明填充根本没落到那块区域上 → **阻断**；
- 成品帧数必须与源帧数一致（渲染过程不能丢帧/多帧）→ 不一致就**阻断**；
- 计划框相对源帧没有变化 → 说明框位置错了，记为**警告**（源区域本身就是纯色时属正常）。

这条校验的价值在于：它不信任检测、不信任跟踪、也不信任渲染代码，
只检查"**最终文件里那块到底是不是被涂掉了**"。

### 5. fail-closed：守不住就丢弃，绝不输出未打码的帧

审计不通过时，默认行为是**删除草稿并以非零码退出**，而不是"输出但打个警告"：

```
审计未通过(N 项), fail-closed 已丢弃草稿, 未产出成品
```

具体地：

- 渲染先写 `out.mp4.draft.mp4`，审计通过后才 `os.replace` 原子发布为最终文件；
- 检测链路抛错（引擎崩溃、外部子进程退出、返回非法 JSON）→ **立即中止并删除草稿**，
  绝不会"跳过这一帧继续渲染"；
- 运行开始前若发现同名旧输出，先删除它，保证**文件存在 == 本次审计通过**，
  不会让上一次的旧文件冒充本次结果；
- 摄像头（数字索引）源被直接拒绝：实时流没有有限的、可复核的帧序列，
  无法做双遍校验，请先录制为文件。

顺带一个取舍：检测候选的过滤一律是**关闭**的（`--plate-filter` 默认 0），
车辆检测只用于"**多打一块**"（车尾窄条兜底）而**不用于过滤候选**。
因为过滤候选会漏打码，而多打一块只会多一块色块——隐私场景下两个方向的代价不对称。

---

## 二、文件与用法

| 文件 | 作用 |
| --- | --- |
| `adapters.py` | 可插拔检测器接口、注册表、并集、外部进程桥、车辆检测适配器 |
| `plate_redact.py` | 完整引擎：多引擎并集 + 跟踪 + 回溯 + 双遍渲染 + 审计发布（图片/视频） |
| `plate_redact_eu.py` | 轻量参考实现：单引擎 + 简单帧间保持（`BoxTracker`），另含白底车牌的几何约束 |
| `plate_tracker.py` | 多目标跟踪器（纯 numpy，ByteTrack 风格两级关联） |
| `plate_llm.py` | 视觉大模型兜底检测（Qwen2.5-VL，只判"像不像车牌"，不识别文字） |
| `examples/cmd_detector_example.py` | 外部检测器桥的协议示例 |

### 快速开始

```bash
pip install numpy opencv-python
pip install fast-alpr                       # 默认检测器，MIT
python plate_redact.py --source input.mp4 --output output.mp4 --pre-frames 15
```

```bash
# 图片目录
python plate_redact.py --source photos/ --output out/

# 多引擎并集 + 放大兜底（远距小目标）
python plate_redact.py --source in.mp4 --output out.mp4 \
    --detector fast-alpr --detector hyperlpr3 --zoom-fallback 1

# 接入自己的检测器（不改本仓库代码）
python plate_redact.py --source in.mp4 --output out.mp4 \
    --detector cmd:python3 examples/cmd_detector_example.py

# 车辆尾部窄条兜底（需要自行安装 ultralytics，AGPL-3.0）
python plate_redact.py --source in.mp4 --output out.mp4 \
    --vehicle-detector ultralytics:yolov5nu.pt --vehicle-fallback 1
```

### 可插拔检测器

适配器契约只有一个方法（`adapters.PlateDetector`）：

```python
detect(frame_bgr) -> [Detection(rect=(x1, y1, x2, y2), score, text, source), ...]
```

命令行规格为 `name[:参数[:参数]]`，可重复传入取并集：

| 规格 | 依赖 | 许可证 |
| --- | --- | --- |
| `fast-alpr[:检测模型[:OCR模型[:设备]]]` | `pip install fast-alpr` | MIT |
| `hyperlpr3` | `pip install hyperlpr3` | Apache-2.0 |
| `vlm:<权重目录>` | `torch` + `transformers` + Qwen2.5-VL 权重 | Apache-2.0 |
| `cmd:<命令>` | 无（子进程） | 由你自行选择 |

注册自己的适配器（无需 fork）：

```python
from adapters import register_detector, PlateDetector

@register_detector('my-engine')
def _make(arg, kw):
    return MyDetector(arg, conf=kw.get('conf', 0.3))
```

`cmd:` 桥的协议见 `examples/cmd_detector_example.py`：父进程每帧发一行 JSON
（帧号、宽高、base64 PNG），子进程回一行 JSON（检测框、分数、文本）。
**任何许可证的引擎都可以这样接进来**——包括本项目刻意不打包的 GPL/AGPL 实现，
由你自己安装、自己承担许可义务。

---

## 三、参考与依赖的开源方案

以下项目**只作为思路参考或由使用者自行安装的可选依赖**，本仓库不包含其任何代码与权重：

| 项目 | 许可证 | 在本项目中的角色 |
| --- | --- | --- |
| [fast-alpr](https://github.com/ankandrew/fast-alpr) | MIT | 默认检测引擎：YOLOv9 车牌检测 + MobileViT OCR |
| [HyperLPR3](https://github.com/szad670401/HyperLPR) | Apache-2.0 | 倾斜 / 多视角车牌补充检测 |
| [Chinese_license_plate_detection_recognition](https://github.com/we0091234/Chinese_license_plate_detection_recognition) | GPL-3.0 | 中国车牌检测/识别引擎（自行安装，注意 GPL 传染性） |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 | 检测器骨干网络（同上） |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | 车辆检测兜底（可选，注意 AGPL 传染性） |
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) | Apache-2.0 | 视觉大模型兜底检测 |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | MIT | 跟踪器「两级关联」思路的参考 |
| [OpenCV](https://github.com/opencv/opencv) | Apache-2.0 | 视频解码/编码、几何与颜色运算 |

各项目的许可证以其官方仓库为准。**GPL-3.0 / AGPL-3.0 依赖默认不被 import**，
是否引入、如何合规由使用者自行决定。

---

## 四、已知限制

- **无音轨**：输出视频由 OpenCV 重新编码，只有画面没有声音，需要音轨请自行封装；
- **会重编码**：草稿与第二遍渲染各编码一次，画面有轻微损失（换来的是可回溯与可审计）；
- **不做实时**：逐帧检测 + 放大兜底 + 双遍渲染，速度远低于播放帧率；
- **回溯是线性外推**：`--pre-frames` 用匀速假设反推，急加速/急转向时误差会变大，
  因此单次反推上限 8 帧；
- **颜色填充会盖住车牌周围的车身**：框外扩比例 `--pad` 越大越安全，但色块也越明显；
- **无法证明"永不漏检"**：任何有限测试集上的 100% 都不构成证明。本项目的答案是
  fail-closed 的可执行边界：**审计不通过就不产出文件**，把"不确定"变成"没有结果"，
  而不是"看起来没问题的结果"。

## 五、许可证

本项目代码以 MIT 许可证发布，见 [LICENSE](LICENSE)。
仓库内不包含任何数据素材、模型权重与第三方代码。

## 设计思路

实现只回答「怎么做」；分层决策写在 [DESIGN.md](DESIGN.md)：可证伪的交付口径、fail-closed 生命周期、按车辆轨迹的独立审计、学生-教师-真人的证据权限模型、候选并集与预注册门禁、时序传播的停止条件、评测方法论，以及教师链的负结果与边界。
