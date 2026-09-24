# 数据获取与评测方法论

> 本文写「要构建什么样的 GT、怎么隔离、报什么指标、什么状态算数」。
> 这些是**协议**：工具已实现，但**独立 GT 从未构建完成**，所以从来没有产生过一次
> `PASS_SHADOW_COMPARISON`。所有阈值都只是工程筛选门槛，**不是交付门禁**。

---

## 1. 两套彼此独立的 GT

### 1.1 `readability_gt`：隐私风险标签

对每条人工 GT 轨迹逐解码帧标注 `READABLE` / `UNREADABLE` / `NO_PLATE` / `UNKNOWN`，
按**正常播放速度、原始分辨率、正常观看距离**判断。

- **不得只做均匀抽帧**：首个 / 最后一个可见帧、可读性转变帧、遮挡前后帧全部保留；
- **未标注帧等价于 `UNKNOWN` / 未完成**，不能被跳过；
- `READABLE` 是**唯一**的隐私正例，且**必须逐帧统计**——
  一条轨迹只要有一个可读帧，该帧就是正例，**不能用轨迹平均值掩盖单帧漏打**；
- `UNREADABLE`、`NO_PLATE` 和 `UNKNOWN` **都不是「安全负例」**：
  前两者可单独报告，后者阻断。

### 1.2 `geometry_gt`：物理牌区标签

独立于可读性，标注每个牌区可见帧的 polygon（必要时同时给 clipped `xyxy`）、
`visibility={full, partial, occluded}`、`truncated`、`edge_contact`、车辆类别和 `two_wheel`。

- 一个 `READABLE` 帧**必须同时能定位几何 GT**；若只能读到号却不能定位，
  记 `READABLE + geometry_unknown`，**不得用车辆框代替牌框**；
- 可见几何 GT **不因号码不可读而删除**：`UNREADABLE` 牌区仍可用于几何误差分析，
  但几何 mask 的「有覆盖」**绝不等价于隐私安全**；
- 两套 GT 各自有文件 SHA-256，并以 `(source_sha256, video_id, track_id, frame_index)` 绑定；
  **禁止从一个标签文件推导出另一个**。

## 2. 父视频隔离与划分规则

- manifest 以 `source_sha256` + `parent_video_group_id` 标识素材，**不是以文件名标识**；
- 原始视频的派生片段、同一上传源的 segment 和重编码副本**必须拥有同一个 parent**，
  **不得跨 split**；
- calibration 与 locked holdout **按完整视频分配**，同一 parent 永不跨 split；
- 一条车辆轨迹的 `[first_visible, last_visible]` **及前后 guard 帧**必须整体留在一个 split；
  不得随机抽帧后把同轨迹的邻帧放进另一 split；
- **不用模型轨迹作为 GT**：先由人工在源视频建立稳定的 `track_id`，
  再把各引擎输出**关联**到该 GT 轨迹；引擎丢轨、换轨或重复轨**必须计入失败 / 分歧**；
- 阈值、提示策略或后处理**只能在 calibration 冻结**；locked holdout 只跑一次，
  保留 argv、环境、checkpoint SHA-256、源码 SHA-256；
- 缺模型、短解码、引擎异常、坐标变换不一致**均为 `INCOMPLETE`**。

## 3. 最低样本量（不足即 `INCOMPLETE`，不是通过）

| 门槛 | 数值 | 附加条件 |
| --- | --- | --- |
| 独立 parent video | ≥ 4 个 | 彼此独立 |
| 人工确认的车辆轨迹 | ≥ 8 条 | — |
| 存在 `READABLE` 帧的轨迹 | ≥ 6 条 | — |
| `edge_entry_exit` 轨迹 | ≥ 4 条 | 来自 ≥ 2 个视频 |
| `two_wheel` 轨迹 | ≥ 3 条 | 来自 ≥ 2 个视频 |

普通车辆与关键难例**都要出现在 locked holdout**；有条件时另加雨 / 夜 / 高速 / 竖屏视频。
**只有一个视频的某一类别只能报告该视频结果，不能报告场景泛化。**

> 状态：协议已实现；上述门槛**从未被满足**——冻结 parent/split manifest、独立几何 GT、
> 独立逐帧可读性 GT 和四引擎汇总**均不存在**。

## 4. 四引擎最小运行矩阵

四者读取**同一冻结源**和**同一 GT manifest**；SAM 的 lossless forward/backward 片段只是传输手段，
必须保存 source frame mapping 与像素精确校验。

| 引擎 | 输入约束 | 首要回答的问题 | 主要指标 | 不得混称 |
| --- | --- | --- | --- | --- |
| SAM3 | 每条 track 使用**同一人工** `geometry_gt` anchor bbox，forward + backward | 给定正确锚点后，时序 mask 是否保持在牌区 | mask coverage、mask precision/IoU、首末帧和空洞 | 不测无提示检测召回 |
| SAM2 | 与 SAM3 **完全相同**的 anchor、片段、方向和阈值 | 同上 | 与 SAM3 同口径逐轨迹比较 | 不因 mask 非空就算安全 |
| YOLO12 | 原始整帧、固定低阈值，**禁止按几何先验静默过滤** | 无提示候选召回如何 | box recall@IoU 0.50 / 0.75、首末帧 / 边缘 / 两轮车分层 | 不把 box recall 当 mask 质量 |
| RT-DETR | 与 YOLO12 相同源帧和候选保留规则 | 无提示候选召回如何 | 同上 | 不与 SAM 的 prompted recall 合并 |

配套纪律：

- 同一 track 的**两个方向共享同一锚点**，且锚点在四引擎运行前冻结；
- YOLO12 / RT-DETR **不得获得人工框提示**，否则需另列为 prompted 实验；
- 四者**均保留原始候选 / mask**，即使低置信或几何可疑；
  后处理**只能产生标记，不得删除评测正例**。

## 5. 指标与预注册判定

对每个 `(track_id, frame_index)` 分别计算：

- detector `hit`：最佳候选框与 geometry GT 的 IoU，至少报 `@0.50` 和 `@0.75`；
- mask `coverage = area(mask ∩ GT) / area(GT)`；
- mask `precision = area(mask ∩ GT) / area(mask)`；
- mask IoU、空 mask、最长连续空洞、越界像素、track 级完整区间覆盖率。

预注册的 shadow 比较门槛（源项目建议值）：

| 指标 | 门槛 | 性质 |
| --- | --- | --- |
| `READABLE` 帧 mask coverage | ≥ 0.99 | 工程筛选，**不是交付门禁** |
| `READABLE` 帧 mask precision | ≥ 0.70 | 同上 |
| 候选框 | 按 IoU@0.50 报告 | 同上 |

**任何门槛变化都必须新建 eval id，不能事后调参。**

### 必报分层指标

每个引擎必须同时给出 **frame-level、track-level 和 per-video（含最差视频）** 结果：

- `readable_frame_recall`：所有 `READABLE` 帧的召回，目标是**零漏帧**；
  **不可用轨迹平均掩盖单帧漏检**；
- `first_visible_hit` / `last_visible_hit`：每条轨迹几何 GT 的首 / 末可见帧命中率，
  并报告 mask / candidate 相对 GT 的 **lead / lag 帧数**；
- `edge_frame_recall` 与 `edge_track_recall`：`edge_contact` 或 `truncated` 帧单独计数；
  **越界框、贴边窄条和传播串车必须单列**；
- `two_wheel_exact_recall` 与 `two_wheel_narrow_fallback_rate` **分列**；
  窄条**不得计入 exact success**，除非同帧有可靠视觉锚点并在记录中明确标注；
- `unreadable_geometry_recall`、`no_plate_fp` **只作诊断 / 误报统计**，
  不得反向提升 readable recall 或清除交付阻断。

## 6. 状态机与「阴性不是证据」

| 状态 | 含义 |
| --- | --- |
| `PASS_SHADOW_COMPARISON` | GT 正例完整、所有引擎运行完整，并达到预注册门槛；附带 `status_scope=shadow_harness_only_not_delivery` |
| `FAIL_SHADOW_COMPARISON` | 有完整 GT 和运行结果，但至少一个正例 / 边界 / 两轮车指标失败 |
| `INCOMPLETE` | 缺 checkpoint、缺 GT、解码 / 映射不完整、引擎异常或存在 `UNKNOWN` |
| `NO_EVIDENCE` | 只有阴性，或没有任何可读 / 几何正例——**该状态不是通过** |

**以下事件绝不能作为负证据**，也不能删除学生候选、写 `safe_issue_ids` 或改变交付状态：
无 mask、空 mask、模型未检出、`NO_PLATE`、`UNREADABLE`、教师共同「没看到」、
引擎错误、缺失的引擎输出文件、以及**只有 hard-negative 的视频**。

每个 engine / video / track 的结果应能回溯到
`source_sha256`、GT 文件 SHA、模型 SHA、源码 SHA、argv、frame mapping 和结果 JSONL SHA。
shadow 目录继续写入 `SHADOW_ONLY_DO_NOT_DISTRIBUTE.txt`，
**不得把结果直接喂给正式 review / audit**。

## 7. 校验器以磁盘证据为准

只读完整性校验器的设计口径：

- **磁盘上的二值 mask 是唯一事实来源**：逐张解码并**重算**正像素面积、非零 `bbox` 和全帧 `fraction`，
  再与 JSONL / manifest 声明**逐字段比对**；
- **对象槽位存在但 PNG 全零仍是没有证据**；
- 全流（或全部对象）全零必须标记 `NO_EVIDENCE`，
  **不能当作阴性召回或安全通过**；
- 没有独立、哈希绑定的 `readability_gt` / `geometry_gt` 与完整逐帧正例，
  **永远不得产生 `PASS_SHADOW_COMPARISON`**；
- **verifier 的完整性通过 ≠ 比较通过**：它只证明所检查的文件结构 / 哈希；
- 非保留模式会拒绝残留视频、overlay 或未列入 manifest 的文件；
- 校验器**不是** READABLE 判定器，也不会修改 mask plan 或交付状态。

> 一个具体的身份教训：某次 shadow 队列的目录树整体迁移后，`st_dev` 从声明的
> `16777232` 变为 `16777231`，而 inode、size 和内容 SHA **都没有漂移**。
> 严格的身份合同仍然必须 fail-closed——**「字节相同」不等于旧的 inode / device 绑定仍有效**。
> 那次 fresh verifier 直接给出 `DENIED`、合格结果文件数为 0，
> 覆盖了历史封装给出的 `QUALIFIED_SHADOW_ONLY`；唯一允许的恢复方式是
> 以当前绑定的 source / draft / 旧队列为**只读输入**，在新的随机 `0700` shadow run 中重新提取。

## 8. 素材目录与入库门禁

`catalog` 校验器是**采集 / 标注边界**，它刻意**不**提升素材、不写 Gold、不改 split、
不触碰 formal / shadow 产物。所以：

- **结构有效 ≠ READY**。READY 由 rights、parent 隔离、独立 GT 和场景覆盖推出；
- 它检查 source SHA、普通文件 / symlink / hardlink 边界、parent 与派生 lineage 不跨 split、
  许可用途、教师 / 真人 authority、双 GT 结构、近重复 cluster 和 required scene 的独立 parent 计数；
- 批准的 rights / GT / 真人 attestation / teacher proposal / 非零人工轨迹
  还**必须提供实际 sidecar 路径与 SHA-256**；
- 候选状态固定为 `quarantine / eval_only / rights_pending`，校验器不会自动改 catalog；
- **容器探测不等于完整解码**：常见视频容器会核对尺寸、FPS 和可取得的容器帧数，
  但正式的 EOF 帧链和双遍像素一致性**只能由 fail-closed 视频链或隔离入库探针证明**。

退出码口径（这是最容易误用的一处）：

| 运行方式 | 退出码 0 | 退出码 2 | 退出码 3 |
| --- | --- | --- | --- |
| 默认 | catalog 结构有效且声明的源文件哈希一致（报告**仍可为** `INCOMPLETE`） | 结构 / 哈希 / 权限边界无效 | — |
| 加 `--require-ready` | 达到 READY | 结构 / 哈希 / 权限边界无效 | `INCOMPLETE` |
| 加 `--no-verify-files` | 跳过源文件读取，**只能用于快速结构诊断**，不能作为素材入库 / 训练 / shadow 评测的通过证据 | — | — |

报告固定带有 `scope=inventory_and_shadow_eval_readiness_only` 和
`training_or_delivery_authorized=false`。**即使未来达到 catalog `READY`**，
也只表示目录中已绑定的独立 parent、场景、sidecar 和人工轨迹数量**达到 shadow 比较门槛**；
sidecar 的真人 / 法律真实性以及 source / proposal / processing fingerprint 的语义，
**仍须由对应独立 verifier 和人工责任人确认**。
**此校验器的返回码不得直接触发训练、Gold、正式 review 或交付 promotion。**

## 9. 新素材入库探针

新视频先跑隔离入库探针，而不是直接进目录：

- 把源视频**冻结**到随机 `0700` 私有目录，**完整解码两遍**，
  绑定源 SHA、EOF、媒体字段、逐帧像素链和抽样感知指纹；
- **精确清理冻结副本之后**才写 `0600` 报告；
- 候选始终固定为 `quarantine / eval_only / rights_pending`，**不会自动改 catalog**。

若源不在 catalog root、源或父目录是 symlink、容器 / 全片解码失败、两遍像素链不一致
或冻结副本清理失败 → 候选保持 `BLOCKED`；**清理失败会留下独立 marker，必须人工隔离**。

## 10. 缺口规划与定向采集

- 缺口排序原则：**独立 parent 增益 > 人眼可读风险 > 首末 / 时序风险 > 引擎分歧 > 车型与颜色多样性**；
- **按轨迹去重，不按帧数或重复图数量排序**；每一项保留前后邻帧，保证可判断时序传播；
- 规划器默认只做结构化快速规划，并**明确标记「未重新校验源文件」**；
  只有加 `--verify-files` 才会重新哈希和探测全部 catalog 源。

| 数字 | 测什么 | 样本 | 状态 |
| --- | --- | --- | --- |
| 31 类 | catalog 的场景类别总数 | 一次快照 | 只读规划 |
| 14 类没有任何真实 parent | 场景缺口 | 同上 | 未解决 |
| 7 类只有一个 parent | 场景缺口（不能满足「≥2 独立 parent」） | 同上 | 未解决 |

最高优先级的**零素材**类别：工程车、雾、过曝、严重逆光、侧置牌、雪、特殊用途牌、临牌、
极小可读牌、挂车。规划结论是把天气 / 运动 / 车型三批合并成**少量连续、可授权的定向采集任务**，
而不是为每一类下载重复单图。

### 采集批次与最低交付证据

| 批次 | 目标 | 最低交付证据 |
| --- | --- | --- |
| D1 基线与阴天 | 白天城市、阴天、轿车 / SUV，至少 2 个新 parent | 每 parent 全片 hash / EOF 绑定；每类至少 2 个 parent 候选 |
| D2 天气与光照 | 雨、雾、雪、严重逆光 / 过曝、强眩光、隧道出入口 | 每个难类至少 2 个 parent；保留曝光突变首尾和原始音视频元数据 |
| D3 运动与边界 | 快速 / 高速、贴边进出、运动模糊、密集遮挡、极小可读牌 | 贴边至少 4 条轨迹且来自 2 个视频；每条含首末可见帧 |
| D4 车型与牌位 | 两轮、卡客挂、工程车、侧置牌、黑车暗牌、蓝黄绿白、临牌 / 特殊用途 | 两轮至少 3 条轨迹且来自 2 个视频；车型 / 牌位由真人确认 |
| D5 设备与编码 | 4K、低清 / 强压缩、竖屏、不同帧率和远近景 | 每个分辨率 / 码率族至少 2 个 parent；编码后独立审计 |

### 停止条件（任一出现立即停止该批次并保持隔离）

源 hash / FD / `O_NOFOLLOW` 绑定异常；parent 无法证明或跨 split；
**任何 GT 只有教师没有真人 attestation**；有 `UNKNOWN`、未解释车辆或未遮挡的 `READABLE`；
完整解码、双遍像素链、编码后审计或 sidecar 缺失；目录出现 symlink / 目录 / 特殊文件替代预期普通文件；
授权范围不清；或出现任一 output lifecycle marker / 未释放 lease。

**停止不等于失败裁定**，更不允许删除、抢占或覆盖既有产物。

### 退出门槛

至少 2 个独立 parent；硬基准达到 4 parent / 8 条人工轨迹 / 6 条 `READABLE` 轨迹；
贴边和两轮车分层门槛满足；双 GT 和授权齐全；去重 / parent 泄漏检查通过；
全片多引擎、车辆覆盖、时序回溯、编码后复检和生命周期审计**全部有可复核 sidecar**。

> 达到退出门槛只允许把某场景从「缺口」改为「待验证」，
> **仍不能直接改为交付通过**——最终交付仍需正式入口的全部 fail-closed 门禁。

## 11. 为什么「1,848 张几何图片」不算广度数据集

| 数字 | 测什么 | 样本 | 状态 |
| --- | --- | --- | --- |
| 1,848 张 / 5,185 框 | 单类车牌几何数据集规模 | 训练 / 验证 / 测试三 split | 只有几何框，`readability_gt` 缺失 |
| 1,245 / 3,548 | 训练 split 图像 / 框 | 7 个 parent | 开发集 |
| 483 / 1,418 | 验证 split 图像 / 框 | 3 个 parent | 验证集 |
| 120 / 219 | 测试 split 图像 / 框 | **仅 1 个 parent** | 不能代表泛化 |
| 11 个 parent | 数据来源数 | 整份 manifest | geometry-only |
| 6 组精确重复图像 | 重复度 | 同上 | 不能用重复帧增加有效样本量 |
| 30 个相同帧索引 | 两个子集的重叠 | 其中两个子集 | 二者同属**同一 parent**，不能当两个独立视频 |

结论：这份数据**是** CPU 训练 smoke 数据，**不是**视频隐私安全或跨视频泛化证据；
`independent_test_allowed=false` 已被明确记录。

## 12. 公开数据集的权利结论（截至源项目的一次核查）

核查的事实层面结论：

- **仓库 LICENSE 不自动覆盖**仓库中的现实车辆图像、车牌、旁观者或派生标注的权利；
- 实际情况分布：带 MIT 文件但**没有**拍摄者 / 车主授权和再许可链（单图单牌、多为停车场）；
  GPL-3.0 的扩展 + 合成条件（其「天气」子集并非独立真实雨雾雪视频）；
  3 万 + 真实道路电子监控、多牌、含四点和内容标注但**仓库无 LICENSE**；
  真实来源混合样本但含互联网搜集内容、无 LICENSE；带 MIT 文件的 **GAN 合成牌图**
  （可补牌型 / 字符外观，但**没有**道路视频、车辆 ROI、parent、track 或 readability GT）；
  仅声明「全类别」而**无下载与 LICENSE**；
  一个真实连续道路视频线索，但**未能从官方站取得可核验的当前下载条款 / 车牌 GT / 数据权利文件**，
  第三方转载页不能替代原始授权。

**结论**：那次核查**没有发现**可直接升级为 `audited_pass`、同时带许可、独立 parent、
轨迹和 readability GT 的国内真实视频集。因此公开数据**最多**用于隔离的 warm-start、
负样本和主动学习候选，**不能**替代定向自采 / 授权素材，也不能解除隐私交付门禁。

## 13. 复核方法论的两条负结果

1. **整片 VLM 审计的误报率很高**。对一段 1800 帧成品逐帧审计曾报出 **140 帧「漏打」**；
   裁剪放大 8 倍 + 双模型复查后，除 **1 帧**真实漏打外**全部是误报**：
   模糊不可读的车牌被整帧误报为「可读」、路沿 / 导流线 / 反光被当成车牌、
   以及对模糊牌**读出号码的幻觉**。
   教训：**「模型读出一串号码」不能作为可读性证明**，必须回原始分辨率和正常观看条件；
   整帧审计要配「裁剪放大 + 双模型交叉验证」才有判据。
2. **拿「已经遮挡的帧」去复问同一个模型，得不到可用的安全负证据**。
   同一模型对已按计划遮挡的两帧再问一次，两帧都是 `UNKNOWN`（其中一帧还是解析失败导致的降级），
   没有形成任何「遮挡安全」的结论。

## 14. 未完成、无结论的部分（如实写）

- 独立双 GT（`readability_gt` / `geometry_gt`）与 track manifest：**从未构建**；
  因此 `PASS_SHADOW_COMPARISON` **从未产生**；
- 教师结果**从未获得真人 attestation**，`UNKNOWN` **从未被清除**；
- 三教师齐备下的汇总结果：**从未齐备**；
- 端侧实机的实际帧率与功耗：**未测**，在拿到实机之前不作任何声明；
- 部分片段的教师复核只跑了一部分（例如索引区间未跑完），
  **不能把分片结果外推为整片安全**。
