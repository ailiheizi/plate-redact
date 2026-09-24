# 生命周期与审计实现细节

> 本文写「从输入冻结到原子发布」这条链上每一道门禁的**实现细节与理由**，
> 以及它们在源项目里被安全审查逐条钉住的位置。凡是本仓库尚未实现、
> 只作为设计记录的一层，都单独标了「本仓库未实现」。

[`fail_closed.py`](../fail_closed.py) 已经实现了这套生命周期的核心：
`FrozenInputSet` / `OutputLease` / `FrameChain` / `decoded_frame_chain` /
`render_masked_draft` / `two_pass_issues` / `publish_canonical_draft` /
`audit_decoded_output` / `validate_plan_bundle` / `issue_digest` /
`normalize_safe_issue_ids` / `write_lifecycle_failure_marker` /
`quarantine_final_output`。
下面第 1–7 节对应这些实现；第 8–14 节是内部链路上更严的门禁，属于设计记录。

---

## 1. 输入冻结：先切断「路径名」这个信任入口

同一份视频和权重会被打开不止一次。**先哈希一个调用方给的路径、之后再按同一个路径名打开，不构成绑定**——
中间那段时间里叶子可以被换掉。所以冻结的做法是：

- 用**一个** `O_NOFOLLOW` 描述符把 source、车牌权重、车辆权重拷进私有 staging 目录；
- 记录**实际拷贝的字节**，之后解码与模型加载**只接受冻结路径**；
- staged 叶子以 `O_EXCL|O_NOFOLLOW` 创建、fsync 后收紧为只读；
- 扫描（或探针 preflight）之后、清理之前，复核**目录、设备/inode、尺寸、时间戳、哈希**；
- 冻结快照**刻意是拷贝而不是硬链接**：硬链接能稳定路径名，但仍会与原件共享 inode 的原地写入；
- 缺失或不安全的 checkpoint 会绑定到一个**故意不存在的私有路径**并失败关闭——不能被并发创建「补成成功」；
- 冻结验证失败、未渲染源快照、快照清理失败，都会留下**不可豁免的 UNKNOWN**，阻断交付。
  **清理是门禁，不是尽力而为的 housekeeping。**

## 2. 排他 lease：一个 output stem 只有一个执行者

- `<output>.lock` 用 `O_EXCL|O_NOFOLLOW` 创建（0600）并 fsync；
- **已存在的 lease 永不自动抢占**；
- 释放时校验它仍是本次创建的那个 regular-file inode；**被替换、消失或释放失败都硬阻断**；
- lease 覆盖 review、promotion 和新运行的整个过程，不只是渲染那几秒。

## 3. 生命周期 failure marker：中间态不能留下当成功

| marker | 含义 | 状态 |
| --- | --- | --- |
| `*.frozen_input_cleanup_failure.json` | 冻结副本清理失败 | 本仓库实现 |
| `*.output_lock_release_failure.json` | output lease 释放失败 | 本仓库实现 |
| `*.draft_cleanup_failure.json` | 草稿 / 参考遍临时产物清理失败 | 本仓库实现 |
| `*.promotion_failure.json` | 提升事务提交失败且回滚也失败 | 设计记录（本仓库未实现） |

规则是统一的：**任一 marker 存在，该 output stem 就是硬阻断**——不能重跑、不能提升、
不能被普通重试当作成功证据，必须人工隔离核查后才可显式归档。
marker 只允许授权 review 读取。

## 4. 提升是一个可回滚事务

发布顺序固定在「所有证据准备完毕」之后：

1. 准备并 fsync 新的 mask-plan、报告和原始 sidecar 恢复副本；
2. 移动 canonical `.draft.mp4`；
3. 提交 sidecar；
4. 最后一次 `os.replace(draft, output)` 原子发布正式文件名。

任一提交失败 → 尝试回滚视频与两个 sidecar。**只要回滚也失败**，事务保持 hard-block：
残留的最终命名产物被隔离到私有 quarantine（必要时最后安全阀移除），并写入
`promotion_failure` sentinel / 审计快照，记录**原错误、回滚错误和隔离路径**。
此后要重新核查，必须使用**绑定了完整 provenance 的授权 review 快照**。

> 设计记录与实现上的差别只有一个：本仓库把「草稿清理失败」也算一类 marker，
> 并保留 `quarantine_final_output()` 的取证隔离语义；`promotion_failure` 目前只写在文档里。

## 5. 独立双遍渲染与三条帧链

正式渲染执行两次**完全独立**的 `source decode -> 同一 mask-plan -> encode`：

- 两遍各自重新打开 source 和 writer，写入**不同的随机私有临时文件**，
  并校验 lexical leaf、普通文件类型和 **inode 不同**；
- 每遍为**遮挡前 source 帧**和**遮挡后提交帧**生成带帧号 / 尺寸 / dtype 边界的 SHA-256 **顺序链**；
- 两遍的**提交链必须精确相同**；
- 两个有损编码临时视频随后都**完整解码到 EOF**：解码帧数、每帧尺寸、全序列 SHA-256 链必须精确一致；
- 扫描阶段的 `scan_source_frame_chain_sha256` 必须**同时等于两遍各自读取的源帧链**
  （不是只比较两次编码后的输出链）；
- 音频 remux 仍为 `-c:v copy`；编码后校验再完整解码最终文件，要求其帧链与主编码链精确一致。

缺少、格式错误或不匹配 → 生成不可豁免的 `output:frame_alignment*` / UNKNOWN，
从而阻断单次渲染、封装或后续文件变更引入的丢帧、等帧数重复帧和顺序错位。

**哪些是本仓库已实现、哪些是内部链路更严的一层**：

| 检查 | 状态 |
| --- | --- |
| 两遍独立解码源、按同一 mask-plan 遮挡、各自编码到**不同叶子** | 本仓库实现（`render_masked_draft` 每遍自己 `VideoCapture` + `VideoWriter`） |
| 扫描侧源帧链 `scan_source_frame_chain_sha256` **同时等于**两遍各自读取的源链 | 内部链路的更严一层（本仓库只比较两遍之间） |
| 两遍源链 / 遮挡后提交链逐项一致，且提交帧数等于期望帧数 | 本仓库实现（`two_pass_issues`） |
| 两个有损临时视频各自完整解码，解码帧链与解码帧数一致 | 本仓库实现（`decoded_frame_chain` + `two_pass_issues`） |
| 两遍的临时叶子必须是**普通文件**、inode 不同 | 本仓库只校验普通文件与拒绝覆盖非普通文件；inode 不同属于内部链路的更严一层 |
| method / schema 进 processing fingerprint，实际帧链进报告与 mask-plan；提升时重做校验 | 内部链路的更严一层 |

**证据边界（必须写清楚）**：

- 这个比较**不是**把有损输出像素与源像素做感知哈希 / MAE 阈值比较，
  所以正常编码误差不会被当成错位；
- 它**不是**对「两遍都以完全相同方式发生的系统性代码 / 解码器错误」的数学证明；
- 它证明的是「当前渲染程序可重复得到同一帧序，且 remux / 最终产物保留了该帧序」；
- 它**不能**代替编码后车牌 / 车辆审计、人工复核或真实素材矩阵；
- 若当前编码后端无法对同一输入稳定重现，入口**fail-closed**——
  而不是放宽一个未定标的阈值。

## 6. 输出侧阻断项：只看成品文件，不看检测器说了什么

| 前缀 | 含义 | 可否人工豁免 |
| --- | --- | --- |
| `engine:*` | 引擎未启用 / 初始化失败 / 运行失败 / 返回非法 JSON | 不可 |
| `decode:*` | 解码短读 / 帧数不符 / 元数据异常 | 不可 |
| `render:*` | 计划框在成品里**不是**近似纯色块 | 不可 |
| `encoding:*` | 编码 / 封装异常 | 不可 |
| `output:*` | 输出绑定、FPS 缺失、帧对齐不一致 | 不可 |
| `lease:*` / `cleanup:*` / `audit:*` / `review:*` | 生命周期与审计基础设施问题 | 不可 |
| `coverage:no_vehicle_or_plate_evidence` | 全片零候选（四引擎「初始化成功」也没出框） | 不可 dismiss |
| `coverage:*`（其他） | 畸形候选输出、帧数不符、**一条车辆轨迹都没有** | 不可 dismiss |
| `vehicle_track:*` | 某条车辆轨迹找不到「可解释牌区」 | **可**——只有绑定问题摘要的真人裁决 |

两条容易被绕过的语义在这里被钉死：

- **全片零候选不等于空路**。一段真的没有车的路必须由独立审计 / 真人明确确认，
  **不能以「post-audit 没有候选」当安全证据**——否则「检测器共同漏掉整片」会被当成「通过」。
  本仓库用 `validate_plan_bundle` 覆盖「畸形候选输出 / 帧数不符 / 一条车辆轨迹都没有」这几种
  `coverage:*`；内部链路还额外有一条针对**全片零候选**的专门 issue。
- **车辆框 ≠ 牌框**。一条车辆轨迹在每一帧都要有属于它自己的计划遮挡，
  且车辆框内不能还有未被计划覆盖的亮且带字符纹理的牌区；缺一条就记 UNKNOWN 并保持 `reviewable`。

## 7. 稳定问题摘要：位置性 ID 不是证据

候选问题 ID 里曾包含 `len(issues)` 形成的定位序号（形如 `candidate:{frame}:{len(issues)}`），
**单靠 ID 抵抗不了非确定性后端或异常恢复造成的顺序变化**。现在：

- `issue_digest` **排除易变 ID 与错误文本**，保留帧 / 框 / 来源等证据；
- 报告生成 `review_issue_bindings`，review 里每个 `safe_issue_id` 必须**携带并匹配**摘要；
- 重复 ID、未知 ID、摘要不符、基础设施问题**全部不可豁免**。

因此**旧的「只有 ID」的 review 会继续阻断**，不能直接用来清除新一轮问题。

## 8. strict-audit 参数预注册：先拒绝危险配置，再加载引擎

正式入口会在**加载模型之前**拒绝会降低召回的配置。下面表格的「值 / 允许范围」来自源项目的安全审查记录，
属于**已实现并有回归**的门禁；它们不是标定过的性能指标，只是高召回配置的上限 / 下限。

| 参数 | 默认值 | 允许范围 | 拒绝方式 |
| --- | --- | --- | --- |
| `plate_conf` | 0.03 | `0 … 0.20` | 超上限直接阻断 |
| `vehicle_conf` | 0.20 | `0 … 0.50` | 同上 |
| `post_audit_conf` | 0.08 | `0 … 0.20` | 同上 |
| `fast_alpr_conf` | 0.20 | `0 … 0.20` | 同上 |
| `hyper_conf` | 0.35 | `0 … 0.35` | 同上 |
| `tiles` / `tile_stride` | 2 / 1 | `tiles>=2`、stride 固定为 1 | 每帧执行重叠切片 |
| `vehicle_crops` / `vehicle_crop_stride` | 1 / 1 | 固定 | 每帧执行车辆 ROI |
| `vehicle_crop_only_empty` | 0 | 必须为 0 | `1` 直接失败 |
| `vehicle_crop_max` | 0 | 必须为 0（不限车辆数） | 任意非零值直接失败 |
| 输入尺寸下限 | — | `plate>=640`、`vehicle>=960`、`tile>=960`、`crop>=640` | 低于下限阻断 |
| `strict_audit` / `post_audit` | 1 / 1 | 必须为 1 | 关闭即失败 |

配套细节：

- **任一补充引擎不可被静默关闭**：`--fast-alpr 0` / `--hyperlpr 0` 一律拒绝。
  在其他引擎没有候选时，缺失的补充引擎会被表现成「没有车牌」而不是失败——这正是要防的混淆。
- **阈值被写进 processing fingerprint 也不算数**：fingerprint 只绑定审查，
  不能阻止危险设置产生 PASS；未批准配置在加载引擎前直接阻断。
- **ROI 对每个车辆执行**，即使全帧 / 切片已有候选。任何有限 cap 都可能在拥挤帧静默跳车，
  所以正式入口拒绝 `vehicle_crop_max` 的非零值。
- **负数、NaN、Inf 和布尔伪装值**同样拒绝。

## 9. 车辆投影与归属：不允许「借别人的覆盖」

想让某条轨迹被清除，投影必须具备：

- 同一车辆**至少两帧**的原始视觉锚点；
- 锚点与车辆的**实质重叠**、合理的相对位置 / 尺寸**离散范围**；
- **唯一**轨迹 ID。

单锚点、投影框、两轮车窄条兜底**都不能充当覆盖证据**，仍会让轨迹保持 UNKNOWN。
车辆与车牌归属采用**同帧独占匹配**，避免一个相邻车辆的宽框同时清除两条轨迹。

## 10. 输出 FPS 与元数据一致性

source 或 output 的缺失 / 非有限 FPS 会被写成**不可豁免**的 `output_fps_unknown`
（source 侧同时保留 metadata issue），并比较 OpenCV 与 ffprobe 的**有效值**。
技术性 25 FPS fallback 只用于生成 draft，**不改变阻断状态**。

## 11. 人工复核入口的信任门禁

独立审计的 review 入口统一补成两阶段真人门禁：

- 任何安全 verdict 必须绑定**源 / 输出哈希、audit fingerprint、完整轨迹摘要、
  proposal digest、真人身份和带时区时间戳**；否则**全部 decision 原子拒绝**、状态 `INCOMPLETE`；
- `READABLE` 与 `UNKNOWN` **永远留在 unresolved**；
  只有真人确认的 `MASKED_SAFE` / `FALSE_POSITIVE` / `UNREADABLE` / `NO_PLATE` 能进入 `PASS_REVIEWED`；
- plate audit 与 vehicle audit 的 fingerprint 是**不同域**，分别复算并验证原始 review evidence，
  **不能相互替换或强制相等**；
- 无 review 的普通初审路径行为不变，仍保持 fail-closed。

## 12. shadow 与源像素的生命周期（本仓库未实现，仅记录）

shadow 探针会短暂持有**未打码源像素**，所以它有一套更严的私有生命周期：

- run 目录在写入任何 marker 之前必须先收紧为 owner-only `0700`；
- 每个无损 FFV1/BGR0 clip **先用 `O_EXCL|O_NOFOLLOW` 保留随机 0600 regular-file 的 FD**，
  并在整个 ffmpeg 编码期间保持该 FD 打开；ffmpeg 只接收 `/dev/fd/<fd>`；
- 编码完成后用 FD 的 `fstat` / SHA-256 与 lexical 路径的 inode / 哈希交叉校验，再原子发布；
- 验证或异常清理**只允许删除预期 inode**；路径被替换、clip 残留或 cleanup 失败 → `INCOMPLETE`；
- 逐帧 `source_pixel_sha256` 与解码像素哈希用来证明 **lossless frame mapping**；
  它们仍是 shadow-only 隐私证据。

| 数字 | 测什么 | 样本 | 状态 |
| --- | --- | --- | --- |
| 19 个目录 `0755` → `0700` | shadow 目录权限修复 | 7 个 SAM shadow run | 已修复并复验（内容未改写） |
| 237 个证据叶 `0644` → `0600` | shadow 证据叶权限修复 | 同上 | 同上，`nlink` 均为 1 |
| 总字节数 827011 | 上述证据叶集合大小 | 同上 | 只读核对 |
| 内容 manifest SHA-256 前后一致 | 是否只改了权限 | 同上 | 未改写内容 |
| 无 symlink / 特殊文件 / 打开句柄 | 目录安全形态 | 同上 | 复验通过 |

> 原始报告里出现的 `0755` 是**权限修复前的历史状态**；修复不改变任何模型比较或交付状态。

## 13. 调度与接力：一次一项，不猜隔离理由

- dispatcher 消费 **immutable** 队列 manifest，**每次最多启动一个** detached 子进程；
- 对 manifest、模型和源片做 SHA-256 / lexical `O_NOFOLLOW` 校验，
  并按 manifest 的**最低空闲磁盘**重做 `statvfs` 门禁；复用 0700 scheduler 目录与排他 lease；
- 已有活动 lock → 只返回 `RUNNING`；已有完整 draft/audit/review → 只返回 `NEEDS_ACCEPTANCE`；
- **任何残缺产物、stale lock 或 lifecycle marker 都是永久 `BLOCKED`**，不会清理或重试；
- 推进需要人工 / 独立 verifier 先写入**不可变 receipt**（`PASS`，或 `NEEDS_REVIEW` 且 `errors: []`，
  并明确 `lock_present: false`、`markers: []`），dispatcher 才允许调用；
  receipt **从不**由 dispatcher 写——这正是调用方独立验证的证据；
- 需要隔离残缺项时，必须先提供声明 `immutable: true` 的 isolation ledger；
  dispatcher **只跳过 ledger 明确列出的项，不删除其残留物**，也不替调用者猜隔离理由；
- heartbeat 编排只接受固定 recovery manifest 和显式 SHA-256，每次最多调用一次 dispatch；
  扫到任一活动 canonical lock / PID 时只返回 `MONITORING` **且不写文件**；
  stale lock、marker、未知 leaf、低磁盘或任一 provenance 不一致均为永久 `BLOCKED`；
  最后一项完成后只返回 `QUEUE_COMPLETE`。

| 数字 | 测什么 | 样本 | 状态 |
| --- | --- | --- | --- |
| launcher lock wait 默认 60 秒 | 接力启动等待上限 | 正式 heartbeat 配置 | 已实现；允许值 5–120 秒 |

## 14. 密钥、字节码与历史脚本

- 外部模型的密钥**只从运行时环境变量读取**（历史脚本只接受环境变量形式的 key）；
  缺少变量即拒绝运行。不要把 token 写进脚本、命令行、日志、队列或 JSON 证据。
- 仓库早期实验中**曾出现过已泄露的 token**，使用前应在服务端**撤销并轮换**；任何文档都不记录 token 内容。
- 提交 / 交接前跑一次**不联网**的密钥卫生回归：只读取业务脚本与脚本目录源码，
  查找常见 provider token 字面量、`Bearer` 字面量以及**未从环境变量派生**的 key 别名；
  失败信息**只包含路径、行号和不可逆摘要**，不回显匹配行或环境变量值。
- **Python 编译缓存不属于证据**：交接前删除 `__pycache__/*.pyc`，避免旧字节码继续携带历史源码。
- 文件名里带 `v2` / `v7` / `v14` 或「final / 终极版」的旧脚本**不因此获得交付资格**。
  它们可能抽帧、单引擎运行、绕过车辆覆盖审计或输出旧式「PASS」字样，
  一律不得用于交付；复现历史结果时把产物标记为 `legacy/diagnostic`，并**从源片重跑正式入口**。
- `plate_redact.py` 这类兼容入口的边界要写清楚：文件视频路径在加载旧模型前**统一转发**到严格入口；
  扩展名白名单和输出路径检查在转发前执行；`--fast-alpr 0` / `--hyperlpr 0` 在兼容层直接拒绝；
  `--thorough` 只是兼容参数；图片路径保留旧工具逻辑；**数字摄像头源明确 fail-closed 拒绝**
  （实时流没有有限、可复核的帧序列，必须先录成文件）。

## 15. 一句话

审计能证明的是：**计划框在成品里确实成了纯色块、帧序可重现、绑定一致、中间态没有留下当成功**。
它**不能**证明画面中不存在未被计划的漏检车牌——正因如此，才有第 6 节的轨迹级 UNKNOWN 阻断，
和第 11 节的真人两阶段裁决。
