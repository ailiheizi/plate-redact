# 教师链

## 为什么需要教师

单靠一个检测器，「这块牌到底能不能读」没有可复核的答案。教师链把「判定」拆成几种
**各自只能产出一种证据**的角色，任何角色都不能单独放行。

## 四类教师

| 教师 | 看什么 | 产出什么 | 状态 |
| --- | --- | --- | --- |
| 开放词表检测教师（本机轻量模型） | 全帧 / ROI，低阈值枚举 | 候选框 | 已实现，有回归 |
| 小 VLM（本机 2B） | 整帧或显式 ROI | 四值语义判定，**不产框** | 已实现；已判定不能当框教师 |
| 分割 / 时序教师（SAM2.1-L，公开权重） | 输入锚点框 → 双向传播 | mask | 已实现（230/230 成功）；SAM3 因权重授权阻塞 |
| 外部视觉教师 | 整帧 | 四值 + 精确框 | 已实现 |
| 真人 | — | **不产出新证据**，只裁分歧与精确框 | 两阶段确认已实现 |

## 证据权限：只增不删

- 教师只能**增加证据**，不能删除别的教师给出的证据；
- **共同阴性不等于安全**：多个教师都说「没牌」，仍然只记 UNKNOWN；
- 结论值四态：`READABLE / UNREADABLE / NO_PLATE / UNKNOWN`，其中 UNKNOWN **永久阻断**；
- 只有「VLM 判可读 + 几何教师支持」才产出提案，且明确标注**不是 Gold**。

## 保守汇总规则

汇总器（`teacher_shadow_ensemble.py`，非交付路径）逐帧要求三教师证据齐备，
出现下面任一条即记 UNKNOWN 并整体 `INCOMPLETE`：

- 缺任一教师证据；
- 引擎报错 / 输出解析失败 / 坐标越界；
- 教师内部自相矛盾；
- 跨教师存在性分歧；
- 空间上没有支持（框对不上）。

缺模型时是 `BLOCKED_MISSING_MODEL`——**不是跳过**。

## 两阶段 attestation

教师结论要生效必须经真人两阶段确认：先对协议（`teacher_review_state = unconfirmed`），
再提交 attestation，并绑定四样摘要——**源哈希、处理指纹、证据摘要、提案摘要**。
任何字段漂移即拒。

具体到实现：

- **第一阶段**：把已绑定的 `READABLE` 结果整理成精确框提案，写出
  `teacher_review_state="unconfirmed"`、`teacher_evidence_sha256` 和覆盖整个未确认提案的
  `teacher_proposal_sha256`；**保留原始教师结果**，不覆盖。
- **第二阶段**：必须用与第一阶段**完全相同的原始 base review、queue / results / source**
  重新构建，并**显式提交已经检查过的那份 proposal 摘要**——
  不能把 proposal 文件自身当 base review 再叠加一次。
- 生成的 `human_attestation` 必须含 `state="confirmed"`、非空 `reviewer_id` / 时间戳，
  以及与提案**逐字匹配**的 `source_sha256`、`processing_fingerprint`、
  `teacher_evidence_sha256`、`teacher_proposal_sha256`。
- **任一框、safe ID / binding、队列 / 结果 / audit 绑定或其它提案字段在人工查看后变化，
  摘要都会失配并拒绝确认。**
- 补充/修正精确框时，要先写回**原始 base review**、重新生成一份**新的未确认提案**，
  **不得直接改冻结提案后沿用旧摘要**。

教师提案即使被确认，也**只是解除了对这条断言的怀疑**：结论仍绑定源哈希与处理指纹，
换视频即失效；Gold / Hard-Neg 仍要按真人逐项裁决另行入库。

## 汇总器与人工门禁

汇总器**只读私有 shadow sidecar**，先校验源哈希、模型哈希、结果哈希、帧覆盖、
权限和教师角色，再生成仍然 `delivery_eligible=false` 的汇总。它不接受任何"外部结论"。
规模与状态（详见 [机器链规模与队列快照](scale.md)）：

- 一次真实证据上的汇总：开放词表教师 `COMPLETE`、小 VLM `COMPLETE`、
  分割教师 `BLOCKED_MISSING_MODEL` → 被考察的三帧**全部 `UNKNOWN`**，
  顶层 `BLOCKED/INCOMPLETE`，`three_teacher_inference_complete=false`；联合回归 77 项通过。
- 高风险扩展的选择是**确定性**的，不是"挑好看的帧"：按固定帧距分层取风险分最高的轨迹，
  再补若干"命中 1–2 次、几何合理"的单帧洞；风险排序考虑车牌形状、独立检测家族数、
  轨迹命中数和字符多样性。相同/重叠时间窗口只扫一次，但**同帧不同空间位置不合并**。
- **只给车辆框的两轮车项禁止直接当车牌 prompt**——必须先由别的教师给出车牌级锚点。

人工复核入口本身也补了信任门禁：任何安全 verdict 必须绑定源/输出哈希、audit fingerprint、
完整轨迹摘要、proposal digest、真人身份和带时区时间戳，否则**全部 decision 原子拒绝**并记
`INCOMPLETE`；`READABLE` 与 `UNKNOWN` **永远留在 unresolved**。
详见 [生命周期与审计实现细节](lifecycle.md#11-人工复核入口的信任门禁)。

## 状态与阻塞（如实写）

- 探针、汇总器、门禁：已实现并有回归；
- **独立的可读性 GT 与几何 GT 缺失** → 从未产生过一次 `PASS_SHADOW_COMPARISON`；
- 教师结果**从未获得真人 attestation** → 从未清除过 UNKNOWN；
- SAM3 被权重授权阻塞（preflight 只证明了隔离与清理正确）；
- 三教师汇总**从未齐备**（缺分割教师权重）；
- **小 VLM 已判定不能当框教师**：已知难例 4/4 报"可读"但与人工框最大 IoU = 0；
  对已遮挡帧复问也拿不到可用的安全负证据。

一句话：**机器侧链路已串通，人工侧尚未清零**。这正是它现在只能叫「影子」的原因。

