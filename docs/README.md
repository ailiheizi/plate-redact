# 文档索引

- [README](../README.md)：怎么用、产物、限制。
- [DESIGN.md](../DESIGN.md)：判定口径、fail-closed 生命周期、边界。
- 这里放**完整的分层设计**——包括还没实现的部分、以及已经被判定不可行的部分。

| 文档 | 内容 |
| --- | --- |
| [teacher-chain.md](teacher-chain.md) | 教师链：四类教师、证据权限模型、保守汇总规则、两阶段 attestation |
| [multi-route.md](multi-route.md) | 多重路线：候选生成、双审计复核、架构分叉史、模型替换 |
| [self-training.md](self-training.md) | 自训练与数据回流：四条通道、伪标签筛法、端侧部署 |
| [measurements.md](measurements.md) | 实测与负结果：消融、耗时、缺陷复现，以及判定不可行的方案 |

**共同前提**：这里所有数字都是**自测口径**——单源、自标注、非独立盲测。文档里没有独立盲测的召回率，也没有一次全片 `UNKNOWN = 0` 的通过记录。这不是回避，而是这套流程用 fail-closed 说话的原因。
