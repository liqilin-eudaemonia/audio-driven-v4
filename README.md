# Audio driven v4 总索引

**当前 P2 训练代码备份（2026-10-08）**：[backup/p2-no-time-frozen-dit-20261008](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p2-no-time-frozen-dit-20261008)。包含无时间提示缓存刷新、原子保存修复、三机训练及GT／Omni整份计划验证代码。当前结构为 **Gemma文本条件与Connector输出U拼接 → 原有text cross-attention；LTX全部冻结，仅训练Connector与辅助decoder**。没有启用独立Omni attention或P2 text cross-attention LoRA。运行配置、启动源码哈希对应关系、验证范围和复现见[备份说明](docs/P2_NO_TIME_GITHUB_BACKUP_20261008.md)及[启动溯源](docs/P2_NO_TIME_LAUNCH_PROVENANCE_20261008.json)。 代码快照提交：[05e6489](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/05e64892daa1d8abd0e4e696a496368a37926271)（详细中文提交说明）；后续文档提交仅登记该固定代码版本。


**GT／Omni 整份计划验证代码已准备（2026-10-08）**：固定Dev48，共96个待生成视频；两组均重新走训练同一hidden提取路径，Gemma与hidden使用同一份计划，固定原尺寸/帧数/seed。71项CPU测试及真实清单准备通过；未运行GPU生成，不打断当前训练。入口、命令和验证边界见[配对验证说明](docs/P2_FULL_PLAN_VALIDATION_20261008.md)。

**无时间提示新训练已启动（2026-10-08 08:38 UTC）**：使用新Train91,009缓存，三机24卡，目标1000步，每100步保存，关闭验证；从新初始化开始。TensorBoard选择`no_time_prompt`，实际进度和权重路径见[新训练记录](docs/P2_NO_TIME_TRAIN_1000STEP_20261008.md)。

**无时间提示缓存已完成（2026-10-08 08:25 UTC）**：Omni user message 不再提供结束时间或输出时长，训练提取与推理共用。三机24卡已刷新Train91,009+Dev48的Omni条件，91,057个旧features文件及2个旧测试副本全部删除；监督特征逐项保持相等，Gemma复用，VAE未改。全量身份/布局/GT/删除审计及576条Train+48条Dev独立数值检查零失败，63项CPU回归通过；24卡已释放。新缓存、故障恢复和验证范围见[刷新记录](docs/P2_NO_TIME_CACHE_20261008.md)及[结果摘要](docs/P2_NO_TIME_CACHE_RESULTS_20261008.json)。新协议已按下方记录启动训练；未实现实时规划训练。

**旧时间提示训练已按用户授权提前停止（2026-10-08 07:58 UTC）**：原目标1000步，实际完成29步，完整状态及Connector权重已保存；三机正常退出，24卡已释放。为对齐仅使用已观测信息的推理目标，先修正规划与时间条件设计。停止时保留缓存；后续已完成无时间提示缓存替换并启动新训练，见上文。原因、权重和TensorBoard位置见[本次训练记录](docs/P2_FULL_TRAIN_1000STEP_20261008.md)。

**旧协议全量缓存历史记录（hidden现已删除并替换，2026-10-08）**：按用户指定路径，三机24卡已完成 Train91,009 的 P2 条件缓存，根目录为 `/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/`。Dev48 已校验并复制到同一目录；Train全量身份/GT/时空/文本检查及576条数值抽查通过。提取与合并耗时45分52秒，约2.04TiB；本次仅准备缓存。运行位置、进度口径和完成标志见[缓存运行记录](docs/P2_FULL_CACHE_20261008.md)。

**此前优化备份（2026-10-08）**：[backup/p2-3n24g-balanced-20261008](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p2-3n24g-balanced-20261008)，从 [51e18fe](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/51e18fe22c58cab79b6abd5001ad5caa9af46e5d) 分出，包含 Bernini 审计修复、数据与多机训练能力、三机24卡负载均衡优化及验证记录。原有 `main` 历史保留。代码快照提交：[38ceb34](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/38ceb349657289c2c76e08ae5a5c68b3a7f57b15)（详细中文说明）；后续索引提交仅登记此固定哈希。

**此前三机工程验证（历史记录）**：[完整实验说明](docs/P2_FULL_DATA_3NODE_20261008.md) · [随仓库保存的三机结果摘要](docs/P2_3NODE_RESULTS_20261008.json) · [全量实验配置](configs/p2_full91009_dev48_experiment.json)。51 项 CPU 回归通过；三台物理机器上的 NCCL、完整预训练 LTX 2.3 原尺寸 joint 训练、Dev48 周期验证、权重恢复和 TensorBoard 均已实际验证。

- 正式清单为 **Train 91,009 / Dev 48**，保留现有 **720p-area 宽高和比例**，不新增正方形裁剪。全量 P2 条件缓存现已完成，旧时间提示训练在29步停止；当前无时间提示训练目标1000步，每100步保存且关闭验证；历史模型门禁使用独立 Train24 与固定 Dev48。
- 三机：node12 / js-public（172.29.252.24，master）、node17 / avgen-2（172.29.15.126）、node16 / avgen-1（172.29.112.17），各8张 H200，共24 rank 联合训练。
- 配对短测速中，同样72次样本曝光、global batch72、一次更新，由 **313.66秒降至162.94秒，吞吐提升1.925×**；三节点平均单卡功耗达到约631–651W。每组仅一次更新，不代表长期吞吐、逐位等价或生成质量结论。
- 通用入口支持每100步保存/验证；历史门禁每2步保存/验证已通过。当前无时间提示训练每100步保存，关闭Dev loss与验证集推理，TensorBoard记录Train指标。训练权重保存在 `/public/2_CKPT/0_WM/llq` 的独立实验目录，具体路径见实验说明。

阶段记录：[Bernini 对照与八卡联合验证](docs/BERNINI_AUDIT_20261008.md)、[早期 Train128/Dev24 与多机功能开发](docs/P2_DATA_MULTINODE_20261008.md)。其中41/49项测试、小集选择及“GPU尚未验证”均描述当时状态；最新范围以上述三机报告为准。

|Pipeline|路线|状态|
|---|---|---|
|Pipeline 1|Qwen2.5-Omni 连续音视频规划，文本 + hidden states 指导 LTX|已实现工程原型并备份；验证范围见下文|
|[Pipeline 2](docs/PIPELINE2_PLAN.md)|冻结 Omni/DiT，融合 Thinker 8/18/28/35 层；最终 U 共享特征 FM 与 AV 监督|多层真实训练/自主 AV 推理工程验证通过；未做正式效果实验|

`src/omni_av` / `scripts/v4.py` 属于 **Pipeline 1**；`src/omni_av_p2` / `scripts/pipeline2.py` 属于 **Pipeline 2**。两条路线分别记录状态、缓存与验证边界。

**版本导航**：[版本变更记录](CHANGELOG.md)集中说明 P1 → P2、P2 单层 → 多层的变化、提交链接、缓存/权重兼容性及分版本验证。P2 单层历史基线是 `4ecd091`，多层结构历史基线是 `bfc873c`；当前训练工程与性能优化在上述备份分支，继续保持多层最终 U 共享及冻结 Omni/LTX 的设计。

**Pipeline 1 仓库与基线**：私密仓库 [liqilin-eudaemonia/audio-driven-v4](https://github.com/liqilin-eudaemonia/audio-driven-v4)，分支 `main`。初始代码备份为 [9a36c02](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/9a36c02c3771889d95eb199d021420ac1caa81f7)（2026-10-07）；此提交固定记录 Pipeline 1 初始版本，不表示后续最新提交。

Pipeline 1 使用同一个 Qwen2.5-Omni-3B：原生回复阶段关闭规划 LoRA，连续规划和渲染条件阶段开启规划 LoRA。Gemma 文本条件与 Omni hidden states 共同指导 LTX。

|文档|内容|
|---|---|
|[三机24卡最新实验](docs/P2_FULL_DATA_3NODE_20261008.md)|Train91,009/Dev48、720p-area审计、真实多机训练/恢复、负载均衡配对测速与复现|
|[三机结果摘要](docs/P2_3NODE_RESULTS_20261008.json)|随 Git 保存的实际结果、运行参数、原始证据哈希和未验证范围|
|[Bernini 审计与八卡实验](docs/BERNINI_AUDIT_20261008.md)|参考工作结构对照、梯度/冻结检查、解码修复及早期八卡工程验证|
|[Pipeline 1 路线方案](docs/PLAN.md)|任务、模型、监督、训练与推理的确定设计|
|[Pipeline 1 数据契约](docs/DATA.md)|GT 复用、新特征、文本与时间约定|
|[Pipeline 1 实现和验证](docs/IMPLEMENTATION.md)|代码入口、命令、实测结果与未验证范围|
|[Pipeline 2 结构与实验计划](docs/PIPELINE2_PLAN.md)|冻结边界、最终 U 共享、decoder、梯度与训推契约、实验矩阵及推进条件|
|[Pipeline 2 实现与验证](docs/PIPELINE2_IMPLEMENTATION.md)|Bernini 结构对照、多层/加权/单层配置、运行命令、32 项测试及分版本的真实模型记录|
|[Pipeline 2 多层验证摘要](docs/PIPELINE2_MULTILAYER_RESULTS_20261008.json)|四层梯度、共享 U、冻结 LTX、原尺寸 latent 更新及自主 AV 推理的结果|

Pipeline 1 状态：规划、渲染适配、联合训练及自主规划导出入口已实现。初始版本19项 CPU 测试及真实 Omni GPU 工程测试通过；本轮审计补充了完整预训练 LTX 的 tiny latent gate、梯度与恢复验证，详见 Bernini 审计报告。P1 自主完整 AV 和正式生成质量仍未验证。旧 v2/v3 工程、数据和权重只读复用。

入口：[scripts/v4.py](scripts/v4.py)。实测：[Omni GPU 结果](outputs/omni_gate_003/completed.json)。命令与未完成项见[实现和验证](docs/IMPLEMENTATION.md)。

多机入口：[集群启动器](scripts/pipeline2_cluster.py)、[DDP训练入口](scripts/pipeline2_parallel.py)、[数据清单生成](scripts/pipeline2_dataset.py)、[缓存分片合并](scripts/pipeline2_merge_cache.py)。准备、启动及恢复命令见实验文档。

约束：只使用用户授权的上述三台机器的空闲 GPU；不停止其他任务。测试完成不代表生成质量通过。输出采用新的运行目录，保留失败记录。

GitHub 备份包含代码、配置、测试、文档和机器可读结果摘要；`outputs/`、模型权重、数据缓存及本机外部依赖不随仓库上传。指向 `outputs/` 的详细产物链接仅在本机可用；GitHub 阅读请使用 `docs/` 下的结果摘要，包括[本轮三机摘要](docs/P2_3NODE_RESULTS_20261008.json)和[初始 P1 摘要](docs/TEST_RESULTS_20261007.json)。
