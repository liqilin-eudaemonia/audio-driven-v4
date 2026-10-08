# Audio driven v4 总索引

**最新备份（2026-10-08）**：[backup/p2-3n24g-balanced-20261008](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p2-3n24g-balanced-20261008)，从 [51e18fe](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/51e18fe22c58cab79b6abd5001ad5caa9af46e5d) 分出，包含 Bernini 审计修复、数据与多机训练能力、三机24卡负载均衡优化及验证记录。原有 `main` 历史保留。代码快照提交：[38ceb34](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/38ceb349657289c2c76e08ae5a5c68b3a7f57b15)（详细中文说明）；后续索引提交仅登记此固定哈希。

**当前实验状态**：[完整实验说明](docs/P2_FULL_DATA_3NODE_20261008.md) · [随仓库保存的三机结果摘要](docs/P2_3NODE_RESULTS_20261008.json) · [全量实验配置](configs/p2_full91009_dev48_experiment.json)。51 项 CPU 回归通过；三台物理机器上的 NCCL、完整预训练 LTX 2.3 原尺寸 joint 训练、Dev48 周期验证、权重恢复和 TensorBoard 均已实际验证。

- 正式清单为 **Train 91,009 / Dev 48**，保留现有 **720p-area 宽高和比例**，不新增正方形裁剪。全量 P2 条件缓存尚未准备完成，91,009 条正式训练尚未启动；本轮模型门禁使用独立 Train24 与固定 Dev48。
- 三机：node12 / js-public（172.29.252.24，master）、node17 / avgen-2（172.29.15.126）、node16 / avgen-1（172.29.112.17），各8张 H200，共24 rank 联合训练。
- 配对短测速中，同样72次样本曝光、global batch72、一次更新，由 **313.66秒降至162.94秒，吞吐提升1.925×**；三节点平均单卡功耗达到约631–651W。每组仅一次更新，不代表长期吞吐、逐位等价或生成质量结论。
- 默认每100步保存/验证，TensorBoard 记录 Train/Dev loss、梯度和耗时；本轮每2步保存/验证的门禁已通过。训练权重保存在 `/public/2_CKPT/0_WM/llq` 的独立实验目录，具体路径见实验说明。

阶段记录：[Bernini 对照与八卡联合验证](docs/BERNINI_AUDIT_20261008.md)、[早期 Train128/Dev24 与多机功能开发](docs/P2_DATA_MULTINODE_20261008.md)。其中41/49项测试、小集选择及“GPU尚未验证”均描述当时状态；最新范围以上述三机报告为准。

|Pipeline|路线|状态|
|---|---|---|
|Pipeline 1|Qwen2.5-Omni 连续音视频规划，文本 + hidden states 指导 LTX|已实现工程原型并备份；验证范围见下文|
|[Pipeline 2](docs/PIPELINE2_PLAN.md)|冻结 Omni/DiT，融合 Thinker 8/18/28/35 层；最终 U 共享特征 FM 与 AV 监督|多层真实训练/自主 AV 推理工程验证通过；未做正式效果实验|

`src/omni_av` / `scripts/v4.py` 属于 **Pipeline 1**；`src/omni_av_p2` / `scripts/pipeline2.py` 属于 **Pipeline 2**。两条路线分别记录状态，保留 Pipeline 1 的基线与验证边界。

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
