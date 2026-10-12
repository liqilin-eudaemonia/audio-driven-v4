# Audio driven v4 总索引

## 当前P3版本备份（2026-10-12）

GitHub分支：[backup/p3-offline-oracle-pred25-c0-a-20261012](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p3-offline-oracle-pred25-c0-a-20261012)；[版本说明、训练边界与复现入口](docs/PIPELINE3_OFFLINE_C0_A_BACKUP_20261012.md)。本次归档**固定GT计划／时间表＋完整Oracle/Pred25离线hidden混合**版本：C0只训生成Connector，A训Connector＋DiT text-attention LoRA，Omni始终冻结于C0/A，AV loss不回传规划器。**不包含刚讨论的Bernini式在线mask分流联合训练。** 包含C0 step700转A的已部署调度代码；原2000步recipe保留用于身份追溯。已推送固定代码提交：[6ee6719](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/6ee67193c31a9b2ba18874f96736cb5ace406696)，66项CPU回归与8个CLI检查通过。运行进度为下文标注时间的快照。


**Pipeline 3 代码备份（2026-10-10）**：[独立备份分支](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p3-3n24g-step1000-20261010)。覆盖MASK规划、冻结/LoRA对照、三机训练与恢复、全量缓存、合批提速、监控、配置、测试和实验文档；详见[备份范围与复现](docs/PIPELINE3_GITHUB_BACKUP_20261010.md)。

固定代码提交：[a56d4ca](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/a56d4ca4939d49b3b2b6c4c17ab627a029bee32f)，已推送并保存在原私密仓库；后续索引提交仅登记该代码版本。独立备份工作树23项P3测试、7个CLI入口检查通过，正式训练25个源码哈希全部匹配。

**Pipeline 3 全量三机训练已完成 step1000（2026-10-09）**：Train91,009／Dev48，node17/node12/node16三机24卡，Thinker LoRA与规划头训练，global batch96。四组实测选择每卡batch2×累积2，相对基准吞吐提高19.7%；正式1000步平均0.619秒/步，24rank权重一致，最终检查点与Dev48结果已保存。[完整训练记录](docs/PIPELINE3_FULL_TRAIN_20261009.md) · [最终结果](docs/PIPELINE3_FULL_TRAIN_FINAL_20261010.json) · [测速证据](docs/PIPELINE3_OPTIMIZATION_RESULTS_20261009.json)。本轮仅训练规划器，尚未接入LTX验证生成质量。

**Pipeline 3 多机工程验证历史（2026-10-09）**：配置式SSH／torchrun、DDP梯度累积、分布式Dev验证及完整状态恢复已测试。18项CPU回归、真实Omni三机24卡更新、两机16卡LoRA恢复及冻结组更新通过；严格24卡恢复／24→4卡转换未验证或实现。[多机说明](docs/PIPELINE3_MULTINODE_20261009.md) · [工程结果](docs/PIPELINE3_MULTINODE_RESULTS_20261009.json)。后续全量step1000完成及GitHub备份见顶部。

**P2／P2.1生成质量指标已接入（2026-10-09）**：沿用旧版ArcFace ID、FID、FVD、SyncNet-C，保留SyncNet-D及覆盖率；GT／Omni分别评分并提供共同成功样本对照，记录规划合法率和生成成功率。支持已有视频独立重算、CSV及TensorBoard。37项CPU回归、真实CPU ID/FID/FVD自检通过；当前未执行Dev生成或新视频SyncNet评分。[指标口径与命令](docs/P2_P21_GENERATION_METRICS_20261009.md) · [检查证据](docs/P2_P21_GENERATION_METRICS_RESULTS_20261009.json)。代码已备份至实验分支，固定提交：[e5773d5](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/e5773d592026148c46e612e3ea7ffba638cbc2ba)。

**P2 → P2.1 切换任务（2026-10-08）**：P2到第200步保存后停止；P2.1使用原始LTX-2.3与新初始化Extractor/LoRA独立训练，**不加载P2权重**。24卡、global batch72、目标1000步、每100步保存、不推理Dev；新增每15分钟三机GPU/进度检查、当前任务告警和受限checkpoint恢复。[运行与监控记录](docs/P21_FRESH_3NODE_TRAIN_20261008.md) · [检查结果](docs/P21_FRESH_START_RESULTS_20261008.json)。14:34 UTC已在原三机24卡启动；四卡完整模型更新/恢复测试已通过。[GitHub实验分支](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/experiment/p21-text-lora-fresh-20261008)，初始实现提交`b07988d`；GPU检查与四卡续训提交[8c7dd05](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/8c7dd057d844145b7281e3994adb0aab0009e14d)。P2第200步完整保存后，按后续指示迁至172.29.104.194的0–3卡续训，见[四卡续训记录](docs/P2_RESUME_4GPU_20261008.md)。

**Pipeline 3 四卡实验已完成（2026-10-08）**：[加速与实验报告](docs/PIPELINE3_FOUR_GPU_20261008.md) · [数值证据](docs/PIPELINE3_FOUR_GPU_RESULTS_20261008.json)。0–3卡并行完成Train16续训及Train2条件检查，四组均到2000步；实测加速2.21×/2.98×。LoRA已学到训练条件依赖，但Dev泛化尚未建立；下一步扩大规划数据和细分条件消融。

**Pipeline 2.1（2026-10-08）**：新增 **Connector + 原生 text cross-attention LoRA**，仅开放视频`attn2`与音频`audio_attn2`的Q/K/V/输出投影增量；其余DiT参数冻结。独立训练/三机/配对验证入口、adapter与Connector共同保存恢复，复用当前P2缓存。默认rank8，完整四卡训练/保存/恢复门禁通过，三机24卡独立训练已启动；P2冻结版另在四卡续训。[实现与命令](docs/PIPELINE21_IMPLEMENTATION.md) · [测试记录](docs/PIPELINE21_RESULTS_20261008.json)。

**当前 P2 训练代码备份（2026-10-08）**：[backup/p2-no-time-frozen-dit-20261008](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p2-no-time-frozen-dit-20261008)。包含无时间提示缓存刷新、原子保存修复、三机训练及GT／Omni整份计划验证代码。当前结构为 **Gemma文本条件与Connector输出U拼接 → 原有text cross-attention；LTX全部冻结，仅训练Connector与辅助decoder**。没有启用独立Omni attention或P2 text cross-attention LoRA。运行配置、启动源码哈希对应关系、验证范围和复现见[备份说明](docs/P2_NO_TIME_GITHUB_BACKUP_20261008.md)及[启动溯源](docs/P2_NO_TIME_LAUNCH_PROVENANCE_20261008.json)。 代码快照提交：[05e6489](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/05e64892daa1d8abd0e4e696a496368a37926271)（详细中文提交说明）；后续文档提交仅登记该固定代码版本。

**Pipeline 3 首轮历史（200步）**：[路线方案](docs/PIPELINE3_PLAN.md) · [首轮结果](docs/PIPELINE3_FIRST_EXPERIMENT.md) · [首轮数值证据](docs/PIPELINE3_RESULTS_20261008.json)。当时两组条件利用均弱；最新2000步结论以上方四卡报告为准。基础入口 `scripts/pipeline3.py`，加速/续训入口 `scripts/pipeline3_extended.py`。

**GT／Omni 整份计划验证代码已准备（2026-10-08）**：固定Dev48，共96个待生成视频；两组均重新走训练同一hidden提取路径，Gemma与hidden使用同一份计划，固定原尺寸/帧数/seed。71项CPU测试及真实清单准备通过；未运行GPU生成，不打断当前训练。入口、命令和验证边界见[配对验证说明](docs/P2_FULL_PLAN_VALIDATION_20261008.md)。

**无时间提示P2训练历史（2026-10-08 08:38 UTC启动）**：使用新Train91,009缓存，三机24卡，目标1000步，每100步保存，关闭验证；从新初始化开始。原三机运行已到200步保存并正常退出，后迁至四卡续训。TensorBoard原记录选择`no_time_prompt`，四卡续训选择`p2_resume_gpu4`；历史权重路径见[新训练记录](docs/P2_NO_TIME_TRAIN_1000STEP_20261008.md)。

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
|[Pipeline 2.1](docs/PIPELINE21_IMPLEMENTATION.md)|P2条件拼接 + 原生text cross-attention LoRA；训练Connector和辅助decoder|完整四卡更新/恢复通过；三机24卡独立训练中，生成质量未验证|
|[Pipeline 3](docs/PIPELINE3_PLAN.md)|Bernini 式 MASK 目标原生特征规划；冻结与 Thinker LoRA 对照|四卡对照及全量三机LoRA step1000完成；代码备份见顶部，生成质量尚未验证|

`src/omni_av` / `scripts/v4.py` 属于 **Pipeline 1**；`src/omni_av_p2` / `scripts/pipeline2.py` 属于 **Pipeline 2**；`src/omni_av_p3` / `scripts/pipeline3.py` 属于 **Pipeline 3**。三条路线分别记录状态、缓存与验证边界。

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
