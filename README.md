# Audio driven v4 总索引

|Pipeline|路线|状态|
|---|---|---|
|Pipeline 1|Qwen2.5-Omni 连续音视频规划，文本 + hidden states 指导 LTX|已实现工程原型并备份；验证范围见下文|
|[Pipeline 2](docs/PIPELINE2_PLAN.md)|冻结 Omni/DiT，融合 Thinker 8/18/28/35 层；最终 U 共享特征 FM 与 AV 监督|多层真实训练/自主 AV 推理工程验证通过；未做正式效果实验|

`src/omni_av` / `scripts/v4.py` 属于 **Pipeline 1**；`src/omni_av_p2` / `scripts/pipeline2.py` 属于 **Pipeline 2**。两条路线分别记录状态，保留 Pipeline 1 的基线与验证边界。

**Pipeline 1 仓库与基线**：私密仓库 [liqilin-eudaemonia/audio-driven-v4](https://github.com/liqilin-eudaemonia/audio-driven-v4)，分支 `main`。初始代码备份为 [9a36c02](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/9a36c02c3771889d95eb199d021420ac1caa81f7)（2026-10-07）；此提交固定记录 Pipeline 1 初始版本，不表示后续最新提交。

Pipeline 1 使用同一个 Qwen2.5-Omni-3B：原生回复阶段关闭规划 LoRA，连续规划和渲染条件阶段开启规划 LoRA。Gemma 文本条件与 Omni hidden states 共同指导 LTX。

|文档|内容|
|---|---|
|[Pipeline 1 路线方案](docs/PLAN.md)|任务、模型、监督、训练与推理的确定设计|
|[Pipeline 1 数据契约](docs/DATA.md)|GT 复用、新特征、文本与时间约定|
|[Pipeline 1 实现和验证](docs/IMPLEMENTATION.md)|代码入口、命令、实测结果与未验证范围|
|[Pipeline 2 结构与实验计划](docs/PIPELINE2_PLAN.md)|冻结边界、最终 U 共享、decoder、梯度与训推契约、实验矩阵及推进条件|
|[Pipeline 2 实现与验证](docs/PIPELINE2_IMPLEMENTATION.md)|Bernini 结构对照、多层/加权/单层配置、运行命令、32 项测试及分版本的真实模型记录|
|[Pipeline 2 多层验证摘要](docs/PIPELINE2_MULTILAYER_RESULTS_20261008.json)|四层梯度、共享 U、冻结 LTX、原尺寸 latent 更新及自主 AV 推理的结果|

Pipeline 1 状态：规划、渲染适配、联合训练及自主规划导出入口已实现。真实 Qwen2.5-Omni-3B GPU 工程测试通过，19 项 CPU 测试通过；完整预训练 LTX GPU 联测待空卡验证。未启动正式训练，未验证生成质量。旧 v2/v3 工程、数据和权重只读复用。

入口：[scripts/v4.py](scripts/v4.py)。实测：[Omni GPU 结果](outputs/omni_gate_003/completed.json)。命令与未完成项见[实现和验证](docs/IMPLEMENTATION.md)。

约束：只使用本机空闲 GPU；不停止其他任务。测试完成不代表生成质量通过。输出采用新的运行目录，保留失败记录。

GitHub 备份包含代码、测试和文档；`outputs/`、模型权重、数据缓存及本机外部依赖不随仓库上传。上面的详细运行产物链接仅在本机可用，仓库内保留 [测试摘要](docs/TEST_RESULTS_20261007.json)。
