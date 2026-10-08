# 版本变更记录

本文件区分研究路线、代码版本和工程验证。`audio-driven-v4` 是项目名；Pipeline 1 / Pipeline 2 是两条研究路线；下文 P2 单层版 / 多层版是 Pipeline 2 内部的实现版本，不对应旧项目 `audio-driven-v2/v3`。这些提交不是正式训练效果或模型发布版本。

## 版本导航

|日期|版本 / 提交|主要变化|验证范围|
|---|---|---|---|
|2026-10-07|[Pipeline 1 初始实现 · 9a36c02](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/9a36c02c3771889d95eb199d021420ac1caa81f7)|Qwen2.5-Omni 音视频连续规划；规划 LoRA、特征预测及渲染适配入口|19 项 CPU 测试及真实 Omni 工程检查；完整预训练 LTX 联测未完成|
|2026-10-07|[索引分流 · 0cbf62b](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/0cbf62b204cdc7325576925a04b2db160a11b6eb)|将既有路线标为 Pipeline 1，记录私密仓库与基线，预留 Pipeline 2|文档变更|
|2026-10-08|[Pipeline 2 设计 · 2ead931](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/2ead931defef0ee2e6072547deebb02c97bc5a6c)|确定冻结 Omni/DiT、融合 Connector、最终 U 共享两类监督|设计与实验计划；不是 P2 代码完成点|
|2026-10-08|[P2 单层实现 · 4ecd091](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/4ecd091a1866c83560c8ecac27a6b259c96227d1)|独立 P2 代码，倒数第二层 hidden→Connector→U；辅助 decoder 与冻结 LTX 共同监督 U|26 项 CPU 测试及真实模型训练、部署、自主 AV 推理工程检查|
|2026-10-08|[P2 多层实现 · bfc873c](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/bfc873c59171dc07ecd6933e8c89ecef1b6a13e7)|默认融合 Thinker 8/18/28/35 层，新增单层及加权融合消融，升级缓存/权重版本|32 项 CPU 测试及多层真实模型梯度、训练和自主 AV 推理工程检查|

## Pipeline 1 → Pipeline 2：研究路线变化

|方面|Pipeline 1|Pipeline 2|
|---|---|---|
|Omni|原生回复时关闭规划 LoRA；规划/条件阶段启用可训练规划 LoRA|全部冻结，不使用规划 LoRA|
|目标特征|训练遮盖目标特征，推理迭代预测并回填|只用于辅助特征损失，不进入 Omni / Connector 条件输入|
|推理路径|连续规划采样→回填 Omni→生成条件|观测与文字→冻结 Omni→融合 Connector→最终 U|
|特征 decoder|参与连续规划采样|训练辅助模块；部署移除|
|生成端训练|渲染适配及联合阶段训练 LoRA / 新增分支|冻结 LTX、Gemma、VAE；训练 Connector，joint 配方另训 decoder|
|代码入口|`src/omni_av`、`scripts/v4.py`|`src/omni_av_p2`、`scripts/pipeline2.py`|

两条路线均保留文字条件和 Listener 首帧。Pipeline 2 没有替换或删除 Pipeline 1；P2 的验证结果也不代表 P1 的完整 LTX 联测已完成。

## P2 单层 → 多层：实现变化

|方面|单层版 `4ecd091`|多层版 `bfc873c`|
|---|---|---|
|默认 hidden|`hidden_states[-2]`，即第 35 个 Thinker block 输出|第 8、18、28、35 个 Thinker block 输出|
|缓存 H 形状|`[B,L,2048]`|`[B,S,L,2048]`，默认 S=4；显式保存有序层号|
|输入处理|LayerNorm + 输入投影|各层独立 LayerNorm + 共享输入投影 + 层身份 embedding|
|query attention 上下文|`[B,L,512]`|主方案按层拼接为 `[B,4L,512]`，mask 同序复制|
|可选消融|初始单层实现|单层 35、四层可学习 softmax 加权、四层独立 memory attention|
|缓存 / checkpoint 版本|`p2_shared_condition_v1`|`p2_multilayer_condition_v2`|
|默认可训练参数|223,161,344|223,175,680|
|回归测试总数|26|32，包含原有测试及层选择、融合梯度、兼容性检查|

本地 Thinker 共 36 个 block，HF hidden tuple 的第 0 项为输入 embedding，第 36 项经过最终 norm；本版只选中间 block 输出。这里的层号不指视觉塔、音频塔或 Talker。8/18/28/35 是待消融的工程起点，不是已验证最优组合。

多层版保持：Omni / LTX 冻结、同一最终 U 同时供特征 FM 和 AV loss 使用、Uv/Ua 宽度 4096/2048、最终 U 的 token 数、Gemma 文本与首帧条件。Bernini 式 AdaLN 残差 MLP decoder 和最终 Connector 投影拓扑保持原实现。新增投影发生在 U 形成之前，不是在 U 后拆出两套 loss 专用 adapter。

Omni 仍只需一次 forward。保存的 hidden 张量约为单层四倍，主方案 attention memory 长度也为四倍；整个缓存文件还含教师特征等内容，因此文件总大小并非必然四倍。LTX 条件 token 数不增加。

### 兼容性与复现

- 旧单层缓存和 checkpoint **不能直接用于新版**，代码会拒绝版本或契约不匹配；需在新版重新 `prepare` 并训练匹配的 Connector。
- 新版的 `pipeline2_single_layer.json` 是当前代码中的单层消融，不是旧 checkpoint 的兼容加载模式。要复现旧实现，应使用 `4ecd091` 对应代码和旧版产物。
- 新版单层与加权模式分别使用 [单层配置](configs/pipeline2_single_layer.json)、[加权配置](configs/pipeline2_weighted_layers.json)。`prepare`、`train/gate`、`infer/export-condition` 应传相同配置；当前完整配置纳入契约，各模式分别准备缓存。
- 旧运行目录、缓存和权重保留。GitHub 只备份代码、配置、文档和测试摘要；`outputs/`、模型及数据不上传。
- 本次文档补充只集中整理已有版本事实，不改变代码、权重版本或训练设置，也不改写已推送的 Git 历史。

### 验证证据与边界

|版本|随仓库保存的结果|本机产物|
|---|---|---|
|Pipeline 1|[初始测试摘要](docs/TEST_RESULTS_20261007.json)|`outputs/omni_gate_003` 等|
|P2 单层|[单层测试摘要](docs/PIPELINE2_TEST_RESULTS_20261008.json)|`outputs/p2_gate_default_001`、`p2_infer_001` 等，详见实现文档第 6 节|
|P2 多层|[多层测试摘要](docs/PIPELINE2_MULTILAYER_RESULTS_20261008.json)|`outputs/p2_multilayer_prepare_001`、`p2_multilayer_gate_001`、`p2_multilayer_full_latent_001`、`p2_multilayer_infer_001`|

多层版验证了每个选定层的融合参数梯度、两类 loss 对共享 U 的梯度、冻结 LTX 参数、部署条件逐值一致、默认配置原尺寸 latent 的一次更新，以及无 GT cache / 无辅助 decoder 的自主文本→233 帧 AV 推理。单层与加权消融只完成当前版本的 CPU 通路验证，尚未开展三组正式效果比较。

所有版本均未由这些工程检查证明生成质量达标；尚无多层优于单层、辅助监督优于仅 AV loss 的正式结论。详细命令、历史失败记录和各测试限制见 [实现与验证](docs/PIPELINE2_IMPLEMENTATION.md)，待开展实验见 [结构与实验计划](docs/PIPELINE2_PLAN.md)。
