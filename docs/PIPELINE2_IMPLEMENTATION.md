# Pipeline 2 实现与验证

日期：2026-10-08。独立代码位于 `src/omni_av_p2`，入口为 `scripts/pipeline2.py`。Pipeline 1 源码未改动。已完成真实模型工程验证，未启动正式实验训练，未评定生成质量。

环境复用本机 `Omni-LTX-stage2/.venv`：PyTorch 2.9.1+cu128、Transformers 4.57.6、PEFT 0.18.1，以及该项目已安装的 `ltx_core` / `ltx_trainer`。本仓库不携带模型文件或 LTX 外部源码。新机器需配置对应模型路径和这两个 LTX 包；普通 Python 环境不能仅凭本仓库直接运行完整模型。

## 1. 已实现结构与 Bernini 对照

```text
观测 Speaker 音频 + Listener 首帧 + 文字计划
 → 冻结 Qwen2.5-Omni-3B 原生 causal forward
 → Thinker 第 8、18、28、35 层 H [B,4,L,2048]
 → 各层 LayerNorm + 共享投影 + 层身份 embedding
 → 四层独立 memory [B,4L,512]
 → 坐标 query 融合 Connector
 → 最终 Uv / Ua
       ├→ 原生 Gemma 条件后拼接 → 冻结 LTX → AV FM
       └→ Bernini 结构 embedding decoder → 感知特征 FM
```

- `decoder.py`：移植 Bernini `TimestepEmbedder`、`ResBlock`、`FinalLayer`、`SimpleMLPAdaLN` 的默认拓扑。保留 256 维正弦时间编码、SiLU、等宽残差 MLP、AdaLN shift/scale/gate、最终 AdaLN，以及调制/输出层零初始化；无 P1 的 tanh gate 或 4 倍扩宽残差 FFN。
- 参考等价测试：加载本机原始 Bernini 同名权重，将零初始化参数改成非零测试值，对比前向与条件梯度，均通过。运行时不依赖 Bernini 源码；只有该对照测试在源文件缺失时跳过。
- `connector.py`：最终两路投影沿用 Bernini `for_gen` 的 `Linear → GELU → RMSNorm → Linear`，最后 Linear 零初始化。没有额外 `for_vit` 投影；decoder 直接使用最终 U。
- 新增的坐标 query cross-attention/self-attention 负责冻结 Omni 条件下的跨层、跨 token 提取。这是 P2 的任务适配，不是 Bernini 原 Connector 的组成。多层只改变 U 的提取过程，decoder 与 DiT 仍直接共享最终 U。
- `ltx.py`：Uv/Ua 分别在官方 Gemma 处理后的原生视频/音频条件后拼接，再走 LTX 已训练的 attention；没有新 DiT 分支、LoRA 或可训练 DiT 参数。
- 目标是冻结 Omni 原生感知 embedding，均为 2048 维；VAE latent 只用于 AV loss。
- 版权和改动说明见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，附 Apache-2.0 许可全文。

## 2. 配置

|项|默认配置|`--engineering-small`|
|---|---|---|
|Omni / LTX / Gemma|完整本地预训练权重|相同完整权重|
|Thinker 层号 / 融合|8、18、28、35 / 分层 memory attention|相同|
|融合宽度、层数、heads|512 / 2 / 8|128 / 1 / 4|
|最终 Uv / Ua 宽度|4096 / 2048|相同|
|decoder 宽度、深度|1024 / 16|128 / 2|
|噪声重复次数|16|2|
|视频采样 / 音频目标|2 FPS、原生时间合并 / 4 Hz|相同|
|优化器|AdamW，1e-4，grad clip=1|相同|
|loss 权重|AV、视觉特征、音频特征均为 1|相同|

默认缩小内部宽度以控制两路 decoder 的成本，保留 Bernini 的 16 层与重复噪声机制。若需要 4096 宽度、16 层和 16 次噪声重复，使用 [pipeline2_bernini_scale.json](../configs/pipeline2_bernini_scale.json)；这一大配置尚未 GPU 实测，不应把默认测试当成它已通过。

FM 采用显式 uniform σ∈[0,1]，xσ=(1−σ)z+σε，目标 ε−z；时间送入网络前乘 1000。没有复刻未完整开放的 Bernini 训练 scheduler/权重。训练配置与目标特征契约均写入缓存和 checkpoint；改变配置需要匹配缓存，不能静默混用。

`hidden_layers` 指完成的 Thinker block 编号：本地模型共 36 层，第 35 层等于此前的 `hidden_states[-2]`。不包括输入 embedding 和最终 norm 输出。保留两种对照配置：[单层 35](../configs/pipeline2_single_layer.json)、[四层可学习加权求和](../configs/pipeline2_weighted_layers.json)。主方案添加层身份并将上下文扩到 4L；加权方案通过 softmax 标量权重压回 L。最终 U 长度和 LTX 条件长度不变，Omni 仅一次 forward，hidden 缓存体积约四倍。

辅助 decoder 同时零初始化输出层和条件调制层，因此前两次前向对 U 的辅助梯度为零是预期行为，第三次前向应打开。AV→U 路径从第一步起即有梯度。零 U token 拼接不保持原 LTX 输出完全不变，文档不作这项承诺。

## 3. 数据与部署约束

- 条件输入 `ConditionInputs` 只包含冻结多层 H、层号、token mask、目标布局坐标和布局 mask；目标 tensor 是独立参数，无法作为 Connector 的输入字段。
- query 布局由外部输出预算和首帧空间网格决定。GT 有效范围只控制特征 loss，不改变 U 的布局 mask。
- 特征缓存重新从冻结 Omni 提取，未使用 P1 规划 LoRA 或其 hidden。
- 当前缓存/权重版本为 `p2_multilayer_condition_v2`。旧单层 3D hidden 与旧 checkpoint 被明确拒绝，必须重新准备。保留原产物；旧版代码位于提交 `4ecd091`。不同融合模式的完整配置也纳入契约，不能静默跨配置加载。
- 对复用的 VAE 缓存，教师媒体先限制到其实际目标时间窗口，再提取特征；视频、音频目标位置仅在完整采样支撑位于该窗口内时计入监督。
- 若继承了旧 VAE 的终端裁帧，训练必须显式传 `--allow-terminal-trim`，并记录原素材/渲染时长。该选项确认知晓裁帧，不代表已审计所有台词尾音；正式训练仍需清洗或重做配对窗口。
- 推理只接受严格 Observation JSON 和可选 TextPlan JSON，不接受训练 cache/manifest。没有文字时调用冻结 Omni 原生回复，再使用同一文本产生 Omni hidden 和 Gemma 条件。
- 输出帧数从外部预算确定，向下对齐到 8n+1，不读取 GT 长度。音频 latent 离散化可能导致短于视频一个时间刻度内的差异，解码只允许至多 40 ms 的末尾零补齐，并记录补齐样本数；更大差异报错。
- `export-checkpoint` 生成只含 Connector 权重的部署文件，没有 decoder 或优化器状态。`infer` / `export-condition` 不实例化辅助 decoder。

## 4. 入口和支持范围

|命令|作用|
|---|---|
|`test`|CPU 回归测试，包含 P1 原测试|
|`prepare`|冻结 Omni/Gemma 提取，生成独立 P2 cache；支持显式 Train manifest|
|`gate`|真实冻结 LTX、共享 U 梯度、恢复/部署一致性、短采样和 MP4 解码|
|`train --recipe av`|E1：只训练 Connector，只有 AV loss|
|`train --recipe joint`|E2：相同 Connector + 音视频辅助 decoder|
|`export-checkpoint`|CPU 导出纯 Connector 权重|
|`export-condition`|用部署权重和观测产生 U，不加载 DiT|
|`infer`|指定文本或自主文本的完整 AV 推理|
|`infer --text-only`|E0：相同文本 + 首帧，不使用 U 或 checkpoint|
|`infer --zero-condition`|保留 U token 数量但置零，作为条件利用诊断|

训练目前单卡、batch size 1、确定性轮流取样；每步 AV 噪声种子固定，辅助噪声使用独立 generator，以便 E1/E2 公平比较。支持周期 checkpoint、跨进程 `--resume`、STOP 文件后保存、严格新输出目录。保存训练 checkpoint 时同时导出 `*_connector.pt`。所有大模型冻结，但 DiT 前向保留对 U 求导的计算图。

当前 `prepare/train` 面向 Train 数据；Dev/Test 的正式批量评价、ASR/同步指标自动汇总、多种子大样本统计及跨样本 U 打乱批处理尚未实现。可以用独立观测/文本逐样本推理，但不能据此称完整实验评估套件已完成。当前 Euler sampler 无 CFG，是工程基线；正式质量比较前需固定更充分的采样设置。

## 5. 可复现命令

从 `/private/liqi_workspace` 运行。先检查空卡，GPU 编号仅为示例；入口会拒绝超过 1 GB 占用或超过 5% 利用率的卡。输出目录必须不存在。以下默认配置和小配置不能交叉混用缓存/权重。

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
P2_CLI=project_code/audio-driven-v4/scripts/pipeline2.py

"$P2_PY" "$P2_CLI" test

# 默认配置的数据准备和小 AV latent 工程 gate
"$P2_PY" "$P2_CLI" prepare --gpu 0 --samples 1 --output /absolute/path/new_p2_cache
"$P2_PY" "$P2_CLI" gate --gpu 0 --tiny --steps 3 --allow-terminal-trim --cache /absolute/path/new_p2_cache/cache_index.json --output /absolute/path/new_p2_gate

# E1/E2 同一数据、初始化 seed、步数；正式执行前先审计数据和目标窗口
"$P2_PY" "$P2_CLI" train --gpu 0 --recipe av --steps 100 --cache /absolute/path/cache_index.json --output /absolute/path/new_e1
"$P2_PY" "$P2_CLI" train --gpu 0 --recipe joint --steps 100 --cache /absolute/path/cache_index.json --output /absolute/path/new_e2

# 导出部署文件；不需要 GPU
"$P2_PY" "$P2_CLI" export-checkpoint --checkpoint /absolute/path/extractor.pt --output /absolute/path/new_deploy

# 自主文字→条件→音视频；若指定文字，追加 --text-plan /absolute/path/plan.json
"$P2_PY" "$P2_CLI" infer --gpu 0 --observation /absolute/path/observation.json --checkpoint /absolute/path/new_deploy/connector.pt --height 256 --width 256 --sampling-steps 30 --output /absolute/path/new_infer

# 相同指定文字的 E0 对照
"$P2_PY" "$P2_CLI" infer --gpu 0 --text-only --text-plan /absolute/path/plan.json --observation /absolute/path/observation.json --height 256 --width 256 --sampling-steps 30 --output /absolute/path/new_e0
```

`--tiny` 只裁 AV latent，用于验证通路，不是正式训练设置。所有 GPU 工程运行的完整参数保留在各 `outputs/p2_*/run.json`。输入 JSON 的字段与 [数据契约](DATA.md) 的 Observation/TextPlan 示例一致，P2 的 cache 是独立格式。

运行层消融时，在 `prepare`、`gate/train`、`infer/export-condition` 的每一步传入同一个 `--config`（例如 `project_code/audio-driven-v4/configs/pipeline2_single_layer.json`），并使用独立缓存/输出目录。未传 `--config` 即使用四层 attention 主方案。

## 6. 旧单层版本的已完成验证（提交 4ecd091）

以下为升级前的历史记录，不能代替多层版验证。旧版 26 项 CPU 测试通过（包含 P1 的 19 项，P2 新增 7 项）。覆盖 Bernini 数值/梯度对照、padding、原生 LTX 中两类 loss 对相同 U 的梯度、冻结参数、目标隔离、纯 Connector 部署、RNG 恢复、外部帧数预算、原生音视频 latent 打包/还原。

真实 GPU 验证使用本机 H200 空闲卡，未停止其他任务：

|运行|结果|
|---|---|
|`p2_prepare_001`|小配置缓存；真实 Omni hidden [1,414,2048]、视觉目标 [1,576,2048]、音频目标 [1,32,2048]；真实 Gemma 条件成功|
|`p2_gate_002`|小配置完整预训练 LTX + 3 步联合更新；两类梯度、冻结参数、部署一致性通过；17 帧 128×128 MP4；峰值分配约 42.13 GB|
|`p2_full_latent_001`|小配置头、未裁原始 AV latent，一步联合训练通过；视频 [1,128,30,41,22]，对应 233 帧 1312×704；音频 [1,8,195,16]|
|`p2_infer_001`|原生自主文本、无 GT cache、无辅助 decoder；输出 233 帧 128×128、48 kHz 音频 MP4，约 7.767 秒|
|`p2_av_only_001`|E1 仅 AV loss 入口通过一次更新|
|`p2_resume_001`|新进程从小配置第 3 步恢复，成功执行第 4 步|
|`p2_text_only_001`|E0 原生文本 + 首帧，使用与自主推理同一份文本，输出相同尺寸和时长的 MP4|
|`p2_prepare_default_001`|默认配置对应缓存提取成功|
|`p2_gate_default_001`|默认 16 层、1024 宽、16 次重复的两路 decoder + 完整冻结 LTX，3 步联合 gate 与 MP4 解码通过；可训练 223,161,344 参数，峰值分配约 43.47 GB|
|`p2_deploy_default_001`|默认配置纯 Connector checkpoint 导出成功|
|`p2_export_condition_001`|新进程只加载默认配置部署权重及冻结 Omni，成功输出条件，无 decoder 模块或 GT cache|

默认配置第三步对最终条件的梯度范数：AV→Uv 约 1.66e-3，AV→Ua 约 1.34e-3；特征→Uv 约 5.47e-5，特征→Ua 约 8.35e-5。只用于验证路径存在，不能作为 loss 权重最优或生成质量提高的证据。

两段已验证的工程/自主 MP4 通过 PyAV 全视频帧及音频包解码。短测试权重、3 步采样和 tiny 画幅都不代表可用生成质量。没有开展正式 E1/E2 效果比较。

失败记录保留：`p2_gate_001` 在第二步检查辅助梯度时失败，原因是检查未考虑 Bernini 双重零初始化的两次更新延迟；修正检查后小/默认配置均通过。CPU 测试另修正了原生 AudioPatchifier 构造需要 patch_size 的调用。未通过删除失败记录掩盖问题。

`outputs/` 中的媒体、缓存、权重不上传 GitHub；仓库保存代码、配置、本说明和测试摘要。P1 的旧验证状态不因 P2 通过而被改写。

## 7. 多层版本验证（2026-10-08）

当前 32 项 CPU 测试通过，包含 P1 的 19 项。新增验证通过原生小尺寸 Qwen forward hook 核对 block 编号；三种融合模式中，每个选定层的归一化参数都获得梯度、每层输入变化都影响 U，而冻结 hidden 不回传梯度；同时检查 memory mask、配置序列化、错误层号和旧版本拒绝加载。既有 Bernini decoder 数值/梯度对照继续通过。

真实模型使用默认融合宽度 512、深度 2、8 heads，两路 decoder 宽度 1024、深度 16、噪声重复 16；未使用 `--engineering-small`。使用空闲 H200 GPU 0/1，保留旧运行目录。

|运行|验证范围与结果|
|---|---|
|`p2_multilayer_prepare_001`|真实 Omni 四层 hidden [1,4,414,2048]，对应 attention memory 长度 1656；视觉目标 [1,576,2048]、音频目标 [1,32,2048]，真实 Gemma 缓存通过|
|`p2_multilayer_gate_001`|完整冻结 LTX、裁小 AV latent，3 步 joint 更新通过；四层融合梯度、两类 loss→U、冻结参数和 Connector-only 部署逐值一致检查通过；生成并解码 17 帧 128×128 AV；峰值分配约 43.47 GB|
|`p2_multilayer_full_latent_001`|默认头配置、未裁小 AV latent，一次 joint 更新通过；同一原始 VAE 样本，保留已记录的 0.2333 秒终端裁帧确认|
|`p2_multilayer_infer_001`|新进程仅加载部署 Connector，自主文本→四层提取→Gemma/LTX→AV；未访问 GT cache，未加载辅助 decoder；3 步 Euler 采样输出 233 帧 128×128、48 kHz 音频，7.7667 秒；离散长度造成的音频补齐为 1760 samples（36.67 ms），在既有 40 ms 限制内|

主方案可训练参数为 223,175,680，较旧单层版增加 14,336；上下文变长造成的计算增长不能只用参数增量衡量。第三步各层 LayerNorm 权重梯度范数（8/18/28/35）约为 1.73e-4 / 1.63e-4 / 2.00e-4 / 2.69e-4。AV→Uv/Ua 约为 1.32e-3 / 1.01e-3，特征 FM→Uv/Ua 约为 5.41e-5 / 8.21e-5。

这些梯度仅证明计算路径存在，不证明各层携带互补语义或多层方案优于单层。单层与加权融合已完成 CPU 通路测试，正式三组效果消融尚未开展。4096 宽度的 Bernini scale 配置仍未 GPU 测试。

多层版两段 MP4 均通过完整视频帧与音频解码检查。可随仓库备份的结果见 [多层测试摘要](PIPELINE2_MULTILAYER_RESULTS_20261008.json)；本机完整参数、日志和媒体保留在上表对应输出目录中。三步训练/采样仅作工程检查，不作为生成质量样例。
