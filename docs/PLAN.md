# Omni 音视频连续规划方案

日期：2026-10-07。目标是在保留原生 Omni 合理回复能力的前提下，学习 Bernini 式目标特征规划，用文本与 hidden states 共同指导 Listener 音视频生成。

## 任务与输入

输入为可观测 Speaker 音频、Listener 首帧、Speaker 已知时间边界及外部指定的输出时长预算。输出包含倾听、微反应、开口回应的 Listener 音视频。第一版为完整话轮离线生成，不宣称流式因果交互。目标音视频、GT 台词及真实未来时长只作为训练监督，不能成为自主推理输入。

## 模型结构

1. 冻结 Qwen2.5-Omni-3B 完整底座，关闭规划 LoRA，从输入生成 Reply、Delivery、Behavior 三行文本 P。
2. 同一个 Omni 开启规划 LoRA，以 X、P 和目标音视频 MASK 为输入，迭代补全目标特征 Z。
3. 将完整 Z 与 X、P 再送入规划态 Omni，提取倒数第二层的文本及目标位置 hidden H。
4. P 经冻结 Gemma 和官方 LTX text processor 产生原生文本条件；H 经独立音视频 Connector 产生规划条件。
5. LTX 同时使用文本、规划和 Listener 首帧生成音视频。

```text
X → Omni 原生回复（LoRA 关闭）→ P → Gemma → 文本条件 ─────────┐
X + P + MASK → Omni 规划（LoRA 开启）→ 特征 Flow Matching → Z │
X + P + Z → 同一规划态 Omni → H → 音视频 Connector ──────────┤
Listener 首帧 ──────────────────────────────────────────────┤
                                                         LTX → AV
```

规划 LoRA 仅改变规划态行为，不修改底座。冻结回复路径不等于完全不训练 MLLM；本路线明确训练 MLLM 的连续规划能力。第一版仅在 Thinker attention q/k/v/o 上启用 LoRA，排除不会影响倒数第二层输出的最后一层；不改感知塔、embedding、LM head。特征头、MASK、模态 embedding 和连接器单独训练。2026-10-07 用户明确选择 Qwen2.5 Omni，替代此前讨论的 Qwen3；不引入 Qwen3 专用的 DeepStack 或 MoE 逻辑。

## 文本计划

```text
Reply: 没关系，我们可以再试一次。
Delivery: 温和、从容，句末放轻。
Behavior: 倾听时保持注视，回应前轻轻点头。
```

Reply 使用目标原语言并保持实际台词；描述采用统一语言。允许无主回复，允许单独保留微反应事件；不能把未标注动作补成事实。确定性编译为自然语言供 Gemma 编码，Gemma 不再自由生成另一份回复。文本与连续计划允许信息重叠。

## 连续目标与注意力

Z_v = E_v(Y_v)，Z_a = E_a(Y_a)，由冻结 Omni 原生感知模块提取、位于进入 Thinker 的特征空间。视觉起点约 2 FPS、低分辨率且保留空间合并网格；音频按时间池化到约 4 Hz。采样率是工程起点，信息保真与任务收益需验证。两路共享目标时间轴；低分辨率特征不能替代 LTX VAE latent。

输入前缀使用原生多模态位置编码；目标位置使用显式目标时间和空间位置，避免把输入 Speaker 时间槽当成 Listener 未来。前缀保持因果、不能读目标；目标区间可以双向读取前缀和其他目标模态。padding 不参与条件或 loss。

训练对目标施加高比例随机遮盖、时间段遮盖及全 MASK。各模态 decoder 在 Omni hidden 条件下学习 x_sigma=(1-sigma)z+sigma*epsilon，速度目标 epsilon-z，以被遮盖的有效位置计算均值 MSE。先不采用未验证的时间权重。推理由 sigma=1 到 0 积分，外层按余弦剩余遮盖比例分批填入，不覆盖已接受位置。

## 生成接口

Gemma 保持官方文本 token、多层特征与 processor/mask/register 语义。新增并行规划 cross-attention 残差分支，输出投影零初始化；音视频分支各有 Connector，均可访问上下文化的两种目标模态。零向量 token 拼接不等于零影响初始化，因此不使用这种替代。

目标投影到对应 LTX context 宽度，保留 token mask。规划分支初始化时对原 LTX 输出应严格无影响；首步输出投影更新后上游梯度才能进入 Connector，这一延迟是预期行为，必须通过第二步梯度检查。

## 训练阶段

|阶段|输入和监督|训练参数|
|---|---|---|
|I 连续规划|X + 匹配 GT 文本 + 遮盖目标特征；音视频特征 FM|规划 LoRA、MASK、模态 embedding、decoder；时间/空间坐标固定|
|II 渲染适配|匹配 GT 文本 + 首帧；AV latent FM|LTX LoRA|
|III 联合对齐|遮盖位置做规划 FM，文本与未遮盖目标 hidden 提供渲染条件；AV FM|规划 LoRA、decoder、Connector、规划 attention、LTX LoRA|

冻结 Omni 底座、感知塔、Gemma、VAE。第一版不加回复 CE，不穿过原生回复的离散采样反传。损失分别按各自有效位置/元素归一化，然后加权。联合训练不展开完整多轮规划采样；不能将任意自主回复或自由动作采样配上另一份 GT 做重建。

LTX 以官方预训练权重为主初始化；step800 是历史证据与对照，不是兼容限制。新文本下的渲染适配不代表重新否定 Stage1 已验证的正确条件生成能力。

## 验证与推进条件

先检查回复 LoRA 开关隔离、前缀无目标泄漏、特征/时间对应、两种 FM 梯度、全 MASK 推理和 checkpoint 恢复。随后做真实模型、真实数据的小规模 AV 前后向及推理。

质量比较：GT 文本；同一 GT 文本 + GT 计划；同一 GT 文本 + 预测计划；自主文本 + 预测计划。分开评价回复相关性、语音对选定台词的忠实度、发声/动作时间、身份和同步。正确回复存在多解，不强制自主生成复现唯一 GT。工程通过不能作为质量结论。

## 依据

- [Bernini 训练](https://arxiv.org/html/2605.22344v1#S4.SS1)：训练 MLLM 预测目标 ViT 特征，联合阶段分开使用遮盖/可见位置。
- [Bernini 实现](https://arxiv.org/html/2605.22344v1#S6.SS1)：MLLM hidden 与原生 T5 条件共同进入渲染器。
- [v2/v3 汇报](../../audio-driven-v3/OMNI_LTX_V2_V3_REPORT_OUTLINE_20260929.md)：已有生成基础与连续接口失败边界。

本方案扩展到音频、采用规划态 LoRA 和并行规划 attention；这些属于本任务设计，不冒充 Bernini 原始实现。
