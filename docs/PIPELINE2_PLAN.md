# Pipeline 2：冻结 Omni 的 hidden 融合提取与生成对齐

日期：2026-10-08。本文件保留设计与实验计划；独立实现及真实模型工程验证现已完成，具体配置、命令和验证边界见 [实现与验证](PIPELINE2_IMPLEMENTATION.md)。未启动正式实验训练、未评定生成质量。下文“待锁定”事项应结合实现文档中的当前工程配置阅读，不能将工程起点视为调优结论。

## 1. 研究目标与边界

训练一个好的 MLLM hidden states 融合提取器：从完整 Speaker 音频、Listener 首帧以及 Listener 文字回应中，提取 LTX 能使用的音视频生成条件。

固定约束：

- 使用 Qwen2.5-Omni-3B，不使用 Qwen3-Omni。
- 冻结 Omni 全部参数，不启用 Pipeline 1 的规划 LoRA。
- 冻结 LTX DiT、Gemma 及其官方条件处理模块、音视频 VAE、目标感知编码器。
- 只训练融合 Connector 和音视频 embedding decoder；Connector 内部的投影、attention、查询参数均属于这一训练范围。
- 保留 Gemma 文本条件及 Listener 首帧条件。
- 最终 LTX 条件直接接受辅助特征监督，不在其后额外设置“辅助任务专用 adapter”和“生成任务专用 adapter”。

本路线最符合“冻结大模型、训练融合提取器”的约束；不预先声称它比 Pipeline 1 更接近 Bernini，或成功概率已经得到证明。辅助特征监督能否改善实际生成，是需要实验回答的问题。

## 2. 与 Bernini、Pipeline 1 的关系

|项目|Bernini|Pipeline 1|Pipeline 2|
|---|---|---|---|
|MLLM|参与规划预训练和联合训练|规划态 LoRA 可训练，原生回复关闭 LoRA|全冻结|
|目标特征进入 MLLM|训练部分真实特征可见，推理逐轮回填预测|采用相应音视频扩展|不进入；只供损失监督|
|特征 decoder|生成目标 ViT embedding|生成目标视觉与音频 embedding|辅助训练最终生成条件|
|推理主路径|迭代预测特征→回填 MLLM→hidden→渲染|迭代音视频特征规划→回填→渲染|冻结 Omni→Connector→LTX|
|共同被两路监督优化的模块|MLLM 规划器等|规划 LoRA 等|共享融合 Connector|
|DiT|参与训练|渲染适配及联合阶段训练 LoRA/新增分支|冻结|

Bernini 联合训练使用遮盖目标位置的 hidden 计算特征 loss，使用条件位置和未遮盖目标位置的 hidden 作为渲染条件。其 decoder 监督处于 ViT embedding 空间，DiT loss 处于 VAE latent 空间；不能把两个空间混为一谈。

Pipeline 2 借鉴“两类监督共同塑造条件表示”的思想，没有复现 Bernini 的完整迭代语义规划。它不需要将目标 MASK 序列塞入未经相应训练的冻结 Omni。

### Bernini decoder 的本地结构参考

按本地配置与构造代码，特征分支为：

```text
MLLM hidden [3584]
 → Linear → GELU → Linear → RMSNorm → Linear
 → 特征条件 c [3584]

加噪目标 xσ [3584] → Linear 3584→4096
 → 16 个 AdaLN 残差 MLP block
 → 最终 AdaLN + Linear 4096→3584
 → 速度预测

c → Linear 3584→4096 ──────────────────────────┐
σ → 正弦/余弦编码256 → Linear→SiLU→Linear ────┴→ 各 block 条件
```

内部 MLP 默认为 `Linear 4096→4096 → SiLU → Linear 4096→4096`；没有跨 token attention。各 block 的 AdaLN 调制层、最终调制层与输出层零初始化。16 层来自构造默认值，4096 宽度来自本地配置。

这些数值是 Bernini 参考，不是已选定的 Pipeline 2 超参数。Pipeline 1 的 512 宽度、3 层实现也不能称为 Bernini 原结构。

## 3. 输入与监督定义

定义观测输入 X：

- 协议允许的完整 Speaker 话轮音频及已知时间边界。
- Listener 首帧。
- 外部指定的输出时长和画面布局；不能从 GT 未来实际时长或内容推断。

P 为 Listener 文字计划：实际回复台词，加上已有的 delivery、behavior、附和等标注。训练采用与目标音视频匹配的 GT 文本；允许规范格式，不允许自由改写成另一段回应或虚构目标中没有的动作。缺失字段保持缺失，不要求全量重新打标。

第一版是完整话轮离线生成。能够读取整个 Speaker 话轮，不代表能够在真实时间线上提前作出同样反应。

监督分为两类：

1. 语义/感知特征：Z_v*=E_vision(Y_v)，Z_a*=Pool(E_audio(Y_a))。使用冻结 Omni 的原生感知特征，当前模型宽度为 2048。
2. 生成 latent：A_v*=VAE_video(Y_v)，A_a*=VAE_audio(Y_a)。使用冻结 LTX VAE 的输出。

辅助 embedding decoder 预测第一类特征；冻结 DiT 的 AV loss 使用第二类 latent。直接让辅助头预测 VAE latent 会变成另一种方案，本版本不采用。

## 4. 完整计算图

```text
完整 Speaker 音频 + Listener 首帧 + 匹配文字计划 P
                             ↓
                   冻结 Qwen2.5-Omni
                             ↓
               倒数第二层 hidden H [L,2048]
                             ↓
              可训练融合 Connector Cψ
        目标布局 queries → 融合、提取、维度适配
                             ↓
                最终生成条件 U=(Uv,Ua)
                  ┌──────────┴───────────┐
                  ↓                      ↓
          特征 decoder Dv/Da         冻结 LTX DiT
          ↑ 加噪 GT 感知特征          ↑ 加噪 GT VAE latent
                  ↓                      ↓
           音视频特征 FM loss           AV FM loss

文字 P → 冻结 Gemma + 官方处理 → LTX 原生文本条件
Listener 首帧 ─────────────────→ LTX 首帧条件
```

H=stopgrad(F_Omni(X,P))，U=Cψ(H,Q)。Q 由已知目标布局构造，不读取 GT 音视频内容。

### 4.1 冻结 Omni 与 hidden 选择

- 保留原生多模态模板、位置编码和 causal attention。
- 观测条件置于回复文本之前，使回复位置能够读取全部观测条件。
- 起点是倒数第二层 hidden，保留文本、音频、首帧等有效位置和模态边界；先不做全局平均。
- 无需 MLLM 反向，可缓存 H。但缓存必须绑定模型、模板、层号、输入媒体、文字计划、时间协议与预处理版本。
- GT 文本 hidden 缓存只能用于对应训练/诊断；自主文本改变后必须重新提取，不能偷用 GT hidden。

### 4.2 融合 Connector

Connector 包含共享融合主体和形成最终音视频条件所需的输出映射：

```text
H → LayerNorm + 输入投影 → 作为 attention 的上下文
目标时间/空间/模态 queries → cross-attention + 必要的 self-attention/MLP
                         → 最终 Uv、Ua
```

逐 token MLP 只能转换已有上下文表示；小型 attention 模块用于学习跨 token 选择、汇总与目标位置组织。它仍属于 Connector，不引入 MLLM LoRA。

queries 对应明确的目标坐标，数量随输出布局变化；不能把第 i 个文字 token 默认对齐到第 i 个目标特征，也不采用未经验证的随机全局压缩。视觉起点可沿用 2 FPS 原生采样及 temporal patch=2 后约 1 Hz 的空间网格，音频约 4 Hz；这两项是候选配置，实施前必须核对目标特征与 query 一一对应。

学习参数可以包括模态/query 内容 embedding；具体时间与空间坐标由外部布局计算。缺失 GT 区间仅影响监督 mask，不能导致同一外部预算在训练时使用更短的 U 条件序列而推理时使用完整序列。

### 4.3 最终条件直接共享

按当前 LTX 原生条件接口，设计 Uv 为 4096 维、Ua 为 2048 维，实际接入点需在工程 gate 验证：

```text
Uv ──→ LTX 视频条件接口
 └───→ 视觉 embedding decoder 的条件输入

Ua ──→ LTX 音频条件接口
 └───→ 音频 embedding decoder 的条件输入
```

两种 loss 使用相同的最终 U 张量。音视频维度适配属于 Connector 输出的形成过程；最终 U 后不再放两套面向不同 loss 的可训练 adapter。

decoder 自己的条件输入层和 DiT 原有 K/V 等投影仍然存在，这是各网络自身的运算，不能误称为两条新的条件生产路径。共享 U 不保证两个网络关注完全相同的 U 子空间，也不能保证辅助 loss 一定有益。

### 4.4 辅助特征 decoder

视觉和音频使用独立参数的条件 AdaLN 残差 MLP，目标输出均为对应感知 embedding 的速度预测。内部流程参考 Bernini：

```text
xσ [2048] → 输入 Linear → AdaLN 残差 MLP × K → 输出 Linear → 速度 [2048]
U_i → decoder 内部 cond_embed ──┐
σ → 时间编码与 MLP ────────────┴→ 调制残差 block
```

每个位置使用对应的最终条件 U_i。跨位置关系由融合 Connector 提供，decoder 主要负责位置内的条件特征生成。内部宽度、深度、噪声重复次数尚未锁定；优先用可承受的最小配置做工程 gate，再决定是否需要 Bernini 规模。

加噪 GT 感知特征只能进入 decoder。它们不能进入 Connector、U 或 DiT 条件路径。decoder 是训练辅助模块，部署时可以完全移除。

### 4.5 冻结 LTX 的接入

候选接口为在官方 Gemma 特征处理之后，将 Uv/Ua 与相应的原生文本条件沿 token 维拼接，保留有效 mask，再走已预训练的条件 attention。维度、mask、register 与位置处理需要真实模型核查，不能因为维度相同就默认兼容。

不沿用 Pipeline 1 的“新建零初始化规划 attention、再将它冻结”：那会让条件分支始终没有作用。新增可训练 DiT attention 或 LTX LoRA 均超出当前冻结约束，若后续确需使用，必须作为单独变体记录。

零向量 token 拼接也不保证原输出不变，因为会影响 attention 归一化；当前不承诺零初始化等价性。最合适的输出尺度/初始化方式待接口实验确定。

## 5. 损失与梯度

对于模态 m 的真实感知特征 z_i*，采样 ε~N(0,I)、σ~U(0,1)：

```text
xσ = (1-σ) z_i* + σ ε
目标速度 = ε - z_i*
L_feature,m = E[ mean_valid_positions_and_channels(
    || Dφm(xσ,σ,U_m) - (ε-z*) ||²
) ]
```

Pipeline 2 不做目标 token 的 masked completion，特征 loss 计算在可用 GT 位置上。不能照搬 Pipeline 1 的“仅遮盖位置 loss”概念。

L_AV 为 LTX 音视频 VAE latent 空间的 Flow Matching loss；按各模态有效生成元素计算，保持首帧条件及损失 mask 的正确性。

```text
L = λv L_feature,v + λa L_feature,a + λAV L_AV
```

损失先按各自有效元素归一化，显式记录三个权重和各路对 Connector 的梯度范数；不因为都是 MSE 就假定尺度一致。第一轮可用 λAV=1、辅助项等模态权重作起点，再根据梯度量级做小范围预注册比较；具体数值在试跑前配置锁定，不随验证样本逐例调参。

|模块|特征 loss 梯度|AV loss 梯度|更新参数|
|---|---|---|---|
|Omni、目标编码器|停止|停止|否|
|Connector 共享融合主体|有|有|是|
|最终 U 的输出映射|对应模态有|对应分支有|是|
|辅助 decoder|有|无|是|
|LTX DiT|不在此路径|需对输入 U 求导|否|
|Gemma、VAE|停止|停止|否|

冻结 DiT 参数不等于对整个 DiT 使用 no_grad。AV 反向必须穿过其运算到 U；Omni/教师特征/Gemma/VAE 提取可以 no_grad。冻结 DiT 仍有反向激活成本，不能假定显存与纯推理相同。

两路独立头直接接固定 H、却没有共享可训练模块，会使特征 loss 无法帮助渲染 Connector；本结构通过最终共享 U 避免这一断开。

## 6. 训练与推理一致性契约

训练生成路径：X + P_GT → 冻结 Omni → Connector → U → 冻结 DiT。

指定文本推理：X + P_given → 冻结 Omni → 同一 Connector → U → 同一冻结 DiT。

自主推理：X → 冻结 Omni 生成 P_pred；随后用 X + P_pred 重新提取 H，经同一路径生成。Gemma 与 Omni 必须使用同一份文字计划。

必须遵守：

1. U 的生产过程不读目标音视频、目标感知特征、加噪目标、GT 未来时长。
2. 推理不把 U 替换为辅助 decoder 的采样结果，不增加回填 Omni 的流程。
3. 同一文字、观测和外部布局下，训练条件生产与推理条件生产在 eval 模式产生同一 U；训练态 dropout 等随机性需单独控制。
4. GT padding 只进入监督 mask，不隐式提供推理时缺失的目标长度条件。
5. GT 文本与自主文本仍有分布差异；共享 U 不会消除这一问题，需分开评价。
6. 自主生成的新台词不能与原始另一段台词的 GT 音频配对做重建训练。自由生成评估使用对自身文本的忠实度，不要求唯一 GT 回复。

两个固定分支投影本身不构成训推不一致；取消它们主要减少辅助约束绕过最终条件的空间，不应宣称完全消除所有分布差异。

## 7. 数据准备与复用

- 复用原始 qualified split、Speaker/Listener 媒体和结构化 GT，保持数据划分与身份隔离规则。
- 审计 GT 台词、附和、行为描述与音轨/视频是否一致；仅对错误样本定向修正。
- 新提取 P2 冻结 hidden 缓存；P1 缓存主要是输入 embedding/目标特征，不能当成 P2 最终 hidden 使用。
- 可复用感知特征和 VAE 缓存的前提是样本、模型、时间裁切、采样和预处理一致；不直接宣称 P1 缓存兼容。
- 已知旧 VAE 存在为满足 8n+1 帧数而裁掉尾帧的情况。正式比较前统一目标窗口，确认未截断台词/动作；需要时重建匹配特征及 latent。仅记录容差不能代替对齐完成。
- 输出时长桶由外部实验协议确定；过长样本重新分桶，不能为凑窗口偷偷截断回复。
- 缓存记录 sample ID、split、原始 GT 哈希、文本、模型版本、hidden 层号、采样率、时间窗口、网格、mask 和指纹。新缓存使用独立 P2 命名，不能覆盖 P1 产物。

## 8. 实验计划

### P2-G0：工程与梯度 gate

先做小尺寸真实模型类测试，再在空闲 GPU 用预训练 Omni/Gemma/LTX 和少量真实素材验证。顺序：

1. 仅观测和文字产生 H、queries、U，检查 shape、dtype、mask、坐标和无目标输入。
2. 接入原生 LTX 条件接口，完成 AV 前向、反向；验证两个模态的 U 和 Connector 均有有限有效梯度。
3. 加入两路特征 decoder，单独反向每种 loss，检查它们更新同一 Connector；禁止只靠总梯度非零掩盖分支断开。
4. 验证所有冻结参数无梯度、无更新；验证 U 不是 detach 后送入 DiT。
5. 验证辅助 decoder 输入变化不改变 U；移除整个辅助 decoder 后相同输入/随机种子的生成路径保持一致。
6. 验证保存/恢复、优化器与 RNG、跨进程恢复，以及训练/推理条件一致性。
7. 真正执行短采样和 VAE 解码，导出可播放 AV，检查首帧、数值、音视频时长。裁小 latent 的测试只作工程诊断，不能代替原尺寸测试。

通过条件：数据契约正确、预期梯度存在、冻结条件成立、无非有限数值、产物可读、恢复一致。通过不代表生成质量通过。

### P2-E0：匹配文本的冻结渲染基线

条件仅为相同 Listener 文本经 Gemma 编码及 Listener 首帧，不含 U。记录当前冻结 LTX、当前文本格式和目标窗口下的执行能力。

这不是否定历史 Stage1 step800 的 GT 条件生成证据，而是为新文本/新冻结设置建立可比对照。step800 可作历史参考，不能在组件和条件不同的情况下视为严格同配置基线。

### P2-E1：Connector + 仅 AV loss

冻结大模型，只训练 Connector；不训练辅助 decoder。验证最直接的 hidden→生成条件适配是否成立。

先用少量干净训练样本检查可拟合性，再在固定 Train/Dev 子集检查泛化。不能只用训练样本漂亮结果作结论。

### P2-E2：同一 Connector + AV loss + 特征 FM loss（主方案）

与 E1 使用相同 Connector 架构、初始化、数据、文本、优化步数、数据顺序及评估采样预算，增加辅助特征监督。记录额外计算时间和显存，不将相同步数误称为相同 FLOPs。

主问题：E2 相比 E1 是否在真实生成质量、表达/时间执行或训练稳定性上有可复现增益，而非只有辅助 loss 下降。

首轮不对 E2 单独增加预训练优势。若后续尝试 feature-only warmup，应列作新变体并匹配训练预算。

### P2-D：条件利用与辅助监督诊断

|诊断|控制变量|目的|
|---|---|---|
|移除/置零 U|保持文本、首帧、噪声种子|是否实际依赖附加条件；注意移除 token 与置零 token 不等价|
|跨样本打乱 U|保持原文本与首帧；尽量匹配布局长度|检查生成是否受 U 影响，不作为正常质量样本|
|仅改变 Speaker 音频|文字计划及首帧固定，重新算 U|提取器是否利用 Speaker 信息，重点观察时序/表达|
|decoder 使用匹配/打乱条件|固定加噪特征与噪声时间，按 σ 分桶|检查辅助头是否忽略 U，尤其避免低噪声去噪掩盖条件无效|
|移除辅助 decoder 做推理|固定完整生成输入与种子|证明部署路径无需该辅助分支|

打乱条件只用于诊断，不将错配文字/音视频用于重建训练。任何单项敏感性结果都不单独证明语义对齐正确。

### P2-E3：自主文本泛化

在匹配文本下接口成立后，使用冻结 Omni 自主生成的 P。E1/E2 使用同一份自主文本、观测与采样种子比较，不能让不同台词混淆条件收益。

回复内容评估其与 Speaker 的相关性；合成语音评估对选定 P 的忠实度；不要求自由回复逐字复现原始唯一 GT。此实验检查 GT 文本→自主文本的分布差异。

## 9. 指标与决策规则

分别记录：

- 内容执行：ASR 对选定回复的 WER/CER，附和是否遗漏、额外台词/重复、文字与发声一致性。
- 时间与表达：发声起止、附和/回应顺序、语速韵律、表情与动作；只有存在可靠对应标签时才计算精确时间误差。
- AV 质量：口型同步、身份稳定、画面连续性、声音质量及可见伪影；自动指标配合盲评样例。
- 条件利用：上述干预诊断，以及各路对 Connector 的梯度量级。
- 代价：训练显存、每步时间、推理延迟、条件 token 数与输出时长关系。

评估使用固定样本与种子；先看工程与小样本结果，再扩大验证集并增加独立重复。若做统计比较，按样本汇总，报告区间与失败例。样本数量、正式步数和数值接受阈值需在试跑结果后、正式评估前写入实验配置。

决策规则：

1. G0 不通过：先修接口/数据/梯度，不开始全量训练。
2. E1 不成立：优先定位冻结 LTX 的条件适配与文本/时间问题，不假定辅助 loss 自动解决；需要解冻 DiT 时另立变体。
3. E1 成立、E2 无收益或退化：保留 E1；不能为维持“仿 Bernini”名义而宣称辅助 decoder 有效。
4. E2 在内容执行不明显退化的前提下，取得可复现的表达/质量收益：再扩大训练和自主文本验证。
5. 特征 loss 下降但生成无改善：仅说明辅助任务拟合，不能证明 U 是更好的生成条件。

## 10. 尚待实施前锁定的参数

- Connector 内部宽度、层数、heads，以及 query 的精确时间/空间布局。
- LTX 既有条件接口的最终接入点、mask/register 处理、U 初始化尺度。
- 两个特征 decoder 的宽度、深度、重复噪声次数和是否采用完整 Bernini 零初始化方式。
- 感知特征的尺度处理；如采用归一化，统计量只能由 Train 估计并固定记录。
- loss 权重、学习率、batch/累积、checkpoint 周期、训练步数与验证频率。
- 冻结 LTX 的具体初始化权重；主线倾向官方预训练权重，step800 不能在未核对依赖时直接套用。
- 训练/验证子集、统一目标时间窗口及长片段分桶。

这些是待验证的工程选择，不改变“冻结大模型、最终 U 共享、两路监督共同更新 Connector”的主结构。

## 11. 文档与代码隔离

Pipeline 1 初始代码基线为提交 `9a36c02c3771889d95eb199d021420ac1caa81f7`，原实现仍在 `src/omni_av`，原入口仍为 `scripts/v4.py`。

Pipeline 2 独立模块为 `src/omni_av_p2`，入口为 `scripts/pipeline2.py`，输出为 `outputs/p2_*`。新增 7 项测试，连同 P1 原 19 项共 26 项通过；真实模型验证单独登记在 [实现文档](PIPELINE2_IMPLEMENTATION.md)，不把 P1 旧结果计作 P2 的验证证据。

## 12. 依据与相关文档

- [总索引](../README.md)
- [Pipeline 1 方案](PLAN.md)、[数据契约](DATA.md)、[实现与验证](IMPLEMENTATION.md)
- [Bernini 训练章节](https://arxiv.org/html/2605.22344v1#S4.SS1)、[实现细节](https://arxiv.org/html/2605.22344v1#S6.SS1)
- 本机 Bernini：`project_code/Bernini/bernini/models/bernini.py` 的 `MLPConnector`、`feat_from_planner_to_renderer`；`models/diffloss_fm.py` 的 `SimpleMLPAdaLN`；`configs/bernini/config.json`。这些文件不随本仓库备份上传。
