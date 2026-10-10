# Pipeline 3：Bernini 式目标特征规划

2026-10-08。此文记录已讨论的路线；首轮仅执行固定条件下的规划对照，后续渲染和自主时间规划仍待结果决定。实现入口：`scripts/pipeline3.py`；独立模块：`src/omni_av_p3`。不替换 Pipeline 1/2，不使用 P2 缓存或检查点。

进展：[2000步四卡加速与条件诊断](PIPELINE3_FOUR_GPU_20261008.md)已完成。LoRA能够学习训练条件关联，Dev泛化仍不足；保留LoRA主候选和冻结对照，下一步扩大数据并细分条件消融。下文的200步配方描述最初pilot，后续配置和结果以四卡报告为准。

2026-10-09工程补充：[多机训练说明](PIPELINE3_MULTINODE_20261009.md)。新增 `pipeline3_cluster.py`／`pipeline3_parallel.py`，默认 node17/node12/node16 三机24卡；真实预训练Omni联合更新已验证。多机版使用新的确定性全局采样顺序及有效batch，恢复契约与原单卡入口分开，详见该文。

## 1. 要解决的问题

输入完整 Speaker 音频与 Listener 首帧，生成 Listener 的回应音视频。保留“内部回复文本 + 连续规划 hidden states”共同指导 LTX；文本可重新设计，不以兼容旧 step800 权重为约束。

P3 学习目标是预测**目标音视频在冻结 Omni 感知空间中的特征**，并在推理时逐轮回填目标 token。不是 P2 的“只从已知条件提取一个 U，再做辅助重建”。用户已接受比较 Thinker LoRA；冻结 Omni 是必要对照，LoRA 是待验证的主要候选。

## 2. 与 Bernini 的对应关系

参考：[Bernini v2](https://arxiv.org/html/2605.22344v2)，尤其 §2.1.1、§4.1、§4.2、§6.1；本机源码 `/private/liqi_workspace/project_code/Bernini`。

|项目|Bernini|Pipeline 3|
|---|---|---|
|MLLM|Qwen2.5-VL-7B-Instruct|Qwen2.5-Omni-3B Thinker|
|目标表示|目标视频的原生视觉特征|目标视频、音频的原生感知特征，均为 2048 维|
|输入目标|部分真实特征 + MASK|同样规则，扩展为音视频目标 token|
|上下文|目标视觉段内双向注意力|目标音视频联合双向，前缀因果且不能读取后面的目标|
|特征生成|倒数第二层 hidden → 特征 Connector → AdaLN MLP FM decoder|同样拓扑，音视频各一套特征头|
|推理|全 MASK 起步，随机顺序、余弦比例逐轮填入|相同外层逻辑，默认 25 轮、每轮内层 5 步 Euler|
|渲染条件|最终再过一次 MLLM，生成 Connector 输出与 T5 token 拼接|拟采用最终 Omni hidden 的生成 Connector + LTX 原生 Gemma 文本条件|
|训练更新|规划与联合训练更新 MLLM；连接器阶段另有冻结策略|首轮冻结与 Thinker LoRA 对照；不直接开始全量微调|

注意论文版本：v2 §4.1 将 **Stage III 定义为 Connector training**，冻结 MLLM/DiT；**Stage IV 才是 joint training**。旧讨论把三阶段论文口径中的联合阶段也称 Stage III，后续文档按 v2 的名称区分。连接器阶段用生成端 DiT 损失；联合阶段为 `0.2 L_ntp + L_visual + L_dit`。本任务扩展音频规划及音频生成损失，不声称系数可原样迁移。

## 3. 第一阶段的精确计算图

```text
完整 Speaker 音频 + Listener 首帧 + GT 回复/表达描述/时间表
                       ↓ 冻结输入 embedding 与感知塔
                    条件前缀 X

GT Listener 视频 ─→ 冻结视觉塔 ─→ Z*v（空间合并后）
GT Listener 音频 ─→ 冻结音频塔 ─→ Z*a（原生约 25 Hz，不做 4 Hz 池化）
                       ↓ 随机遮盖
                [X, visible Z*, MASK]
                       ↓ Thinker
             目标位置的倒数第二层 hidden
               ┌───────┴────────┐
        视频特征 Connector    音频特征 Connector
               ↓                ↓
        视频 AdaLN decoder   音频 AdaLN decoder
               ↓                ↓
            L_plan,v          L_plan,a
```

目标不是 VAE latent。规划结果需要重新填入 Omni 输入，因此选择其原生感知特征空间。后续 LTX VAE latent 才承担音视频渲染损失。原生 audio feature 是否足够保留精确语音内容、语气和韵律，要由后续解码及生成实验验证；不能仅由特征维度推断。

两组都训练两套特征 Connector、两套 decoder、两种模态各自共享的 MASK 向量和模态 embedding。LoRA 组额外训练 Thinker 第 1–35 层注意力 q/k/v/o 投影的 rank16、alpha32 LoRA；第36层在倒数第二层读出之后，不参与规划优化。所有原始权重、输入词 embedding、视觉塔、音频塔保持冻结。冻结组仍允许梯度穿过 Thinker 到 MASK，不能将整个 Thinker 包在 `no_grad()` 里。

### 结构

特征 Connector 对齐 Bernini `MLPConnector.for_vit()`：

`Linear(D,D) → GELU → Linear(D,D) → RMSNorm(D) → Linear(D,D)`，D=2048。

Decoder 复用仓库中已做 Bernini 数值对齐测试的 `SimpleMLPAdaLN`：输入投影 + 正弦时间 embedding 的两层 SiLU MLP + 条件线性投影；16 个 AdaLN 残差 MLP block；末端 AdaLN、归一化、输出线性层。每个 block 的条件产生 shift/scale/gate。AdaLN 和最终输出层采用 Bernini 的零初始化。

首轮配置宽度 **512**、深度 **16**、噪声重复 **4**，保留拓扑但缩小容量和计算量；这不是 Bernini 发布模型的原始宽度。后续正式实验可扩大宽度、使用 diffusion_batch_mul=16。零初始化意味着最初若干步上游梯度为零，必须经过更新后检查特征 Connector、MASK 与 LoRA 梯度，不能用第一步判断支路失效。

### 优化目标

对模态 m 的被遮盖真实向量 z*，采样 sigma~Uniform(0,1)、epsilon~N(0,I)：

`x_sigma = (1-sigma) z* + sigma epsilon`

`u* = epsilon - z*`

`L_plan,m = mean_(被遮盖 token、噪声重复、特征维度) ||u_phi(x_sigma,1000 sigma,c)-u*||²`

首轮 `L = L_plan,v + L_plan,a`，分别归一化后相加，记录各自数值；没有 NTP、AV 重建或持续时间预测损失。这里明确实现 uniform sigma、单位权重，不能声称复刻了未开放的原始训练 scheduler。若两个模态量级严重失衡，再以训练集统计为依据设计下一轮，不能事后悄悄改变本轮权重。

遮盖率采用 Beta(12,0.9)，按模态抽取位置，另设 20% 全 MASK 样本。这是本任务的初始配方，不代表 Bernini 对双模态任务的官方设置。所有有效目标 token 在注意力中互相可见，MASK 仅改变内容。

## 4. 时间与序列长度

第一步固定现有 GT 文本和 GT 时间表，不训练时间头，避免把内容、时长与连续规划误差混在一起。条件文本包含 Reply/Delivery/Behavior/Vocal，以及既有 vocal_track、floor_track 和目标时长。直接复用标注，不重新打标。

视觉以 2 FPS 采样，保留源宽高比，最多 224² 像素；按原生 temporal_patch=2、空间 merge 得到目标网格。音频特征数量由原生 mel 有效帧及音频塔输出长度公式确定（mel hop10ms，卷积和池化合计约4倍下采样）。不直接把 Speaker 输入 token 个数当成 Listener 输出长度。

目标 token 含时间、空间坐标，音视频共享秒级时间轴；时间位置按 25 ticks/s 转为 MRoPE。视觉末端不足一组时按原生预处理重复最后一帧，并记录这一边界约定。首轮每条样本从零开始独立编码，不拼接不同样本的目标。

最终自主推理设想：原生 Omni 先输出回复和粗时间表，检查时间表后得到预算 T，再由 T、图像网格及音频编码器长度公式创建 MASK 布局。随后以预测的全部特征再运行一次 Omni，为渲染器产生条件。**这个自主长度策略尚未实现或验证**，不会用首轮的 GT 时间结果代替它。

## 5. 固定条件的配对实验

配置：[pipeline3_pilot.json](../configs/pipeline3_pilot.json)。从已有 Train91,009 与 Dev48 的清单中，各自按固定哈希选择有文字回复且长度≤8秒的 **Train16 / Dev8**，保留 ID 和原始路径；Test 不参与。此为可学习性 pilot，不是效果结论或完整数据训练。

|控制项|约定|
|---|---|
|主干/目标编码器|同一预训练 Omni，同一冻结目标缓存|
|公共头初始化|独立 RNG 初始化，保存 SHA256 并要求两组一致|
|样本曝光|每组200步、batch1、完全相同的样本顺序|
|随机性|同一步的遮盖、sigma、epsilon 相同；与模型参数量无关|
|优化器|AdamW，头/LoRA lr均1e-4，weight decay0，梯度裁剪1|
|训练预算|相同更新数与样本曝光；不声称相同 FLOPs 或耗时|
|评估|固定 Dev8，每条两个噪声 seed；训练前 FM，训练后 FM 与全 MASK 采样|

必须报告：

1. 两个模态各自的随机遮盖 FM loss、全 MASK 条件 FM loss。
2. 全 MASK 起步的一轮预测、25轮逐步回填，均内层5步；相同位置的初始噪声在两个采样方式间固定。
3. 特征 MSE、cosine、预测与目标 RMS；不能把 cosine 高直接等价为细节正确。
4. 训练集每模态平均特征基线，以及预测/目标的 RMS 幅度诊断。生成模型会采样不同可能结果，单样本 GT MSE 可能天然高于条件均值预测，因此均值 MSE 不是生成能力的硬性通过线；还必须检查条件敏感性与后续可解码性。
5. 固定相同目标网格/噪声，循环打乱 Dev 条件前缀的对照；只有正确条件能降低误差，才支持模型使用了对应条件。该对照包含身份/外观变化，不足以单独证明 Speaker 语义被使用。
6. 每个 Dev 样本先平均两个 seed，再给冻结/LoRA 的配对差及探索性 bootstrap 区间。单训练seed、8条Dev不支持强统计结论。
7. 梯度连通、冻结参数、有限数值、初始化/缓存/顺序契约。采样 API 只有 Layout，不接收 GT target 参数。

## 6. 结果如何决定下一步

若两组只降低 teacher-forced loss，但全 MASK 不改善、打乱条件不恶化，先排查采样步数、条件利用、特征分布与模型容量；不直接接渲染器。

若 LoRA 在全 MASK 条件 FM 损失及采样诊断中稳定优于冻结组，并且在固定同一目标、sigma、噪声下正确条件优于打乱条件，下一步扩大训练/Dev覆盖并重复训练seed，确认改善可复现。均值预测仅辅助解释特征尺度和采样误差，不单独否决随机生成模型。若两组相近，则保留冻结方案作为对照，先检查是否已经充分训练、数据量、梯度幅度及 decoder 是否掩盖主干差别；不能用短跑断言无需训练 Omni。

规划能力建立后，再做渲染条件对齐：固定同一句 GT 文本与 Listener 首帧，比较真实目标特征条件、预测目标特征条件和已有原生文本条件。过去 Stage1 step800 已证明其原条件路径的生成能力，因此不重复泛泛的“给 GT 能否生成”；新增问题是 **P3 新特征和连接方式是否被 LTX 利用，预测特征相比 Oracle 损失多少**。

### 后续联合训练的梯度关系

拟保持 Bernini 两类 Connector：特征分支服务 FM，生成分支服务 DiT。它们任务/输出空间不同，不因分支不同就必然优化冲突。Thinker LoRA 可共同接收两类损失；冻结主干则失去这条共享可训练表示的协调通道。

Bernini 式 teacher-forced 联合前向中，被遮盖目标位置训练 decoder；文字、源条件及未遮盖目标位置 hidden 给渲染器。当前这一轮 decoder 的采样结果并不进入渲染分支，所以 L_AV 不直接训练 decoder。这与 LoRA 的共享表示协调是两个不同问题。后续需要预测条件评估及受控混入预测特征来检查分布差异，不能宣称打开 LoRA 已自动消除训练/推理差异。

未来损失候选为 `lambda_text L_NTP + lambda_v L_plan,v + lambda_a L_plan,a + lambda_AV L_AV`。理解/语言保持数据、冻结策略、生成 Connector 的尺寸与原生 LTX 接口、预测条件混入策略均待第一步结果后细化。此轮不实现生成 Connector 或启动 LTX 训练。

## 7. 实现和结果入口

- [首轮运行记录](PIPELINE3_FIRST_EXPERIMENT.md)：命令、工程验证、实测状态及后续判断。
- `src/omni_av_p3/model.py`：MASK、混合注意力、特征 Connector、FM 和迭代采样。
- `src/omni_av_p3/data.py`：固定 GT 条件、原生目标特征、独立缓存契约。
- `src/omni_av_p3/experiment.py`：配对训练、全 MASK 评估、条件打乱与结果比较。
- `tests/test_p3.py`：真实 tiny Omni 的梯度、目标隔离、噪声一致性与采样方向测试。

GitHub 仍为私密仓库 [liqilin-eudaemonia/audio-driven-v4](https://github.com/liqilin-eudaemonia/audio-driven-v4)。本页描述工作区新增 P3，是否已提交/上传以运行记录为准；不将现有 P2 提交当作 P3 备份。
