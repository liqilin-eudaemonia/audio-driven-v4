# Bernini 对照审计与八卡联合验证

日期：2026-10-08。审计起点 `51e18fe`；本地 Bernini 参考版本 `e6c2cf11843470b08491745947abcba9c3c6b079`。本阶段修改现已纳入[优化备份分支](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p2-3n24g-balanced-20261008)；未修改 Bernini、LTX 外部源码或旧运行产物。后续三机结果见[最新报告](P2_FULL_DATA_3NODE_20261008.md)。机器为八张 H200 约 140 GiB；所有任务启动前检查空卡。

结论：在下述范围内，P2 的最终 U 共享、三项 loss、冻结边界和部署路径通过验证；发现并修复了原尺寸 VAE 解码 OOM，以及缓存读取未核验 GT 哈希、准备阶段未直接拒绝超预算原始 GT 的问题。新增八卡 DDP 训练和等效单卡梯度累积入口，短测吞吐提高约 7.76 倍。不能将有限测试表述为“所有输入和所有训练规模均完全正确”，也没有生成质量结论。

机器可读结果：[完整测试摘要](BERNINI_AUDIT_RESULTS_20261008.json)。本机日志、checkpoint、MP4、原失败记录位于 `outputs/p2_audit_20261008_01/`；不上传模型与数据产物。

## 1. 与官方代码的逐项对照

|检查项|官方参考|P2 结论|
|---|---|---|
|生成投影|`bernini/models/bernini.py::MLPConnector.for_gen`|`Linear→GELU→RMSNorm→Linear` 拓扑相同；从参考类直接取权重，前向、输入梯度和所有参数梯度对照通过|
|特征 decoder|`bernini/models/diffloss_fm.py::SimpleMLPAdaLN`|时间编码、SiLU、等宽残差 MLP、AdaLN shift/scale/gate、最终层及双重零初始化相同；非零参数下数值/条件梯度对照继续通过|
|特征 loss|`DiffLoss_FM.forward`|P2 显式使用 uniform σ、速度 `ε−z`、有效元素均值；加噪符号、16 次独立噪声重复、mask、GT detach 与条件梯度验证通过；重复的排列顺序不同但期望目标相同|
|scheduler|`bernini/models/scheduler.py`|本地类标注 inference only，缺少 `add_noise`、`training_target`、`training_weight`；不能声称 P2 复刻了完整官方训练时间分布或权重|
|共享表示|`feat_from_planner_to_renderer`、`for_vit`|Bernini 有独立 `for_gen/for_vit` 分支，并在规划时使用目标 MASK/可见特征；P2 有意改为最终 U 同时进入 decoder 和冻结 LTX，不是 Bernini 等价复现|
|输出初始化|参考 `MLPConnector` 构造及配置|本地参考配置写了 zero-init 意图，但该类构造未执行末层清零；P2 显式清零是本项目选择，不能从配置字段推断参考构造已清零|
|条件接入|本地 LTX 原生 context attention|Uv/Ua 拼接到官方 Gemma processor 输出后，mask 同序拼接；没有额外可训练 DiT 分支，保留首帧条件|
|监督空间|冻结 Omni 感知编码器、LTX VAE|辅助目标为 2048 维感知 embedding；AV 目标为 VAE latent，二者没有混用|
|冻结与反向|P2 `data/model/ltx/runner`|Omni/Gemma/VAE 提取停止梯度；LTX 参数冻结但计算图保留到 U；每项 loss 分别对每个选定层检查梯度|
|自主推理|P2 `inference.py`|新文本重新提取 hidden；GT cache 与 manifest 被拒绝；部署不实例化 decoder；同一份文字进入 Omni 与 Gemma|

P1 的规划 LoRA、目标遮盖/迭代回填与 P2 的冻结提取路线保持分离。本次另外完成 P1 完整预训练 LTX 的 tiny latent GPU gate：新增规划分支零初始化保持输出逐值一致，第二步 AV→Omni hidden 梯度约 `1.03e-5`，AV→规划 LoRA 梯度约 `6.06e-3`，冻结参数检查和 checkpoint 恢复通过。它使用 Oracle 工程采样，不是 P1 自主完整 AV 或正式质量验证。

## 2. 修复与训练加速

### 原尺寸视频 VAE 解码

原 `decode_and_save` 一次性解码整个视频。在 `multi_full_gate` 中，三步联合训练与采样已完成，233 帧、1184×768 的解码请求额外分配约 85 GiB 工作空间，导致 OOM。这里不同样本的画幅不同：第一条训练样本是 1312×704，触发解码失败的是第三条 1184×768 样本。

现改为 LTX 原生 `decode_video` 和默认时空重叠分块；各 chunk 及时转 CPU，并严格核对最终帧数/高/宽。大视频不再整段占用 GPU 激活空间，未裁帧、缩小画面或改变目标预算。原失败 latent 在新目录成功解码，三种融合模式也重新完整通过。分块运算不承诺与整段解码逐值相等。

### 数据契约

- `prepare_cache` 在读取模型目标前检查原始 GT duration 不超过外部预算；过长必须重新分桶。
- `read_sample` 实际核验保存的 GT SHA256、sample ID、split、GT→TextPlan 一致性与预算。此前只是保存哈希，读取时未核验。
- 新训练入口的有界 device cache 减少重复磁盘读取、VAE 元数据扫描和主机到 GPU 传输；每个样本首次载入时检查契约，一个训练进程内视为不可变。
- 仍未对全部媒体和数十 GB 权重逐文件计算内容哈希；原路径下修改媒体、模型、特征缓存或 VAE 后必须重新准备，并重启训练进程。

### 并行入口

新增 `scripts/pipeline2_parallel.py`、`src/omni_av_p2/distributed.py` 和 `performance.py`：

- 每卡复制冻结 LTX，仅对 Connector/辅助 decoder 使用 DDP；没有把 DiT 解冻或改成可训练分支。
- 按全局样本编号分配数据及 AV/特征噪声种子，单卡累积与多卡相同 global batch 使用相同样本和噪声。
- 辅助 loss 纳入 Extractor forward，支持 `no_sync` 梯度累积；每次优化前对累积 loss 取均值。
- 保留默认完整 DiT 检查点；可配置 decoder 检查点、DiT 检查点间隔和 fused AdamW。默认有界缓存两个样本。
- rank 0 保存 checkpoint/纯 Connector，所有 rank 协调 STOP、保存和退出；训练结束逐 rank 校验全部可训练状态的 SHA256 完全一致。
- 记录逐步各项 loss、层梯度、样本/噪声种子、时间和显存。DDP checkpoint 的 identity 包含 world size/累积设置，恢复需匹配；不直接作为旧单卡训练入口的 resume 文件，纯 Connector 仍可用原推理入口。

原单卡 gate 也合并了对两个 U 的 AV 梯度检查，避免重复遍历昂贵 DiT；同时分别检查 AV、视觉特征、音频特征三项 loss 对每个选定层的梯度，防止只靠总梯度掩盖分支断开。

## 3. 验证结果

CPU 共 **41 项通过**，包括参考投影/decoder 数值对照、独立三路梯度、缓存失效回归、检查点策略等价性，以及真实双进程 Gloo DDP 与串行累积的梯度和参数更新对照。最后一次全套耗时 12.50 秒。

|GPU 验证|范围|结果|
|---|---|---|
|原始单卡默认多层|6 步 joint、6 步 AV-only、另一 seed 6 步 joint，原尺寸 latent|通过|
|三种融合|各自独立匹配缓存；单层/加权/多层原尺寸三步 gate|全部通过；每项 loss→每层梯度、冻结参数、部署逐值一致、233 帧 1184×768 AV 解码通过|
|八样本多层|8 步原尺寸 joint、8 步 AV-only|通过，属于工程短测|
|八卡 DDP|8 个真实样本，global batch=8，6 个 optimizer step，共 48 次样本前后向|通过；8 rank 参数哈希一致；LTX 未更新；每卡峰值分配约 64.96–66.24 GB|
|八卡恢复|从第 3 步启动新进程，执行到第 6 步|通过；每 rank 一致，但与不中断运行非逐位一致，见下文|
|DDP 部署|仅 Connector + 指定文本；正常 U / 零 U / text-only，各 10 步采样|均完成 233 帧 256×256、48 kHz AV；没有加载辅助 decoder 或 GT cache|
|自主文本|默认多层原生回复→条件→10 步采样|通过，233 帧 256×256 AV|
|Bernini-scale|两路 decoder 4096 宽、16 层、16 次噪声重复，完整预训练 LTX + tiny latent 三步 gate|通过；约 28.80 亿可训练参数，峰值 107.54 GB；不代表这一规模的原尺寸训练通过|
|P1 额外检查|完整预训练 Omni/Gemma/LTX，tiny latent，两步 AV 反向|通过；峰值 53.89 GB；未改变 P1 自主完整 AV 的未验证边界|

以上 **11 个 MP4** 已逐帧解码视频和音频，无解码异常。媒体可读不等于语音可懂、口型准确或生成质量达标。

恢复误差：相对不中断训练，Connector 参数相对 L2 差约 `3.33e-4`，最大绝对差约 `2.27e-4`；视频/音频 decoder 相对 L2 差约 `1.17e-5 / 5.28e-6`。数据/噪声序列与 checkpoint 状态恢复通过；当前未启用确定性 kernel 的 BF16 GPU 运算不能据此称逐位可复现。CPU 的 DDP/串行累积数学对照通过；也不能将 GPU 小幅浮点差异伪装成完全相等。

## 4. 吞吐与失败配置

固定八个真实样本、初始化 seed、噪声分配、global batch=8、模型与优化器设置；都保留完整 DiT/decoder 检查点。单卡一次优化累积八个样本，八卡每 rank 一个样本。

|执行方式|稳态优化步耗时|样本/秒|
|---|---:|---:|
|单卡累积 8|120.684 s|0.06629|
|八卡 DDP|15.550 s|0.51448|

实测吞吐比 **7.76×**。第一步两种方式的全局平均总 loss、AV loss 和两路特征 loss 一致。该短测速排除了加载和 checkpoint 保存；单卡取第二步一个观测，八卡取第 2–6 步均值，尚非长时间统计基准。

`nvidia-smi` 每两秒记录一次。在匹配八卡运行时间窗口、模型已载入且至少一张卡忙碌的 33 个采样点中，29 个点八卡利用率均 ≥95%；每卡平均约 88.0%、100%、99.9%、100%、100%、100%、100%、100%。rank 0 的 checkpoint/同步影响采样，NVIDIA utilization 也不是有效 FLOPs 利用率；不宣称从加载到结束每秒八卡均 100%。

失败配置完整保留：

- `perf_c0_d0`：关闭 DiT 与 decoder 检查点，原尺寸前向超过约 140 GiB，OOM。
- `perf_c2_d0`：仅对交替 DiT block 重算，原尺寸也 OOM。
- `perf_c1_d0`：保留 DiT 检查点、只关闭 decoder 检查点，通过但速度基本不变（14.93 对 14.97 s/单样本），峰值从 64.74 增至 68.90 GB；不足以支持改默认配置。
- `multi_full_gate`：旧整段 VAE 解码 OOM；已由分块解码及新目录复测修复。

## 5. 复现

在仓库根目录运行；以下输出目录必须不存在，GPU 必须空闲。测试缓存只含八个工程样本，不能直接当全量训练集。

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python

"$P2_PY" scripts/pipeline2.py test

# 新机器/新数据先准备匹配缓存；这里复用本次已审计的八样本缓存。
"$P2_PY" scripts/pipeline2_parallel.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --cache outputs/p2_audit_20261008_01/multi_prepare/cache_index.json \
  --steps 6 --save-every 3 --allow-terminal-trim \
  --output outputs/new_ddp8_joint

# 等效 global batch=8 的单卡对照。
"$P2_PY" scripts/pipeline2_parallel.py \
  --gpus 0 --accumulation 8 \
  --cache outputs/p2_audit_20261008_01/multi_prepare/cache_index.json \
  --steps 2 --allow-terminal-trim --output outputs/new_serial_batch8

# 新进程恢复：保持 world size、recipe、accumulation、seed 和缓存。
"$P2_PY" scripts/pipeline2_parallel.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --cache outputs/p2_audit_20261008_01/multi_prepare/cache_index.json \
  --resume outputs/new_ddp8_joint/extractor_step_000003.pt \
  --steps 3 --allow-terminal-trim --output outputs/new_ddp8_resume
```

仅 AV 配方使用 `--recipe av`。当前推荐保留默认检查点策略；`--checkpoint-every 0/2` 已在原尺寸下失败。`--allow-terminal-trim` 仅确认继承裁帧，本次没有完成全量台词尾音审计。

正式生成质量、辅助 loss 的增益、三种融合的效果优劣、全量 Train/Dev/Test 评估、长时间稳定性仍需独立实验。本次没有以短测权重或 loss 下降替代这些结论。
