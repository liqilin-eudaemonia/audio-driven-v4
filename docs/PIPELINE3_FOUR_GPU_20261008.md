# Pipeline 3 四卡加速与规划收敛检查

2026-10-08。延续[首轮固定条件对照](PIPELINE3_FIRST_EXPERIMENT.md)。用户授权使用本机空闲0–3卡进行测试、修改代码并优化速度；启动前已核对这些卡没有计算任务。4–7卡的其他工作保持运行。

## 本轮实验

|GPU|实验|初始化与训练范围|
|---|---|---|
|0|冻结 Omni 的 Train16 延长对照|先配对测速，再从原200步检查点恢复至总2000步|
|1|Omni LoRA 的 Train16 延长对照|同上，保持相同样本曝光和噪声序列|
|2|冻结 Omni 的 Train2 条件学习检查|原清单前2条，从原配方的相同初始化训练2000步|
|3|Omni LoRA 的 Train2 条件学习检查|与GPU2同数据/初始化/噪声，增加Thinker LoRA|

四组均batch1，保持原头结构、学习率、遮盖率、FM噪声约定，不以扩大batch改变本轮研究问题。每500步保存检查点，对Train及Dev检查正确/打乱条件；最终在Dev8上执行全MASK25轮/5内步采样。Train2仅检查记忆和条件利用，不代表泛化。Train16延长到2000是总步数，新增1800步。

入口：`scripts/pipeline3_extended.py`；运行根目录：`outputs/p3_four_gpu_20261008/`。正式有效训练目录是`train16_frozen`、`train16_lora`、`train2_frozen_v2`、`train2_lora_v2`。最初两个Train2任务在梯度汇总函数上报错，尚未完成第一次optimizer更新；错误为遍历字典键而非值，已修复并补充真实梯度测试，原目录和日志保留。

## 速度优化

新增`src/omni_av_p3/accelerated.py`，不改P1/P2模块及原始P3参考路径：

- 小数据集的输入embedding、目标特征常驻GPU，避免每步传输。
- 在CPU提前生成与原实现一致的mask/目标位置索引，再传到GPU，减少GPU动态索引带来的同步。
- 验证数据后缓存静态注意力矩阵和MRoPE位置；缓存保留原对象，防止Python对象ID复用误命中。
- 利用空闲显存关闭Thinker/decoder的激活重计算。
- 使用fused AdamW；恢复旧优化器状态时显式处理fused参数和step张量设备，并复制状态以避免污染原检查点。
- 改为周期性、向量化梯度审计，保留每步的有限梯度范数检查。
- 四个独立实验并行，避免为小模型对照增加DDP通信开销。目标是有效吞吐，不是填满显存。

13项P3/加速相关测试通过；修复梯度汇总后4项加速回归再次通过。测试覆盖非零条件下优化前后损失/梯度、遮盖索引、续训顺序、优化器恢复不污染检查点。历史首轮24项与本轮测试有重叠，不相加充当唯一测试总数。

### 实测配对速度

每种模式从同一200步检查点（含optimizer）开始，5步预热、30步计时，同样本、sigma、epsilon和更新数。参考模式保留原激活重计算、逐参数同步检查和每步传输；加速模式使用上述改动。

|组别|参考平均ms/步|加速平均ms/步|速度提升|
|---|---:|---:|---:|
|冻结 Omni|280.61|127.01|2.21×|
|Omni LoRA|376.01|126.01|2.98×|

加速模式峰值allocated显存约15.2/15.5GiB。fused optimizer和重计算策略会改变浮点运算顺序：35步轨迹的最大绝对loss差分别为0.000133/0.001889，最大单参数差0.001339/0.003953。因此记录为数学目标一致、短期loss相近，**不是逐位等价或长期轨迹等价**。两个研究组均用同一加速配置；原200步结果保留。测速在其他实验同时运行时完成，均值和中位数均存入JSON，不声称是稳定生产吞吐。

原始测速结果：`benchmark_frozen/completed.json`、`benchmark_lora/completed.json`。GPU利用率、显存、功率每2秒记录到`gpu_samples.jsonl`；更新耗时与包含验证/保存的实验循环耗时分开报告，后者不包含启动时的模型加载、缓存读取和mask预准备。

## 条件诊断

除了FM和采样指标，新增固定同一GT/noise的sigma=1检查，并沿信息路径测量切换条件前缀导致的变化：目标hidden → 特征Connector输出 → decoder速度。这样可以区分“Omni hidden对条件不敏感”和“hidden已有差异，但decoder没有利用”。

正确/打乱条件的目标网格和噪声相同，打乱对象记录sample ID。暂不将整段前缀打乱解释为只改变Speaker语义。所有结果仍是固定GT回复/时间的规划诊断，没有启动生成Connector或LTX训练。

## 复现

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" scripts/pipeline3_extended.py benchmark \
  --arm frozen --cache outputs/p3_fixed_condition_pilot_20261008/cache/index.json \
  --resume outputs/p3_fixed_condition_pilot_20261008/frozen/final.pt \
  --output outputs/p3_four_gpu_20261008/benchmark_frozen

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$PY" scripts/pipeline3_extended.py train \
  --arm frozen --cache outputs/p3_fixed_condition_pilot_20261008/cache/index.json \
  --resume outputs/p3_fixed_condition_pilot_20261008/frozen/final.pt \
  --total-steps 2000 --validate-every 500 \
  --output outputs/p3_four_gpu_20261008/train16_frozen

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 "$PY" scripts/pipeline3_extended.py train \
  --arm frozen --cache outputs/p3_fixed_condition_pilot_20261008/cache/index.json \
  --train-limit 2 --total-steps 2000 --validate-every 500 \
  --output outputs/p3_four_gpu_20261008/train2_frozen_v2
```

LoRA组改`--arm lora`、检查点和输出目录；仅在确认空卡后启动。示例中的本次产物已存在时，需选新的输出目录，不覆盖历史结果。缓存配置保持旧值，延长步数是独立运行参数，既能复用原生特征缓存，也明确记录每次训练的总步数与起点。

## 完成结果

**四组均已完成2000步及最终全MASK评估，进程正常退出。** [完整机器可读结果](PIPELINE3_FOUR_GPU_RESULTS_20261008.json)保存配对检查、各阶段条件诊断、最终采样结果、GPU遥测、失败记录和代码/检查点哈希。没有运行渲染器，也没有自动推送GitHub。

|实验|本轮更新数|更新总耗时|平均更新ms/步|含验证/保存的循环耗时|
|---|---:|---:|---:|---:|
|Train16 冻结|1800|177.12s|98.40|251.89s|
|Train16 LoRA|1800|226.60s|125.89|318.67s|
|Train2 冻结|2000|195.49s|97.74|254.40s|
|Train2 LoRA|2000|248.36s|124.18|318.81s|

四组并行总计执行7600次新增更新。每组的训练样本顺序和噪声seed在冻结／LoRA之间逐项核对一致；Train2公共头初始化哈希一致。Train16各自恢复原200步权重和优化器，保留先前两组已经形成的参数差异。

GPU0–3在模型显存>8GiB的采样窗口内，平均利用率分别为66.6%、60.9%、73.8%、64.7%，中位数73.5%、67%、80%、69%。这些数值**包含加载、保存、验证、测速及早期失败启动**，不是纯更新区间利用率，也没有将低利用率样本剔除后宣称满载。后续扩大有效batch可以进一步提高吞吐，但会改变优化统计，需建立新配对实验，不能悄悄替换本轮batch1。

### 1. LoRA已经能学到训练条件依赖

固定相同GT、sigma、epsilon，将条件前缀换为另一条样本；下面是均匀sigma FM损失的相对增幅，正值表示正确条件更有利。

|训练集检查|冻结 Omni|Omni LoRA|
|---|---:|---:|
|Train16 视频|+0.0002%|+0.125%|
|Train16 音频|+0.662%|**+18.817%**|
|Train2 视频|+0.046%|**+6.316%**|
|Train2 音频|+0.524%|**+10.034%**|

在sigma=1、完全没有干净目标信息的噪声端，Train16音频的对应增幅为冻结2.90%、LoRA33.10%。这改变了200步时“两组几乎完全忽略条件”的观察：LoRA这条梯度路径能够学习条件关联，尤其音频分支。它仍可能主要利用身份、时间表或记忆样本，不能由整段前缀打乱直接推断理解了Speaker语义。Train2的目标布局也不同，存在几何/长度辅助区分样本的可能；小集结果只作为可学习性证据。

### 2. Dev尚未出现可靠的条件泛化

Train16模型在Dev8的全MASK条件FM：

|Dev指标|冻结 Omni|Omni LoRA|
|---|---:|---:|
|视频：正确条件FM|1.91426|1.92829|
|音频：正确条件FM|1.01281|1.06241|
|视频：打乱条件后的FM增幅|−0.00076%|+0.0654%|
|音频：打乱条件后的FM增幅|−0.119%|−0.105%|
|25轮/5内步视频采样MSE|4.02232|3.99671|
|25轮/5内步音频采样MSE|1.38103|1.28375|

LoRA采样视频MSE低0.64%、音频低7.04%，但Dev条件FM更高，而且音频条件打乱没有恶化，因此**不能仅凭采样MSE选择LoRA并宣布泛化成功**。随机采样误差还受方差、幅度和采样器影响。两条样本模型也没有可靠Dev收益，符合小集过拟合的风险；完整结果见JSON。

### 3. 信息变化主要在哪里减弱

以第一条Dev为例，Train16 LoRA切换前缀后的RMS变化：

|分支|hidden变化 / 自身RMS|Connector变化 / 自身RMS|decoder速度变化 / 自身RMS|
|---|---:|---:|---:|
|视频|3.468 / 6.312|0.286 / 2.195|0.0199 / 0.923|
|音频|3.389 / 5.743|0.461 / 1.155|0.199 / 0.791|

这是一个样本的诊断，不是跨样本统计结论；它表明hidden并非完全不随条件变化，视频分支的变化在Connector/decoder后明显减弱。因此不应仅凭输出不敏感就推断Omni完全没读到输入。不同模块的维度和尺度不同，这些比值只用于定位后续检查，不作为信息量估计。

## 下一步建议

保留LoRA为主要候选、冻结Omni为对照；当前证据支持继续研究LoRA的条件学习能力，尚不支持进入渲染训练或取消冻结对照。

1. 将Train扩大到至少128–512条，使用完整Dev48及更多训练seed，检查训练条件收益能否泛化；本轮只使用原Train16/Dev8，未预先查看额外Dev来调参。
2. 分别替换回复文本、Speaker音频和Listener首帧，重新编码条件前缀，区分语义、身份和时间线关联；避免仅靠整体前缀置换解释任务理解。
3. 优先诊断视频Connector/decoder的条件抑制，再决定是否改变mask比例、宽度或损失权重；不要同时改结构和数据量后难以归因。
4. 后续更大数据训练再配对测试更大有效batch/多卡累积。当前batch1四个独立任务已获得实测2–3倍加速，目标函数和实验变量保持清楚；进一步追求高利用率也应以每秒有效样本和验证结果为准。

上述扩大训练、细分消融及新batch实验尚未启动。
