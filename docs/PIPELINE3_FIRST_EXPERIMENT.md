# Pipeline 3 首轮：固定条件的冻结／LoRA 规划对照

2026-10-08。方案见 [PIPELINE3_PLAN.md](PIPELINE3_PLAN.md)。本记录会区分代码检查、真实模型工程验证与能力比较。

后续已完成：[四卡2000步与加速报告](PIPELINE3_FOUR_GPU_20261008.md)。本页保留200步时的原始结论和当时的下一步计划；其中小集/延长对照已在后续执行，不代表当前仍未执行。

## 当前范围

**首轮真实模型对照已完成**：Train16 / Dev8，冻结与 LoRA 各200步，每条 Dev 两个采样 seed。代码和梯度路径通过检查，但两组对对应条件的利用都很弱，暂不能选定 LoRA 或冻结路线。下一步继续检查规划训练收敛与条件利用，不进入渲染训练。

机器可读证据：[PIPELINE3_RESULTS_20261008.json](PIPELINE3_RESULTS_20261008.json)，包含完整配对结果、条件诊断、权重及代码哈希。已有 P2 训练、缓存、未提交改动均保留；本次尚未新增 GitHub 提交或推送。

三台授权机器最初均无空卡；后续本机 GPU4/5/6 释放，已检查计算进程列表，先使用本机 GPU4 准备缓存。仅使用空卡，不停止其他任务。

## 复现

Python 环境：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。以下在仓库根目录执行，`PY` 指向该环境，输出目录必须是新的，入口拒绝覆盖历史产物。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
OMP_NUM_THREADS=2 "$PY" -m pytest tests/test_p3.py tests/test_p2.py::test_p2_decoder_matches_bernini_forward_and_backward -q

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 "$PY" scripts/pipeline3.py prepare \
  --config configs/pipeline3_pilot.json \
  --train-manifest outputs/p2_full91009_dev48_20261008/train.json \
  --dev-manifest outputs/p2_full91009_dev48_20261008/dev.json \
  --output outputs/p3_fixed_condition_pilot_20261008/cache

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 "$PY" scripts/pipeline3.py train \
  --config configs/pipeline3_pilot.json \
  --cache outputs/p3_fixed_condition_pilot_20261008/cache/index.json \
  --arm frozen --output outputs/p3_fixed_condition_pilot_20261008/frozen

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 "$PY" scripts/pipeline3.py train \
  --config configs/pipeline3_pilot.json \
  --cache outputs/p3_fixed_condition_pilot_20261008/cache/index.json \
  --arm lora --output outputs/p3_fixed_condition_pilot_20261008/lora

"$PY" scripts/pipeline3.py compare \
  --frozen outputs/p3_fixed_condition_pilot_20261008/frozen \
  --lora outputs/p3_fixed_condition_pilot_20261008/lora \
  --output outputs/p3_fixed_condition_pilot_20261008/comparison.json
```

GPU编号仅记录本次位置，复现前需重新核对空闲状态。固定条件包含 GT 回复、表达描述和时间，不能将结果描述为自主回应质量。视频分辨率是特征编码预算，不涉及改写已有720p-area素材。

## 结果

所有主实验、检查点恢复和附加条件诊断均成功退出，无失败重跑或被替换的结果。22项规划/参考结构测试通过，新增的2项严格权重恢复测试也通过，共24项。其范围不代表全仓所有 P2 功能或生成质量验证。

两组公共可训练头均为75,384,832参数，初始化 SHA256 完全相同；LoRA组额外7,168,000参数。每步样本ID和随机seed逐一匹配。冻结组仅训练用时49.31秒，LoRA组74.53秒，不包括缓存、加载与评估；峰值 allocated 显存约10.45/10.56 GiB。两组不同物理卡并发，不能把此短跑耗时比当作正式吞吐结论。

|Dev 指标（越低越好）|冻结 Omni|Omni LoRA|
|---|---:|---:|
|训练前：全 MASK 条件 FM，视频|4.48873|4.48873|
|训练后：全 MASK 条件 FM，视频|2.31294|2.31162|
|训练前：全 MASK 条件 FM，音频|1.68114|1.68114|
|训练后：全 MASK 条件 FM，音频|1.20191|1.20624|
|全 MASK、25轮、5内步采样：视频特征 MSE|4.15718|4.14425|
|全 MASK、25轮、5内步采样：音频特征 MSE|1.66104|1.68572|

LoRA的视频采样MSE相对低0.31%，音频反而高1.49%，没有一致优势。训练均值基线的视频/音频MSE为2.58316/0.55152；音频预测RMS为1.119/1.140，也高于目标的0.826。**随机生成结果对单个GT的MSE可能天然高于均值预测，不能仅凭这一项判定失败。** 对应 cosine、RMS、一轮采样、样本配对差和探索性bootstrap区间均在JSON中。

### 条件是否被利用

主实验的采样对照中，换成另一条Dev的条件前缀，MSE几乎不变。为排除单次生成误差的歧义，训练完成后附加一次诊断：恢复同一检查点，保持同一GT向量、sigma、epsilon，比较正确/打乱条件的**全MASK条件FM损失**。不追加训练，不挑超参数。

|条件打乱后的FM损失相对变化|冻结 Omni|Omni LoRA|
|---|---:|---:|
|视频|+0.000664%|−0.000292%|
|音频|−0.000203%|+0.000084%|

这些差异极小且正负不稳定，没有提供模型有效使用匹配条件的证据。打乱整个前缀同时改变Speaker、回复、身份和时间，后续还需逐项消融；这轮不声称已测出各条件的独立作用。

条件诊断可复现：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 "$PY" scripts/pipeline3.py diagnose \
  --run outputs/p3_fixed_condition_pilot_20261008/frozen \
  --cache outputs/p3_fixed_condition_pilot_20261008/cache/index.json \
  --output outputs/p3_fixed_condition_pilot_20261008/frozen/condition_diagnostic.json
```

LoRA组替换run/output中的组名。输出存在时会拒绝覆盖，复现须另选新路径。两组从磁盘恢复后可训练参数逐值一致，诊断的正确条件FM与主实验记录也一致。

### 可以与不能得出的结论

- 可以确认：真实 Omni 感知特征提取、遮盖训练、冻结/LoRA梯度、Bernini拓扑decoder、全MASK迭代回填、配对评估和权重恢复已经打通。
- 当前观察：200步内两组损失均下降，但条件打乱几乎不影响全MASK条件FM；还不能把下降解释成学会了依照Speaker与计划预测目标。
- 不能确认：LoRA无用、冻结Omni足够、目标原生音频特征足以控制台词、最终音视频生成会改善。没有训练/评估LTX，没有自主回复/时间预测，只有一个训练seed和8条Dev。

## 下一步计划（尚未执行）

下一轮仍属于规划能力验证，保持文本+时间固定，不增加渲染损失。

1. **先做条件可学习性检查。** 选择2–4条差别明显的训练样本，延长到能观察收敛的步数。检查正确/打乱条件在同一target和noise下的差异，以及 hidden、Connector输出、decoder速度对条件的变化，定位信息在哪一层被忽略。小集记忆成功也不等于泛化，但若这一步失败，不应直接扩数据或接DiT。
2. **延长受控对照。** 为两个200步检查点增加受控续训入口，保持原样本/初始化/噪声规则，将总步数预先设为2000，至少在200/500/1000/2000步固定评估。第一版不同时改宽度、学习率或mask比例，避免混淆训练不足与结构影响。原始200步产物永久保留，新输出目录记录训练契约；当前CLI尚未提供续训功能。
3. **分离采样误差和条件学习。** 增加sigma区间的条件FM诊断及5/20内步采样对照；同一个target/noise测试正确条件、仅替换回复文本、仅替换Speaker音频。输入消融需要重新编码对应前缀，不能把整段随机置零当作特定语义干预。
4. **出现可靠条件增益后再扩大。** 使用更大的Train集合、完整Dev48和多个训练seed，判断LoRA收益是否可复现，再决定进入生成Connector/渲染对齐。GT特征→渲染的Oracle对照只验证P3新接口，不重复声称旧step800从未验证过GT生成。

当前最重要的结论是“先让规划器明确利用条件”，而不是在这次短跑后直接选冻结或LoRA。后续时间预测与自主回复依然独立验证。
