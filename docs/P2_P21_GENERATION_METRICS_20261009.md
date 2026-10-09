# P2／P2.1：沿用旧版四项生成指标

对照来源：`audio-driven-v2/OMNI_LTX_GLOBAL_INDEX.md`第0.1节与第10.5节、`Omni-LTX-stage2/packages/ltx-trainer/src/ltx_trainer/validation_benchmark.py`以及旧Generated-Plan指标重算脚本。旧正式协议为 **ArcFace身份相似度、FID、FVD、SyncNet-C**；同时记录规划合法率、条件AV成功率、端到端成功率和SyncNet覆盖率。

实现入口：[pipeline2_metrics.py](../scripts/pipeline2_metrics.py)，P2与P2.1共用；两个原验证CLI均支持`metrics`子命令。实现：[generation_metrics.py](../src/omni_av_p2/generation_metrics.py)，配置：[p2_generation_metrics.json](../configs/p2_generation_metrics.json)。只评价已有生成视频，不加载生成模型、不改变训练任务。当前训练仍关闭验证，本次没有生成Dev视频。

## 指标定义及沿用范围

|指标|方向|本次使用的旧实现|
|---|---|---|
|ArcFace ID similarity|高更好|buffalo_l；参考目标视频首帧与生成视频8个均匀采样帧的人脸embedding余弦相似度，保留逐样本和人脸覆盖率|
|FID|低更好|torchvision ImageNet Inception-v3池化特征，每视频4帧；集合级Fréchet距离|
|FVD|低更好|PyTorchVideo I3D-R50 Kinetics特征，每视频16帧；集合级Fréchet距离|
|SyncNet-C|高更好|旧vocal-separated SyncNet，单个可见人脸轨迹门控；生成视频的自身音画同步|
|SyncNet-D（随旧后端保留）|低更好|同一次SyncNet执行产生的音画embedding最小距离，不额外运行模型|

FID/FVD严格复用旧特征提取及距离实现，**不是**pytorch-fid/TF-I3D实现；不可与不同实现的公开数值直接比较。指标特征提取沿用旧版299×299 resize、224中心crop等内部预处理，只作用于评测特征副本；训练、生成和落盘视频仍为原720p-area非正方形几何，不裁改源视频。

此处Dev48与旧固定24例并非相同样本集合；当前生成也使用固定训练几何、Euler无CFG，与旧生成采样协议有差异。沿用的是指标计算口径，不表示实验整体相同。48例分布指标仍属小样本诊断；单例FVD输出null/not_computable，空生成集合全部为null，不伪造零分。

## GT／Omni与失败口径

- 两组分别使用各自实际保存的Reply确定SyncNet候选；Omni绝不回填GT回复，不把Delivery／Behavior／Vocal当成主回复文本。无主回复（包括只有非语言Vocal）不进入SyncNet，仍进入ID/FID/FVD。候选不代表一定有可评分语音；实际音轨/人脸检测失败由旧后端记录为失败或不适用。
- `planning_started.json`、`planning_result.json`区分有效、拒绝、运行错误；在hidden提取前记录合法计划，避免把之后的OOM误算为解析失败。旧产物无法证明规划阶段时记unknown，未尝试记not_attempted。
- 修复验证流程：Omni计划被ValueError拒绝后，GT分支可继续提取条件和生成；不把GT作为Omni替代。配对任务仍标为partial，不冒充两组成功。运行错误仍明确失败。
- 总清单48始终作为端到端成功率分母；另外报告规划尝试数、合法数、合法率（全清单/已尝试两个分母）、合法计划条件下生成成功率、拒绝/错误详情、GT fallback数（必须0）。
- `gt_available`、`omni_available`各评分自身成功生成的子集；`gt_common`、`omni_common`使用交集做对照。集合相同时复用已算结果，避免重复执行。不可比较不同成功子集而忽略覆盖差异。
- ArcFace逐样本结果、SyncNet C/D/offset及status、eligible/success/dataset计数均保留。无脸、多人脸、缺失音频/检测失败不会静默丢失；覆盖不足标partial，后端错误标failed。`quality_pass`保持null，工程完成不等于质量通过。

## 参考视频隔离与溯源

GT目标视频只通过评测命令的`--metadata`读取，**不写回生成suite，不输入Omni/LTX推理**。metadata SHA必须与suite准备时相同，sample_id/split/目标几何严格匹配。

每个成功视频必须具备分支完成标志、计划、几何、case和worker checkpoint/suite哈希。拒绝不同checkpoint混用、GT计划改变、条件来源不符或目标媒体用于生成。评测准备记录生成/参考视频SHA；score前重新核验。原生成目录保持不变，新评测输出目录必须不存在。

`protocol.json`保存旧后端源码/模型SHA、帧数设置及运行配置。资产包括ArcFace两份ONNX、Inception/I3D、SyncNet、人声分离和人脸检测权重。运行环境沿用本机ovi/avbench/vocal_sync。

## 使用命令

以下命令不自动串入正在进行的训练；先完成生成，再单独评分。`GENERATION`须指向某一个checkpoint的完整生成输出（与suite匹配），不可填训练目录或混合不同checkpoint。

```bash
# 1. CPU资产预检；不启动任何GPU模型
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline21_validate.py metrics preflight \
  --output /tmp/p21_metric_assets.json

# 2. CPU准备参考清单和覆盖统计；变量应填写实际生成结果与新输出目录
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline21_validate.py metrics prepare \
  --suite outputs/p21_dev48_full_plan_validation_20261008/suite.json \
  --generation "$GENERATION" \
  --output "$METRIC_PLAN"

# 3. 指定空闲物理GPU；不是CUDA重映射后的逻辑编号
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline21_validate.py metrics score \
  --plan "$METRIC_PLAN/evaluation_plan.json" \
  --output "$METRIC_RESULTS" --gpu 0
```

P2将命令入口换成`pipeline2_validate.py metrics`，或两者都直接使用`pipeline2_metrics.py`。prepare/score分离支持已有视频重算，无需重新生成AV。若新增生成结果，需要在新目录重新prepare形成新快照；不改动旧报告。

score先检查指定GPU空闲，再依次运行各指标；不会停止训练或操作其他GPU。修正了旧SyncNet子进程会重设`CUDA_VISIBLE_DEVICES`的集成陷阱：传入选择的**物理GPU编号**，不能始终传0。超时只终止本次指标子进程组。

输出：

- `evaluation_plan.json`及四组JSONL：样本、checkpoint/视频哈希、规划/生成覆盖率及失败原因。
- 各组`identity.json`、`visual.json`、`sync.json`、`sync_per_sample.jsonl`与`commands.log`：原始结果和诊断。
- `metrics.json`、`metrics.csv`：GT/Omni、可用/共同子集的汇总。
- `tensorboard/`：`generation/<mode_subset>/id_similarity|fid|fvd|sync_c|sync_d`及覆盖率、成功率、指标缺失标记；横轴为checkpoint step。与现有视频`comparison.html`配合查看。

## 已完成检查与边界

2026-10-09：

- 37项相关CPU回归通过，覆盖原P2/P2.1条件/LoRA、规划拒绝仅阻止Omni、指标清单/失败分母/共同子集、空集合与单例、错误/NaN处理、GPU编号隔离、忙卡拒绝、媒体篡改拒绝及CSV/TensorBoard输出。
- 13项真实资产存在性/SHA预检通过；真实Dev48参考清单准备通过。未生成状态明确记录48项not_attempted，不当成规划失败率。
- 两条真实参考视频在CPU上执行旧ArcFace/Inception/I3D：ID覆盖2/2；reference-vs-itself FID=0、FVD=0。此为后端自检，不是P2/P2.1生成结果。ArcFace比较的是首帧与多帧，不要求自身视频ID值等于1。
- SyncNet无候选样本汇总实际执行通过；另用旧raw结果回放，验证C/D汇总0.5304/10.20735及覆盖2/2。回放不等同于重新执行SyncNet神经模型。
- Ruff F/E9及diff检查通过。**没有占用训练GPU、没有做本次模型Dev生成，也未在新生成视频上执行非静音SyncNet评分。** 这部分仍需空闲GPU上一对真实生成样本门禁，再跑Dev48全量。

检查摘要：[P2_P21_GENERATION_METRICS_RESULTS_20261009.json](P2_P21_GENERATION_METRICS_RESULTS_20261009.json)。
