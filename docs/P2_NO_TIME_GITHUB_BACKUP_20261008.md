# 当前P2训练GitHub备份（2026-10-08）

代码快照提交：[05e6489](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/05e64892daa1d8abd0e4e696a496368a37926271)（详细中文提交说明）；后续文档提交仅登记该固定代码版本。

分支：[backup/p2-no-time-frozen-dit-20261008](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p2-no-time-frozen-dit-20261008)。基于此前备份`9732ad3`，保留main和旧备份历史。本分支归档当前P2方案及其配对验证工具，P3独立工作不纳入本次备份。

## 当前实际训练结构

冻结Qwen2.5-Omni-3B，提取Thinker第8/18/28/35层hidden；可训练Connector将其融合为最终U_video/U_audio。U与Gemma原生文本条件沿token维拼接，共同进入LTX2.3已有的text cross-attention（视频attn2、音频audio_attn2）。这些attention正在参与前向，但权重冻结。

完整LTX/Gemma/Omni/VAE均冻结，仅训练Connector与音视频辅助decoder。总loss为AV flow matching + video feature flow matching + audio feature flow matching，各权重1。没有启用独立omni_attn2分支，P2限定text cross-attention LoRA也尚未实现或启用。不能将P1已有的广范围LoRA辅助函数当作当前P2训练实现。

Omni user message采用observation_no_time_v1，不提供明确时长或时间戳；仍使用完整观测片段与GT assistant计划提取hidden。这不等于已经实现实时自主轮次预测。

## 数据、运行和产物

- Train91,009 / Dev48，保留原720p-area宽高与比例，无正方形裁剪。
- 当前缓存根目录：`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_no_time_cache_20261008`，分别读取`train/cache_index.json`和`dev/cache_index.json`。
- 当前运行：`/public/2_CKPT/0_WM/llq/p2_no_time_full91009_3n24g_1000step_20261008_0838`。
- 三机node12/node17/node16，各8卡，joint配方，microbatch1，累积3，global batch72，seed45；从新Connector/decoder初始化开始。
- 目标1000步，每100步保存完整状态和Connector到运行目录`checkpoints/`，以`.complete.json`为保存完成标志；没有开启Dev loss或验证集推理。
- 本次备份采集日志时到第93步；这是时间点快照，不代表训练完成。实际进度读取`training/train_rank0.jsonl`。
- TensorBoard：运行目录`training/tensorboard/`；node12本地127.0.0.1:6006，选择`no_time_prompt`。SSH转发与完整命令见[训练记录](P2_NO_TIME_TRAIN_1000STEP_20261008.md)。

模型权重、缓存、媒体、日志原件及outputs不上传GitHub。完整checkpoint与Connector权重仍保留在用户指定的`/public/2_CKPT/0_WM/llq`下。

## 精确启动代码与复现

[启动溯源JSON](P2_NO_TIME_LAUNCH_PROVENANCE_20261008.json)保存运行原始命令、索引哈希、47个启动文件的哈希和备份对应关系。46个文件逐字节一致；唯一差异为实验登记配置在启动后补充运行信息，另存[原始登记配置](P2_NO_TIME_LAUNCH_EXPERIMENT_CONFIG_20261008.json)。实际启动没有传`--config`，模型结构和超参数来自本分支`Config`默认值。原始源码还保存在运行目录`source_snapshot/`。

`launch.code_commit=9732ad3`是启动时Git HEAD，不代表当时工作区没有修改；本次提交补齐这些修改，以逐文件SHA256核对实际运行源码。配对验证入口在启动之后新增，不属于原47文件。

复现时检出本分支的固定代码提交，准备已记录的Omni/LTX/Gemma权重、同版外部LTX源码与环境，以及已审计的数据/缓存；参考溯源JSON中的command，将仓库绝对路径和run-id/output/checkpoint-dir/log-dir替换为新的位置，避免复用正在运行的输出目录。依赖环境：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`；LTX依赖来自同项目的`packages/ltx-core`和`packages/ltx-trainer`，不随此仓库复制。仅Git仓库本身不足以离线还原所有大文件依赖。

## 验证与限制

无时间提示全量刷新、故障恢复与审计见[缓存记录](P2_NO_TIME_CACHE_20261008.md)及[结果](P2_NO_TIME_CACHE_RESULTS_20261008.json)。GT/Omni验证切换整份Reply/Delivery/Behavior/Vocal计划，两组均使用训练相同的hidden提取方法；固定Dev48共96个待生成视频。只完成CPU检查和清单准备，尚未运行GPU配对生成，见[配对验证](P2_FULL_PLAN_VALIDATION_20261008.md)。

本次隔离工作树CPU回归为70通过、1跳过（未复制旧P1本机GPU产物，非代码测试失败）；此前原工作区71通过。结果另存`P2_NO_TIME_BACKUP_CHECKS_20261008.json`；检查不加载GPU模型、不暂停训练。训练loss下降不能替代生成质量验证，尚不能断言hidden条件有效性或冻结DiT方案优于LoRA。
