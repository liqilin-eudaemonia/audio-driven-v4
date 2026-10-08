# P2第200步保存后切换P2.1：独立从头训练与15分钟监控

用户要求：P2冻结版到200步保存后停止；P2.1独立训练，不加载P2第200步权重。新运行根目录：`/public/2_CKPT/0_WM/llq/p21_fresh_3n24g_1000step_20261008`。2026-10-08 14:34 UTC已完成P2第200步保存审计，并启动`attempt_000`三机训练。目标步数不是已完成步数，实时阶段读取根目录`state.json`。

## 对照和初始化

P2.1仍使用原始LTX-2.3预训练底座，Connector和两个辅助decoder按P2相同seed45、相同构造顺序重新初始化；为防止加载DiT消耗随机数，Extractor在renderer之前构造。完整公共模块初始SHA256：`12e4514e2bb0af062d9be47f9f5a659fbcac96c04a62d678a9dda8b9a41ba982`（223,175,680个参数）。LoRA A随机、B为零，优化器/步数从零开始；启动命令不包含`--init-p2`或旧checkpoint。恢复故障时只允许使用本次P2.1自己的完整checkpoint。

仍复用无时间提示Train91,009缓存、原720p-area视频形状；三个节点各8卡，global batch72、累积3、样本顺序与噪声种子沿用P2。rank8/alpha8 LoRA仅覆盖原生视频attn2和音频audio_attn2的Q/K/V/out投影；Extractor LR1e-4、LoRA LR1e-5，其他DiT参数冻结。1000步目标，每100步保存；另外第3步保存一次启动安全checkpoint，尽早具备故障恢复点。不运行Dev loss或验证集推理。

## 停止和切换

`scripts/stop_training_at_step.py`观察P2日志到step200后写入该运行的`training/STOP`。P2在下一轮检查STOP之前会完成第200步周期保存，再正常退出；避免强杀写坏权重。需验证完整checkpoint、Connector导出、24rank RNG、完成标记和24rank最终参数一致性后，才写新运行根目录`P2_CHECKPOINT_VERIFIED.json`。

`scripts/pipeline21_supervise.py`只在P2完成、源码检查READY、checkpoint验证通过、三机GPU空闲后启动P2.1。每次尝试写到`attempt_000`等独立目录，训练目录`training/`、权重`checkpoints/`、集群日志`agents/`。代码原文保存在`source_snapshot/`，源哈希和测试结果在`READY.json`。

## GPU监控与异常处理

监督脚本每15秒采样三台机器的GPU利用率/显存/功耗/温度，每900秒汇总一次健康检查。检查逐rank进度、loss有限性、长时间无新step、连续低利用率、持续高利用率但低功耗（可能NCCL等待）、节点失联和权重磁盘空间；低利用率使用多次采样均值，不因保存权重时的一次下降触发重启。

- `gpu_telemetry.jsonl`：原始GPU采样。
- `health.json`、`health_history.jsonl`：最新与历史15分钟报告。
- `state.json`：当前phase、attempt、PID和逐rank步数。
- `alerts.jsonl`、`events.jsonl`：告警、通知投递、启动/恢复动作。
- `supervisor.json`：守护PID、检查频率、恢复上限。

告警通过本机`codex queue --thread ... --message ...`排入当前Codex任务，由当前任务读取最新状态后处理。该CLI通道已经实际排队自检；后续主动处理依赖本机Codex app-server和任务仍可用，投递失败会写入事件记录。没有向其他人或外部聊天发送消息。

自动恢复最多两次，只处理已有完整checkpoint时的明确通信/IO故障，或节点可达且持续超过15分钟无训练进度。仅终止本次launcher所拥有的guard/torchrun进程组，等待所有指定GPU空闲后，使用最新P2.1完整状态恢复剩余步数，保留旧attempt日志。OOM、非有限loss、契约错误不会盲目重跑，告警后等待诊断。低利用率本身只告警，不修改超参数或重启。

用户创建新运行根目录`STOP`可要求保存后停止本次训练/守护；`DISABLE_AUTO_RECOVERY`关闭自动恢复。不会删除数据缓存或其他任务。

## 验证范围

当前CPU回归105项通过，包含P2/P2.1/P3既有检查、公共模块初始化一致、双进程LoRA梯度/恢复、精确停止、误报过滤及恢复白名单。三节点当前源码哈希一致。监控功耗规则补充后11项监控回归通过，与105项有重叠。完整四卡GPU测试已通过，详见下文。

## 实际GPU检查与并行P2基线

额外机器172.29.104.194仅使用0–3卡：完整LTX2.3、实际缓存和原视频尺寸，先独立训练2步，再恢复到第3步。384个LoRA B矩阵均有更新；模型/优化器有限、部署Connector+LoRA与完整状态一致、四rank参数一致、LTX底座冻结。最长样本峰值已分配显存约114 GB。结果目录：`/public/2_CKPT/0_WM/llq/p21_gpu4_gate_20261008_1415`，完整核验记录`verified.json`。测试完成后四卡已释放，随后按用户指示用于P2续训。

三机正式运行首步loss 6.392409880956014、AV 0.6543121797343094，与原P2首步完全一致；初始Extractor SHA与公共参考一致，LoRA梯度非零。24个rank均完成首步。初期单步约110–120秒，尚不足以承诺长期耗时。

P2原三机200步状态通过权重/优化器有限性、Connector导出逐项相等和24rank一致性检查。最新指示要求它在172.29.104.194的0–3卡继续，见[续训记录](P2_RESUME_4GPU_20261008.md)。

TensorBoard运行于node12的`127.0.0.1:6006`；SSH端口转发后访问。选择`p21_fresh/attempt_000/training/tensorboard`查看P2.1，`no_time_prompt`查看旧P2，`p2_resume_gpu4`查看续训。

GitHub分支：[experiment/p21-text-lora-fresh-20261008](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/experiment/p21-text-lora-fresh-20261008)，初始实现提交`b07988d`；后续提交[8c7dd05](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/8c7dd057d844145b7281e3994adb0aab0009e14d)补充GPU证据、功耗监控和P2四卡续训。仓库仅保存源码和审计摘要，不上传模型/缓存。

14:44 UTC：24rank均已完成第4步，第3步安全checkpoint经CPU完整读取核验通过（全部模型/优化器有限、optimizer step3、24份rank RNG、384个LoRA B全部更新、部署子集与完整状态逐项相等）。审计文件`STARTUP_CHECKPOINT_VERIFIED.json`；前三步耗时110.26/120.32/114.08秒。三机最新采样24卡利用率均100%、功耗约625–687W。这是工程运行检查，不代表收敛或生成质量达标。
