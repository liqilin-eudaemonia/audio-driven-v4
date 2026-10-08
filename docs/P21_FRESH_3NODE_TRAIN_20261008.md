# P2第200步保存后切换P2.1：独立从头训练与15分钟监控

用户要求：P2冻结版到200步保存后停止；P2.1独立训练，不加载P2第200步权重。新运行根目录：`/public/2_CKPT/0_WM/llq/p21_fresh_3n24g_1000step_20261008`。本文开始记录时仍在等待旧训练完成，实际阶段读取根目录`state.json`，不得将目标步数当作已完成步数。

## 对照和初始化

P2.1仍使用原始LTX-2.3预训练底座，Connector和两个辅助decoder按P2相同seed45、相同构造顺序重新初始化；为防止加载DiT消耗随机数，Extractor在renderer之前构造。完整公共模块初始SHA256：`12e4514e2bb0af062d9be47f9f5a659fbcac96c04a62d678a9dda8b9a41ba982`（223,175,680个参数）。LoRA A随机、B为零，优化器/步数从零开始；启动命令不包含`--init-p2`或旧checkpoint。恢复故障时只允许使用本次P2.1自己的完整checkpoint。

仍复用无时间提示Train91,009缓存、原720p-area视频形状；三个节点各8卡，global batch72、累积3、样本顺序与噪声种子沿用P2。rank8/alpha8 LoRA仅覆盖原生视频attn2和音频audio_attn2的Q/K/V/out投影；Extractor LR1e-4、LoRA LR1e-5，其他DiT参数冻结。1000步目标，每100步保存；另外第3步保存一次启动安全checkpoint，尽早具备故障恢复点。不运行Dev loss或验证集推理。

## 停止和切换

`scripts/stop_training_at_step.py`观察P2日志到step200后写入该运行的`training/STOP`。P2在下一轮检查STOP之前会完成第200步周期保存，再正常退出；避免强杀写坏权重。需验证完整checkpoint、Connector导出、24rank RNG、完成标记和24rank最终参数一致性后，才写新运行根目录`P2_CHECKPOINT_VERIFIED.json`。

`scripts/pipeline21_supervise.py`只在P2完成、源码检查READY、checkpoint验证通过、三机GPU空闲后启动P2.1。每次尝试写到`attempt_000`等独立目录，训练目录`training/`、权重`checkpoints/`、集群日志`agents/`。代码原文保存在`source_snapshot/`，源哈希和测试结果在`READY.json`。

## GPU监控与异常处理

监督脚本每15秒采样三台机器的GPU利用率/显存/功耗/温度，每900秒汇总一次健康检查。检查逐rank进度、loss有限性、长时间无新step、连续低利用率、节点失联和权重磁盘空间；低利用率使用多次采样均值，不因保存权重时的一次下降触发重启。

- `gpu_telemetry.jsonl`：原始GPU采样。
- `health.json`、`health_history.jsonl`：最新与历史15分钟报告。
- `state.json`：当前phase、attempt、PID和逐rank步数。
- `alerts.jsonl`、`events.jsonl`：告警、通知投递、启动/恢复动作。
- `supervisor.json`：守护PID、检查频率、恢复上限。

告警通过本机`codex queue --thread ... --message ...`排入当前Codex任务，由当前任务读取最新状态后处理。该CLI通道已经实际排队自检；后续主动处理依赖本机Codex app-server和任务仍可用，投递失败会写入事件记录。没有向其他人或外部聊天发送消息。

自动恢复最多两次，只处理已有完整checkpoint时的明确通信/IO故障，或节点可达且持续超过15分钟无训练进度。仅终止本次launcher所拥有的guard/torchrun进程组，等待所有指定GPU空闲后，使用最新P2.1完整状态恢复剩余步数，保留旧attempt日志。OOM、非有限loss、契约错误不会盲目重跑，告警后等待诊断。低利用率本身只告警，不修改超参数或重启。

用户创建新运行根目录`STOP`可要求保存后停止本次训练/守护；`DISABLE_AUTO_RECOVERY`关闭自动恢复。不会删除数据缓存或其他任务。

## 验证范围

当前CPU回归105项通过，包含P2/P2.1/P3既有检查、公共模块初始化一致、双进程LoRA梯度/恢复、精确停止、误报过滤及恢复白名单。三节点当前源码哈希一致。实际完整GPU训练、资源利用与第3步安全checkpoint验证会在启动后补充，不以CPU结果替代。
