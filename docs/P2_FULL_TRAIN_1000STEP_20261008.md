# P2 Train91,009 三机24卡正式训练（目标1000步）

用户在全量缓存完成并审计后授权启动；随后授权必要时提前停止，以对齐仅使用已观测信息的推理目标。2026-10-08 07:58 UTC 已在第29步安全停止，未完成原定1000步。

## 固定配置

- Train：91,009条全量缓存，索引 `/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/train/cache_index.json`；已核验与最终审计中的SHA256一致。
- 三机：node12 / js-public（172.29.252.24，master），node17 / avgen-2（172.29.15.126），node16 / avgen-1（172.29.112.17），各8张H200，共24rank DDP。
- 目标1000次optimizer更新；microbatch1、每rank累积3次，global batch72；开启按视频token计算量均衡的样本分配，seed45，joint配方。
- 使用冻结的完整LTX2.3、冻结Omni/Gemma/VAE；从新初始化的多层Connector和两个辅助decoder开始，未恢复工程门禁权重。
- 保留现有720p-area宽高和比例；不使用 `--tiny`。`--allow-terminal-trim` 仅接受既有VAE的末端时间对齐裁剪。
- 每100步保存完整训练状态、Connector导出与完成标记；第1000步也保存。
- **不传入validation-cache**，因此训练开始、中间及结束均不执行Dev loss，也不进行验证集推理。
- TensorBoard开启；重计算策略、fused Adam和device cache容量3沿用已通过三机真实模型测试的配置。

## 产物位置

|内容|路径|
|---|---|
|本次运行根目录|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704`|
|权重|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/checkpoints/extractor_step_000029.pt`；另有 `_connector.pt` 与 `.complete.json`，未到第100步|
|训练及逐rank日志|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/training/`|
|TensorBoard事件|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/training/tensorboard/`|
|三机agent日志及预检结果|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/agents/`|
|启动参数与源码哈希|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/launch.json`|
|实际worker配置|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/training/run.json`|
|控制器PID|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/controller.json`|
|主日志|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/launcher.log`|
|GPU功耗/温度/显存采样|`/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/gpu_power_telemetry.jsonl`|

`launch.json`包含精确命令、代码基线提交和未提交源码的逐文件哈希，便于锁定本次实际运行版本。原代码备份分支和缓存生成运行均保留。

安全停止：创建 `/public/2_CKPT/0_WM/llq/p2_full91009_3n24g_1000step_20261008_0704/training/STOP`。各rank在当前更新结束后的步边界共同停止，补存最后一个尚未保存的step；不会停止其他进程。无需按PID强杀训练。

读取 `training/train_rank0.jsonl` 查看已实际完成的step；初期GPU满载不代表optimizer已完成。完整终止结果为 `training/completed.json`；运行中不存在此文件是正常现象。


## 启动实测

24rank已完成首步：global batch72的样本ID与噪声seed逐项匹配全量清单，72条互不重复；所有rank的loss和梯度范数一致，冻结renderer运行时检查通过。首步loss6.39241，用时108.73秒，最高rank allocated显存110.55GB。TensorBoard已读取到step1及step2的真实Train loss；step2 loss6.23280。未出现Dev标签。第100步保存是运行配置，当前尚未到首次周期保存点。

首步训练期间三节点平均单卡功耗约650–663W，GPU-Util均100%；原始时序采样保留。上述为启动短窗口，不作为1000步稳定吞吐或质量结论。机器可读核验：运行根目录 `startup_verified.json`。


## TensorBoard可视化服务与早期耗时估计

TensorBoard2.21.0已启动在node12的 `127.0.0.1:6006`，HTTP页面和scalars API已实测返回训练曲线。客户端通过SSH转发查看：

```bash
ssh -N -L 6006:127.0.0.1:6006 root@172.29.252.24
```

随后打开 `http://127.0.0.1:6006`。自动审批拒绝了监听所有网卡的操作，因此使用仅本机监听。服务信息和PID保存在运行目录 `tensorboard_service.json`，核验结果为 `tensorboard_verified.json`。

截至第4步：排除首步后的平均步时108.70秒，按剩余996步线性估算约30.07小时，尚未计入后续checkpoint保存；早期样本少，不能视为精确完成时间。


## 提前停训与路线修正（2026-10-08 07:58 UTC）

用户要求判断是否应本版就采用“仅输入已知信息→预测回应计划与相对时间→条件生成”，并明确授权必要时立即停训。07:57:12 UTC 创建本次运行专用 `training/STOP`，当前更新结束后24rank共同在29步停止，保存完整训练状态和Connector导出。三节点agent退出码均为0；07:58:37 UTC 实查三台机器GPU均无计算进程。原始核验为运行根目录 `stop_verified.json`。

- 最后step29，loss=4.7221135431；训练完成29×72=2088次样本曝光，不代表完整epoch或质量通过。
- 完整权重：`checkpoints/extractor_step_000029.pt`（2,704,675,452 bytes），含优化器及各rank RNG状态。
- 部署权重：`checkpoints/extractor_step_000029_connector.pt`（160,448,492 bytes）。
- 24rank参数摘要一致；最终 `training/completed.json` 标记 `stopped=true`、`quality_pass=null`、`validations=[]`。
- TensorBoard事件及已有localhost服务保留，可继续查看历史曲线。全量缓存保持原样。

停训原因是任务与信息边界需要调整，不是已证实的数值计算故障。当前Omni冻结，GT文本计划拼入assistant续文提取条件，整个截取的说话人音频参与编码。这适用于已拿到完整输入片段的离线条件生成，但不能证明只凭当前前缀进行实时轮次决策的能力。删除prompt中的一句时间文字不会改变音频可见范围，也不会使冻结Omni学会新决策。

本版下一阶段应先确定观测截止点，以此之后的相对时间监督回应计划；未来真实终点仅可用于标签，不可作为观测输入。输出窗口作为系统预算保留。GT计划用于teacher forcing不等于偷看输入，但下游还需评估预测计划带来的误差。时间标签抽查已发现vocal_track/floor_track/visual_events，来源为teacher标注，时间精度仍需对照真实音频核验。该方案尚未实现、未开始重建缓存或重新训练。

兼容性：现有VAE和监督数据可作为复用候选；凡改变观测截止点、prompt或Omni权重，旧Omni hidden不能直接充当新协议条件。如果输出目标时间窗口也改变，需另外核验旧VAE与teacher特征的时间对齐，不能宣称全部可直接复用。保留本轮权重作为旧协议基线，新路线先完成小样本契约与梯度测试，再安排全量计算。
