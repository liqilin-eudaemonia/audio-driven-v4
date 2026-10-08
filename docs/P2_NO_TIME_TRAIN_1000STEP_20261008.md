# 无时间提示缓存：三机24卡1000步训练（2026-10-08）

代码快照提交：[05e6489](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/05e64892daa1d8abd0e4e696a496368a37926271)（详细中文提交说明）；后续文档提交仅登记该固定代码版本。

用户要求使用新缓存开始训练，已于2026-10-08 08:38 UTC启动。当前正在运行，不能将目标1000步解释为已完成1000步。最新实际进度以逐rank日志和TensorBoard为准。

- Train91,009，缓存：`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_no_time_cache_20261008/train/cache_index.json`；启动前核对完整审计、索引SHA256及当前prompt contract。
- Omni user message不含结束时间、时长或时间戳，协议`observation_no_time_v1`；训练与推理共用模板。Omni仍冻结，不新增实时轮次规划训练。
- node12/node17/node16各8卡，共24rank；joint，microbatch1，累积3，global batch72，seed45，工作量均衡。
- 从新Connector和辅助decoder初始化开始，未恢复旧29步权重。冻结Omni/Gemma/VAE/完整LTX2.3；原始720p-area宽高与比例，不新增正方形裁剪。
- 目标1000步，每100步保存完整状态和Connector；不传validation-cache，无Dev loss与验证集推理。
- 沿用重计算、fused Adam、device cache容量3。训练相关源码、脚本、配置原文已保存至本次运行`source_snapshot/`，哈希和精确命令见`launch.json`。

## 运行位置

根目录：`/public/2_CKPT/0_WM/llq/p2_no_time_full91009_3n24g_1000step_20261008_0838`。

|内容|根目录下路径|
|---|---|
|逐rank训练记录|`training/train_rank0.jsonl`至`train_rank23.jsonl`|
|权重|`checkpoints/extractor_step_000100.pt`，每100步递增，另有Connector导出和完成标记|
|TensorBoard事件|`training/tensorboard/`|
|三机日志与预检|`agents/`|
|GPU利用率/功耗/温度|`gpu_power_telemetry.jsonl`|
|训练控制器|`controller.json`，PID2292154|
|GPU监控|`monitor_controller.json`，PID2292155|

首个周期checkpoint尚未产生前，只能声明保存配置已启用，不能声明100步权重已存在。安全停止：创建`training/STOP`，所有rank在当前更新后共同停止并补存最后一步；不需要强杀进程。

## TensorBoard

本次服务仅监听node12的127.0.0.1:6006；同一页面同时显示`no_time_prompt`（新训练）和`legacy_time_prompt`（旧29步）。已确认仅替换本任务原有服务，未操作其他TensorBoard。客户端执行：

```bash
ssh -N -L 6006:127.0.0.1:6006 root@172.29.252.24
```

打开http://127.0.0.1:6006，选择`no_time_prompt`。服务与PID记录在`tensorboard_service.json`。


## 首步实际核验

24rank完成step1，loss6.392409880956014、grad_norm13.257633209228516，用时108.73125980235636秒；72条样本互不重复，ID和noise seed逐项匹配新缓存全量清单。所有rank的loss/grad_norm一致，冻结renderer运行时检查通过。零初始化输出投影使首步层融合梯度为0属于预期，不以首步证明所有内部层梯度已非零。

TensorBoard真实新曲线`no_time_prompt/.`已读取step1 Train loss，无Dev标签。随后GPU采样24卡利用率97–100%，三节点平均单卡功耗约652–671W；这是启动窗口，非长期吞吐保证。机器可读证据为`startup_verified.json`及`tensorboard_verified.json`。当前仍在后台训练，尚未达到100步首次周期保存。
