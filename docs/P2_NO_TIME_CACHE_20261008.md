# Omni 无时间 user message 与条件缓存刷新（2026-10-08）

用户明确要求修改训练代码，在user message中不提供时间信息，并删除、重新提取受影响的hidden缓存。本次仅修改提示及对应缓存协议；未实现新的实时规划训练，也未启动新的正式训练。

## 当前输入与兼容性

固定user message：

```text
Plan a context-specific listener response based only on the provided speaker audio and listener reference image.
```

System prompt保持原有Reply/Delivery/Behavior要求。训练条件提取和自主推理均调用`prepare_observation`；不再把speaker_end_s、output_duration_s或其他数值时间拼入user message。音频读取仍使用已有观测范围，输出布局仍使用内部窗口长度，因此本改动不是流式或任意前缀的轮次预测实现。缓存计划仍来自GT assistant续文；Omni仍冻结。

P2 contract增加`observation_prompt`：policy=observation_no_time_v1，并存储system/user prompt的SHA256。旧缓存与旧权重的contract会被当前训练/推理入口拒绝；不能直接将29步旧权重恢复为同协议训练。结构未改变，但语义条件已变化。

## 路径与资源

- 新缓存根目录：`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_no_time_cache_20261008`。
- 完成后的Train索引：`train/cache_index.json`，91,009条；Dev索引：`dev/cache_index.json`，48条。
- 旧根目录：`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008`。
- 初始控制器PID：2265632，08:11:35 UTC启动；断点恢复控制器PID：2276761，08:20:34 UTC启动；三机24卡，每卡6worker，共144并发槽。Train192分片、Dev8分片。
- 进度：新根目录`progress.json`；日志：`logs/`及根目录旁同名`.launcher.log`。
- 原始命令/PID：根目录旁同名`.controller.json`；协议：`identity.json`；运行与源码快照：`run_*.json`、`source_sha256_*.json`。
- 全量已于08:25:00 UTC发布；独立审计于08:25:52 UTC完成，零失败；08:26:20 UTC核验三机GPU均无计算进程。

## 替换与删除策略

新路径重新计算Omni多层hidden及其前缀mask。输出坐标、输出mask、层ID与旧文件逐项相等后，复用旧teacher张量及window元数据；Gemma prompt核验后以硬链接复用文件，不重复计算或复制其存储。VAE及原始媒体不修改。

每条新文件原子发布后，完整回读，验证身份、GT、数值有限性及形状；新condition与提取结果相等，teacher所有张量与旧文件逐项完全相等，才删除该条旧features文件。旧文件同时装有hidden和teacher，删除前teacher已保存在新文件内。删除保护限制到指定旧根目录、features_*.pt，并核对inode/size/mtime未被其他进程改变。

旧全局索引改名为`cache_index.superseded_no_time.json`保留来源记录，停止作为可用入口；`SUPERSEDED.json`指向新路径。旧Gemma硬链接及历史日志保留，不存在重新复制的Gemma占用。中断时已完成的新文件可复用，尚未成功替换的旧文件仍保留；不会因停止而全量重来。

## 验证

- CPU全套63项测试通过，包含时间文字独立性、提示哈希失效、真实GT身份核验、刷新恢复、磁盘写入失败保留旧文件、布局改变拒绝复用、删除边界及硬链接复用。
- 3条真实Dev GPU门禁通过：新hidden有限，输出布局不变，teacher回读逐项相等，Gemma复用；门禁未删除旧文件。门禁目录：`/public/2_CKPT/0_WM/llq/p2_no_time_refresh_gate_20261008`。
- Ruff错误类检查与git diff --check通过。
- 未进行新协议训练或生成质量验证；用户要求暂不推理验证集继续有效。


## 中断与恢复记录

初轮刷新约66,025条时，Train shard045 的 features_0447.pt 在写入后的立即回读中报告缺少zip central directory；复查新文件大小9,437,184 bytes，仍不可读，而旧源11,574,530 bytes可正常读取。其余worker由控制器保护性停止，已验证的新文件未删除、未重新提取。该现象证实新归档不完整，未据此断言共享存储的具体底层故障原因。

修补`atomic_save`：通过Python文件对象写入，flush和fsync完成后才发布硬链接；仍执行完整新文件回读。损坏新文件只有在旧源身份及内容校验通过后才隔离，允许单条恢复；读回失败可利用内存中的提取结果重写一次，无需重新运行Omni。添加归档损坏恢复及无旧源时拒绝丢弃的回归测试，完整63项通过。

实际先恢复shard045：447条新结果复用，27条提取（含失败样本），1个损坏新归档隔离；24.77秒完成，随后三机断点恢复。该分片恢复日志为`recovery_shard045.log`，历史失败状态在`failed_before_*.json`，每次运行源码哈希分别保留。被中断写入的partial文件和损坏归档仅作故障证据，不被索引引用。


## 最终完成结果

Train91,009、Dev48共91,057条全部完成；200个分片成功，精确覆盖原清单，无Train/Dev重叠。新索引已发布，旧索引所列91,057个features文件全部不存在。另有旧根目录resume_smoke中的2条旧提示副本，对照新全量文件确认同一example及teacher逐项相等后删除，记录为old_smoke_cleanup.json。旧根目录递归检查features_*.pt结果为0。

新teacher在每条替换时完整回读并与旧tensor逐项相等；新hidden每条校验有限性、形状和GT身份后才删除旧源。独立审计覆盖全部Train身份/GT/布局/时空mask/文本/旧文件删除/文本硬链接，并对每个Train分片首中尾共576条及Dev48再次数值检查，零失败。独立审计并非全Train逐元素数值复查；逐条数值与teacher相等验证发生在刷新阶段。final_audit.json的seconds包含等待提取完成，不能当作纯审计耗时。

新Train逻辑文件大小2,199,537,515,427 bytes（含复用Gemma硬链接）。首次08:11:35启动至08:25:00发布约13分25秒，包含一次保护性中断、修复和恢复，非纯GPU提取耗时。三机24卡已释放，没有重启训练。结果：final_summary.json、final_audit.json、gpu_release_verified.json；仓库同步摘要为[P2_NO_TIME_CACHE_RESULTS_20261008.json](P2_NO_TIME_CACHE_RESULTS_20261008.json)。
