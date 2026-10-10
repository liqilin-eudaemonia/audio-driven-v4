# Pipeline 3 全量三机训练：目标 step1000

用户指定使用完整 Train91,009／Dev48，三机训练并监控 GPU／训练速度。启动日期2026-10-09。本文件记录本次运行，不将早期Train16工程测试当作正式全量训练。

## 数据和配方

- 完整复用 `outputs/p2_full91009_dev48_20261008/{train,dev}.json` 的样本清单、GT与媒体，重新提取P3特征，不复用P2融合hidden或池化teacher。
- Train91,009，Dev48；训练时长3.139–24秒，均值约11.864秒。保留37条无文字回复训练样本；不使用旧8秒／必须有回复的筛选，不重新人工打标。
- 本轮仍是固定GT回复及目标时间表的Bernini式MASK规划研究；不训练LTX，不声称实现了自主时长预测。
- 主路线Thinker LoRA；原生感知塔／embedding冻结，特征Connector与AdaLN decoder训练。宽度512、深度16、噪声重复4次、头与LoRA学习率均1e-4，保持既有目标与结构。
- 从预训练Qwen2.5-Omni-3B及新头／LoRA初始化训练到总step1000，不加载Train16试验权重。24rank，实测每卡batch1/2/4、分别累积4/2/1次，始终global batch96；1000步共96,000次样本曝光，约1.055遍训练集。
- 每100步保存完整状态并在固定Dev48上验证；验证是全MASK特征FM与打乱前缀对照，不是渲染视频评分。

## 机器

`configs/p3_full_three_nodes.json`：node17/172.29.15.126为master，node12/172.29.252.24与node16/172.29.112.17为worker，各GPU0–7。启动前24卡均空闲，用户明确授权使用。172.29.104.194上的其他训练／推理任务不参与本次计算。

## 位置与运行链路

|产物|路径|
|---|---|
|完整P3缓存|`/public/2_CKPT/0_WM/llq/p3_full91009_dev48_cache_20261009`|
|当前根目录|`/public/2_CKPT/0_WM/llq/p3_full91009_1000step_20261009_perf`|
|正式训练输出|根目录下`training/`|
|完整检查点|`training/step_000100.pt`等，带`.complete.json`校验标记|
|TensorBoard|`training/tensorboard/`|
|分节点启动日志|`training_logs/`|
|配对测速|`benchmark_reference/`、`benchmark_optimized_b1/`、`benchmark_optimized_b2/`、`benchmark_optimized_b4/`、`benchmark.json`|
|当前启动记录|`outputs/p3_full_training_20261009/launch_three_nodes_restored.json`|
|GPU与进度采样|`outputs/p3_full_training_20261009/monitor_three_nodes/telemetry.jsonl`|

`pipeline3_full_cache.py`保留96个时长平衡的不可变分片，支持多个进程通过文件锁领取同一分片中的不同样本。提取并发从每GPU4路增至8路，后增至12路。node12被LCOT占用时，剩余提取临时映射到node17/node16的16卡、192进程；已经完成的记录不重复提取。每个样本发布独立SHA256记录；中断文件保留，不覆盖历史数据。只有91,057条全部齐备且身份／split对应后才发布完整索引。训练器首次读取时再次检查样本文件哈希。

特征提取只需视觉塔、音频塔、embedding lookup和MRoPE。加载原权重后在内存中释放不参与提取的Thinker blocks／LM head，真实模型对照确认输出embedding逐元素一致。额外测试24秒完整样本和无文字回复样本通过。提取输入仍是完整Speaker音频和Listener首帧，目标使用原生2FPS视觉特征与约25Hz音频特征。

`pipeline3_full_run.py`是独立后台监督进程：等待完整缓存 → 四组三机20步测速 → 选择更快配置 → 新初始化正式训练1000步。基准失败会停止；可选优化组失败则记录原因、释放该组进程后比较其他成功组，不选择失败组。缓存不完整不以部分缓存冒充全量训练。原始失败记录保留。训练完成写`COMPLETED.json`。没有无限自动重试或重置失败状态。

## 性能优化与衡量

基准和优化组均global batch96、同一初始化、相同全局样本与噪声集合、20次更新；前5步预热，后15步统计平均／中位更新耗时和样本/秒。优化组增加CPU预取与同一个global batch内的长度负载平衡，仅改变样本分配到rank/microstep的方式，不更换训练样本、noise seed或损失权重。浮点累加顺序改变，因此不声称逐位等价。

四组都沿用BF16、fused AdamW、不做激活重计算、RoCE和有界GPU缓存。新增真正的多样本Thinker合批：分别填充prefix和target，保留每个样本的MRoPE位置与有效mask；FM按样本分别归一化后取均值，保留原noise seed。选择平均每步耗时更低的一组开展正式训练，测速结果不混进正式模型训练轨迹。每步记录loss、grad norm、更新耗时、数据等待、rank本地耗时和峰值allocated显存；TensorBoard记录loss、Dev、吞吐和数据等待。

每10秒通过SSH读取所有GPU的利用率、显存、功率、温度，并记录缓存进度、各rank训练步数及退出情况。SSH复用每个节点的连接；提取使用每GPU一个父进程管理多个worker，启动等待握手确认，降低连接风暴；预检对SSH 255暂态断连有限重试。以有效样本/秒、rank等待及显存余量判断瓶颈，不把单次GPU利用率100%等同于高吞吐。缓存阶段包含媒体读取／视频解码，其利用率与正式训练分开汇报。

缓存生产记录中，4路/GPU时约54样本/秒，8路/GPU时约80–84样本/秒。这是不同时间段的实际吞吐，样本时长有变化，不能作为严格同工作量加速比。后续正式训练以配对测速记录为准。

最新回归：`test_p3.py`、`test_p3_acceleration.py`、`test_p3_distributed.py`、`test_p3_cache_concurrency.py`共23项通过。新增冻结/LoRA可变长度合批与逐样本loss/gradient/noise等价检查、竞争领取样本不重复检查；四CPU rank batch2×accumulation2与串行参考检查通过。GPU batch2/4的速度及显存上限由上述真实测速决定。

验证：20项P3相关CPU测试通过；新增预取与同步数据一致性、缓存上限、负载平衡保持样本／noise序号、减少构造长短样本偏斜的检查。四CPU rank使用可变长度tiny Omni、预取及负载平衡完成4步，与同一global batch串行参考最大参数差2.98e-8。真实三机优化速度及正式训练进度以运行结果为准，尚未完成时不提前宣称通过。

## 状态与边界

**最新状态：正式三机24卡LoRA训练已正常完成step1000。** 24rank最终权重一致，`stopped=false`，无失败记录。`training/step_001000.pt`及完整校验标记、`dev_001000.json`、`completed.json`均已保存，根目录`COMPLETED.json`确认监督流程结束。正式1000步平均0.619秒/步；去掉前5步平均0.617秒/步、155.59样本/秒。上述更新耗时不包含周期保存/验证，不能直接当作完整墙钟耗时。[最终结果与源码核对](PIPELINE3_FULL_TRAIN_FINAL_20261010.json) · [GitHub备份](PIPELINE3_GITHUB_BACKUP_20261010.md)。

最终Dev48正确条件video/audio FM loss为1.67795/0.95597，打乱条件为1.67852/0.97594。音频对条件存在差异，视频差异很小；单次特征损失对照不能证明生成质量或泛化达标。本轮没有运行LTX渲染评估。此前[step100快照](PIPELINE3_FULL_TRAIN_STEP100_20261009.json)保留为历史记录。

### 三机实际测速结果

完整缓存已发布：Train91,009／Dev48，634,051,336,705 bytes；索引SHA256 `dab1ab1cf06c9cdd2bb8d84cafec7409932c799e49a6dee71a9926a2a823bf52`。

每组均三机24卡、20步，丢弃前5步预热，后三组启用预取与负载平衡。逐步核对四组全部rank的全局样本ID／noise seed集合完全一致；四组均24rank权重一致、Dev48验证完整、无运行失败。

|配方|平均秒/步|样本/秒|24卡中最大allocated显存（GiB）|
|---|---:|---:|---:|
|基准 batch1 × 累积4|0.9775|98.21|23.66|
|优化 batch1 × 累积4|0.8805|109.03|23.32|
|优化 batch2 × 累积2|**0.8166**|**117.56**|34.61|
|优化 batch4 × 累积1|0.9390|102.24|57.51|

正式训练选择batch2×累积2，吞吐相对基准提高19.7%；不按显存占用最多选择batch4。测速是短窗口结果，正式训练持续记录长窗口吞吐／GPU利用率。每组均从同一初始化开始，但浮点计算及累加顺序不同，不宣称权重逐位等价；Dev数值接近并不代表已学会利用条件。

对应根目录`benchmark.json`保存完整候选指标和选择，`training_command.json`保存正式1000步命令。正式运行从新初始化开始，不加载测速权重。

2026-10-09后续恢复：用户明确要求停止node12的LCOT任务及自动拉起程序，继续三机训练。已关闭对应专用tmux pane `%604`（shell PID3888160，任务`repeat_r1_h200_20261009_142959`），该任务启动器及LCOT GPU/编译进程均已退出；再次检查LCOT进程为0，8卡均1MiB空闲显存。未改动或删除其他项目文件。三机监督与监控已重新启动，记录见上表。旧根目录中的SSH失败／GPU占用预检失败记录保留，不代表正式训练已更新。

历史初始状态：全量缓存准备已启动，训练监督进程等待完整缓存；启动记录中包含各后台PID和完整命令。该历史记录已被上面的完成状态取代。原始监控日志保留；2026-10-10的代码备份与训练结果归档见[备份说明](PIPELINE3_GITHUB_BACKUP_20261010.md)。

启动检查已完成：[随仓库保存的启动证据](PIPELINE3_FULL_TRAIN_STARTUP_20261009.json)。额外抽查24路提取进程的样本文件哈希、GT身份、原生目标维度／有限值和布局均通过；真实长样本LoRA累积4次、更新4步的检查通过，峰值allocated显存约23.63GiB，所有规划头／LoRA梯度有效且基座不变。该检查与本任务的缓存进程共用node17 GPU0，仅检查正确性和显存，不用于吞吐对照，检查进程已退出。启动证据快照时缓存已超过28,000条，近期约46条／秒；尚未开始正式optimizer更新，后续实际状态读取实时日志和根目录阶段标记。

全量训练只有约一个epoch，step1000完成不代表规划泛化已成立。下一步应检查Dev正确／打乱条件差异及条件消融，再决定增加训练预算或接入生成Connector／LTX。
