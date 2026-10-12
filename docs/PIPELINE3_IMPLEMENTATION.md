# Pipeline 3 完整流程与交接入口

更新：2026-10-11。P3不是P2.1 Qwen残差路线；规划器24,000步不等于DiT训练24,000步。

## 交接阅读顺序

1. [总索引](../README.md)：各Pipeline现状、历史实验、HTML选项对应权重。
2. 本页：输入输出、模型结构、loss、冻结边界、代码入口。
3. [C0→A运行手册](PIPELINE3_STAGED_RENDER_PLAN_20261011.md)：本次配方、队列、监控、恢复、停止和验证边界。
4. [P3多机说明](PIPELINE3_MULTINODE_20261009.md)、[全量规划器训练](PIPELINE3_FULL_TRAIN_20261009.md)：上游历史工程证据，不能代替本次C0/A实测。
5. 结果合集说明（主工作区docs/RESULT_COLLECTION_20261011.md）：HTML查看和对比；旧P3视频是step17000规划器＋Connector100＋冻结DiT。
6. 实际运行目录的plan/state/health、当前日志和checkpoint完成标志：交接时的实时状态以它们为准。

本次根目录：`/public/2_CKPT/0_WM/llq/p3_full_connector_then_dit_20261011`。
固定源码（审查修复版）：`/private/liqi_workspace/project_code/audio-driven-v4-p3-staged-auditfix-20261011`。开发目录后续修改不会自动进入快照。

## 输入到输出

当前受控任务输入：Speaker已观测音频、Listener首帧、固定GT Reply／Delivery／Behavior／可选Vocal及时间表。**Pred25仅预测目标音视频特征，不代表自主预测回复文本或时间。** 这与P2无时间提示实验不同。

1. 冻结感知编码器提取原生音视频特征。P3音频采用原生约25 Hz序列，不是P2的4 Hz池化缓存。
2. Omni接收观测条件、目标位置MASK或回填特征及计划。两路特征Connector和Flow Matching预测头学习目标感知特征。
3. Oracle回填真实目标感知特征；Pred25从全MASK起步，25轮预测回填。固定规划器再前向，读取 `hidden_states[-2]`，分别取目标音频／视频位置。
4. 生成Connector映射到LTX两路条件维度，与Gemma文本条件沿token维拼接，进入原DiT text cross-attention。没有独立Qwen残差attention。
5. LTX-2.3以Listener首帧为条件，原生调度生成音视频latent，冻结VAE/vocoder解码为带声音的视频。

生成不加载GT目标VAE latent。Oracle仍读取GT感知特征，只作上界诊断；Pred25不读取目标感知特征作为模型输入。两组均保留固定GT回复和时间表。

## 阶段、目标与梯度

|阶段|更新模块|冻结边界与目标|
|---|---|---|
|P：上游规划器，历史已完成|特征Connector、两路FM预测头、MASK等；LoRA组还更新Omni Thinker LoRA|感知塔和原始基座冻结；Frozen对照额外冻结Thinker；感知特征FM，无LTX AV loss|
|旧渲染pilot，历史已完成|生成Connector|Train24训练100步；Omni和DiT冻结，AV梯度穿过DiT到Connector，规划hidden detach|
|C0，本次排队|生成Connector|Train91,009，2000步；冻结step17000规划器与原始DiT；AV loss更新Connector|
|A，本次排队|Connector＋DiT text cross-attention LoRA|C0后独立1000步；Omni、其规划头和其余DiT冻结；同一AV loss更新两者|
|B，未实现／不自动启动|拟在线Omni LoRA＋规划头＋Connector＋DiT LoRA|拟联合AV与规划FM；须在线前向并允许AV回传Omni，旧hidden缓存不能代替在线计算|
|C，可选后续|拟扩大DiT self/AV attention LoRA|仅在受控验证支持后决定；未开展|

AV loss是在LTX音视频latent加噪后的速度预测误差，沿用原生strategy的权重、有效区域和首帧遮罩，不是波形或像素MSE。分别记录总AV／video／audio；C0/A没有新增规划特征辅助loss。

A LoRA位于每个block的视频 `attn2`、音频 `audio_attn2` 的Q/K/V/输出投影，rank8/alpha8、零增量初始化；不是全DiT训练。Connector LR3e-5，DiT LR5e-6、50步warmup。感知塔、Gemma、VAE/vocoder始终冻结。

## 数据与缓存

- Train91,009／Dev48严格按split和sample_id关联。运行根目录manifest记录原始清单、文本、VAE、特征与初始权重哈希。
- 原生特征：`/public/2_CKPT/0_WM/llq/p3_full91009_dev48_cache_20261009/index.json`。
- 新readouts只缓存冻结step17000规划器的Oracle／Pred25最终hidden，不替换旧原生特征。
- 按全局曝光序号确定性各50%选择Oracle／Pred25；恢复保留顺序、条件选择和噪声seed。
- 保留原始720p-area宽高比，只沿用时间末端对齐，不裁方、不空间降采样；长视频原生分块解码。
- 全量91057条两种条件逐项身份／哈希审核，发布readouts/complete.json后才训练。

## 本轮工程证据

本机8H200已完成16条短／长样本条件缓存，以及C0/A各8份独立真实运行：每份更新2步、完整保存后严格恢复1步，总计48次更新。短样本约7秒／步，长样本约107秒／步；不能当作24卡global batch72的吞吐估算。

13项CPU测试通过：真实tiny LTX梯度/冻结、恢复下一次更新一致、双rank梯度累积、数据契约、依赖完成判定、空卡分配和只清理子进程组。

本机测试后另修补多rank共享输出目录检查竞态，以及最终step已保存但结束验证失败时只补验证/完成记录的恢复路径。新快照必须再通过队列内24卡更新/恢复和24条原生采样解码门禁；不能把8份单卡成功写成24卡DDP已通过。

## 代码导航

|模块|入口|
|---|---|
|原规划器／原生FM|src/omni_av_p3/；scripts/pipeline3_parallel.py|
|C0/A参数|configs/pipeline3_staged_render.json|
|清单／缓存|pipeline3_staged_prepare.py、pipeline3_staged_cache.py|
|生成训练模型／检查点|src/omni_av_p3_staged/model.py、checkpoint.py|
|DDP／TensorBoard／Dev loss|scripts/pipeline3_staged_train.py|
|队列／GPU检查／子进程隔离|pipeline3_staged_queue.py、pipeline3_staged_worker.py、orchestration.py|
|20分钟监控|scripts/pipeline3_staged_watch.py|
|原生CFG／C0质量检查|pipeline3_staged_generate.py、pipeline3_staged_quality.py、sampling.py、media.py|
|测试|tests/test_p3_staged_render.py、test_p3_staged_queue.py|

交接时先读实时状态、日志及GPU进程，不重复启动队列，不把旧告警当成当前故障。只处理本目录所属任务。24卡检查点禁止静默改卡数恢复。

Dev48 AV loss和Dev8视频评分不可互相替代。采样固定原生Euler30、CFG video3/audio7，不复用已淘汰的旧采样。完整路线包括未来B/C，不代表所有阶段已实现。

本轮部署证据：GPU/CPU/队列记录（运行根目录与主工作区docs/PIPELINE3_STAGED_DEPLOYMENT_20261011.json）。记录为快照，实时状态以运行目录为准。
