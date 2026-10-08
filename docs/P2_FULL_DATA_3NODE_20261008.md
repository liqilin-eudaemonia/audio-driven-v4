# P2 全量数据、三机联合测试与负载均衡

2026-10-08。当前正式数据清单改为 **Train 91,009 / Dev 48**。本页取代前一轮 Train128/Dev24 的试验选择；旧产物保留。权重统一放在用户指定的 `/public/2_CKPT/0_WM/llq` 下，各实验使用独立子目录。

## 数据与空间尺寸

- 清单：[Train](../outputs/p2_full91009_dev48_20261008/train.json)、[Dev48](../outputs/p2_full91009_dev48_20261008/dev.json)、[源数据审计](../outputs/p2_full91009_dev48_20261008/audit.json)。Dev48 保留原 Dev24，再从其余 qualified Dev 确定性增加24条。Test不参与训练或调参。
- 全部91,009条 GT 可用性、媒体路径和文本转换已检查；取消8秒筛选，按视频长度分配8/16/24秒预算，各有20,836 / 52,358 / 17,815条。源媒体不因预算而截断。
- 使用现有 `processed_pairs_720p_area` 视频和 `ltx23_precomputed_720p_area_qualified_20260920` VAE latent。**保留现有视频宽高，不新增正方形裁剪或缩小 LTX 训练目标。**没有启用 `--tiny`。
- 720p-area 是约720×1280的面积预算，按长宽比缩放并对齐到32倍数，不是强制1280×720。Train有103种尺寸，面积860,160–983,040像素。实际视频头抽查包括宽×高 `768×1184`、`288×3072`、`1632×576`。
- 全部91,009条Train和48条Dev的清单尺寸与VAE metadata一致：[空间审计](../outputs/p2_full91009_dev48_20261008/full_spatial_audit.json)。原数据自身有656条正方形视频，不等于训练新增了正方形裁剪。
- `TrainingSamples` 在读取实际 latent 后断言其 H/W×32 与样本目标尺寸相符；逐rank训练日志记录实际 latent shape。例如 `[1,128,30,42,22]` 对应宽704、高1344，空间尺寸保持不变。
- Omni 感知/辅助特征支路使用自身的小尺寸输入和时间采样；它不是 LTX 的720p-area目标，也不会替换生成塔的原尺寸latent。
- 修复了VAE按30FPS对齐后比原音频多15–31ms的边界：辅助teacher仅在 `min(GT真实时长, VAE窗口)` 内取样、计算有效mask，padding不作为真实监督。没有放松speaker因果音频的越界检查。

[全量实验配置](../configs/p2_full91009_dev48_experiment.json)记录正式数据路径、保存/验证周期和空间约束。**全量P2条件缓存尚未准备完成，91,009条正式训练尚未启动**。本轮完整模型联合门禁采用独立24条代表性Train（8/16/24秒各8条），Dev使用完整固定48条；不能把门禁写成全量训练完成。

## 三台物理机器

|节点|IP|全局rank|GPU|
|---|---|---|---|
|node12 / js-public，本机/master|172.29.252.24|0–7|8×H200|
|node17 / avgen-2|172.29.15.126|8–15|8×H200|
|node16 / avgen-1|172.29.112.17|16–23|8×H200|

[集群配置](../configs/p2_three_nodes.json)与[SSH启动器](../scripts/pipeline2_cluster.py)。先对全部节点预检，成功后共同启动。每节点使用独立日志，SSH/控制器结束时仅清理本次启动的进程组；不使用按配置名匹配的全局kill。网络参数沿用参考脚本的eth0/mlx5_0/GID3配置。

首次资源预检发现本机lcot任务占用，未启动冲突任务；用户随后手动停止，才继续。最初远端准备有一次SSH握手被重置，重试成功。首次轻量probe训练本身完成，但守护器退出时stdin daemon线程触发Python退出错误；改为select监视stdin后，三个节点均正常返回0。失败日志保留。

## 已完成的真实GPU验证

- **三机24rank NCCL probe及恢复通过**：非整除验证集23条分给24rank，计数准确；参数在24rank间一致；串行global-batch参考与DDP更新一致。真实hostname确认是三台物理机器。
- **完整预训练LTX、原尺寸3步joint门禁通过**：24rank同步更新同一Extractor，冻结renderer审计通过，各rank参数SHA一致。不是三个独立训练。
- **Dev48周期验证通过**：step0 / step2 / step3，固定噪声与GT text，loss分别为6.16529 / 6.01590 / 5.89836。短门禁loss下降不表示生成质量通过。
- **真实保存与恢复通过**：step2恢复到step3，loss与连续训练一致；Connector相对L2差5.59e-6，两个decoder约1.7e-9，非逐位一致。24份rank RNG完整保存。
- **TensorBoard实际曲线已核验**：`train/*`、`dev/*`和时间/吞吐等标签均有真实GPU训练事件，不再只是CPU probe。
- 完整门禁峰值显存约65.6–115.7 GB/卡；最长24秒样本通过，无缩图或裁正方形的OOM兜底。

GitHub 可直接阅读的[三机结果摘要](P2_3NODE_RESULTS_20261008.json)包含下述原始 JSON 的内容、SHA256、运行配置与样本清单哈希。`outputs/` 不上传，以下原始产物链接仅在本机有效。

证据目录：[`outputs/p2_3n24g_20261008`](../outputs/p2_3n24g_20261008/)。主要文件：`probe_v2/completed.json`、`probe_resume/completed.json`、`full/completed.json`、`full_resume/completed.json`、`full_resume_audit.json`、`full/tensorboard_audit.json`。

完整模型权重：

```text
/public/2_CKPT/0_WM/llq/p2_3n24g_20261008_full/extractor_step_000002.pt
/public/2_CKPT/0_WM/llq/p2_3n24g_20261008_full/extractor_step_000003.pt
/public/2_CKPT/0_WM/llq/p2_3n24g_20261008_full_resume/extractor_step_000003.pt
```

每个完整checkpoint约2.5GiB，另有约129MiB Connector-only导出及 `.complete.json`。正式默认每100步保存/验证；本轮门禁改为每2步，以验证中途保存逻辑。

## 用户指出的低功耗/低温问题

问题成立：原门禁把短、中、长视频分别分配到三个节点，形成严重计算不均衡。实际观察短/中样本节点只有约120–140W，而长样本节点约620–680W，两边GPU-Util却都显示100%。短任务等待NCCL同步时，活跃的通信kernel也会计入GPU-Util；此前用100%判断有效计算充分不严谨。

原global batch24的三步分别105.97 / 104.98 / 102.43秒。本机恢复测试的短样本前向约3.78秒，反向（含同步等待）约101.32秒，说明低温的主要原因是负载分配不均；显存未占满本身不是性能错误。

新增 `--balance-workload`：在**同一个累积global batch内部**，按视频token数平方估算工作量，用最长任务优先的调度分配到各rank。全局样本ordinal、每个样本的噪声seed、loss权重、样本总数均保持不变，只改变rank归属和累积顺序。各rank的microbatch数保持相同，DDP归约语义不变。使用 `--accumulation 3` 时global batch为72，microbatch仍为1，不靠增加单次显存峰值解决等待。

两组配对性能测试使用同样的72次样本曝光、相同seed、同样global batch72、一次optimizer更新；分别关闭/开启balance-workload。结果已在同目录 `performance_comparison.json` 汇总，并收录至[随仓库保存的结果摘要](P2_3NODE_RESULTS_20261008.json)。单步短基准包含预热，不能视为长程稳定吞吐或质量比较。


### 配对性能结果

|指标|原始分配|按计算量平衡|
|---|---:|---:|
|同样72次样本、一次更新|313.655秒|162.943秒|
|吞吐|0.22955 次/秒|0.44187 次/秒|
|node12平均单卡功耗|201.1W|651.5W|
|node17平均单卡功耗|363.6W|648.3W|
|node16平均单卡功耗|612.7W|630.6W|
|最大 allocated 显存|114.81GB|113.91GB|

吞吐提升 **1.925×**，更新耗时减少约48%。两组样本ID、噪声seed、latent形状逐项匹配，global batch均72；loss差8.88e-16。24个rank在各自运行内的参数SHA一致。不同累积顺序/设备分配产生小幅浮点差异：Connector相对L2差5.51e-5、max abs约1.99e-4；decoder相对L2约1e-8。不能称逐位等价，也不据此推断长程学习效果。

[完整比较](../outputs/p2_3n24g_20261008/performance_comparison.json)包含采样时间窗与功耗统计；原始功耗记录为同目录 `gpu_power_telemetry.jsonl`。这是各1次完整更新的配对短基准，计入更新内预热，不含加载模型和保存权重。显存/温度只用于诊断，优化结论以实际吞吐及减少等待为依据。

当前回归共 **51项通过**。正式运行建议沿用 `--accumulation 3 --balance-workload`；三机24rank对应global batch72。全量条件缓存准备完毕后，传入全量cache索引，不能把本轮24条门禁cache当成91,009条数据。


## GitHub 备份与复现

- 分支：[backup/p2-3n24g-balanced-20261008](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p2-3n24g-balanced-20261008)，从 `51e18fe` 分出，代码快照为 [38ceb34](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/38ceb349657289c2c76e08ae5a5c68b3a7f57b15)；[总索引](../README.md)与[变更记录](../CHANGELOG.md)记录版本关系。
- 本次备份覆盖源码、脚本、配置、测试、文档与精简结果，不含媒体、条件缓存、模型权重或 TensorBoard 原始事件。外部 Bernini/LTX 源码保持只读。
- 实测解释器：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`；Torch `2.9.1+cu128`、Transformers `4.57.6`、TensorBoard `2.21.0`，使用该环境已有的原生 `ltx_core/ltx_trainer`。
- [摘要](P2_3NODE_RESULTS_20261008.json)保存各次实测的 `run.json` 参数、模型配置、数据身份和原始文件哈希；早期八卡摘要中的源码哈希只代表当时版本，不代替本轮代码。

在同一共享盘环境下，可用保留的 Train24/Dev48 缓存复现原尺寸3步门禁（新运行前须确保三机空闲，修改下面唯一运行名；启动器拒绝覆盖旧输出）：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v4
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
P2_RUN=p2_3n24g_backup_repro_001
"$P2_PY" scripts/pipeline2_cluster.py \
  --cluster configs/p2_three_nodes.json --log-dir "outputs/${P2_RUN}_agents" -- \
  --cache outputs/p2_3n24g_20261008/cache/train/cache_index.json \
  --validation-cache outputs/p2_3n24g_20261008/cache/dev/cache_index.json \
  --output "outputs/$P2_RUN" --checkpoint-dir "/public/2_CKPT/0_WM/llq/$P2_RUN" \
  --network-config configs/p2_roce.json --run-id "$P2_RUN" \
  --recipe joint --steps 3 --accumulation 1 --save-every 2 --validate-every 2 \
  --allow-terminal-trim --tensorboard
```

上述 `--allow-terminal-trim` 显式接受已有 VAE 时间对齐裁剪，不是新增空间裁剪。恢复时使用新的运行名/输出目录，增加 `--resume /public/2_CKPT/0_WM/llq/<原运行名>/extractor_step_000002.pt`，设 `--steps 1` 从step2续训至step3，保持同一缓存、world size、累积数与模型配置（`--steps` 是本次追加更新数）。

复现负载均衡配对短测时，使用两个新运行名，移除 `--validation-cache`，改为 `--steps 1 --accumulation 3 --cache-samples 3 --save-every 100 --validate-every 100 --seed 45`（保留 TensorBoard）；一组不传 `--balance-workload`，另一组增加该参数，其余一致。72次曝光来自24个不同样本各重复3次，不能表述为72条不同训练数据。逐rank记录含样本ordinal、噪声seed和latent尺寸；计时不含模型加载和checkpoint保存。

全量清单生成及缓存分片/合并入口见[数据训练文档](P2_DATA_MULTINODE_20261008.md)。正式配置以[Train91,009/Dev48实验配置](../configs/p2_full91009_dev48_experiment.json)为准，旧文档的128/24为历史小集；必须完成全量缓存并核验覆盖后再切换正式训练输入。
