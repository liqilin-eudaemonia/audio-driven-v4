# P2 全量条件缓存准备记录（2026-10-08）

**完成状态**：2026-10-08 06:58:03 UTC，Train91,009/192分片全部完成并合并，Dev48已完成。提取与合并从06:12:11开始，总耗时45分52秒，包含并发调优与断点恢复；Train缓存2,237,901,403,299字节（约2.04TiB）。三机24卡已释放，没有启动训练。下面调度与升级记录保留为过程历史。

用户指定缓存落盘至 `/public/2_CKPT/0_WM/llq`。本次只准备缓存，没有自动启动后续训练。当前配置为每卡6个进程（共144个任务槽）、Gemma batch4、每卡最多2个Gemma进程；Omni仍逐样本提取，由多个进程重叠CPU解码与GPU计算。Train91,009 清单、固定 Dev48、720p-area VAE 源数据不变；不新增空间裁剪。

## 位置

|内容|绝对路径|
|---|---|
|本次缓存根目录|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/`|
|Train分片产物|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/train/shard000/` 至 `shard191/`|
|Train合并索引（全部完成后发布）|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/train/cache_index.json`|
|Dev48缓存及索引（已完成复制与校验）|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/dev/cache_index.json`|
|全任务进度|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/overall_progress.json`|
|主队列分片状态|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/progress.json`|
|分片日志|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/logs/`|
|主控制器日志|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/launcher_6workers_batch4.log`|
|增补进程日志|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/assist.log`|
|GPU采样|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/gpu_node12.jsonl`、`gpu_node17.jsonl`、`gpu_node16.jsonl`|
|运行参数与来源|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/run.json`、`assist_run.json`、`source_sha256.json`|
|全量完成标志|`/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/completed.json`|

进度文件中 `feature_files` 和 `text_files` 分别表示已创建的 Omni 特征包与 Gemma 文本包数量，不能把只有 features 的样本算作完整可训练样本。新一代进度统计包含已读回校验的旧文件和新完整写入文件；刚恢复时计数可能暂时低于磁盘文件数。`completed_shards` 为已成功完成两阶段的分片数。最终必须看到根目录 `completed.json` 与覆盖91,009条的合并索引，才算全量可用。

每条包含冻结 Omni 四层 hidden、目标音视频感知特征、坐标/mask、GT身份/哈希及冻结 Gemma 条件；最终 U 不缓存，训练时仍由可训练 Connector 生成。已有720p-area VAE latent保持原位置，不复制或重编码。

## 启动与调度

- node12（172.29.252.24）、node17（172.29.15.126）、node16（172.29.112.17），各 GPU0–7，共24卡。
- [主控制器](../scripts/pipeline2_cache_cluster.py)先检查全部GPU空闲，按原清单每192条轮转切分，产生192个互斥小分片；初始每卡2个进程，共48个动态领取任务。
- 单卡两个进程仍有媒体解码等待，增加[辅助队列](../scripts/pipeline2_cache_assist.py)处理尚未领取的后96个分片，再增加每卡2进程，目标总并发96。启动前核验现有GPU进程全部属于本次缓存任务。
- 分片文件锁防止两个队列重复写入；后到队列核验完成分片的契约、样本顺序及文件存在性后复用。已在两台远端验证共享盘文件锁互斥。
- 最新提取入口为 [cache_prepare.py](../src/omni_av_p2/cache_prepare.py)，复用原生 Omni 提取函数。先提取该分片的 Omni 特征，再卸载 Omni，取得每GPU的Gemma并发槽后加载模型并按batch4编码。模型权重、精度、文字、1024-token原生tokenization、mask和processor语义保持；批处理有下述浮点差异，不声称逐位一致。
- 守护器只清理本次子进程组；原失败日志保留。任意分片失败或缺失时不发布全量索引。最终合并检查精确样本集合、无重复、split/contract一致及所有引用文件存在；不是全量 payload 二次读回的声明。
- Dev48 已全量读回，核验契约、GT哈希与文字匹配；复制前后逐文件 SHA256 相同，旧 Dev 缓存保留。

查看进度：

```bash
cat /public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/overall_progress.json
tail -f /public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008.launcher.log
```

如需停止缓存任务，创建 `/public/2_CKPT/0_WM/llq/p2_full91009_dev48_cache_20261008/STOP`；两个控制器均响应，只停止本次任务。中断分片不会伪装为完成。新版 `--resume` 检查清单哈希/分片数量；对已有特征逐样本读回，检查GT哈希、契约、样本内容、hidden/target有效性。未写完整的文件改名为 `.interrupted.<时间戳>` 保留证据，再重算；新tensor文件采用原子发布。已有文本检查prompt、shape和有限值后复用。

当前状态以公共盘进度文件为准。本记录中的启动安排不表示缓存已经全部完成，也不表示1000步训练已开始。


## 并发与batch实测、升级记录

最初48进程（每卡2个）扩至96进程（每卡4个）；观察到逐条媒体处理造成GPU空隙后，按用户要求进一步优化。旧两个控制器已停止，保留约37,691个features文件、15,678个text文件和全部日志。当前仅新主控制器运行，辅助控制器不再运行；最新PID/命令见 `launcher_current.json`。

同16条Dev文本在生产负载旁的Gemma批处理短测（不含模型加载，包含CPU返回）：

|batch|条/秒|单进程peak allocated GB|相对串行输出最大相对L2|
|---|---:|---:|---:|
|1|4.03|33.87|0|
|2|5.17|36.96|0.00234|
|4|6.24|43.14|0.00516|
|8|5.40|55.52|0.00516|
|16|5.68|80.19|0.00516|

选择batch4，不能把batch无限加大当作必然提速。逐项文本和mask一致；FP16/BF16批量内核差异不等于逐位相同（本次实际使用BF16）。结果为 `text_batch_benchmark.json` / `.log`；仅一次短测且与其他任务并发，不是隔离环境下长期性能结论。

每卡6个worker并不意味着6个Gemma同时加载：共享文件锁限制每卡最多2个Gemma进程，其余处理Omni或卸载后等待。batch4实测每进程约43.1GB峰值，因此必须做阶段限流。GPU采样和实际吞吐用于继续判断是否需要调整。

升级验证：53项CPU测试通过（15.80s）；新增中断缓存保留/复用、并发上限及异常释放测试。实际GPU恢复冒烟检查复用了2个既有Omni缓存，隔离1个模拟不完整文本文件并成功生成batch2尾批。缓存主目录未迁移，仍在用户指定公共盘。


升级后实测快照（06:34 UTC，运行仍在继续）：48/192分片完成，45,411份Omni特征、22,753份文本缓存，无失败分片。三个节点最近3次采样的平均GPU利用率为88.5%/75.9%/71.7%，最高显存约130,451MiB；这些是短窗口观测，不声称持续100%利用率。`optimized_gpu_sampling.json`记录完整采样窗口。对3个新完成分片各抽首/中/末样本共9条，原生read_sample检查通过，见`optimized_payload_readback.json`。


## 最终只读审计与收尾修补

- 91,009条全部检查：ID唯一、清单顺序、Train/Dev无交叉、缓存契约、源GT哈希/文字、四层hidden及mask结构、教师时间支持mask、VAE时间窗口、720p-area metadata尺寸、Gemma文本及输出shape。零失败。
- 对每个分片首/中/末样本共576条读取实际数值，检查hidden、teacher、文本embedding及mask的有限值，主要embedding非零。不是2.04TiB全部张量逐元素二次扫描。
- 审计54.15秒，仅CPU读回已有产物，无GPU重新提取；结果见公共缓存目录 `final_audit.json`、`final_summary.json`，审计脚本归档在 `audit_cache.py`。
- 发现并修复了最终合并后 `global_index_published` 未刷新问题：索引实际已成功发布，旧进度字段仍为false；当前进度已依实际索引修正，控制器在合并后再次刷新状态。
- 为防止将来中断留下半写索引，将分片/总索引改为原子JSON发布，拒绝覆盖已有索引。此改动只加固未来写入，不改变本次tensor缓存，不需要重算。相关13项回归通过；此前整体53项测试已通过。
- 本次调优切换中有7个旧方式写入的文本文件未写完整，已隔离为 `.interrupted.<时间戳>` 并仅补齐对应文件；最终索引不引用这些文件。并非全量重做；原始证据保留。
