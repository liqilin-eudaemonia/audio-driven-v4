# P2 数据、验证、TensorBoard 与多机训练

**最新更新**：[Train91,009 / Dev48、720p-area尺寸审计、三机24卡实际验证与负载均衡](P2_FULL_DATA_3NODE_20261008.md)。以下旧小集和“GPU尚未验证”描述保留为历史记录。

2026-10-08。本轮沿用 Gemma text + Connector U 拼接到 LTX 原生 cross-attention 的方案；没有添加事件时间戳控制。冻结 Omni / Gemma / LTX-2.3，仅训练 Connector 和两路辅助 decoder。

## 已确认的数据

来源：[v2 总索引](../../audio-driven-v2/OMNI_LTX_GLOBAL_INDEX.md)第 7.6 节、[v3 实验索引](../../audio-driven-v3/EXPERIMENT_DOCUMENT_INDEX.md)及其链接的数据审计。实际读取 qualified manifest 与 VAE metadata，结果如下。下列计数是文件记录与 ID 集合核对，不代表本轮逐个解码了全部媒体。

根目录：`/private/liqi_workspace/data/seamless/subset_300h_pure`。

|用途|样本数|根目录下的位置|
|---|---:|---|
|Train|91,009|`train_gt_qualified_20260920/train_part{0,1}_manifest.json`|
|Dev|1,964|`manifests_gt_qualified_20260920/dev_manifest.json`|
|Test|1,699|`manifests_gt_qualified_20260920/test_manifest.json`|
|媒体|三个 split|`processed_pairs_720p_area/{train,dev,test}`|
|Train GT|两个 part|`omni_gt_train_part{0,1}_30fps/<id>/avatar_plan.validated.json`|
|Dev/Test GT|各自独立|`omni_gt_{dev,test}/<id>/avatar_plan.validated.json`|
|可复用 VAE|三个 split|`ltx23_precomputed_720p_area_qualified_20260920/metadata_{train,dev,test}.jsonl`，及 `latents` / `audio_latents`|

三个 split 的 ID 两两不交叉，VAE metadata 与 qualified manifest 的 ID 集合、条数一致。旧索引中“VAE metadata 尚待重建”是历史状态；当前 qualified 路径已经匹配，不能因此重复编码 VAE。旧 Omni 条件包不符合 P2 当前四层 prefix/teacher 契约，不能仅因 shape 相同而复用。

Train 原视频长 3.139–24 秒。8 秒预算只包含 20,836 条 Train、458 条 Dev；它不是全量训练。当前已生成固定 **Train128 / Dev24** 清单，按 `SHA256(seed:sample_id)` 排序选取，seed45，预算8秒，不截断长样本。独立 Test 不参与选模型。训练仍消费 GT TextPlan，验证是 **paired-GT-text Dev loss**，不是自主回复生成的质量评测。

清单、源文件 SHA 与审计：[`outputs/p2_data_20261008_v1`](../outputs/p2_data_20261008_v1/)。工具：[pipeline2_dataset.py](../scripts/pipeline2_dataset.py)。修复了无主回复样本 `response.prosody=null` 导致 TextPlan 转换报错的问题。

## 保存与验证

新入口：[pipeline2_parallel.py](../scripts/pipeline2_parallel.py)，旧的 `pipeline2.py train/gate` 不自动获得这里的周期 Dev/TensorBoard 功能。

- `--save-every 100`：每100个 optimizer step、最后一步、STOP 后保存；包含 Connector、decoder、Adam、step、各 rank 的 CPU/CUDA/Python/NumPy RNG，另导出仅 Connector 权重。
- 临时文件写完后才发布最终 `.pt`，全部导出完成后发布 `.complete.json`；不覆盖已有结果。恢复使用新 output 目录。
- `--validation-cache <Dev cache_index.json> --validate-every 100`：启动时、每100步、结束时验证固定 Dev，写 `validation_step_XXXXXX.json`。未传 Dev cache 时不运行验证。
- Dev 使用固定噪声种子1045，各 rank 按索引分片，每条恰好统计一次；没有样本的 rank 仍参加归约。结果为逐样本 loss 的平均值。验证 `no_grad/eval`，恢复训练模式和 RNG，不更新参数。
- Train/Dev 的 ID、split 与缓存契约必须匹配；旧工程 Train index 缺少 ID/split 时，需要新准备或经过审计重建索引才能搭配 Dev。
- `--steps N` 表示恢复后**追加 N 步**。恢复要求相同训练/验证清单、seed、world size、梯度累积及模型契约。GPU 浮点续训不承诺逐位一致。
- 当前样本按清单循环，microbatch=1，全局 batch=`总 rank 数 × accumulation`。它适合当前固定小集实验；尚未实现全量变长分桶、逐 epoch 尾批协议。

## TensorBoard

默认启用，依赖 `.[monitoring]`；已在当前 LTX venv 安装 TensorBoard 2.21.0。只有全局 rank0 写 `<output>/tensorboard`。

标签包括 `train/{loss,av,video_feature,audio_feature,av_video,av_audio,grad_norm,lr,samples_seen}`、`dev/{loss,av,video_feature,audio_feature,av_video,av_audio}`、最慢 rank 的 step 秒数、samples/sec、rank0 峰值显存。步数沿 checkpoint 接续；日志保存在新 run 目录，避免覆盖原曲线。

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m tensorboard.main \
  --logdir outputs/p2_train128_dev24 --host 127.0.0.1 --port 6008
```

目前只有 CPU probe 的事件文件得到实际写入/读取验证，**新完整 P2 训练曲线尚未产生**，也未启动 TensorBoard 服务。

## 缓存准备与实验命令

工作目录 `audio-driven-v4`，Python 使用已有 LTX 环境。资源空闲后，下面的分片循环在八卡同时准备当前 P2 缓存；训练和验证清单分开准备。任一分片失败，合并时会拒绝缺失/重复 ID，原始数据不改写。

```bash
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
for split in train dev; do
  pids=()
  for gpu in {0..7}; do
    "$P2_PY" scripts/pipeline2.py prepare --gpu "$gpu" \
      --manifest "outputs/p2_data_20261008_v1/$split.json" \
      --shards 8 --shard-index "$gpu" \
      --output "outputs/p2_cache128_dev24/$split/shard$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
  "$P2_PY" scripts/pipeline2_merge_cache.py \
    --indexes outputs/p2_cache128_dev24/"$split"/shard*/cache_index.json \
    --manifest "outputs/p2_data_20261008_v1/$split.json" \
    --output "outputs/p2_cache128_dev24/$split/cache_index.json"
done
"$P2_PY" scripts/pipeline2_parallel.py --gpus 0,1,2,3,4,5,6,7 \
  --cache outputs/p2_cache128_dev24/train/cache_index.json \
  --validation-cache outputs/p2_cache128_dev24/dev/cache_index.json \
  --output outputs/p2_train128_dev24 --steps 200 --recipe joint \
  --save-every 100 --validate-every 100 --allow-terminal-trim
```

200步是可复现实验命令，**尚未执行**。`--allow-terminal-trim` 明确沿用已有 VAE 的末端对齐裁剪，具体每例窗口在准备日志中记录。

## 多机

参考用户指定 `VA-LTX/packages/ltx-trainer/scripts/bx_train_ltx_online_mm.sh`，支持每节点一个 torchrun agent、指定解释器、master地址/端口、node rank、run ID、独立节点日志、NCCL 错误处理和预检。网络示例：[configs/p2_roce.json](../configs/p2_roce.json)，包含 eth0 / mlx5_0 / GID3 等参考值；仅在实际网卡匹配时传入，不强制所有机器使用该值。

每个节点执行同一训练命令，再追加：

```bash
--nnodes 2 --node-rank NODE_RANK --master-addr MASTER_IP --master-port 29518 \
--run-id p2_train128_dev24_v1 --network-config configs/p2_roce.json
```

`NODE_RANK` 分别为0、1；替换真实 master IP。所有节点必须使用同一共享 output/cache 路径、相同代码和环境、相同每节点 GPU 数。仅全局 rank0 创建 output，错误广播给所有 rank，避免多个节点抢建目录。加载 LTX 前核对训练/验证契约、源代码 SHA、PyTorch 版本、恢复路径及关键训练参数。`--preflight-only` 检查本节点 GPU 空闲和输入文件，打印实际命令但不训练；不代替网络通信门禁。

没有照搬参考脚本的 Full-SFT FSDP：P2 的冻结 DiT 单卡可容纳，DDP 只同步可训练头。也没有自动 SSH 到旧节点或终止其他实验。需要作业平台/操作者在各节点启动上述同一任务。

## 本轮验证及限制

- 49项 CPU 回归通过，包括 GT null prosody、Train/Dev 隔离、VAE split、固定噪声验证不改变权重/RNG、逐 rank RNG 恢复、原子保存、缓存分片完整覆盖与既有 Bernini/梯度测试。
- 两个独立 torchrun agent、每个2 rank，**同一物理机 CPU/Gloo**：3步联合更新、从step2恢复到step3、3条Dev分配给4rank均通过。各rank权重一致，更新与串行global batch参考一致。
- 连续/恢复模型参数完全相同；Adam状态最大绝对差 `2.9802322387695312e-08`。两段TensorBoard step分别 `[1,2,3]` 与 `[3]`，已实际读取。
- 证据：[初始结果](../outputs/p2_multinode_cpu_20261008_v2/initial/completed.json)、[恢复结果](../outputs/p2_multinode_cpu_20261008_v2/resumed/completed.json)、[状态与事件审计](../outputs/p2_multinode_cpu_20261008_v2/state_and_tensorboard_audit.json)。首次 probe 参考模型误共享 Adam tensor 的失败保留在 v1；修正测试参考副本后 v2 通过。
- **尚未验证**：跨物理机器 NCCL/RoCE；本轮新增 Dev 验证和保存逻辑在完整 GPU 模型上的联合运行；Train128/Dev24 特征准备及正式训练；独立Dev的真实音视频生成质量。
- 本轮检查时八卡均被其他任务占用，每卡约30.8GiB，故没有抢占资源或启动 GPU 任务。此前八卡吞吐/冻结/梯度证据见 [Bernini审计](BERNINI_AUDIT_20261008.md)，不能替代本轮新增逻辑的 GPU 复测。
