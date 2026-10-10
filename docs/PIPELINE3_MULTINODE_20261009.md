# Pipeline 3 多机训练与恢复

2026-10-09。本次补充 P3 **目标原生音视频特征规划器**的 DDP 工程，不改变 MASK、两路特征 Connector、AdaLN decoder 或 Flow Matching 目标，也没有接入 LTX。结构与研究结论分别见[路线方案](PIPELINE3_PLAN.md)和[2000步对照](PIPELINE3_FOUR_GPU_20261008.md)。

## 节点与入口

默认配置 `configs/p3_three_nodes.json`：

|节点|地址|训练角色|GPU|
|---|---|---|---|
|node17 / avgen-2|172.29.15.126|master，rank 0–7|0–7|
|node12 / js-public|172.29.252.24|rank 8–15|0–7|
|node16 / avgen-1|172.29.112.17|rank 16–23|0–7|

`172.29.104.194`（node8 / tk-1）按用户最新说明另作四卡续训／推理机器，不在默认三机训练配置中。启动器可以从任一有权限的机器发起，按实际 IP 自动选择本地执行或 SSH；master 不必是发起机器。本次工具实际从 tk-1 发起，三机正式测试的计算在上表节点进行。

内网 SSH 使用 root、22端口；群公告中的20341/20474等外部端口不作内网 SSH 或 NCCL rendezvous 端口。三机默认 rendezvous 端口29875。`p3_five_nodes.json`仅保留五节点清单配置，未验证五机联合训练；`p3_three_node_gate.json`是早期 node17/tk-1/node16 每机两卡的工程测试配置，不能误认为当前正式三机。

- `scripts/bx_train_pipeline3_mm.sh`：参考用户给出的 `bx_train_ltx_online_mm.sh` 提供配置式 shell 入口。
- `scripts/pipeline3_cluster.py`：多节点预检、SSH、远程 master、分节点日志及本次任务进程组清理。
- `scripts/pipeline3_parallel.py`：单机／多机 torchrun 入口，可选择 frozen 或 lora。
- `src/omni_av_p3/distributed.py`：联合更新、全局采样序列、分布式验证、保存与恢复。

## 同步和训练语义

采用 DDP，同步 MASK／modality embedding、两个特征 Connector、两个 decoder，以及 LoRA 组的 Thinker adapters。冻结的 Omni 权重每卡本地加载，不注册进 DDP 广播；前向仍运行真实 Omni，并保留经过其冻结算子的输入梯度。训练中逐步检查冻结参数版本及梯度；结束时所有 rank 的可训练参数 SHA256 必须完全一致。

每卡每个 microstep 一条可变长度样本，global batch = world size × accumulation。采用确定性全局样本序号生成数据顺序、遮盖和噪声，累积期间使用 `no_sync`，最后一次反向才同步。每模态先按本样本被遮盖特征求均值，再对 microstep／rank 平均。不能把此目标误称为整个 batch 所有 token 的加权均值。

不同 epoch 采用确定性排列，不为分片对齐补齐数据；batch 大于训练集时会跨 epoch 重复样本。本次 Train16、global batch48 是工程验证设置，不是扩大训练数据后的正式配方，也不直接与旧 batch1 的2000步效果比较。

保留 BF16、fused AdamW、可选激活重计算；GPU 特征采用有界 LRU 缓存（默认每 split 每卡4条），防止全量数据全部驻留显存。没有缓存训练中的 Omni hidden。验证按 rank 分片，汇总逐样本损失和计数，即使 Dev 数量小于 world size，也不重复补齐样本。当前验证是固定噪声的全 MASK FM 及打乱前缀对照，不是视频渲染或多轮采样质量评估。

## 网络、预检及故障处理

所有节点先检查指定卡空闲且没有计算进程、共享代码签名、缓存契约、文件路径、初始化／恢复文件和网络配置。代码路径、Python环境及缓存路径必须在所有节点可读；本机环境使用共享目录。显式使用同一个 Python 的 `-m torch.distributed.run`，不依赖 shell 中的 torchrun。

`p3_nccl_socket.json`用于 TCP 回退；`p3_nccl_roce_example.json`来自参考脚本的 RoCE 参数，指定 `mlx5_0`、GID3、eth0，且显式设置 `NCCL_NET=IB`。本次日志确认 `NET/IB ... /RoCE` 与 `Using network IB`，并非静默回退 TCP。换集群须重新核对网卡和 GID。

某节点失败会关闭本次启动器拥有的 worker 进程组，不按进程名称广泛杀任务。输出和日志必须使用新目录，历史／失败记录保留。可在运行目录新建 `STOP` 文件，所有 rank 在同一 optimizer 更新边界退出并保存；该机制已实现，但本次未做跨机 STOP 故障注入。

## 保存和恢复边界

只由 rank0 保存 trainable weights、optimizer、训练步数、各 rank RNG 和完整身份契约。写入采用新文件排他创建，flush/fsync 完成后才发布带 SHA256 的 `.complete.json`；中断残留且没有完成标记的文件禁止恢复，不覆盖既有文件。

`--resume`要求相同数据缓存、arm、world size、accumulation、seed、模型配置与优化器／checkpointing设置，`--total-steps`是恢复后希望达到的总步数。恢复目录必须是新目录。

原单卡检查点可用 `--initialize-from`加载可训练权重，但会重置 optimizer、步数和全局采样序列；当前要求完整模型配置相同。它不是原训练轨迹的严格续训。**目前不支持把24卡 checkpoint 用 `--resume`直接迁到4卡**，需要明确设计样本曝光／优化器恢复策略后再补充拓扑转换；四卡机器的用途标注不代表这项转换已实现。

## 启动示例

在仓库根目录执行，所有输出示例都需要尚不存在：

```bash
P3_ROOT=/private/liqi_workspace/project_code/audio-driven-v4
export RUN_LOG_DIR="$P3_ROOT/outputs/p3_next_run_logs"
export CLUSTER_CONFIG="$P3_ROOT/configs/p3_three_nodes.json"
bash scripts/bx_train_pipeline3_mm.sh \
  --cache "$P3_ROOT/outputs/p3_fixed_condition_pilot_20261008/cache/index.json" \
  --output "$P3_ROOT/outputs/p3_next_run" \
  --arm lora --total-steps 6 --accumulation 2 \
  --save-every 3 --validate-every 3 --threads 2 \
  --network-config "$P3_ROOT/configs/p3_nccl_roce_example.json"
```

仅检查时增加环境变量 `PREFLIGHT_ONLY=1`；冻结组改 `--arm frozen` 并使用新的输出目录。恢复时保持原训练契约，增加 `--resume /绝对路径/step_000006.pt`，更换输出／日志目录并把 `--total-steps`设为8或更大。每次正式运行前预检都会重新检查空卡，历史空闲情况不是持续占用许可。

## 本轮验证记录

原始产物：`outputs/p3_multinode_20261009/`。随仓库保存的数值与哈希见[结果摘要](PIPELINE3_MULTINODE_RESULTS_20261009.json)。

- 18项 P3、加速与分布式 CPU 回归通过。
- CPU 四 rank：DDP 更新与同一全局 batch 串行参考最大参数差 `2.98e-8`；Dev3小于world4的验证计数正确。
- CPU 第4步恢复至第6步，与从头不中断的第6步比较模型／optimizer／RNG／身份：通过容差检查，最大张量差 `1.49e-8`，不声称逐位一致。
- 早期三台物理机、六卡：Socket与RoCE均完成 tiny Omni LoRA 4步；DDP与串行参考对齐、rank权重一致。tiny 指真实缩小的 Omni 架构，未载入预训练权重。
- 正式 node17/node12/node16 三机24卡：真实 Qwen2.5-Omni-3B LoRA 从初始化完成6步，global batch48；周期保存／Dev8验证通过，24 rank最终参数哈希一致。平均纯更新约0.696秒／步，仅6步，排除模型加载、保存和验证，不代表长期吞吐。

三机测试完成后，node12出现其他 `avbench` 计算任务。随后三机恢复请求被空卡预检拒绝，尚未启动训练；原始记录保留在 `omni24_resume_logs/preflight_js-public.log`。没有停止其他任务。恢复测试改为仍空闲的node17/node16、两机16卡，使用 `configs/p3_two_nodes_gate.json`；先建立相同16卡拓扑的独立检查点，再恢复，未绕过24→16的契约限制。

两机16卡真实 LoRA：从初始化训练4步，保存后恢复到第6步，global batch32。恢复前后的第4步 Dev8 逐样本结果完全一致，后续两步更新成功，16 rank最终权重哈希一致。严格三机24卡恢复未实测，本次真实多机恢复证据来自两机16卡。

两机16卡真实冻结组：独立初始化完成4步，global batch32，平均纯更新约0.674秒／步。MASK／modality、Connector与decoder均有非零梯度；LoRA不存在、Omni基座保持冻结，16 rank最终权重哈希一致。真实LoRA测试的上述梯度及LoRA梯度也均非零。所有运行的逐rank样本噪声seed与全局序号公式核对通过，最终检查点完成标记的SHA256核对通过。

本轮所有测试已正常退出；最终检查node16、node17的全部GPU均为1MiB、0%利用率且无计算进程。本次GPU占用已释放，node12的其他任务未被停止。

以上证明多机工程链路可运行，不证明规划泛化或生成效果提升。仍需扩大训练集、匹配冻结／LoRA曝光预算、做细分条件消融；生成 Connector／LTX、自主回复与时长属于之后的研究步骤。当前修改保存在工作区，本轮未新建 GitHub 提交或推送。
