# Pipeline 2.1：Connector + 原生 text cross-attention LoRA

2026-10-08。独立入口`src/omni_av_p21`与`scripts/pipeline21_*`，保留P2冻结DiT训练入口与正在运行的实验。P2备份固定提交为[05e6489](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/05e64892daa1d8abd0e4e696a496368a37926271)。本版实现代码与CPU验证，不表示已启动GPU训练或证明生成收益。

## 结构和训练范围

仍然使用冻结Omni的8/18/28/35层hidden，经可训练Connector得到U_video/U_audio；与冻结Gemma条件沿token维拼接，进入LTX2.3原有text cross-attention。没有新增独立Omni attention。

- 在每个block的`attn2`、`audio_attn2`上，为`to_q`、`to_k`、`to_v`、`to_out.0`安装LoRA。
- 默认rank8/alpha8，A采用Kaiming初始化，B为零；相对于使用相同Connector的P2模型，初始增量为零。
- 对实际LTX2.3 checkpoint头部核查：48个block、384个投影、18,874,368个rank8 LoRA参数（FP32主参数）。此为权重布局核查，不是完整GPU训练测试。
- 原始投影权重/偏置、QK norm、gated-attention门控、AdaLN、self-attention、AV cross-attention和FFN都冻结。
- Omni/Gemma/VAE冻结；Connector、joint配方的两个辅助decoder继续训练。
- 默认Extractor学习率1e-4，LoRA学习率1e-5；两组由同一AdamW更新，所有可训练参数共同以max_norm1裁剪。三项loss权重仍为1，暂不改变损失设计。

## 数据与多卡

缓存契约继续采用P2 Config及`observation_no_time_v1`，LoRA配置独立存储，不污染冻结感知缓存身份。已有Train91,009/Dev48 hidden、teacher、Gemma和VAE缓存均可复用；不重新提取。原720p-area宽高、帧数规则及首帧条件保持。

单个TrainingModel前向包含Connector、辅助loss和带LoRA的LTX前向；DDP注册Extractor和LoRA，梯度累积no_sync覆盖整个计算，避免LoRA仅在单卡更新。22B冻结底座不注册到DDP根模块，不写入训练checkpoint；renderer持有同一组adapter对象，运行时检查参数归属与冻结底座版本。

三机入口延续空闲卡检查、各rank配置/源码一致性校验、共享目录与失败退出保护；`configs/p21_three_nodes.json`采用独立端口29851。P2的transport probe不用于证明LoRA同步，P2.1另外通过真实小型LTX的双进程CPU/Gloo梯度和恢复测试。

## 保存、恢复与推理

独立checkpoint版本为`p21_text_attention_lora_v1`，不能当作P2权重使用。

- 完整状态：`extractor_step_000100.pt`，包含Extractor、text LoRA、优化器、逐rank RNG、配置和精确投影名单。
- 部署权重：`extractor_step_000100_deployment.pt`，同时包含Connector与LoRA，排除辅助decoder及优化器；不能仅导出Connector。
- 同步保存二者后写`.complete.json`完成标志。
- `--resume`只接受完整P2.1 checkpoint，恢复全部训练状态。
- `--init-p2`可从同契约、同recipe的完整P2 checkpoint加载Extractor；LoRA为零增量、优化器新建、步数从0开始。这是显式初始化，不是无缝续训；不接受P2 Connector-only导出，与`--resume`互斥。
- `pipeline21_validate.py`复用P2 Dev48观测/整份计划提取协议，要求P2.1 Connector+LoRA一同加载；漏LoRA、投影名单不符、Train/Dev重叠均拒绝。GT/Omni两组都使用相同LoRA，不用默认冻结LTX替代。

## 独立训练启动示例

在三机GPU空闲后，于仓库目录执行。使用新的run-id与输出目录；命令不传validation-cache，故关闭周期Dev loss和验证集推理。

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline21_cluster.py  \
  --cluster configs/p21_three_nodes.json  \
  --log-dir /public/2_CKPT/0_WM/llq/p21_text_lora_run01/agents --  \
  --config configs/pipeline21_text_lora.json  \
  --cache /public/2_CKPT/0_WM/llq/p2_full91009_dev48_no_time_cache_20261008/train/cache_index.json  \
  --output /public/2_CKPT/0_WM/llq/p21_text_lora_run01/training  \
  --checkpoint-dir /public/2_CKPT/0_WM/llq/p21_text_lora_run01/checkpoints  \
  --network-config configs/p2_roce.json --run-id p21_text_lora_run01  \
  --recipe joint --steps 1000 --accumulation 3 --balance-workload  \
  --save-every 100 --cache-samples 3 --seed 45 --allow-terminal-trim --tensorboard
```

本次正式P2.1必须独立初始化，不传`--init-p2`或P2的`--resume`；旧P2权重仅作为基线保留。实际三机运行命令与路径见[独立训练记录](P21_FRESH_3NODE_TRAIN_20261008.md)。

配对验证在GPU空闲、保存完成后执行；已有P2准备的`suite.json`可直接复用，因为观测和计划协议未改变：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline21_validate.py run  \
  --suite outputs/p2_dev48_full_plan_validation_20261008/suite.json  \
  --checkpoint /public/2_CKPT/0_WM/llq/p21_text_lora_run01/checkpoints/extractor_step_000100_deployment.pt  \
  --output /public/2_CKPT/0_WM/llq/p21_text_lora_step100_first_pair  \
  --gpu 0 --shards 48 --shard-index 0
```

TensorBoard在新运行的`training/tensorboard/`，新增`train/lora_grad_norm`、`train/connector_grad_norm`（均为裁剪前总loss梯度范数）和`train/lora_lr`；不把这些指标解释成各loss梯度贡献。

## 验证边界

本次完整CPU回归90项通过，其中P2.1新增10项测试；Ruff F/E9和git diff --check通过。Dev48已通过P2.1入口完成CPU任务准备，保存到`outputs/p21_dev48_full_plan_validation_20261008/suite.json`，96段视频均未生成。

测试包括精确注入名单、零增量等价、AV loss向Connector和两路LoRA回传、冻结底座不变、梯度重计算、BF16底座/FP32 adapter、LTX2.3 gated attention + text AdaLN路径、双进程累积对照、保存/恢复与部署输出一致、旧P2显式初始化、LoRA漏载拒绝、固定噪声Dev/RNG恢复、多机入口与忙卡保护。结果见[测试记录](PIPELINE21_RESULTS_20261008.json)。

后续验证：105项CPU回归通过；监控功耗检查修改后11项监控回归通过（与前者重叠，不相加）。172.29.104.194的0–3卡完整22B模型完成2步独立训练、保存、加载并续训至第3步；LoRA梯度、冻结底座、四rank参数一致、部署权重与完整权重一致均通过。三机24卡已于14:34 UTC独立启动，最新进度与checkpoint审计见[运行记录](P21_FRESH_3NODE_TRAIN_20261008.md)。Dev生成质量仍未验证。
