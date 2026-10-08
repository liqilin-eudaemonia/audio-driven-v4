# 实现与验证记录

**2026-10-08 补测**：完整预训练 Omni/Gemma/LTX 的 P1 tiny latent gate 已通过，两步 AV 反向、零初始化等价、AV→规划 LoRA、冻结参数与恢复均验证。详见 [审计记录](BERNINI_AUDIT_20261008.md)。下文是 2026-10-07 历史状态；仍未验证 P1 原尺寸联合训练及完整自主 AV 质量。

状态：2026-10-07，独立实现；真实 Omni GPU 工程验证通过，完整预训练 LTX GPU 联测尚未运行。没有启动正式训练，没有生成质量结论。

## 环境

- Qwen2.5-Omni-3B：`/public/2_CKPT/3_PreTRAINED/Qwen2.5-Omni-3B`。
- LTX：`/public/2_CKPT/2_AVATAR/LTX23/LTX_23/ltx-2.3-22b-dev.safetensors`。
- 初次 GPU 检查：本机 0–3 空闲，4–7 忙；仅使用空闲卡且每次运行前复查。
- 不修改 v2/v3 的实现与权重，不自动开始全量训练。
- Python：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`；Python 3.13.12、PyTorch 2.9.1+cu128、Transformers 4.57.6、PEFT 0.18.1。
- 复用本机 `Omni-LTX-stage2/packages/ltx-core` 与 `ltx-trainer`，不修改其源码。

## 已完成的验证

19 项 CPU 测试覆盖：混合注意力、前缀不能读取目标、padding、两模态遮盖、FM 速度符号、全 MASK 补全、LoRA 开关隔离、checkpoint/RNG 恢复、禁止覆盖已有输出、跨阶段初始化完整性、因果输入契约、真实缓存和 VAE 数据结构。LTX 集成测试使用**真实模型类的小尺寸随机配置**：零初始化分支严格保持原输出，第二次 AV 反向可回传 Omni 规划 LoRA，采样不读取目标尾部且保持首帧。这不等于完整预训练 LTX 已通过。

[真实 Omni GPU 测试记录](../outputs/omni_gate_003/completed.json)：

- Qwen2.5-Omni-3B，真实两条 Train 素材，原生视觉与音频塔提取目标特征，宽度 2048。
- 三次参数更新，两路 FM、MASK 和规划 LoRA 均有有限非零梯度；总 loss 为 5.940、6.655、5.952，仅为通路检查。
- 冻结底座检查通过；更新前后关闭规划 LoRA 的原生回复逐字相同。
- 保存、参数恢复、随机数状态恢复通过。
- 使用自主文本和外部 8 秒预算，从全 MASK 完成三轮工程采样，剩余位置 526 → 304 → 0。
- 峰值已分配显存约 10.54 GB；可训练参数 31,858,688。
- 修复的真实错误：Qwen2.5 chat template 的系统消息需要内容列表；目标视频预处理空间网格必须与首帧决定的规划网格一致。失败记录保留在 `omni_gate_001/002`。

## 代码地图

|模块|职责|
|---|---|
|`data.py`、`omni.py`|严格输入/GT 文本、原生模型加载、回复隔离、冻结特征提取和坐标|
|`lora.py`、`masking.py`、`flow.py`、`planner.py`|规划态 LoRA、混合注意力、两路条件 FM、多轮补全|
|`renderer.py`、`ltx_runtime.py`|官方 Gemma 条件、VAE 读取、规划连接器、LTX 并行规划 attention 与 AV FM|
|`runner.py`、`training.py`|缓存、Stage I/II/III、独立自主规划导出|
|`checkpoint.py`、`ltx_gate.py`|只保存新文件的增量权重、恢复、完整预训练模型验证入口|

各阶段目前为单卡、batch size 1、确定性轮流取样。Stage I AdamW 1e-4，Stage II/III 1e-5；LTX 使用 uniform timestep sampling。它们是工程起点，不是调优后的训练配方。Stage I 默认只保存最后一次权重；Stage II/III 按 `--save-every` 保存，也支持创建运行目录内的 `STOP` 文件后完成当前步并保存。恢复使用新输出目录和 `--resume`，不覆盖旧运行。

## 命令

以下均从 `/private/liqi_workspace` 运行。`--gpu 0` 仅为示例；入口会拒绝已有超过 1 GB 显存占用或利用率超过 5% 的卡。所有输出目录必须尚不存在。

```bash
V4_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
V4_CLI=project_code/audio-driven-v4/scripts/v4.py

"$V4_PY" "$V4_CLI" test

# 提取少量真实样本并跑规划更新/自主规划验证
"$V4_PY" "$V4_CLI" omni-gate --gpu 0 --samples 2 --steps 3 --budget 8 --output project_code/audio-driven-v4/outputs/omni_gate_next

# 下一个必要的 GPU 验证；tiny 仅裁小 AV latent，模型仍为完整预训练模型
"$V4_PY" "$V4_CLI" ltx-gate --gpu 0 --tiny --cache project_code/audio-driven-v4/outputs/omni_gate_003/cache_index.json --output project_code/audio-driven-v4/outputs/ltx_gate_next

# 以下为训练入口示例，尚未启动正式训练
"$V4_PY" "$V4_CLI" train-planner --gpu 0 --steps 100 --cache /absolute/path/cache_index.json --output /absolute/path/new_planner_run
"$V4_PY" "$V4_CLI" train-renderer --gpu 0 --steps 100 --cache /absolute/path/cache_index.json --output /absolute/path/new_renderer_run
"$V4_PY" "$V4_CLI" train-joint --gpu 0 --steps 100 --cache /absolute/path/cache_index.json --planner-checkpoint /absolute/path/planner.pt --renderer-checkpoint /absolute/path/renderer.pt --output /absolute/path/new_joint_run

# 推理只读取因果观测 JSON，不接受 GT 缓存作为观测
"$V4_PY" "$V4_CLI" infer-plan --gpu 0 --observation /absolute/path/observation.json --planner-checkpoint /absolute/path/planner.pt --output /absolute/path/new_plan_run
```

`prepare --manifest /absolute/path/examples.json` 可单独提取显式 Train 列表；格式见 DATA.md。省略 manifest 时仅选择 `--samples` 条工程样本，不能把默认两条误认为完整训练集。

## 未验证范围与下一步

1. Omni 测试后，8 张 GPU 均出现其他任务占用；完整 LTX gate 与额外恢复运行被入口拒绝，没有停止其他任务。待空卡首先执行上述 `ltx-gate --tiny`，再做不裁 latent 的原尺寸检查。
2. Stage II/III 入口完成静态核查，核心损失/梯度由小尺寸真实模型测试覆盖；完整预训练 Gemma→LTX 加载、联合更新和该训练入口仍需 GPU 实测，不能宣称已无运行错误。
3. 已有自主规划导出和渲染 latent 的工程采样；尚无完整自主规划→最终 MP4 的发布级推理入口、VAE 解码与质量评测。
4. 旧 VAE 的终端裁帧与新特征的完整素材存在小幅时间范围差异，已检查并记录；全量训练前需审计裁帧是否截掉台词/动作，必要时重建匹配缓存。
5. 全量数据审计、验证集指标、正式训练超参数、长时/多卡训练和生成质量均未验证。短测权重只作工程验证，不作为可用规划器。
