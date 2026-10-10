# Pipeline 3 GitHub备份与复现（2026-10-10）

私密仓库：[liqilin-eudaemonia/audio-driven-v4](https://github.com/liqilin-eudaemonia/audio-driven-v4)。独立分支：[backup/p3-3n24g-step1000-20261010](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p3-3n24g-step1000-20261010)。以`d3e366744cc72ebc11bbf24acac558fcbcecd449`为基础，继承已备份的P1/P2/P2.1及P3需要的共享工具；本次新增P3文件，不改变其他分支。

## 本次版本内容

- `src/omni_av_p3/`：原生音视频目标特征、混合注意力与MASK规划、Bernini式特征Connector/AdaLN embedding decoder、冻结/Thinker LoRA对照、迭代全MASK采样与条件诊断。
- `scripts/pipeline3*.py`、`scripts/bx_train_pipeline3_mm.sh`：单卡/四卡试验、配置式SSH/torchrun三机DDP、严格续训、完整状态保存、Dev验证、全量特征提取和自动测速/训练/监控。
- 配置与测试：三机/五机拓扑、RoCE/socket示例、pilot配置，以及目标布局、梯度、合批、并发缓存与分布式一致性测试。
- 文档与结果：首轮200步、四卡2000步、多机工程验证、全量Train91,009/Dev48的LoRA step1000、合批提速与本次备份核对。

P3把目标音视频的原生特征部分遮盖后交给Omni，预测被遮盖位置；与P2的多层前缀hidden融合方案分开。本轮全量训练只更新Thinker LoRA及规划头，感知塔和基座冻结，没有训练生成Connector或LTX，也没有实现自主时长预测。

## 实际训练及溯源

运行根目录：`/public/2_CKPT/0_WM/llq/p3_full91009_1000step_20261009_perf`。

- 完成：三机24rank、global batch96、每卡batch2×累积2，step1000；最终rank权重一致，Dev48完整。
- 检查点：`training/step_001000.pt`及`step_001000.complete.json`。
- TensorBoard：`training/tensorboard/`；逐rank训练日志与Dev结果保留在`training/`。
- 结果：[最终训练摘要](PIPELINE3_FULL_TRAIN_FINAL_20261010.json)；性能：[四组配对测速](PIPELINE3_OPTIMIZATION_RESULTS_20261009.json)。
- `training/run.json`记录的25个启动源码文件，与本次备份逐文件SHA256相同；原始Git HEAD较旧，不能用它代替实际工作区源码版本。监控依赖的P2.1工具从已上传基础提交继承。

GitHub包含代码、配置、测试、文档及结果摘要。模型权重、约634GB特征缓存、音视频媒体、`outputs/`和日志原件仍在本地/共享存储，不上传GitHub。

## 复现入口

环境：Python `/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。需另行准备Qwen2.5-Omni-3B权重、数据清单/媒体、缓存以及已记录的Python依赖。绝对路径、SSH key路径和节点IP需按目标环境调整。

使用已有完整P3缓存启动同配方训练的示例：

```bash
python scripts/pipeline3_cluster.py \
  --cluster configs/p3_full_three_nodes.json \
  --log-dir /new/location/p3_run_logs -- \
  --cache /public/2_CKPT/0_WM/llq/p3_full91009_dev48_cache_20261009/index.json \
  --output /new/location/p3_run --arm lora --total-steps 1000 \
  --accumulation 2 --microbatch 2 --save-every 100 --validate-every 100 \
  --threads 2 --cache-samples 8 --timeout 900 \
  --network-config configs/p3_nccl_roce_example.json \
  --prefetch --balance-workload
```

输出路径必须是新的目录。续训需保持world size、采样顺序和batch契约，详见[多机说明](PIPELINE3_MULTINODE_20261009.md)。SSH连接复用可在cluster配置中设置`ssh_control_dir`；历史临时socket目录无需照搬。

完整缓存准备与四组测速监督入口见[全量运行说明](PIPELINE3_FULL_TRAIN_20261009.md)。训练完成与特征损失下降不等于生成质量验证通过；后续应做条件消融和渲染评估。
