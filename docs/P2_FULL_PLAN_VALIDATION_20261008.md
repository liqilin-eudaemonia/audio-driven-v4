# Dev48：GT / Omni 整份计划配对生成验证

用户已确认两组切换整份Reply、Delivery、Behavior、Vocal计划。本入口只准备离线生成验证，未加入训练周期回调，不改变正在运行的1000步训练。

## 两组定义与共享路径

|模式|计划来源|hidden与Gemma|
|---|---|---|
|gt|与训练相同的gt_text_plan转换结果，四个字段均来自标注|该完整计划重新提取hidden；同一计划生成Gemma提示|
|omni|只依据speaker音频和listener参考图自主生成，四个字段均来自其输出|自主计划重新序列化为assistant续文，再完整forward提取hidden；同一计划生成Gemma提示|

两组都调用训练缓存使用的`omni_av_p2.data.extract_condition`，即相同无时间user message、相同Omni处理器和原生音频编码、相同assistant计划序列化与causal Thinker forward、相同8/18/28/35层、相同query布局，然后用同一个checkpoint的Connector生成Uv/Ua。不会把GT组的hidden换个文本标签后交给Omni组，也不会用generate过程中的返回hidden代替训练提取路径。

Vocal是可选字段：Omni没有输出就为空，不从GT回填；保持训练原有system prompt，不为验证修改提示协议。Omni无法完整结束或解析时，该样本配对任务明确失败，不静默替换为GT。GT组是使用完整标注计划的条件生成对照，不能将两组差异仅归因于Reply文字。

## 数据与几何

固定原Dev48，共48对、96个待生成视频，Train/Test不进入本清单。准备时核对GT文本转换、输入媒体SHA256、参考图原始尺寸及训练VAE元数据。生成阶段只读观测音频、参考图、计划和训练checkpoint；不读GT目标视频、GT听者音频或训练hidden/VAE目标缓存。

两组采用同一参考图、checkpoint、样本seed、原始720p-area宽高、训练目标帧数与fps。现有48张参考图均已核验与目标H/W一致，拒绝自动resize/crop；不使用旧推理入口256×256默认值。

输出帧数直接采用训练metadata中的合法8k+1帧数（本Dev48为129–633帧）；不根据浮点duration再次向下取整。条件query仍用原Observation.output_duration_s，以匹配训练布局；这些内部几何时间不会写入Omni user message。这是固定时长的离线配对验证，不评估自主生成时长或流式轮次预测。

原始speaker音频仍按16kHz波形进入Omni；音频条件/辅助监督4Hz和LTX原生音频latent布局保持当前训练协议。

## 采样与资源

沿用现有Euler、无CFG采样器；本验证默认30步，可在prepare时设置，不据此宣称是LTX最佳采样配置。每条样本seed=1045+清单序号，两组在strategy准备前和真正采样噪声前分别重设相同seed，避免Omni生成长度影响采样随机数。

每对共享一次Omni加载、一次Connector、一次Gemma加载、一次参考帧VAE编码和一次LTX加载；两路分别解码。集群入口将48对按序号切成24份，每卡2对，独立推理、不使用DDP。所有GPU须空闲；占用时直接拒绝启动，不停止训练、不自动等待抢占。GPU集群推理尚未实测。

## 入口与命令

代码：`scripts/pipeline2_validate.py`，实现：`src/omni_av_p2/paired_validation.py`。

已执行CPU准备：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline2_validate.py prepare \
  --output outputs/p2_dev48_full_plan_validation_20261008
```

清单：`outputs/p2_dev48_full_plan_validation_20261008/suite.json`。prepare不需要checkpoint、不加载GPU模型、不会生成视频。再次prepare应选新的输出目录。

待有完整保存的checkpoint且GPU空闲后，先执行一对样本的真实GPU门禁（下面命令尚未执行）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline2_validate.py run \
  --suite outputs/p2_dev48_full_plan_validation_20261008/suite.json \
  --checkpoint /public/2_CKPT/0_WM/llq/p2_no_time_full91009_3n24g_1000step_20261008_0838/checkpoints/extractor_step_000100_connector.pt \
  --gpu 0 --shards 48 --shard-index 0 \
  --output /public/2_CKPT/0_WM/llq/p2_dev48_full_plan_step100_first_pair_20261008
```

完整三机24卡命令（尚未执行）：

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python scripts/pipeline2_validate.py launch \
  --suite outputs/p2_dev48_full_plan_validation_20261008/suite.json \
  --checkpoint /public/2_CKPT/0_WM/llq/p2_no_time_full91009_3n24g_1000step_20261008_0838/checkpoints/extractor_step_000100_connector.pt \
  --cluster configs/p2_three_nodes.json \
  --output /public/2_CKPT/0_WM/llq/p2_dev48_full_plan_step100_20261008
```

100-step文件只是命令示例，不表示它已经生成。也可显式选择其他完整保存的同协议checkpoint。读取训练周期checkpoint前检查`.complete.json`，校验提示契约与Train/Dev隔离；推荐Connector导出，避免加载完整优化器状态。每次使用新的输出目录，不覆盖既有结果。

## 产物

每个sample_id目录下：

- `case.json`：观测、GT计划、固定几何、seed。
- `omni_raw.txt`：自主规划原文。
- `gt/`、`omni/`各自的`text_plan.json`、`hidden.pt`、`condition.pt`、`sampled_latents.pt`、`geometry.json`、`generated.mp4`和`completed.json`。
- hidden保存形状、层号、mask与坐标；结果还记录hidden SHA256、Gemma提示、GT使用标志及解码几何，便于排查条件串用。

集群输出根目录另有`comparison.html`，可并排播放两种生成结果并查看各自完整计划；失败样本显示缺失状态，详细原因在`sample_id.failed.json`及worker日志。`completed.json`的passed只表示工程任务完成，quality_pass保持null，不冒充生成质量评价。单卡处理全部48对（run默认shards=1）也输出对照页面。

## 已验证与未验证

- 71项CPU回归通过，含新增完整计划切换、缺失Vocal不回填、Omni失败不回退GT、计划/hidden/Gemma一致性、相同噪声、原始几何、checkpoint发布与数据隔离、对照页面文本转义测试。
- 真实Dev48的CPU清单准备通过；检查了48个GT、48对输入媒体哈希及参考图尺寸。未读取GT目标音视频。
- 在训练占用GPU0时实际调用run，确认在加载模型、创建输出前拒绝运行；当前训练继续，未产生新验证视频。
- GPU生成、24卡生成调度、完整视频解码与效果尚未实测。默认采样器的生成质量尚未验证，必须先做一对真实样本门禁再推进全量。

机器可读记录：[P2_FULL_PLAN_VALIDATION_RESULTS_20261008.json](P2_FULL_PLAN_VALIDATION_RESULTS_20261008.json)。


2026-10-09新增：[历史四项生成指标及覆盖率](P2_P21_GENERATION_METRICS_20261009.md)。支持原验证入口的`metrics`子命令，对已有视频评分；与训练/Dev loss分开记录。GT／Omni规划阶段增加独立状态记录，Omni计划拒绝后GT可继续，配对状态保留partial。
