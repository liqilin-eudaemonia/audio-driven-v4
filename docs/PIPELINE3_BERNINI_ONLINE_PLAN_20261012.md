# P3 Bernini式在线条件分流：C0与A改造

**2026-10-13归档更新：** 39项CPU回归重新通过，包含下文GPU验证后的修复；版本定位和备份入口见[在线版备份说明](PIPELINE3_ONLINE_GITHUB_BACKUP_20261013.md)。本文分日期追加的“尚未验证／未推送”描述当时状态，最终范围以上述备份说明为准。

2026-10-12。用户授权记录方案并实施代码修改；本轮不切换、不停止既有离线训练。
独立分支：`experiment/p3-bernini-online-c0-a-20261012`，基于离线备份`2b1ee36`。

## 与离线版本的区别

离线版C0/A按样本混合完整Oracle/Pred25 hidden，hidden已detach，AV loss不能更新Omni。
新版直接读取已封存的原生感知特征和前缀input embedding，训练时在线Omni forward；不读取最终hidden缓存。
保留Speaker音频、Listener首帧、固定GT回复与时间表，不在本轮增加自主文本／时间预测。

一次输入为`prefix + [GT z1, MASK, GT z3, MASK]`，读取同次forward的倒数第二层：

- 前缀hidden＋未mask视频槽位hidden → 视频生成Connector → Gemma视频文本条件拼接。
- 前缀hidden＋未mask音频槽位hidden → 音频生成Connector → Gemma音频文本条件拼接。
- masked音视频槽位hidden → 各自特征Connector＋Embedding Decoder → 各自特征FM。
- 拼接条件进入原生LTX text cross-attention，得到视频／音频生成loss。没有Qwen残差attention。

前缀包含观察多模态token与计划token，不等同于纯文本；保留因果前缀、目标双向注意力，前缀不能看目标特征。
训练无需先做25轮回填。FM用GT特征加噪、预测noise−target速度，只监督有效masked槽位。
推理从全mask开始预测回填，再完整forward读前缀＋全部目标槽位hidden；不读取GT目标感知特征。

## 阶段、初始化与优化

|阶段|优化模块|目标与梯度|
|---|---|---|
|C0|仅音视频生成Connector|AV loss穿过冻结LTX更新Connector；Omni及其所有规划头冻结|
|A前100步（可配置）|生成Connector＋DiT text-attention LoRA|延续C0条件路径，Planner冻结；让新DiT LoRA先适配|
|A后续|生成Connector、DiT text-attention LoRA、Omni LoRA、特征Connector／Decoder、mask／模态参数|同次Omni forward联合特征FM与AV梯度；不detach hidden|

C0从指定离线Connector温启动（不是严格续训）；冻结规划器使用已选step17000。
A继承**新版C0**权重，新增DiT LoRA零增量、新优化器，独立1000步；不把旧离线A优化器当作在线A续训。
C0默认2000步为可调配方，不继承旧任务700转阶段自动调度。每100步保存，可显式停止；没有自动启动或转阶段。
推荐LR：生成Connector3e-5，DiT LoRA5e-6（50步warmup），Omni LoRA1e-5，特征头1e-5。
这些是新配方起点，不是已验证最优值；Omni/特征头在A联合阶段开始时再warmup50步。

联合目标为`L_AV + λ_fv L_feature_video + λ_fa L_feature_audio`，保留LTX原生AV内部权重；两个特征权重初始为1，记录各分项和各组梯度范数，后续根据实测调整。C0与A暖启动不把冻结FM计入优化目标。
视频／音频特征分别按有效masked token均值归一化，避免token数多的模态自动压过另一模态。

mask使用可复现随机数：部分mask比例默认均匀覆盖0.5–0.95（不是宣称Bernini原配方），20%全mask；部分mask时两模态各保留至少一个可见和一个masked槽位（仅一个槽位时mask该槽位）。全mask可用前缀条件，不构造假GT槽位，不出现空cross-attention。

## 工程边界

- 不重新提取、不删除现有感知／文本／VAE缓存；复用原manifest与输入SHA封存，并逐次校验原生特征字节SHA。旧Oracle/Pred25 readout缓存继续供旧实验使用。
- 新格式`p3.bernini-online.v1`，保存Planner所有适配参数、生成Connector、DiT LoRA、优化器与各rank RNG；不保存冻结基座。严格续训绑定源码、manifest、训练配方、world size与初始化来源。
- 一个DDP根持有全部适配参数，允许A暖启动和动态mask产生未使用参数；梯度累积后统一更新。C0/A不同阶段不能冒充严格恢复。
- 保留原始720p-area、首帧条件、global batch72的三机配方、每100步保存与TensorBoard；不自动抢卡、不操作其他作业。
- 推理使用新checkpoint中的Planner LoRA与特征头，不能继续用旧step17000的Pred25 hidden缓存。可对照新Planner的Oracle，但必须单独标记，不能作为自主生成质量。

## 验收要求

1. 真正的小型Qwen Omni＋原生LTX模型验证条件分流：未mask位置送生成器、masked位置仅做特征FM；前缀参与生成。
2. 单独AV loss能回传Omni LoRA；单独特征FM能回传对应头和Omni LoRA；冻结基座不变；C0及A前100步规划器不更新。
3. 全mask、部分mask、padding、模态空可见集合、不同长度、梯度累积和两rankCPU DDP正常。
4. 完整保存→严格恢复的下一步与连续训练一致；旧离线checkpoint禁止当作新版resume。
5. 推理全mask不读GT特征，使用当前checkpoint；原生调度＋CFG与第一帧保持逻辑复用。
6. 正式大模型24卡内存、速度与画面/声音质量需另做GPU工程测试；CPU通过不能写成已完成真实24卡验证。

## 实现入口

代码工作树：`/private/liqi_workspace/project_code/audio-driven-v4-p3-online-20261012`。
旧离线实现仍保留在`omni_av_p3_staged`；新入口不会被旧训练队列自动调用。

|用途|文件|
|---|---|
|在线Omni、mask分流、两个目标、训练范围|`src/omni_av_p3_online/model.py`|
|原生缓存、文本/VAE读取和完整性检查|`src/omni_av_p3_online/data.py`|
|只保存适配参数、严格恢复、C0→A初始化|`src/omni_av_p3_online/checkpoint.py`|
|真实模型构造与分组学习率|`src/omni_av_p3_online/runtime.py`|
|单/多机torchrun训练入口|`scripts/pipeline3_online_train.py`|
|多机预检、源码/数据一致性和进程守护|`scripts/pipeline3_online_cluster.py`|
|使用当前检查点重新预测的推理入口|`scripts/pipeline3_online_generate.py`|
|训练配方、三机示例|`configs/pipeline3_bernini_online.json`、`configs/pipeline3_online_cluster.example.json`|

`OnlineTraining`把Omni适配参数、生成Connector和DiT LoRA注册在一个DDP根中，但不注册冻结基座，避免保存/同步整套预训练权重。A暖启动时通过无梯度Planner前向冻结其更新，不在DDP构建后动态改变参数集合；联合阶段复用同一组适配参数。默认不启用低精度梯度通信，先保证在线联合梯度的可验证性。

参数检查点通过完整标记与文件SHA校验，`planner_names`防止索引式适配参数错位；旧离线checkpoint只能显式温初始化新C0的Connector，不可当作新版resume或直接初始化新A。实际推理必须使用检查点绑定的源码，且使用当前Omni LoRA、特征预测头、生成Connector和DiT LoRA。

训练中每个rank写入`rankN.jsonl`、`progress_rankN.json`；rank0写TensorBoard的`train/{av,video,audio,feature_video,feature_audio,loss,masked_fraction}`、`grad/{connector,dit_text_lora,omni_lora,feature_heads}`和每步耗时。每100步的Dev目标诊断包含固定seed的partial/all-mask条件，不把这些指标冒充Pred25视频质量验证。

## 运行示例（本轮未执行启动）

先在独立运行根目录做短样本、长样本、更新/恢复的真实GPU工程测试，再开启正式训练。以下输出目录须替换为新的在线实验路径。三机必须能访问同一源码路径与同一份manifest/缓存；示例端口29937需确认未占用。

```bash
python scripts/pipeline3_online_cluster.py \
  --cluster configs/pipeline3_online_cluster.example.json \
  --log-dir /new/online_run/launch_C0 --preflight-only -- \
  --manifest /public/2_CKPT/0_WM/llq/p3_full_connector_then_dit_20261011/manifest.json \
  --integrity-root /public/2_CKPT/0_WM/llq/p3_full_connector_then_dit_20261011/readouts/inputs \
  --config configs/pipeline3_bernini_online.json \
  --stage C0 --initialize /path/to/selected_offline_connector.pt \
  --initialization-sha256 SELECTED_CHECKPOINT_SHA256 \
  --output /new/online_run/C0_attempt0 --control-root /new/online_run
```

去掉`--preflight-only`才会启动；预检目录也必须换成新的目录。空卡检查包含显存、利用率及进程，任一选定卡有其他作业即拒绝。cluster工具只关闭自己启动的进程组，不自动重启失败训练，不调用旧队列。

工程测试额外传`--engineering-indices /path/to/train_indices.json --engineering-steps 3 --stop-after 2`，保留真实模型与真实几何，完成2步后保存；随后改用`--resume <新版step2>`和新的输出目录、去掉`--stop-after`，验证第3步。初始化SHA仍填原温启动源的SHA。工程模式world size可变，但某一检查点的严格恢复不允许改变world size。

A显式启动时使用`--stage A --initialize <新在线C0完整checkpoint> --initialization-sha256 <该C0的SHA>`，新输出目录、新优化器；不将工程C0用于正式A。停止使用新运行根或当前输出目录的`STOP`；下一更新边界保存后退出，恢复前需要用户明确决定是否移除STOP。新代码不会删除任何STOP。

推理入口为`python scripts/pipeline3_online_generate.py --job job.json --gpu N`。job字段：

```json
{
  "version": "p3.bernini-online.v1",
  "manifest": "/path/to/manifest.json",
  "manifest_sha256": "SHA256",
  "integrity_root": "/path/to/readouts/inputs",
  "input_integrity_sha256": "SHA256_OF_COMPLETE_JSON",
  "checkpoint": "/path/to/online_step.pt",
  "checkpoint_sha256": "SHA256",
  "negative_text": "/path/to/native_negative_text.pt",
  "negative_sha256": "SHA256",
  "index": 0,
  "mode": "pred25",
  "seed": 20261012,
  "output": "/new/online_eval/sample0_pred25"
}
```

`mode=oracle`仅作目标特征上限诊断；`pred25`的模型输入不包含GT目标感知特征，读取原生缓存文件时仍在CPU验证整份文件身份，但不把其中GT targets传给模型。首帧由参考图单独编码；不读取目标音视频VAE作为采样初值。保持Euler30、video CFG3/audio CFG7，输出`sampled_latents.pt`、`generated.mp4`、`raw_audio.pt`、`completed.json`，已有正确身份的latent/video可复用。独立评分可复用现有ASR/SyncNet/ID入口。

## 本轮核验与剩余边界

- 新版核心/IO测试17项，加原P3 Planner与离线C0/A回归，合计36项CPU测试通过（25.39秒）。使用真正的小型Qwen2.5-Omni和原生LTX模块，开启非重入激活重计算；包含两rank Gloo梯度累积，从A暖启动切换联合阶段，与串行全局batch更新一致。
- 确认一次前向读出后分流；单独AV loss的Omni LoRA梯度非零，单独特征FM也能更新Omni与特征头；冻结基座不变。新建零初始化头第一步先打开输出层，不能要求所有上游梯度在全零初始化时立即非零。
- 严格恢复下一更新逐参数一致；旧版resume、新A直接加载旧C0、GT缓存篡改和通信配置注入均被拒绝。
- [完整测试记录](PIPELINE3_ONLINE_TESTS_20261012.json)。3个CLI帮助入口与10个新增Python文件语法检查通过；未启动真实模型GPU进程。
- 实际复用正式91009/48 manifest及输入封存，抽查Dev原生特征/文本/VAE读取通过：prefix `[1,513,2048]`，目标特征`[1,434,2048]`，325个视频槽位＋109个音频槽位；几何1600×576保持不变。[数据读取证据](PIPELINE3_ONLINE_DATA_CHECK_20261012.json)。
- **未验证：** 真实22B LTX＋3B Omni同卡在线反向峰值显存、三机24卡吞吐和通信稳定性、新版完整音视频质量。下一步是空卡上的真实短/长样本更新与严格恢复测试，不能直接把CPU结果当作正式训练通过。
- 当前离线C0/A作业、已有权重和缓存均未修改；本轮没有部署监控或启动新训练，也没有推送新实验分支到GitHub。

## 2026-10-12复核修复与真实GPU工程测试

用户确认保留按模态分路；正式训练仍继续原离线A，在线分支本轮只做工程验证。

- 修复Dev `partial`沿用训练全mask概率的问题：partial评估显式将全mask概率设为0；原固定48个seed中8条被错误归入partial的问题已回归覆盖。训练mask分布保持原配方。
- 修复最终检查点恢复漏做Dev验证：恢复点达到验证间隔时先重算该点Dev，再继续更新或发布完成标记；验证失败不能发布训练完成。
- 多机启动器复用既有SSH ControlMaster及跳板配置，使三机检查与训练可沿既有通路运行。
- 修复后39项CPU测试通过，包含新增的partial标签和最终恢复边界回归；不修改生成条件分路。

真实模型工程测试目录：`/public/2_CKPT/0_WM/llq/p3_online_multinode_check_20261012`。冻结测试源码：`/private/liqi_workspace/project_code/audio-driven-v4-p3-online-gpucheck-20261012`。测试用step500仅作工程温启动，不替代正式A选点。

资源：本机6–7卡保留给C0 step700验证；三机各0–5卡共18卡联合，另两台机器各6–7卡分别测试最长样本的全mask／50%mask。每组均执行C0保存step2→严格恢复至step3→A保存step2→严格恢复至step3。工程配方将Planner暖启动缩短为1步，使A第2步就覆盖在线联合梯度；每卡micro1、工程累积1，**不冒充global batch72的正式配方验证**。真实模型、首帧、原始720p-area保持不变。

实时读取该目录`state.json`、`health.json`及各组`state.json`；最终结果为`report.json`。只在全部检查完成后据实际记录判定通过，不把启动或首步成功当作全部通过。

### 实测结果（2026-10-12，逐项更新）

第一轮三组均通过，证据：[多机／最长样本报告](/public/2_CKPT/0_WM/llq/p3_online_multinode_check_20261012/report.json)。真实3B Omni＋22B LTX，覆盖三机18 rank、最长样本全mask及50%mask；C0与A均完成保存step2并严格恢复更新至step3。A暖启动后Connector、DiT LoRA、Omni LoRA和特征头梯度均有限且非零，冻结参数检查未报错。GPU轮询观测到最高107770 MiB（约105.24 GiB）；这是采样观测值，不是CUDA精确峰值。最长测试包含713帧、85500个视频latent token。

第二轮另行验证实际梯度累积：[目录](/public/2_CKPT/0_WM/llq/p3_online_globalbatch_check_20261012)。三机18 rank×micro1×accum4＝global72，另两组2 rank×accum3＝global6；读取完整Train91009/Dev48输入契约，仍只做C0/A各3步短测，Planner暖启动缩短至1步。此轮不使用engineering-indices子集；两卡组沿用历史目录名，但本轮不是固定最长样本子集。

推理功能检查：[目录](/public/2_CKPT/0_WM/llq/p3_online_inference_smoke_20261012)。从第一轮C0／A各自step3检查点重新运行Planner Pred25并生成同一短Dev22样本；固定seed1067、Euler30、video CFG3／audio CFG7。这些工程权重不参与正式离线A选点。

边界：GPU恢复检查覆盖严格载入与下一次实际更新，未检验与不中断GPU训练的逐数值等价；CPU等价测试已通过。三机最大联合规模为18 rank，不能称为24 rank测试。短步工程通过不能证明新版生成质量或联合loss权重已调好；正式在线A仍需100步暖启动及完整验证。

**10:26 UTC最终结果：第二轮与推理检查均通过。** [global72及累积3报告](/public/2_CKPT/0_WM/llq/p3_online_globalbatch_check_20261012/report.json)三组全部passed。三机18 rank×accum4的C0第2步108.41秒、恢复第3步118.45秒；A联合第2步112.15秒、恢复第3步121.31秒。仅短测、样本长短变化，不能作为稳定吞吐或提速百分比。第二轮GPU轮询最大90228 MiB。A恢复step3梯度范数：Connector0.07316、DiT LoRA0.002253、Omni LoRA0.29125、特征头0.45364；loss均有限。

[在线推理功能报告](/public/2_CKPT/0_WM/llq/p3_online_inference_smoke_20261012/report.json)C0／A两条均成功，当前检查点重新Pred25预测，129帧、1600×576、4.3秒，音视频输出与几何契约一致、波形finite，无目标特征或目标VAE输入。仅验证功能可运行，不能据工程3步权重判断模型质量。

三组工程队列已正常完成释放GPU；正式训练按用户选择使用旧离线A，从同批评选的C0 step500独立初始化，新目录`/public/2_CKPT/0_WM/llq/p3_A_best_C0_20261012`。在线源码修复留在本独立工作树，未推送；本次不启动在线正式训练。
