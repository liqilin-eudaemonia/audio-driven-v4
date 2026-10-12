# P3全量Connector→DiT适配运行手册

## 2026-10-12：用户指定C0 step700后转A

**用户指定提前转A（2026-10-12 07:40 UTC）：** 新指令为“等P3 C0的step700权重保存后，提前进入A的训练，同时开展推理验证”，替代此前C0必须到2000及等待质量门槛的安排。自动调度已部署，核对时C0为633步，尚未进入A。第700步更新后通过当前C0输出目录的STOP在下一个更新边界正常退出，保留完整step700权重、优化器/RNG与Dev48评估；不修改正在运行的C0源码或检查点identity。随后A从该Connector初始化、DiT text-attention LoRA零增量初始化，新优化器／独立1000步，Connector LR3e-5、LoRA rank8/alpha8及LR5e-6（前50步warmup），global batch72、原始720p-area、每100步保存。冻结规划器，B不自动启动。

step700同批Dev8 Pred25推理／评分在172.29.104.194的4–7卡并行开展，启动前等待空卡；配置与step600一致，完成后加入HTML“P3同批对照 · C0 Connector700 · Pred25”。A不等待此轮生成评分；这属于明确的用户决策，**不会写成自动质量门槛通过**。绑定权重SHA和manifest SHA的授权记录为`early_transition_authorization.json`，在700完整保存并正常退出后才生成。根STOP始终有效，不删除既有STOP，不干扰其他作业。

调度状态：训练根目录`scheduled_transition.json`、`transition_state.json`、`scheduled_transition.log`；新代码快照`/private/liqi_workspace/project_code/audio-driven-v4-p3-early-a700-20261012`。当前活跃C0仍使用原2000步recipe以保留原始身份，`plan.json.scheduled_transition`记录700边界；接管时才将`stage_steps`设为C0=700/A=1000并保存旧plan/cluster。复用原监控，GPU每60秒检查、队列每15秒检查，单卡低于30%持续300秒告警。切换和验证完成会通知本会话。


2026-10-11更新。用户授权本机8卡先测试，残差P2.1最终验证结束后自动开展三机后续测试与训练；A为独立1000步。结构见[完整流程与交接](PIPELINE3_IMPLEMENTATION.md)。

## 配方

|项目|配置|
|---|---|
|C0|Train91,009，2000步，只训练生成Connector，LR3e-5|
|A|继承通过效果检查的C0 Connector，独立1000步；额外训练DiT attn2/audio_attn2 Q/K/V/输出LoRA；rank8/alpha8、LR5e-6、50步warmup|
|初始化|冻结LoRA规划器step17000；C0从对应pilot Connector100初始化、新优化器；A新建零增量DiT LoRA与新优化器|
|批量与保存|三机24卡，micro1×累积3，global batch72；每100步完整保存、Dev48固定噪声AV loss与TensorBoard|
|条件|Oracle/Pred25确定性各50%；固定GT回复与时间表|
|几何/推理|原始720p-area宽高；原生Euler30、video CFG3/audio CFG7；首帧条件，无目标VAE输入|

不自动启动B，不切换成Qwen残差。C0改善生成映射，不更新规划器；“Connector训练不足”仍是待验证假设。

## 队列

1. 等待残差step1000的96视频及GT/Omni/Reference各48条评分完成，检查完成标志和报告。
2. 本机8卡短/长C0/A更新和严格恢复已通过。随后24卡DDP各阶段更新2步→恢复1步，各rank权重摘要必须一致。
3. Dev8×纯文本/C0/A共24条原生30步生成解码工程测试。工程权重的结果不用于质量结论。
4. 全量Oracle/Pred25 hidden缓存：初始每卡4个独立worker；失败有界降至2、1；已完成样本保留。CPU核验全部91057条后封存。
5. C0全量2000步。
6. Dev8×纯文本/旧Connector100/C0 Oracle/C0 Pred25/打乱条件共40视频；实际ASR、vocal SyncNet、ID、裁剪前音频幅度检查生成绑定SHA的C0_quality_gate.json。
7. 自动非退化检查通过才进入A1000；失败报告并等待诊断，不制造通过标志。

自动门槛：Pred25 WER≤文本和旧Connector较好者+0.05；Sync成功至少6/8且较文本少不超过1；Sync-C下降≤0.3；ID均值下降≤0.05；裁剪前越界比例≤max(0.005,文本+0.001)。这是小样本筛查，不替代动作、漂移和爆音人工判断；Oracle/打乱组只作诊断。

A后应比较纯文本、仅A DiT LoRA、正常P3条件和打乱条件；本队列终点为A1000保存，尚未自动接A后完整生成评估，也不会启动B。

## 2026-10-11代码审查修复

- 质量门槛升级为 `quality_version=2`。身份比较要求基线至少6个有效样本，Pred25和共同有效样本至少 `max(6, 基线有效数-1)` 个；每个有效样本至少50%采样帧可识别人脸，共同样本帧覆盖率下降不得超过0.125。ID均值仅在共同样本上比较。旧版通过报告必须重新汇总，不能直接初始化A。
- 工程阶段恢复核验阶段、步数、停止状态、world size、manifest及完整checkpoint哈希。initial预期step2且stopped=true，resume预期step3且stopped=false。失效阶段及其后续工程结果保留归档后重跑。
- 新增 `pipeline3_staged_integrity.py`，将文本条件及两路VAE文件SHA256封存在 `readouts/inputs/complete.json` 和逐样本记录中。独立于规划hidden缓存，不改变原manifest和已有hidden。正式训练必须先完成全量输入封存；实际反序列化的同一份字节在每次读取时检查SHA256。工程测试允许先固定小子集，正式封存会复核已有记录，拒绝为变化的数据重新建立基线。
- 正式checkpoint的identity包含输入封存文件SHA256。旧checkpoint不会静默按新源码恢复；本次维护若接续已开始的C0，需 `checkpoint_transition.json` 明确绑定旧checkpoint及identity、新代码哈希和输入封存路径。仅允许这两类identity字段变更，world size、模型、阶段配方和manifest保持原值，并记录恢复使用的证书SHA256。该校验保证建立新基线后的读取一致性，不追溯证明旧版本已消费的数据内容。
- 监控使用稳定的阶段开始时间作为首个进度文件缺失时的起点，分别跟踪训练micro-step、缓存完成数量、封存数量和输入哈希进度。单纯更新时间戳不算进展。

## 运行位置

根目录：`/public/2_CKPT/0_WM/llq/p3_full_connector_then_dit_20261011`。
固定源码（审查修复版）：`/private/liqi_workspace/project_code/audio-driven-v4-p3-staged-auditfix-20261011`。
Python：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。

- plan.json：目标、依赖、门槛、超时与通知线程。
- cluster.json：node12 172.29.252.24、node17 172.29.15.126、node16 172.29.112.17，各8卡；master port29913。
- manifest.json：Train91009/Dev48，SHA256 `d9c8aeacb429ae45a3bd791f2a14214a0aff16f5f08b453ba6f6d5232a6e1ee5`。
- source_snapshot.json：源码哈希及本机测试后修补说明。
- launch.json、state.json、queue.log：进程、当前阶段、调度日志。
- watch.json、health.json、health_history.jsonl、alerts.jsonl、notifications.jsonl：监控和通知收据。
- local_gate/、gate24_*、render_gate/：工程测试证据。
- readouts/train和dev：新hidden；readouts/complete.json为封存标志。
- C0_attemptN/、A_attemptN/：rank日志、tensorboard/、dev_XXXXXX.json、step_XXXXXX.pt和.complete.json。
- C0_quality/report.json、C0_quality_gate.json：实际测量和转阶段依据。

队列入口为快照内scripts/pipeline3_staged_queue.py --root <根目录>；监控为pipeline3_staged_watch.py --root <根目录>。已经部署的实例见launch/state，交接时不要重复启动。

## 停止与恢复

在根目录创建STOP文件持久停止。训练下一步边界保存完整checkpoint并退出，不进入下一阶段；队列给保存最多900秒宽限。用户停止后不能自动删除STOP或自行续训。

阶段内最多3次恢复尝试；选最新哈希通过的完整checkpoint，恢复模型、优化器、各rank RNG，校验源码/manifest/world size/配方，不重复更新。最终step已保存但报告失败时只补结束检查。OOM、非有限loss、契约不匹配、冻结参数变化不盲目重试。

20分钟检查节点、显存、温度、利用率和进度；15秒检查队列生命周期。连续两次均值利用率<30%或20分钟无训练进度通知当前线程。队列意外消失可有界重启；guard只清理自身子进程组，SSH探测重试3次。

仅分配无计算进程、显存<1000MiB且利用率≤5%的卡，不停止其他作业凑24卡。网络故障、他人占卡或质量失败仍可能需要干预。

## 实测边界

全量清单、初始化SHA审核通过；本机8GPU的16条条件缓存及C0/A各24次更新、保存和恢复通过；13项CPU测试通过。24卡DDP、原生生成、全量缓存吞吐及长训效果以随后实际产物为准。

[A300历史CPU准备证据](PIPELINE3_STAGED_RENDER_PREP_20261011.json)保留；本次使用A1000新manifest。旧准备清单不用于启动本次训练。

本轮部署证据：[GPU/CPU/队列记录](PIPELINE3_STAGED_DEPLOYMENT_20261011.json)。记录为快照，实时状态以运行目录为准。

## 2026-10-11全量缓存校验提速

全部hidden提取后，单进程全量哈希/反序列化校验成为CPU瓶颈，导致24卡等待。运行快照已改8个CPU进程完整校验，保留每条身份、SHA256、Oracle/Pred25契约，不重新提取缓存。进度见readouts/seal_progress.json；脚本pipeline3_staged_seal.py。修补记录source_amendment_parallel_seal.json，恢复记录parallel_seal_recovery.json。仅替换本任务CPU校验和队列监督进程，不操作其他作业。开发主目录另有未部署的进度补丁，未覆盖；实际以固定快照及修补记录为准。

本次并行校验实际完成91,057/91,057条，耗时144.34秒；正式C0已启动于C0_attempt0，目标2000步，队列PID1642992。实时步数以该目录rank0.jsonl为准。
