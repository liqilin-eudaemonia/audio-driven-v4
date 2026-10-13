# P3 Beta10：无人值守温启动／重新初始化对照

2026-10-13用户授权：按已讨论方案实现自动验证与线路选择，未来十小时持续安排训练和验证，仅管理本任务资源。不会为维持利用率占用其他作业的卡；必要的模型加载、保存、验证和阶段切换会有短暂间隙。硬件故障不能保证零停顿。

## 初始化、数据与资源

沿用已完成P阶段按Dev48选择的P step400：
`/public/2_CKPT/0_WM/llq/p3_beta10_p_c0_a_full_20261013/P_attempt0/step_000400.pt`
SHA256：`f7d8ff18eb296fb3a780208f39815abc7b35625f782f6e2206266dee777ba67f`。

- warm：生成Connector沿用旧离线C0 step500。
- fresh：只重新初始化生成Connector，保留构造器的随机隐藏层和零输出投影；不清空Planner/FM Decoder，不加载旧生成Connector。
- 两路均从C0 step0开始，新优化器；除初始化模式外配方一致，同P、样本顺序、mask和噪声种子。
- 各三机12卡，每台warm用0–3、fresh用4–7，micro1×accum6=global72。合计24卡。全Train91009/Dev48，原始720p-area，Beta(10,1)，固定GT回复/时长。
- 原24卡单路C0在更新边界停止并保留STOP/权重。它不跨world size恢复到这两路。新路每100步保存与Dev48验证，然后在相同12卡拓扑下严格恢复到新attempt目录。

## 自动验证与选择（预先规定的代理指标）

每100步保存后自动排入原同批dev8：4、6、11、15、22、34、42、47。每个checkpoint恢复自己的Planner重做Pred25，Euler30，CFG video3/audio7，seed1045+index。随后Whisper计算CER/WER，Vocal SyncNet计算同步；检查原始音频有限值、截幅比例和静音异常。

推理优先使用172.29.104.194真正空闲的卡。该机无空卡时，在100步块边界将本任务一组12卡用于推理/评分，另一组继续训练。处理完待验证任务后恢复该组训练，不能同时把两个训练组都用于C0验证。A阶段使用24卡；外部无空卡时在保存边界使用本任务卡完成验证后严格恢复A。

两路按相同步数比较，最低各300步；300/400/500以后每100步均可评审。必须具备Dev48和配对dev8生成/识别结果；不会用缺失结果当零分。口型同步重复失败后保留错误证据，允许降级为无同步指标的代理排名，但不提前选优。

比较指标：Dev48 Beta视频loss、音频loss、dev8汇总CER、至少6条共同有效样本的SyncNet距离。各域等权计算对称相对差。提前优势需平均幅度至少2%、单域回退不超过5%，且300与400连续同向。500步后差异≤1%时保守选warm；这些阈值是本次实验预定启发式，并非显著性检验或公认最优标准。波形非有限、截幅比例>0.1%或RMS<1e-5的路线不能作为健康候选，两路都异常则继续训练并保留诊断。

选路不等于立刻切A。胜出路线最近3个验证点的AV总loss最佳改善不足0.5%才视为平台期；否则继续比较和训练，C0预算上限2000步。最终从所选路线已评分的300步及之后候选中，按固定warm300基线归一化的视频loss/音频loss/CER均值选检查点（同分更早）。路线选择纳入同步；检查点排名不混入不同同步覆盖样本。选路、检查点选择均为自动代理规则，不能声称人工感知最优。保存`comparisons/*.json`与`selection.json`。

A从所选C0权重初始化，新优化器，三机24卡、accum3、global72、独立1000步。前100步Connector+DiT LoRA，第101步起加入Omni LoRA和特征头，AV+特征FM；每100步保存/验证。无需用户在场确认，不启动B。

## 恢复与监控

根STOP持久有效，不自动清除。训练失败从本路线最新SHA验证通过的完整检查点恢复（含optimizer、RNG、world、源码/配方身份），输出使用新attempt；最多3次恢复。非有限loss、冻结模块被更新、身份不一致、显存不足等明确错误会隔离路线并通知，不反复盲跑。推理/评分最多2次重试；SyncNet降级明确记录，不伪装成功。主控制器有独立监督进程，心跳失效或退出后最多3次重启；只关闭其拥有的guard进程树，不扫描杀死其他任务。

GPU每60秒采样，逐卡利用率低于30%持续300秒告警；控制器心跳、训练加载/backward进度、GPU进程与显存同时记录。十小时是无人值守覆盖期，队列不会到点强制停训；完成A及待验证任务后正常结束，不以无意义负载维持GPU利用率。

## 入口和产物

- 源码：`project_code/audio-driven-v4-p3-autoselect-20261013`
- 主控制器：`scripts/pipeline3_ab_autopilot.py`
- 选优规则：`scripts/pipeline3_ab_policy.py`
- 训练初始化：`src/omni_av_p3_online/checkpoint.py::configure_connector`
- 运行根：`/public/2_CKPT/0_WM/llq/p3_beta10_ab_autopilot_20261013`
- 监督入口：运行根`code/supervise.py`，主状态`controller.json`/`state.json`，监控`health.json`/`health_history.jsonl`/`alerts.jsonl`。
- 可视化：http://127.0.0.1:18780/omniresponse-media/p3_ab/0
- TensorBoard：http://127.0.0.1:18787/#scalars ，18=warm、19=fresh、20=选优后A。

## 验证证据

52项CPU回归/调度测试通过；追加完整配对选优与平台切换测试1项通过，共53项。真实两组各三机12卡、global72完成warm/fresh初次更新、保存和严格恢复更新，共4组检查通过；包含原始713帧长样本。两路mask统计和样本顺序一致，梯度有限。fresh首次零输出投影梯度较大，经既有全局norm1裁剪后更新，第二次梯度已回落。

工程检查点不进入正式训练。正式warm/fresh从原P400独立开始。真实自动dev8生成/评分将在首次正式100步检查点保存后执行；沿用已验证的生成和评分入口，不能把本轮调度测试称作已经完成新权重的生成质量验证。

## 2026-10-13 dev22近静音评测修复

fresh step100的dev22原始音频峰值0.000431，解码音频RMS约9.276e-5。Vocal分离器连续报告near-silent/empty并跳过输出WAV，SyncNet包装器随后仍调用ffmpeg，产生文件不存在错误。warm同条RMS约0.103824、CER0.06897；fresh CER0.96552。仅这一条证据不足以提前决定路线。

只修改本项目评测入口`pipeline21_residual_eval_metrics.py`：同时核对分离器near-silent日志、准确的缺失stem路径及原始错误，符合时记为`no_detectable_vocal`、`metric_valid=false`，不填写SyncNet分数，也不继续对同一确定性失败重试。其他错误仍抛出；未改共享SyncNet仓库、训练绑定源码、优化器或模型。新增3项针对性测试通过。已完成任务保留原报告备份及`classification_audit.json`，保留实际ASR和RMS，不把音频缺陷转换为有效口型分数。自动选择仅在共同有效同步样本上比较，CER仍包含该样本。
