# Pipeline 3 离线Oracle／Pred25条件版：C0→A代码备份

版本标识：`P3-offline-oracle-pred25-c0-a-20261012`。整理日期：2026-10-12。

GitHub分支：[backup/p3-offline-oracle-pred25-c0-a-20261012](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/backup/p3-offline-oracle-pred25-c0-a-20261012)。已推送。固定代码提交：[6ee6719](https://github.com/liqilin-eudaemonia/audio-driven-v4/commit/6ee67193c31a9b2ba18874f96736cb5ace406696)；提交含详细中文说明。后续索引提交仅登记此代码版本，不改变训练实现。

## 版本定位与边界

这是**固定GT回复／时间表、规划器独立预训练、Oracle／Pred25离线hidden混合、生成Connector C0→DiT text-attention LoRA A**版本。不是刚讨论的Bernini式在线mask分流联合训练。A的“联合”仅指生成Connector与DiT LoRA共同接受AV loss，**不包含Omni**。

|阶段|输入与目标|训练模块|冻结模块|
|---|---|---|---|
|P3规划器|Speaker音频＋Listener首帧＋assistant中的GT回复计划／时间表，追加部分mask的GT目标音视频感知特征；只对masked槽位计算特征FM|音视频特征Connector、Embedding Decoder、mask／模态参数；LoRA组另训Omni LoRA|感知塔与Omni基座；Frozen组连Omni LoRA也不训练；此阶段没有LTX|
|C0|完整Oracle／完整Pred25条件按样本约各50%；仅计算LTX视频／音频生成loss|两路生成Connector|规划器全部参数、特征预测头、DiT、Gemma、VAE|
|A|沿用C0离线条件和AV目标|生成Connector＋DiT视频attn2／音频audio_attn2的Q/K/V/输出投影LoRA|规划器全部参数、DiT其他权重、Gemma、VAE|

Oracle：读取GT Listener目标音视频感知特征，全部回填冻结Planner后读目标槽位hidden。Pred25：从全MASK出发，25轮预测回填后再forward读目标槽位hidden。两套hidden离线保存；一次训练样本只选择一套，不相加、不按token各取一半。生成Connector显式detach输入，AV梯度不回传Omni。

生成路径：`离线目标槽位hidden → 视频/音频生成Connector → 与原生Gemma计划文本条件拼接 → LTX text cross-attention → AV loss`。没有新Qwen残差attention。规划器已有的feature Connector与C0/A的generation Connector是不同模块。

文本：user提供Speaker音频和Listener首帧，没有显式时间文字；assistant前缀仍包含GT回复计划及`Fixed target timeline`（时长、vocal_track、floor_track）。Pred25预测的是音视频特征，不是自主回复文本／时刻。Oracle使用实际推理拿不到的目标特征，结果只能作诊断。更多结构与交接信息见[完整P3说明](PIPELINE3_IMPLEMENTATION.md)。

**尚未实现：** 同一次在线Omni forward中，前缀＋未mask槽位hidden指导LTX、masked槽位hidden接受特征FM、两类loss共同回传Omni LoRA。刚讨论的C0/A调整为下一版本建议，本分支不包含该改造，不自动开启B。

## 配方与备份时运行状态

- Train91,009／Dev48；三机24卡，global batch72，每卡micro batch1、累积3；每100步保存模型／优化器／各rank RNG，并记录TensorBoard与Dev48固定噪声AV loss。
- 原始720p-area几何，保持宽高比；原生Euler30生成，video CFG3／audio CFG7。Dev loss与视频生成验证分开记录。
- 冻结LoRA规划器step17000；C0从先导生成Connector100初始化，C0重新计步、使用新优化器。Connector LR3e-5。
- A继承C0 Connector，DiT text-attention LoRA rank8／alpha8、零增量初始化；新优化器、独立1000步，DiT LR5e-6，前50步warmup。
- 仓库`configs/pipeline3_staged_render.json`保留原C0=2000/A=1000的历史recipe。已部署用户授权调度在C0完整保存step700后停止C0、切换A；不篡改旧checkpoint identity，也不把用户指定切换写成质量门槛通过。
- **2026-10-12 08:44 UTC只读核对：C0训练日志到670步，最新完整保存600步；调度等待700，尚无A正式训练完成的证据。** 这是备份时快照，不能作为之后的实时状态。
- 运行根目录：`/public/2_CKPT/0_WM/llq/p3_full_connector_then_dit_20261011`。实时查看`state.json`、`transition_state.json`、当前attempt的`rank0.jsonl`与`health.json`。

当前C0使用`audio-driven-v4-p3-staged-auditfix-20261011`；700转阶段源码为`audio-driven-v4-p3-early-a700-20261012`。本分支归档工作区最新实现，本次选入的P3及共享src文件与700转阶段快照逐文件相同；训练／调度脚本也一致。源码来源与SHA见[采集清单](PIPELINE3_OFFLINE_SOURCE_MANIFEST_20261012.json)。清单记录拷贝瞬间内容，README/CHANGELOG等随后增加本次备份说明，以Git提交为最终文件版本。

## 代码入口与复现

|用途|入口|
|---|---|
|Planner模型、mask、特征FM、采样、多机训练|`src/omni_av_p3/`；`scripts/pipeline3_parallel.py`、`pipeline3_cluster.py`|
|全量原生特征缓存|`scripts/pipeline3_full_cache.py`|
|Oracle/Pred25 hidden缓存及封存|`scripts/pipeline3_staged_prepare.py`、`pipeline3_staged_cache.py`、`pipeline3_staged_seal.py`、`pipeline3_staged_integrity.py`|
|C0/A模型与训练|`src/omni_av_p3_staged/model.py`；`scripts/pipeline3_staged_train.py`|
|多机队列、STOP、严格恢复、监控|`scripts/pipeline3_staged_queue.py`、`pipeline3_staged_watch.py`、`pipeline3_staged_worker.py`|
|C0 step700转A调度|`scripts/pipeline3_schedule_transition.py`；`tests/test_p3_early_transition.py`|
|原生生成、评分、诊断|`scripts/pipeline3_staged_generate.py`、`pipeline3_staged_quality.py`、`pipeline3_diagnostic_*.py`|

详细配置与启动/恢复边界见[完整P3说明](PIPELINE3_IMPLEMENTATION.md)、[运行手册](PIPELINE3_STAGED_RENDER_PLAN_20261011.md)。队列脚本会操作指定运行目录和节点，**已有实例运行时不要重复启动**。备份操作没有调用这些队列、没有切换运行源码或重启训练。

参考环境：`/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。需另备Qwen2.5-Omni-3B、LTX-2.3、Gemma、LTX训练器依赖、数据媒体与清单、规划器权重、hidden缓存、评分模型/SyncNet。`pyproject.toml`为基础Python依赖声明，不是完整环境锁文件；仓库中的绝对路径、节点IP、SSH key路径和外部仓库路径均需按目标环境配置。密钥内容不在仓库。

建议先执行CPU回归及CLI帮助检查，再按运行手册准备独立输出目录。不能将此Git备份直接代替已部署快照去严格续训；源码身份与运行manifest绑定，需要遵循已有显式迁移逻辑。模型权重、数据、媒体、缓存、TensorBoard事件与运行日志原件仍保存在共享存储，不上传GitHub。

## 验证与效果边界

初版完整工作区快照在独立备份工作树运行P3相关12个测试文件：**66项CPU测试通过（19.50秒）**，覆盖Planner、缓存并发、分布式契约、C0/A训练范围与恢复、输入完整性、质量门槛、STOP/监控、700转阶段等。精简提交后仅导出Git暂存文件，独立再次运行66项CPU测试通过（17.29秒），8个CLI与内部导入检查通过；[本轮验收记录](PIPELINE3_OFFLINE_BACKUP_CHECKS_20261012.json)。没有占用GPU重跑训练或质量验证；历史多卡工程实测见已有运行文档，不能把CPU回归等同于本次新做了24卡验证。

同批Dev8的C0 400→500，Pred25 WER48.4%→32.6%；500→600又到48.4%，AV loss下降并不保证生成质量单调改善。仍有动作偏少、音频不稳定、细节漂移等待定位问题，不宣称质量达标。跨Pipeline报告和合集网页仍保留在主工作区，本次新提交聚焦P3源码与版本说明。本地HTML入口为`http://127.0.0.1:18780/`（需相应端口转发），本分支不上传合集媒体或额外跨Pipeline报告。

本分支继承既有P3备份`92f99210682f69a1153e5b2ffdae6358b7517d80`，补齐C0/A、生成评分与必需共享依赖；P1/P2/P2.1已有内容从旧GitHub提交继承，本次不新增这些路线的无关历史文档或训练实现。独立工作树提交，不移动主工作区分支、不改其暂存区。

## 上传范围收敛与远端检查

自动审批曾拒绝初版106文件推送，理由为额外P2/P2.1历史资料及未确认远端可见性。该推送未成功。随后将新增内容收敛为P3源码、测试、配置、版本说明及直接依赖；不上传额外P2/P2.1实验历史、跨Pipeline分析、合集网页或运行元数据副本。旧版原有文件从同仓库已推送提交继承。

2026-10-12检查：对目标仓库的匿名GitHub API读取返回404（对照公开python/cpython返回200），同一目标仓库使用既有专用SSH凭据能读取已知远端分支。这与历史备份文档标注的私有仓库一致。凭据只用于Git认证，不读取或提交私钥内容；新分支不改变仓库可见性。
