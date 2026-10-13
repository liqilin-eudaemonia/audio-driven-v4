# P3 Bernini式在线联合版：GitHub备份（2026-10-13）

分支：[experiment/p3-bernini-online-c0-a-20261012](https://github.com/liqilin-eudaemonia/audio-driven-v4/tree/experiment/p3-bernini-online-c0-a-20261012)。版本契约`p3.bernini-online.v1`。本次归档原在线实现`33602c3`及后续验证／恢复／SSH修复；固定代码提交在推送后由索引登记。

## 这是哪个版本

这是把Bernini的条件选择思路迁移到Qwen2.5-Omni＋LTX-2.3的在线C0/A版本，不是旧离线Oracle/Pred25缓存版，也不宣称完整复现官方训练配方。输入仍为Speaker音频、Listener首帧及固定GT回复计划／时间表；不包含自主回复／时长实验。

一次在线Omni forward后，前缀＋未mask目标槽位hidden按模态进入生成Connector，与Gemma条件拼接后指导LTX；masked槽位hidden进入特征Connector／FM decoder。特征Connector和生成Connector参数独立，本版没有实现待验证的共享主干。无GT目标特征直接进入生成器；训练未mask特征作为Omni上下文，推理Pred25从全mask预测回填，再读当前规划器hidden。

|阶段|训练模块|监督与冻结边界|
|---|---|---|
|在线C0|生成Connector|AV loss；Omni及全部规划头、原DiT冻结|
|在线A前100步（默认）|生成Connector＋DiT text-attention LoRA|AV loss；规划器前向不保留梯度|
|在线A第101步起（默认）|上述模块＋Omni LoRA、特征Connector／decoder、mask／模态参数|特征FM＋AV在线共同优化；AV可回传Omni，特征头由特征FM监督，DiT LoRA由AV监督；基座／感知塔／Gemma／VAE冻结|

DiT仅视频`attn2`、音频`audio_attn2`的Q/K/V/输出投影加LoRA，不是P2.1独立Qwen残差版的六类attention。原生感知、前缀input embedding、文本与VAE缓存可复用，最终Omni hidden不使用离线缓存。严格恢复绑定输入封存、阶段、配方及各rank状态；不自动删STOP，不自动启动B。

## 本次包含的修复

- Dev partial mask明确排除全mask概率，避免错误混淆partial与all-mask两组。
- 从最终检查点恢复时先补做应执行的Dev验证；验证失败不发布训练完成标记。
- 多机SSH支持既有ControlMaster与ProxyCommand通路；示例配置需按部署环境修改。
- 新增上述边界的CPU回归，并保留在线梯度、条件隔离、严格恢复、原生CFG与两rank累积测试。

## 验证证据与限制

- 本次重新运行4个测试文件，**39项CPU测试通过**；未占用GPU。检查摘要及源码哈希见[备份核验记录](PIPELINE3_ONLINE_BACKUP_CHECKS_20261013.json)。
- 2026-10-12真实GPU记录：三机18 rank和两组2 rank均完成C0／A更新、保存、严格恢复；第一轮覆盖最长样本713帧和全mask／部分mask，第二轮18×micro1×accum4验证global72；工程A将暖启动缩短至1步以覆盖联合梯度。
- 当前checkpoint重新Pred25的C0／A短样本推理通过，保持原始1600×576／4.3秒、Euler30、CFG video3/audio7；仅功能验证。
- **未验证：** 三机24 rank在线联合、长期稳定吞吐、默认100步暖启动后的长训收敛及正式生成质量。GPU恢复测试未做不中断训练逐数值等价比较，CPU恢复等价已测。短步结果不能当作成熟模型效果。

完整历史记录：[在线设计、命令及GPU结果](PIPELINE3_BERNINI_ONLINE_PLAN_20261012.md)。本次备份不启动或切换训练。

## 代码入口与复现依赖

- [在线模型](../src/omni_av_p3_online/model.py)、[输入完整性](../src/omni_av_p3_online/data.py)、[检查点](../src/omni_av_p3_online/checkpoint.py)。
- [训练](../scripts/pipeline3_online_train.py)、[推理](../scripts/pipeline3_online_generate.py)、[多机启动](../scripts/pipeline3_online_cluster.py)。
- [训练配置](../configs/pipeline3_bernini_online.json)、[集群示例](../configs/pipeline3_online_cluster.example.json)。
- 复用仓库P3 Planner、P2原生LTX接线及P2.1 LoRA等共享依赖。外部Omni-LTX-stage2软件环境、预训练模型和数据须另外准备；路径不是便携安装包。
- Git仅备份代码、配置、测试和文档；模型权重、优化器、数据缓存、生成媒体、SSH私钥不上传。备份验证摘要只包含结果与来源哈希，不打包大型运行目录。
