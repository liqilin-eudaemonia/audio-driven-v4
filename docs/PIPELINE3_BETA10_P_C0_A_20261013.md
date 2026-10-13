# P3 Beta(10,1) 全量 P→C0→A

用户于2026-10-13确认：按Bernini的任务相关Beta策略改造，修改并测试后开始全量训练。旧Uniform在线实验独立保留，不改变原配方、检查点或实验名称。

## Mask与梯度

每条样本抽取 r~Beta(10,1)，音视频共享r，各模态独立打乱有效槽位，遮蔽ceil(N*r)个。不额外掺入20%全mask，不要求保留可见槽位；全mask时生成条件只含前缀。padding不参与遮蔽/监督。三阶段采用同一规则，FM只监督masked槽位，生成分支只接前缀与同模态可见槽位hidden。

选择Beta(10,1)参照Bernini论文I2V配置；音视频共享比例与分模态选择是本项目扩展，不是原论文公布的音视频配方。论文：https://arxiv.org/html/2605.22344v2#S4.SS1 。它的期望遮蔽率约90.9%，不是固定遮蔽91%。推理仍全mask起步、随机顺序和余弦回填25轮。

## 正式阶段

- P：旧Planner17000温启动，完整Train91009采样，适配500步。仅训练Omni LoRA、音视频特征Connector/FM Decoder、MASK/模态参数；不加载LTX，不计算AV loss。新优化器，头/LoRA学习率各1e-5、前50步warmup。
- C0：从P候选中选择权重，训练生成Connector2000步。生成Connector另从已选旧离线C0 step500初始化；不导入旧Uniform在线C0或旧A。Planner、DiT冻结。
- A：继承新Beta C0，独立1000步。前100步训练生成Connector和DiT text-attention LoRA；第101步开始加入Omni LoRA与特征头，AV+音视频特征FM联合训练。

统一三机24卡、micro1×accum3=global72；原始720p-area、固定GT回复/时间。每100步保存及Dev48验证。P在step0保存初始基线并评估。C0使用P的0/100/200/300/400/500候选中全mask FM归一化均值最小者：0.5×(视频FM/基线视频FM+音频FM/基线音频FM)，同分选更早步数；分别记录两模态值，不把FM选择当作生成质量结论。基线可胜出，避免强制使用适配更差的候选。

## 身份和停止

新版本p3.bernini-beta.v2；旧Uniform recipe/源码不能严格恢复到本版本。阶段切换只复制匹配参数，不继承优化器；P→C0保留适配后的Planner，C0→A保留Planner/Connector并新建零增量DiT LoRA。工程检查点不能初始化正式C0/A。

STOP保留，更新边界保存退出，不进入下一阶段，不启动B。仅管理本次拥有的进程。配置、源码、manifest、输入seal、初始化SHA、world及逐rank RNG均绑定恢复身份。

## 测试和部署

必须通过CPU的Beta统计/边界、全mask条件、P训练梯度、P→C0→A精确参数传递、严格恢复下一次更新、旧配方拒绝，以及真实多机P/C0/A训练、保存恢复、A暖启动→联合阶段测试。工程测试可缩短A暖启动但必须显式engineering身份；正式A仍保留100步暖启动。完成后从原始已批准权重重新开始正式P，不能把工程更新混入正式训练。

正式运行后保持GPU利用率和进度监控。C0/A每100步完整权重保存后，在172.29.104.194真正空闲卡运行原同批dev8 Pred25推理；P只有特征预测指标，不能直接作为完整渲染检查点。推理必须恢复本次检查点中的Planner重新预测特征。
