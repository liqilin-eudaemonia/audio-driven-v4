# 数据复用与新特征契约

不全量重新打标。复用 GT-qualified 的素材、GT Plan、split；由原始结构化标签构造新文本，新增冻结 Omni 目标特征提取。

## 可复用资产

- 根目录 `/private/liqi_workspace/data/seamless/subset_300h_pure`。
- 目标媒体 `processed_pairs_720p_area/{train,dev,test}` 中 target.mp4、listener_gated.wav、speaker_gated.wav、首帧 jpg。
- GT Plan：`omni_gt_train_part0_30fps`、`omni_gt_train_part1_30fps`、`omni_gt_dev`、`omni_gt_test`。
- VAE：`ltx23_precomputed_720p_area_qualified_20260920`，只有裁切、时间和编码器一致才复用。
- 旧 teacher_forcing_raw hidden 不是目标音视频感知特征，也不能替代可训练规划态 Omni 的在线前向。

## 文本与时间

Reply 取真实台词，Delivery/Behavior 取已有真实描述；缺失保持缺失。附和与无主回复单独表达，避免删除目标音轨上真实存在的短发声。旧短模板的 speaker_end+200ms 是规则，不是实际 GT 时间，不能沿用为真实时序监督。新文本不要求生成精确时间 JSON。

所有时间相对目标首帧。观测音频只读取协议允许的范围。规划输出预算由外部指定，不由 GT 未来时长隐式决定；训练样本必须完整放入预算，过长直接报错或显式重新分桶，不截断 GT 或拉伸媒体。

## 缓存

新缓存记录 sample_id、split、模型和预处理指纹、GT 文本、感知特征、有效 mask、模态和时间/空间坐标。目标视觉与音频特征保持原生 Thinker 宽度；如后续归一化，必须保存固定统计量和回填逆变换。

当前宽度为 2048。视觉 2 FPS 采样后由原生 temporal patch size=2 合并，时间组约 1 Hz；空间网格由输入首帧预处理确定。音频池化到 4 Hz。外部预算尾部仅作存储补齐，超出真实目标时长的 token 通过 valid mask 排除。缓存存冻结输入 embedding；规划 hidden 必须在线计算。

当前指纹包含配置、Omni config 和权重索引的 SHA256，以及原始 GT JSON 的 SHA256；未计算数十 GB 权重文件与全部媒体的逐文件校验和。修改原路径下的权重或媒体后必须重新提取，不能仅依赖路径相同。

旧 VAE 预处理为了满足 8n+1 帧数可能裁去终端至多约 7 帧，规划特征则取完整原素材。AV 入口检查同一 sample ID、VAE 音视频自身时长一致，并记录与原素材的差异；超过 8/fps 容差时报错。这是同一片段的短前缀重建，**不是完全等长对齐**。正式训练前需抽查是否截掉有效回复尾音；不通过的样本应重做匹配预处理。`--tiny` 还会额外裁 latent，只能用于工程测试。

`prepare --manifest` 接受 JSON 数组，每条包含：

```json
{
  "observation": {
    "sample_id": "clip_id",
    "speaker_audio": "/absolute/path/speaker.wav",
    "listener_image": "/absolute/path/first_frame.jpg",
    "speaker_start_s": 0,
    "speaker_end_s": 4.8,
    "output_duration_s": 8
  },
  "target_video": "/absolute/path/target.mp4",
  "target_audio": "/absolute/path/listener.wav",
  "duration_s": 8,
  "text_plan": {"reply": "Actual transcript", "delivery": "", "behavior": "", "vocal_events": ""},
  "gt_path": "/absolute/path/avatar_plan.validated.json",
  "split": "train"
}
```

`text_plan` 必须等于从该 `gt_path` 确定性转换得到的结果，否则报错；完整示例见 `outputs/omni_gate_003/samples.json`。`infer-plan --observation` 仅接受上面的 observation 对象，多余目标字段由严格 dataclass 拒绝。`cache_index.json` 保存样本缓存的绝对路径，因此移动缓存时需要另建索引。

训练缓存和自主推理输入分开。推理接口仅接受输入媒体、Speaker 边界、外部输出预算；拒绝从 GT 缓存构造自主回复。仅在 teacher-forcing 训练/Oracle 诊断时读取 GT 文本和目标特征。

## 需要补标的情况

只对抽查发现的错误台词、漏标附和、与素材矛盾的动作或错位时间做定向修复。不得让 Omni 自由编造一份更理想的描述再与原素材配对重建。当前尚未完成全量新格式审计。
