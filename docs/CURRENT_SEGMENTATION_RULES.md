# Seamless listen + response 当前切分规则

更新：2026-10-22。当前版本为 **V2 句子选窗 → V3 句首修复**。这是已实现规则，不是下一版的设想。

## 1. 任务与角色

每条样本生成目标人物 Listener 的连续「倾听 → 正式回应」音视频。A/B 是会话中的固定人物；Speaker/Listener 是每条样本的动态角色。

|原始相邻话轮|这一对的 Speaker|这一对的 Listener|含义|
|---|---|---|---|
|A₁ → B₁|A|B|B 倾听 A₁，然后回应 B₁|
|B₁ → A₂|B|A|A 倾听 B₁，然后回应 A₂|

因此 **A₁ → B₁ → A₂ 枚举两对**，前一对的 Listener 就是后一对的 Speaker；不是隔轮配对，也不是每两个话轮分成一组后跳过中间一对。

两对都从原始话轮独立选窗，不拿前一条已经裁短的 Listener 片段构造下一条。B₁ 在第一对可能只保留回应前缀，在第二对可能只保留作为输入的句子后缀，时间范围不必相同。每对独立检查，枚举不保证两对都最终保留。每个原始相邻 pair_key 最多留一个窗口。

小时数是样本时长相加，允许相邻配对复用内容；不能当作去重后的原视频时长。所有配对继承原会话 split，不能再随机拆配对造成跨训练/验证集泄漏。

## 2. 来源与初始话轮

- 原始 Seamless：`/public/0_DATA/2_Avatar/seamless_interaction/`，覆盖 improvised/naturalistic 的 train/dev/test。
- 按 label、split、interaction 匹配两人；要求有可读的 MP4、WAV、词级转写 JSON。全量索引为 21,428 个 tar、64,638 个配对会话，不仅是带情绪 NPZ 的子集。
- 复用官方词级转写/时间戳。不运行新 ASR/VAD；原始音频的 10 ms RMS 仅辅助寻找切口。
- `raw_utterances` 用 ≤0.6 秒的词间隔形成处理小段。**这不是最终话轮边界规则**：同一人可以跨小段继续；最终根据实质发言交替与微反应构造话轮。
- 微反应使用词表和时序启发式：通常≤4词、≤1.5秒、Speaker 仍继续说话；包括 yeah/haha 等。紧接正式回应的短确认不能轻易单独视作倾听微反应。纯笑声有稍宽的时序规则，见 `is_micro`。
- 非微反应的另一人发言视为实质切换。保留窗口内实质重叠>0.2秒排除该窗口；≤0.2秒的非零重叠仍标记复查。后续候选可能通过完整句截短避开被删范围内的重叠。

## 3. V2：完整句首至完整句末的连续窗口

1. 对每组相邻实质话轮枚举 Speaker 起点、Listener 句末终点。
2. Speaker 可从话轮起点，或内部一个完整句子的开头开始，保留连续后缀；保留部分至少5词且≥1.2秒。
3. Listener 必须从正式回应起点开始，可保留至一个完整句末，不跳过回应开头，不拼接中间片段，不将剩余回应单独另造配对。
4. 句末主要依据 `.?!`，排除省略号、缩写句点、明显未完连接词；与后续内容紧接的短确认/引导语不单独作为回应截断点。实现是启发式，不等同于语义模型确认。
5. V2 整条目标≤9.95秒，为预览编码预留约50ms；不满足时尝试 Speaker 后部完整句或更短 Listener 完整前缀。
6. **先枚举窗口，再检查保留范围内的问题**。优先选择无自动复查标记的窗口，在候选顺序内优先更早的 Speaker 起点、更晚的 Listener 完整句末；全部需复查时保留候选顺序中的第一个可用结果。

## 4. 实际声音起止

- 开头通常争取首词前约0.2秒，在最多约0.35秒的局部范围内找切口，避开前一段已转写声音；不能删掉已选首词。
- 结尾通常争取回应词尾后约0.8秒，受下一次发声、源视频结尾和时长上限限制；最多向后约1.5秒。
- 安静切口要求双方音量都低于自适应阈值，稳定低能量区间至少约60ms，切点留在其内部。清晰尾部通常至少留120ms，并为后续声音预留约80ms。
- 检查后续发声时不仅看对方接话，也看 Listener 自己下一句、微反应等；不能把下一句起音混进尾部。
- 找不到可信安静切口、余量不足、切点穿词等保留复查标记，不用淡出或静音掩盖边界问题。
- 全量不做音量统一、淡入淡出、门控、头部运动/遮挡筛选或空间裁剪。预览增益只是浏览器试听功能。

## 5. V3：补回同一句的 Speaker 句首

修复对象是 V2 中 `speaker_sentence_start_unconfirmed`。旧小段排序/话轮归属可能让完整句跨小段后，只保留后半句；不能仅把这个问题当作标点误报。

- 回到 Speaker 原始逐词时间线，从当前首词向前追溯，直到可靠句末之后或转写开头，补回**完整连续前缀**；不被旧话轮编号限制，不只固定多截几百毫秒。
- 只修复有充分时间依据的情况。词间隔>3秒、词时间异常，或新增范围任一方存在>1.5秒的单词时间戳，保持待复查；3秒仅用于保守回溯，不是全局强制话轮切分阈值。
- 不跨入对方正式发言。符合既有时序规则的微反应可以保留；对方此前发言已在恢复句首前结束，不阻止回溯。
- **完整句首优先于10秒目标**。仅为这种修复允许最多11.95秒，约等于含编码余量的12秒；这是当前对“略超10秒”的具体配置，普通样本不一并放宽。
- 补齐后仍过长就保留原样待复查，不再从补回的前缀中任意删词凑时长。
- Listener 回应起点、回应词尾、片段终点不变。重新检测新句首的安静切口；仅修复成功才清除句首完整性标记，其他标记不一起清除。
- 不新增配对、不重新组全部话轮，也不恢复 V2 已经排除的配对。新清单包含全部原配对，尚未修复的行保持原样。

## 6. 当前全量结果

|类别|配对数|相加时长|
|---|---:|---:|
|全部候选|395,038|774.706小时|
|自动无复查标记|119,789|236.558小时|
|仍有复查标记|275,249|538.148小时|

V3 补齐8,161条，其中2,146条超过10秒，最长11.947秒；仍有40,398条句首子问题未自动解决。相比V2增加5.382小时，没有增加配对数。自动无标记不等于人工确认合格，复查也不等于应丢弃。

当前生产清单：`outputs_sentence_start_repair_v3/{kept,clean,review}.jsonl`。句首未解决子集为 `unresolved_sentence_start.jsonl`，原因在 `audit.jsonl`。`summary.json`/`validation.json` 为结果依据。

修复行增加 `start_recovery`（原始边界、补回词、延伸量、是否超10秒），`speaker_start_evidence.kind=recovered_same_sentence_prefix`；`original_speaker_start_s` 仍指旧话轮起点，可能晚于新首词。未修复行逐字段保持V2，行内revision也保留V2；全量阶段以V3目录及summary为准。

## 7. 关键代码与执行顺序

|作用|入口/函数|
|---|---|
|原始tar索引、双方匹配|`scan_current_full.py`：`metadata` → `database`|
|旧版本对照清单（首次复现需生成）|`apply_current_full_rules.py`|
|微反应和话轮|`preview_listen_response_v3.py`：`is_micro`、`floors_from_words`|
|每个相邻位置都枚举一对|`apply_sentence_windows_v2.py`：`process_conversation` 中 `range(len(fs)-1)`|
|连续句子选窗与结构检查|同文件：`candidate_windows`、`retained_flags`|
|实际声学起止|同文件：`quiet_points`、`cut_window`|
|角色、时间、文本和来源输出|同文件：`make_output`|
|同句句首恢复|`repair_sentence_starts_v3.py`：`propose`、`apply_repair`|
|全量一致性验证与clean/review划分|`validate_sentence_start_repair_v3.py`|
|双向相邻配对/微反应回归|`test_listener_role_switch.py`|
|句首恢复、12秒例外回归|`test_sentence_start_repair_v3.py`|

已有V2输出时：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python repair_sentence_starts_v3.py audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python repair_sentence_starts_v3.py apply
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_sentence_start_repair_v3.py
```

从零复现还需要先执行：

```bash
python scan_current_full.py metadata --workers 24
python scan_current_full.py database
python apply_current_full_rules.py --workers 16
python apply_sentence_windows_v2.py --workers 48
```

V2 会读取旧V1 kept用于历史对照字段，因此从零复现不能直接跳过该文件的生成。已有全量结果不要仅为了备份重新运行昂贵步骤。各版本指纹和输出目录隔离，修改规则不能混用旧分片。

预览入口：[105条V2分类样本](http://127.0.0.1:8875/sentence_windows_review/)；[V3固定五条修复前后对照](http://127.0.0.1:8875/sentence_start_repair/)。这两个页面版本不同，不能把原105条页面当作V3全量已经重新渲染。
