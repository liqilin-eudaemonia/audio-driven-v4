# V3：补回 Speaker 同一句话的句首

日期：2026-10-22。用户确认：回溯同一句中缺失的若干词语，句首完整优先；修复后略超 10 秒可以接受。

这是 V2 选窗之后的**必要修复阶段**，不修改已经冻结的 V2 代码、输出或预览。处理现有 395,038 对候选，不重新枚举配对，不恢复 V2 已排除的配对。

## 规则

1. 只处理 `speaker_sentence_start_unconfirmed`。从当前 Speaker 首词在原始逐词时间线中的位置向前追溯，直到前一个可靠句末标点之后或原始转写开头；不受旧话轮编号限制，也不只补固定秒数。
2. 必须补回完整连续前缀。相邻词间隔超过 3 秒、词时间顺序异常、恢复范围内任一方存在超过 1.5 秒的单词时间戳时，保持原样待复查；这些阈值仅用于保守恢复，不把全局长停顿自动拆成新话轮。
3. 按原始双方发言小段检查新增范围。可保留既有词表和时序规则认可的微反应；若跨入对方正式发言，暂不恢复。对方已在所恢复句子开始前说完，不阻止恢复。
4. **只为句首恢复放宽到 11.95 秒**，为编码预留约 50 ms，约等于 12 秒上限。此具体上限是本次对“稍微超出 10 秒”的实现选择；普通未修复样本仍≤9.95秒。完整前缀放不下时不截掉几个词凑时长，原条目保留复查。
5. Listener 回应开始、词尾、实际片段终点均不变。重新寻找句首低能量切口；只读新起点附近的原始 PCM，复用 V2 的两路能量阈值，以免改变尾部判据。没有安静切口仍保留相应标记。
6. 只有确认恢复后才去掉句首完整性标记；其他问题不一并清除。新增微反应保留并记录，不做增益、淡入淡出、门控、视觉筛选或新模型推理。

## 全量结果

- 总候选仍为 **395,038 对 / 774.706 小时**，比 V2 增加 5.382 小时，全由向前延伸带来，不是新增配对。
- **8,161 条补齐句首**；其中 **2,146 条超过 10 秒**，最大 11.947 秒。
- 自动无复查标记：**119,789 对 / 236.558 小时**。
- 仍有复查标记：**275,249 对 / 538.148 小时**。
- 句首子问题剩余 **40,398 条**，均保留，不因无法修复而硬删。

无法自动恢复的原因：跨正式发言 13,583；前后词不连续 13,499；完整前缀超过 12 秒 11,809；加句首余量后超过上限 123；恢复范围内长词时间戳 1,384。原因互斥，具体配对查 `audit.jsonl`。这些是保守规则的处理结果，不等价于人工判断全都不合格。

## 固定五条对照

页面：<http://127.0.0.1:8875/sentence_start_repair/>。保持原 056—060 编号，提供修复前、当前片段、前后文及源时间定位。复用现有 8875 服务，原 105 条页面保持 V2 以供对照。

|样本|处理|当前时长|
|---|---|---:|
|056|补回 `How will we find`|8.613 s|
|057|补回 `I am so`|9.095 s|
|058|此前词语与当前起点不连续，保留复查|9.616 s|
|059|此前孤立 `I` 的时间戳距当前发言约 23 秒，保留复查|4.753 s|
|060|回溯会跨正式发言，保留复查|6.084 s|

056、057 虽已补回文本句首，新的句首仍没有满足规则的稳定低能量切口，因此仍有声学复查标记。没有把“文本补齐”宣称为起音已人工确认。

## 清单与版本字段

- `kept.jsonl`：V3 全部候选。
- `clean.jsonl` / `review.jsonl`：经验证后生成的分组清单；不是人工合格/不合格结论。
- `unresolved_sentence_start.jsonl`：尚未解决句首子问题的 40,398 条。
- `audit.jsonl` / `audit_summary.json`：只依赖转写的恢复计划及原因。计划有 2,144 条超过10秒；实际安静切口计算后为2,146条，以 `summary.json` 为准。
- `apply_summary.json` / `summary.json` / `validation.json`：执行统计、最终统计、逐条验证。
- `sample_results.json` / `preview/`：固定五条原始记录和前后对照媒体。

修复行新增 `speaker_start_recovered:true`、`start_recovery`（原始时间、补回词、延伸秒数、超10秒情况）；`speaker_start_evidence.kind` 改为 `recovered_same_sentence_prefix`。`original_speaker_start_s` 仍记录旧话轮起点，新的首词允许早于它；`speaker_trimmed` 保留 V2 是否取了话轮后缀的含义，不能单独用于识别恢复。

未修复行逐字段完全等于 V2，因此行内 `revision` 仍为 V2；整份清单的阶段以目录、`summary.json` 为准。全量仍只生成清单，`media_rendered:false` 不因独立五条预览而改变。

## 代码与复现

```bash
cd /private/liqi_workspace/data/pipeline/seamless
# 首次从原始元数据出发时，先运行原 V2 全量选窗；已有 V2 则不用重跑。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python repair_sentence_starts_v3.py audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python repair_sentence_starts_v3.py apply
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_sentence_start_repair_v3.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_sentence_start_repair_v3.py
# 固定五条预览
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python repair_sentence_starts_v3.py preview
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python publish_sentence_start_repair_v3.py
LD_LIBRARY_PATH=/tmp/seamless_browser_libs/extracted/usr/lib/x86_64-linux-gnu PYTHONPATH=/tmp/seamless_browser_deps python check_sentence_start_repair_v3.py
```

验证覆盖：395,038 对一一对应；修复仅改变获准的句首字段；恢复词逐个与原始连续词序列相同；回应时间不变；未修复行完全不变；12秒例外仅用于修复行；清单分组计数；V2源代码指纹未变。单元测试包含真实类型的句首跨小段、微反应保留、正式插话阻拦、异常时间戳、超过10秒但完整保留、12秒保护及微秒舍入比较。

初次应用因二进制浮点表示与六位小数舍入比较失败（例如59.04099999999994与59.041），修正断言容差后重新从头应用，未改变切点算法。原失败版本及日志保存在 `pre_rounding_assert_fix/`。
