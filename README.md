# Seamless listen + response 切分代码备份

2026-10-22 的冻结实现：**V2 句子选窗 + V3 同句句首恢复**。当前数据为395,038对、774.706小时（相加时长），其中自动无复查标记119,789对、236.558小时。本仓库快照用于备份与复现，不包含数据集或媒体。

**相邻角色已双向覆盖**：A₁→B₁→A₂ 枚举 A₁→B₁、B₁→A₂ 两对，前一对的 Listener 成为后一对的 Speaker。两对从原始话轮独立选窗，保留范围可不同，不要求两对都通过筛选。相邻配对复用内容，小时数不能当作去重后的原始视频时长。

先看 [完整当前切分规则](docs/CURRENT_SEGMENTATION_RULES.md)，再看 [V3实际运行记录](docs/V3_RUN_RECORD.md)。以下是 GitHub 快照的文件组织与验证方式。

## 文件组织

|路径|内容|
|---|---|
|`pipeline/scan_current_full.py`|原始tar索引、转写/时间戳与PCM信息、双方匹配|
|`pipeline/apply_current_full_rules.py`|旧V1基线；V2历史对照字段依赖其kept清单|
|`pipeline/apply_sentence_windows_v2.py`|所有相邻配对、Speaker句子后缀、Listener连续前缀、结构与声音边界|
|`pipeline/repair_sentence_starts_v3.py`|补齐同一句的前半句，修复专用11.95秒上限|
|`pipeline/preview_listen_response_v3.py`|当前使用的微反应与话轮函数；文件名来自早期预览|
|`pipeline/validate_*`、`pipeline/test_*`|生产结果验证、合成回归、角色交换检查|
|`pipeline/build_*`、`pipeline/publish_*`、HTML、浏览器检查|105条分类预览和5条V3前后对照的代码|
|`reports/`|已完成全量运行摘要、生产指纹、预览验证、本快照重放结果|
|`SOURCE_MAP.json`|所有复制文件的原位置、SHA-256、大小和递归本地import依赖图|
|`DATA_ARTIFACTS.json`|本地数据清单、元数据索引的大小和SHA-256；不含其内容|
|`tools/check_backup.py`|独立快照检查；不读取原始数据、不改生产清单|
|`tools/prepare_seamless_github_backup.py`|原工作区的白名单归档脚本存档；恢复至原seamless代码目录后运行|

共22个Python实现/验证文件，保留原文件名及内容，以便匹配生产指纹；没有为整理目录而重构已经验证的规则。部分历史预览模块被生产代码import，因此一并保留，但它们的历史主入口不是当前全量流程。尤其不要误用旧预览的音量归一化流程处理当前全量数据。

## 当前规则摘要

- 仅按规则和已有词时间戳切分，不运行新的ASR/VAD或视觉模型。
- 倾听中的微反应保留；实质发言按话轮切换，保留范围内实质重叠>0.2秒不接受，轻微重叠保留复查。
- Speaker可从完整句首开始取后缀；Listener从正式回应起点开始，在完整句末截取前缀；不剪掉中间内容后拼接。
- 普通V2窗口≤9.95秒。当前V3只为补齐Speaker同一句句首允许≤11.95秒，回应起点与终点不变。
- 全量补齐8,161条，其中2,146条超过10秒，最长11.947秒；40,398条句首问题仍待复查。
- “自动无标记”不是人工保证合格；“待复查”也不是应全部丢弃。没有新增头部运动、遮挡或音量筛选。

## 验证快照

Linux，原运行环境Python3.13.12；依赖版本在 `requirements.txt`，预览还需FFmpeg/FFprobe、Playwright及系统浏览器运行库。

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/check_backup.py
```

检查代码哈希、本地依赖闭包、V2/V3生产指纹，并在临时输出目录重放V2窗口、V3句首修复与双向相邻配对的合成回归。测试不重新处理全量，也不会写入原有生产输出。

`reports/v3_full_validation.json` 是之前完成的395,038条生产清单验证，不是上述合成测试重新生成的结果。`reports/backup_validation.json` 是本独立快照的新检查结果，二者区别保留明确。

## 恢复与全量复现

原代码位置：`/private/liqi_workspace/data/pipeline/seamless/`。原始Seamless位置：`/public/0_DATA/2_Avatar/seamless_interaction/`。

推荐先在独立位置恢复并检查快照，再按 `SOURCE_MAP.json` 对照恢复所需文件；不要覆盖更新的现场改动。脚本是生产代码原样快照，部分数据路径和预览发布路径仍指向原工作区。换机器时应配置相同挂载，或明确修改路径、生成新指纹及独立输出目录，不混用历史分片。

从零处理需要原始数据；它不随Git上传。在 `pipeline/` 对应的运行代码目录依次执行：

```bash
python scan_current_full.py metadata --workers 24
python scan_current_full.py database
python apply_current_full_rules.py --workers 16
python apply_sentence_windows_v2.py --workers 48
python repair_sentence_starts_v3.py audit
python repair_sentence_starts_v3.py apply
python validate_sentence_start_repair_v3.py
```

已有V2清单与元数据时只需最后三个命令。旧V1 kept是V2历史对照字段的输入，所以完整复现不能只备份V2/V3两个脚本。所有入口均依赖本目录保留的本地模块。

最终读取 `outputs_sentence_start_repair_v3/kept.jsonl`，按需使用 `clean.jsonl` / `review.jsonl`。V3未修复行逐字段保持V2，因此行内revision可以还是V2；最终阶段看V3目录及其summary。全量只产出清单；预览媒体属于独立抽样产物。

## 备份范围

上传代码、规则文档、合成测试、聚合统计及哈希记录。原始视频/音频、逐词转写清单、SQLite元数据库、模型、缓存、预览媒体、密钥和认证配置均不包含。数据本身仍由原存储系统保存，代码备份不能替代数据备份。
