# Third-party code notices

`src/omni_av_p2/decoder.py` adapts the `TimestepEmbedder`, `ResBlock`, `FinalLayer`, and `SimpleMLPAdaLN` topology from Bernini's `bernini/models/diffloss_fm.py`. `src/omni_av_p2/connector.py` adapts the RMS normalization and generation projection topology from `bernini/models/bernini.py`.

Original copyright: Copyright (c) 2026 Bytedance Ltd. and/or its affiliate.

These adapted portions are licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0 . Distributed on an AS IS basis, without warranties or conditions of any kind. A copy is included in `licenses/Apache-2.0.txt`.

Modifications: independent module without Bernini training scheduler dependencies; explicit FM objective; separate audio/video heads; configurable widths; coordinate-query fusion over selected frozen Thinker layers before the final generation projections; same final conditions shared with frozen LTX. The upstream optional environment-controlled MLP normalization variant is not included.

Local reference snapshot: `/private/liqi_workspace/project_code/Bernini`. The project is described at https://arxiv.org/html/2605.22344v1 .
