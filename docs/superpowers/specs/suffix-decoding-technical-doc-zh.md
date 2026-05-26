# SuffixDecoding 集成技术文档

> 分支：`codex/ds4-sota-audit`，基于 [antirez/ds4](https://github.com/antirez/ds4)
>
> 参考文献：SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications (arXiv:2411.04975)
>
> 参考实现：[Snowflake ArcticInference](https://github.com/snowflakedb/ArcticInference)、[vLLM suffix decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/)
>
> 从 ArcticInference 移植的特性：概率估计、`min_token_prob` 过滤、自适应 draft cap（`max_spec_factor`/`max_spec_offset`）、best child 缓存索引、score 遥测。

---

## 1. 项目概述

为 ds4（DwarfStar）——一个 DeepSeek V4 专用的 C/Metal/CUDA 推理引擎——实现了一条 opt-in 的免模型投机解码路径。

核心是一个带内存预算的 CPU 驻留 suffix trie，从 prompt、checkpoint 和已接受的生成 token 中学习重复的 token 模式，然后提出 draft token 供已有的目标模型验证器接受或拒绝。不需要单独的 draft model、不需要训练、不需要 GPU kernel。

## 2. 架构设计

### 2.1 数据结构

紧凑型 suffix trie，每个节点代表一个 token ID。子节点存储在按 token_id 排序的数组中，通过二分查找定位——对 LLM token 序列典型的小扇出场景，既缓存友好又分配高效。

```
ds4_suffix_node
  token_id: int              （根节点为 -1）
  freq: uint32_t             （出现次数）
  n_children / cap_children
  best_child_idx: uint32_t   （最高频子节点缓存索引；UINT32_MAX 表示无效）
  children: *ds4_suffix_node （按 token_id 排序）

ds4_suffix_tree
  root: ds4_suffix_node
  node_count / node_budget   （有界内存）
  total_bytes                （估算内存）
  max_depth                  （默认 32）
  遥测计数器                 （query_count, query_hits, draft_tokens_produced,
                               draft_tokens_accepted, draft_score_total）
```

### 2.2 生命周期

| 事件 | Suffix tree 行为 |
|------|------------------|
| `ds4_session_create` | 分配 tree（当启用 `--suffix-decoding` 时） |
| `ds4_session_sync` | 重置 + 从完整 prompt 重新播种 |
| `ds4_session_load_payload` | 重置 + 从恢复的 checkpoint 重新播种 |
| `ds4_session_rewind` | 重置 + 从截断后的 checkpoint 重新播种 |
| `ds4_session_eval`（普通解码） | 通过 `suffix_learned_len` 增量学习——仅追加上次学习以来的新 token |
| 投机验证通过后 | 增量学习已接受的 token |
| `ds4_session_free` | 释放 tree 及所有节点 |

### 2.3 Draft 选择流程

```
ds4_session_eval_speculative_argmax()
  |
  +-- 目标 token 已提交，logits 已就绪
  |
  +-- suffix_available? -------> draft_from_suffix_tree()
  |    |                           |
  |    |                           +-- 第一阶段：ds4_suffix_tree_match_depth()
  |    |                           |   查找最长匹配后缀深度 p
  |    |                           |   要求 p >= 2
  |    |                           |
  |    |                           +-- 第二阶段：计算自适应上限
  |    |                           |   cap = p * suffix_spec_factor + suffix_spec_offset
  |    |                           |   限制在 [1, draft_cap] 范围内
  |    |                           |
  |    |                           +-- 第三阶段：ds4_suffix_tree_query()
  |    |                           |   沿最高频延续路径行进
  |    |                           |   概率估计：prob *= child_freq / parent_freq
  |    |                           |   当 prob < suffix_min_prob 时停止
  |    |                           |   返回 score = 每步概率之和
  |    |                           |
  |    |                           +-- 要求：suffix_n > 0 且 score > 0.0
  |    |                           +-- 否则：返回，尝试 MTP
  |    |
  |    +-- suffix 命中? ---------> drafts[0..n-1] = suffix tree 提案
  |    |                             跳过整个 MTP 递归循环
  |    |                             记录 p, draft_n, score（若 DS4_SUFFIX_SPEC_LOG）
  |    |
  |    +-- suffix 未命中 -------> MTP 可用？ -> MTP 递归 draft
  |                                 MTP 不可用？ -> 仅输出目标 token
  |
  +-- 第一个 draft 免费验证（logits 已在手）
  |
  +-- can_batch_verify?
  |    (s->graph.spec_logits != NULL)
  |    |   在启用 MTP 或 suffix decoding 时分配
  |    |
  |    +-- 是 -> 微批 / 精确解码验证器
  |    +-- 否 -> 逐个验证 fallback
  |
  +-- 所有验证通过的 draft：DS4_SUFFIX_NOTE_ACCEPTED()
       +-- 从 checkpoint 增量学习
       +-- 记录 draft_tokens_accepted
```

### 2.4 Query 语义

**两阶段查询**将匹配与 draft 生成分离：

1. **`ds4_suffix_tree_match_depth()`** — 查找 trie 中完全匹配前缀且有延续的最长后缀。返回匹配深度 `p`，不生成 draft。

2. **`ds4_suffix_tree_query()`** — 已知匹配深度后，沿最高频延续路径提出 draft。每步：
   - 估计概率：`prob *= child_freq / parent_freq`
   - 按 `min_prob` 过滤：当 `prob < min_prob` 时停止
   - 累积 `score += prob` 用于遥测
   - 使用缓存 `best_child_idx` 实现 O(1) 子节点选择（查询时惰性重建）

匹配条件（三个必须同时满足）：

1. `j == prefix_len` — 整个前缀后缀被消费完毕（不允许部分匹配）
2. `depth > match_depth` — 此匹配比之前任何匹配都长
3. `cur->n_children > 0` — 匹配节点有后续节点（非终端叶子）

### 2.5 概率估计与置信度过滤

移植自 ArcticInference 的 `_speculate_path()`。每个 draft token 的条件概率估计为：

```
P(next_token | context) ≈ best_child_freq / parent_freq
```

累积乘积 `prob` 在延续路径中逐步跟踪。当 `prob < min_prob`（可通过 `--suffix-min-prob` 配置）时，路径提前终止，防止低置信度 draft 被提出。

累积 `score`（每步概率之和）用于日志记录，可用于比较 suffix 置信度与其他 draft 来源。

默认：`--suffix-min-prob 0.0`（禁用——所有 draft 均通过，向后兼容）。

### 2.6 自适应 Draft 上限

移植自 ArcticInference 的 `speculate()`。替代原始硬编码的 `alpha=1` 上限（对 p-token 匹配最多 draft p 个 token）：

```
adaptive_cap = p * suffix_spec_factor + suffix_spec_offset
```

默认值：`--suffix-spec-factor 1.0`、`--suffix-spec-offset 0.0`（复现原始 `alpha=1` 行为）。

更高的 factor 值允许对更长匹配进行更激进的投机；offset 为短匹配提供下限。

### 2.7 Best Child 缓存

每个 `ds4_suffix_node` 存储 `best_child_idx`——其最高频子节点的索引。避免查询时每步扫描所有子节点。

**维护策略：**
- **失效**（设为 `UINT32_MAX`）：结构变更时——`insert_child()`（memmove 移动索引）、`remove_zero_leaves()`（剪枝）
- **即时更新**：在 `insert()`/`append()` 中频率增加后通过 `update_best_child()` 更新
- **惰性重建**：查询时若 `best_child_idx >= n_children`，执行一次扫描并缓存结果

### 2.8 内存管理

- **预算分配**：字节预算在分配时转换为节点预算（`byte_budget / (1.5 * sizeof(node))`）
- **插入期剪枝**：批量插入（如从长 prompt 播种）时，节点数超过 `budget + slack` 即触发渐进式剪枝，避免内存尖峰
- **多轮剪枝**：`prune_toward_budget()` 最多执行 16 轮（递减全部频率 + 移除零频叶子），确保完全收敛到预算
- **剪枝策略**：频率老化（所有 freq 减 1，下限为 0）然后移除零频叶子。保留高频模式，丢弃一次性序列

## 3. 变更文件清单

| 文件 | 状态 | 行数 | 用途 |
|------|------|------|------|
| `ds4_suffix_tree.h` | 新增 | 112 | Suffix trie API、`ds4_suffix_stats` 遥测类型、`best_child_idx`、`match_depth()` |
| `ds4_suffix_tree.c` | 新增 | 488 | 排序数组 trie：概率估计、best child 缓存、自适应剪枝 |
| `tests/suffix_tree_test.c` | 新增 | 293 | 8 个单元测试：连续匹配、终端跳过、部分前缀、剪枝/重置、概率估计、min_prob 截断、best child 缓存、append |
| `ds4.h` | 修改 | +17 | 6 个引擎选项、前向声明 `ds4_suffix_stats`、`draft_score_total` |
| `ds4.c` | 修改 | ~260 diff | Session 集成、增量学习、两阶段 draft 选择、score 门控 |
| `ds4_cli.c` | 修改 | +32 | `--suffix-decoding` + 5 个参数；`cli_speculative_decode_enabled()` |
| `ds4_bench.c` | 修改 | +80 | `spec_steps` 列、suffix 遥测 CSV、3 个新配置项、后端保护 |
| `Makefile` | 修改 | +12 | `suffix-tree-test` 目标、构建规则、clean |
| `README.md` | 修改 | +27 | `--suffix-decoding` 文档 |
| `CONTRIBUTING.md` | 修改 | +6 | suffix 遥测列说明 |
| `speed-bench/README.md` | 修改 | +26 | MTP 和 suffix bench sweep 示例 |

### API 表面

**公开 API（`ds4.h`）**：
- `ds4_engine_options.suffix_decoding`（bool）
- `ds4_engine_options.suffix_max_depth`（uint32_t，默认 32）
- `ds4_engine_options.suffix_memory_budget`（uint64_t 字节，默认 64MB）
- `ds4_engine_options.suffix_spec_factor`（float，默认 1.0）
- `ds4_engine_options.suffix_spec_offset`（float，默认 0.0）
- `ds4_engine_options.suffix_min_prob`（float，默认 0.0）
- `ds4_suffix_stats` 遥测快照结构体（含 `draft_score_total`）
- `ds4_session_suffix_stats()` 查询函数

**内部 API（`ds4_suffix_tree.h`，不对下游暴露）**：
- 完整 trie API：`alloc`、`free`、`insert`、`append`、`query`、`match_depth`、`prune`、`reset`、`stats`

### CLI 参数

```
--suffix-decoding              启用 suffix tree 投机解码
--suffix-max-depth N           最大序列深度（默认 32）
--suffix-memory-budget MB      最大 tree 内存，单位 MB（默认 64）
--suffix-spec-factor F         Draft cap 乘数：cap = p * F + offset（默认 1.0）
--suffix-spec-offset F         Draft cap 偏移量（默认 0.0）
--suffix-min-prob F            继续起草的最低条件概率（默认 0.0，禁用）
```

### Benchmark CSV 列

```
spec_steps                     投机解码总尝试次数（MTP 或 suffix）
suffix_tree_nodes              当前 tree 节点数
suffix_tree_bytes              估算 tree 内存
suffix_draft_attempts          Tree 查询次数
suffix_draft_hits              返回候选的查询次数
suffix_accepted_tokens         被目标验证器接受的 draft token 数
suffix_avg_draft_len           每次成功命中的平均 draft 长度
```

### 环境变量

```
DS4_SUFFIX_SPEC_LOG=1          将 suffix spec 命中/未命中及 score 输出到 stderr
```

## 4. 验证模式与加速能力

| 模式 | Draft 来源 | 验证器 | 能否加速 |
|------|-----------|--------|----------|
| 仅 MTP | MTP 递归 | 批量验证（`spec_logits` 可用） | 可以加速 |
| Suffix + MTP | suffix tree（优先），MTP fallback | 批量验证（`spec_logits` 可用） | 可以加速 |
| 仅 Suffix | suffix tree | 批量验证（`spec_logits` 独立分配） | 结构已就绪，待 benchmark 验证 |
| CPU | N/A（提前退出） | N/A | N/A |

### 仅 Suffix 模式的批量验证

`spec_logits` 和 spec frontier 缓冲区在 `enable_spec_verify = e->mtp_ready || e->suffix_decoding` 为 true 时分配（在 `metal_graph_alloc()` 中）。这意味着仅 suffix 模式无需 MTP 权重或 graph state 即可分配验证缓冲区。

结构层面的前提条件已就绪。实际加速效果需要在真实 DeepSeek V4 模型上 benchmark 以确认批量验证 kernel 在无 MTP 状态下正常工作。

## 5. 安全属性

### 正确性保证

- **目标验证器把关所有输出**：无论 draft 来源如何，每个提议的 token 都经过目标模型 logits 验证后才被提交。错误的 draft 被拒绝，投机状态通过 `spec_frontier_snapshot` / `spec_frontier_restore` 回滚。
- **不改变输出分布**：被接受的 token 与目标模型贪心解码的输出完全相同。Suffix tree 只提出 draft；验证器决定一切。
- **MTP SWA 计数器受保护**：`DS4_MTP_KEEP_ACCEPTED()` 受 `using_mtp` 门控，suffix-only draft 不会污染 MTP raw SWA 状态。
- **内存有界**：硬性节点预算 + 多轮剪枝。插入期渐进剪枝防止瞬态内存尖峰。
- **独立验证器暂存区**：`spec_logits` 在启用 MTP 或 suffix decoding 时分配（`enable_spec_verify = e->mtp_ready || e->suffix_decoding`），suffix-only 模式无需依赖 MTP 即可拥有批量验证能力。

### 已处理的边界情况

| 情况 | 行为 |
|------|------|
| 空 tree（首批 token） | 查询返回 `p=0`，回退到 MTP 或单 token |
| 所有 draft 被拒绝 | 验证循环提前终止，仅输出目标 token |
| draft 序列中出现 EOS | draft 在 EOS 处截断 |
| `ds4_session_rewind` | Tree 从截断后的 checkpoint 重新播种 |
| `ds4_session_sync` 接收新 prompt | Tree 重置并从新 prompt 重新播种 |
| 插入期间分配失败 | 静默跳过（tree 继续使用部分数据） |
| CPU 后端启用 `--suffix-decoding` | CLI/bench 的后端保护阻止进入投机路径 |
| `min_prob` 过滤掉所有 draft | 查询返回 `score ≤ 0`，调用方回退到 MTP |

## 6. 测试

### 单元测试（`make suffix-tree-test`）

| 测试 | 验证内容 |
|------|----------|
| `test_repeated_continuation` | 重复模式的最频繁延续；遥测计数器正确性；`match_depth` 正确性 |
| `test_skip_terminal_longest_match` | 终端后缀（无子节点）被跳过，选择有 continuation 的匹配 |
| `test_no_partial_prefix_match` | 前缀 `{20,30,99}` 不匹配 tree 路径 `{20,30,...}`——要求 `j == prefix_len` |
| `test_prune_and_reset` | 对抗性输入下 tree 剪枝到预算内；重置清除所有节点 |
| `test_append_does_not_reinflate_prefix` | `append()` 仅添加新尾部边，不会重新增加已有前缀频率 |
| `test_probability_estimation_and_score` | score > 0 且 ≤ draft_n；`draft_score_total` 遥测已更新 |
| `test_min_prob_cutoff` | 当 P(best_token) ≈ 0.83 时，`min_prob=0.99` 拒绝所有 draft |
| `test_best_child_cache` | 查询填充 `best_child_idx`；第二次查询使用缓存值 |

### 构建验证

```sh
make suffix-tree-test NATIVE_CPU_FLAG=   # 单元测试
make cpu NATIVE_CPU_FLAG=                 # 5 个 CPU 二进制
make ds4-bench NATIVE_CPU_FLAG=           # Metal 二进制
./ds4-eval --self-test-extractors         # 提取器自测
```

### 未覆盖项

- 真实 DeepSeek V4 模型的端到端投机解码（需要 80GB+ GPU）
- 实际吞吐加速测量
- Logprob 分布回归测试（需要模型 + 参考输出）
- 生产级 agent 负载下的内存行为
- 仅 Suffix 批量验证正确性（结构前提已就绪，待真实模型测试）

## 7. 已知限制与后续计划

| 限制 | 影响 | 后续方案 |
|------|------|----------|
| 尚无真实模型 benchmark | 无法确认实际加速效果 | 在 80GB+ GPU 上运行真实 DeepSeek V4 |
| 大 tree 的 sync 重置+重新播种 | sync 时短暂暂停 | 延迟播种或后台重建 |
| `drafts[16]` 缓冲区限制 | 每步最多 15 个 suffix draft | 验证深层投机后扩展缓冲区 |
| 无 logprob 质量回归测试 | 无法声称输出质量不变 | 使用参考输出跑完整模型 benchmark |
| 默认参数不产生行为变化 | `--suffix-min-prob 0.0`、`--suffix-spec-factor 1.0`、`--suffix-spec-offset 0.0` 复现原始行为 | 在真实工作负载上调优参数 |
| CSV 列顺序变更 | 破坏下游 CSV 解析器 | 在 PR 说明中标注为 breaking change |

## 8. 简历描述

### 英文版

> Integrated an opt-in SuffixDecoding-style model-free speculative decoding path into DwarfStar (antirez/ds4), a DeepSeek V4-specific C/Metal/CUDA inference engine. Implemented a bounded CPU-resident suffix trie in pure C (~490 lines), seeded from prompt/checkpoint tokens and updated incrementally from accepted generation tokens, then wired it as an additional draft source before the existing MTP fallback and target verification pipeline. Ported probability estimation, adaptive draft caps, and confidence filtering from Snowflake ArcticInference. Added benchmark telemetry, score-based draft gating, and 8 unit tests — all without requiring an extra draft model or GPU kernels.

### 中文版

> 为 DeepSeek V4 专用本地推理引擎 DwarfStar 设计并实现免模型投机解码路径：使用纯 C 实现带内存预算的 CPU 驻留 suffix trie（约 490 行），从 prompt/checkpoint 与已接受生成 token 中增量学习重复模式，并作为 MTP 之外的 draft source 接入现有 target verification/rollback 管线；从 Snowflake ArcticInference 移植概率估计、自适应 draft cap 和置信度过滤机制；补充 benchmark 遥测、基于 score 的 draft 门控和 8 项单元测试，在不引入额外模型、训练流程或 GPU kernel 的前提下，为 agentic 长上下文重复场景提供可验证的加速基础设施。

## 9. 参考文献

- SuffixDecoding 论文：https://arxiv.org/abs/2411.04975
- SuffixDecoding 项目页：https://suffix-decoding.github.io/
- ArcticInference（Snowflake）：https://github.com/snowflakedb/ArcticInference
- vLLM suffix decoding：https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/
- ds4 仓库：https://github.com/antirez/ds4
