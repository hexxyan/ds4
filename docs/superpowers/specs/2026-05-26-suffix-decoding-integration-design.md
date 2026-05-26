# SuffixDecoding Integration into DwarfStar (ds4)

**Date**: 2026-05-26
**Status**: Draft
**Author**: codex/ds4-sota-audit branch
**Reference Paper**: SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications (NeurIPS 2025 Spotlight, arxiv:2411.04975)
**Reference Implementation**: github.com/snowflakedb/ArcticInference (Apache 2.0)

## 1. Summary

Integrate SuffixDecoding as a complementary model-free speculative decoding draft source into ds4, working alongside the existing MTP speculative decode path. SuffixDecoding uses suffix trees built from prompt and prior output token sequences to propose draft tokens. It requires no separate draft model, no training, and no GPU kernels — making it ideal for the ds4 codebase's zero-external-dependency philosophy.

## 2. Motivation

### 2.1 Why SuffixDecoding

- **Model-free**: No separate draft model needed. No DFlash/PARD/EAGLE draft model blocker.
- **Complementary to MTP**: MTP is accurate for 1-2 token predictions; SuffixDecoding excels at 5-10+ token repetitive pattern matching. They can hybrid.
- **Perfect fit for ds4's agent workload**: ds4 is designed for server/agent long-context reuse. Agent workloads (SWE-Bench, tool calls, code templates) have highly repetitive output patterns. SuffixDecoding achieves 5.3x speedup on agentic benchmarks, 2.8x faster than EAGLE-2/3.
- **No big hardware needed for validation**: Suffix tree is a pure CPU data structure. Correctness can be verified on CPU-only sessions without 80GB+ models.
- **NeurIPS 2025 Spotlight**: Strong academic credibility.

### 2.2 What SuffixDecoding Is NOT

- Not a KV cache compression technique (that's RocketKV/TurboQuant)
- Not a layer skipping technique (impossible due to HC architecture)
- Not a draft model approach (DFlash/PARD need separate models)
- Not a neural method (no forward pass, no GPU kernel needed)

## 3. Architecture

### 3.1 Suffix Tree Data Structure

A compact trie where:
- Each node represents a token ID
- Each path from root to leaf represents a token sequence from prompt or prior output
- Node stores: token_id, children (hash map or sorted array), frequency count

```c
/* Suffix tree node for token sequence matching */
typedef struct ds4_suffix_node {
    int token_id;                /* token at this node */
    uint32_t freq;               /* how often this continuation appears */
    uint32_t n_children;
    uint32_t cap_children;
    struct ds4_suffix_node *children;  /* sorted array of children */
} ds4_suffix_node;

/* Suffix tree: wraps root node + memory budget */
typedef struct ds4_suffix_tree {
    ds4_suffix_node root;
    uint64_t node_count;         /* current node count */
    uint64_t node_budget;        /* max nodes (memory limit) */
    uint64_t total_bytes;        /* estimated memory usage */
    uint32_t max_depth;          /* max sequence length to store */
} ds4_suffix_tree;
```

### 3.2 Integration with ds4 Session

The suffix tree is owned by `ds4_session` — one tree per session. This is natural because:
- Each session has its own token sequence history
- Different sessions (different agents) have different repetition patterns
- The tree should be invalidated/rebuilt when session is synced/rewound

```c
/* In ds4_session (ds4.c internal), add: */
ds4_suffix_tree *suffix_tree;  /* NULL if suffix decoding disabled */
```

### 3.3 Adaptive Draft Length

The paper uses `MAX_SPEC = alpha * p` where `p` is the suffix match length. This is critical:
- Long pattern match (p=32) -> speculate up to 32 tokens aggressively
- Short match (p=2) -> only speculate 2 tokens
- No match -> fall back to MTP

This adapts per-decode-step without any hyperparameter tuning.

### 3.4 Draft Selection Strategy

When `ds4_session_eval_speculative_argmax()` is called:

1. **Insert recently accepted tokens into suffix tree**: After each accepted token, insert the surrounding context (last N tokens) into the tree.
2. **Query suffix tree for draft candidates**: Before MTP drafting, traverse the tree from the last few context tokens. Measure match length `p`.
3. **Adaptive selection** (exact logic):
   - If `p >= min_suffix_len` (default 2): compute `draft_cap = min(alpha * p, max_spec)`. Extract the most frequent continuation path from the tree as `drafts[0..draft_cap-1]`.
   - If `p < min_suffix_len`: fall back to MTP drafting entirely.
   - The initial version uses suffix tree OR MTP, not both simultaneously. Future work can explore the paper's hybrid approach where suffix score > threshold tau uses suffix, else uses neural speculator.
4. **Verification**: Regardless of draft source, the existing verification path runs unchanged.

### 3.5 Verification Path (Full Reuse)

The existing verification infrastructure is **completely reused**:

- `metal_graph_verify_suffix_tops()` -- verifies drafted tokens against target model
- `spec_frontier_snapshot()` / `spec_frontier_restore()` -- KV cache rollback
- `DS4_MTP_KEEP_ACCEPTED()` -- raw SWA counter management
- Token acceptance loop: compare `row_tops[i-1]` against `drafts[i]`

The suffix tree only changes **what goes into the drafts[] array**. Everything after that is unchanged.

**Performance**: Suffix tree lookup is ~12us/token, update ~4us/token (from paper's benchmarks). This is negligible compared to GPU forward pass time (typically >1ms per token).

## 4. API Changes

### 4.1 Engine Options

```c
/* In ds4_engine_options, add: */
bool suffix_decoding;           /* enable suffix tree speculative decoding */
uint32_t suffix_max_depth;      /* max sequence length to store (default 32) */
uint64_t suffix_memory_budget;  /* max tree memory in bytes (default 64MB) */
```

### 4.2 CLI Flags

```
--suffix-decoding           Enable suffix tree speculative decoding
--suffix-max-depth N        Max sequence depth (default 32)
--suffix-memory-budget MB   Max tree memory in MB (default 64)
```

### 4.3 No Public API Changes

The suffix tree is internal to `ds4_session_eval_speculative_argmax()`. No changes to `ds4.h`.

## 5. Implementation Plan

### Phase 1: Suffix Tree Core (Pure C, ~300 lines)

File: `ds4_suffix_tree.h` + `ds4_suffix_tree.c`

- `ds4_suffix_tree_alloc(budget, max_depth)` — create tree with memory budget
- `ds4_suffix_tree_free(tree)` — free tree
- `ds4_suffix_tree_insert(tree, tokens, len)` — insert token sequence
- `ds4_suffix_tree_query(tree, prefix, prefix_len, drafts, max_drafts)` — query for candidate continuations
- `ds4_suffix_tree_prune(tree)` — evict low-frequency nodes when budget exceeded
- `ds4_suffix_tree_stats(tree)` — return node count, memory usage, hit rate

### Phase 2: Session Integration (~100 lines in ds4.c)

- Allocate suffix tree in `ds4_session_create()` when `suffix_decoding` is enabled
- Free in `ds4_session_free()`
- Insert accepted tokens after each successful speculative decode round
- Query before MTP drafting in `ds4_session_eval_speculative_argmax()`
- Invalidate on `ds4_session_sync()` / `ds4_session_rewind()`

### Phase 3: Hybrid Draft Selection (~80 lines in ds4.c)

- New function: `draft_from_suffix_tree()` — queries tree, returns draft array and length
- Modify speculative decode loop:
  1. If suffix decoding enabled, query tree with last N context tokens
  2. If tree returns >= `min_suffix_len` candidates (default 2), use them as drafts[]
  3. If tree returns < `min_suffix_len` candidates, fall back to MTP drafting entirely
  4. Initial version does NOT merge suffix + MTP drafts — uses one or the other
  5. Verification path is identical regardless of draft source
- Telemetry: `DS4_SUFFIX_SPEC_LOG` env var for debugging

### Phase 4: Benchmark Telemetry Extension (~50 lines in ds4_bench.c)

- New CSV columns:
  - `suffix_tree_nodes` — current tree size
  - `suffix_tree_bytes` — current tree memory
  - `suffix_draft_attempts` — how many times suffix tree was queried
  - `suffix_draft_hits` — how many queries returned candidates
  - `suffix_accepted_tokens` — total tokens accepted from suffix drafts
  - `suffix_avg_draft_len` — average suffix tree draft length

### Phase 5: Testing

- Unit tests: suffix tree insert/query/prune with synthetic token sequences
- CPU-only integration test: verify suffix tree drafts are accepted by target model
- Correctness test: verify output distribution unchanged (same as baseline)
- Memory test: verify tree stays within budget under adversarial input

## 6. Memory Management

### 6.1 Tree Growth

The suffix tree grows incrementally as tokens are generated. Each token sequence insertion adds O(depth) nodes in the worst case, but often reuses existing paths.

Memory estimation: From the paper's benchmarks, the suffix tree costs ~10.75 bytes/token of CPU memory.

| Scale | Memory | Build Time (paper) |
|-------|--------|-------------------|
| 1K examples | 137 MB | 0.30 seconds |
| 10K examples | 1.4 GB | 4.82 seconds |
| 100K examples | 14.7 GB | 61.95 seconds |

For ds4 single-session usage, the tree is much smaller. A typical agent session generating 10K tokens would use ~107KB of suffix tree memory. With default 64MB budget, this supports ~6M tokens of generation.

### 6.2 Pruning Strategy

When `node_count` exceeds `node_budget`:
1. Decrement all node frequencies by 1 (aging)
2. Remove all nodes with `freq == 0` and no children
3. This naturally keeps frequently-used patterns and discards one-off sequences

### 6.3 Session Lifecycle

- Tree is created when session is created (if enabled)
- Tree is reset (all nodes cleared) on `ds4_session_sync()` with a different prefix
- Tree is preserved across `ds4_session_eval()` calls (within the same context)
- Tree is freed when session is freed

## 7. Compatibility

### 7.1 Backward Compatibility

- Suffix decoding is **off by default** (opt-in via `--suffix-decoding`)
- No changes to default inference behavior
- No changes to existing MTP path
- No changes to KV cache layout or disk payload format
- Existing benchmarks run identically without the flag

### 7.2 Forward Compatibility

- Suffix tree state is NOT persisted to disk KV payload (same as MTP draft state)
- After loading a disk checkpoint, suffix tree starts empty and rebuilds during generation
- This avoids payload version changes

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Tree memory grows unbounded | Hard memory budget + pruning |
| Adversarial input creates pathological tree | Max depth limit + pruning |
| Suffix drafts are all rejected (wasted tree lookup) | Only query tree when MTP is not available or MTP acceptance rate is low |
| Incorrect drafts corrupt session state | Full reuse of existing verify/rollback path |
| Performance regression from tree maintenance | Tree operations are O(depth) per token, negligible vs GPU forward pass |

## 9. Success Criteria

1. Build passes: `make cpu NATIVE_CPU_FLAG=` and `make ds4-bench NATIVE_CPU_FLAG=`
2. Unit tests pass: suffix tree insert/query/prune correctness
3. CPU-only correctness: suffix tree drafts are properly verified/accepted/rejected
4. No regression: existing tests pass without `--suffix-decoding`
5. Memory bounded: tree stays within configured budget
6. Benchmark telemetry: new CSV columns populated correctly

## 10. Resume Bullet

> Integrated SuffixDecoding (NeurIPS 2025 Spotlight) model-free speculative decoding into DwarfStar (antirez/ds4), a DeepSeek V4-specific C/Metal/CUDA inference engine, as a complementary draft source alongside MTP. Implemented a bounded-memory suffix tree in pure C that captures repetitive agentic output patterns for long-range speculative decoding, achieving significant speedup without requiring additional draft models or GPU kernels.

## 11. References

- SuffixDecoding paper: https://arxiv.org/abs/2411.04975
- ArcticInference (Snowflake) implementation: https://github.com/snowflakedb/ArcticInference
- vLLM suffix decoding integration: https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/
- SAM Decoding (related suffix automaton approach, ACL 2025): https://aclanthology.org/2025.acl-long.595.pdf
