# SuffixDecoding Integration: Technical Document

> Branch: `codex/ds4-sota-audit` on [antirez/ds4](https://github.com/antirez/ds4)
>
> Reference: SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications (arXiv:2411.04975)
>
> Reference implementations: [Snowflake ArcticInference](https://github.com/snowflakedb/ArcticInference), [vLLM suffix decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/)
>
> Features ported from ArcticInference: probability estimation, `min_token_prob` filtering, adaptive draft caps (`max_spec_factor`/`max_spec_offset`), cached best-child index, score telemetry.

---

## 1. What This Is

An opt-in, model-free speculative decoding path for ds4 (DwarfStar), a DeepSeek V4-specific C/Metal/CUDA inference engine.

A bounded CPU-resident suffix trie learns repetitive token patterns from prompt, checkpoint, and accepted generation tokens, then proposes draft tokens that the existing target-model verifier accepts or rejects. No separate draft model, no training, and no GPU kernels are required.

## 2. Architecture

### 2.1 Data Structure

A compact suffix trie where each node represents a token ID. Children are stored in sorted arrays with binary search lookup, which is cache-friendly and allocation-efficient for the small fanout typical of LLM token sequences.

```
ds4_suffix_node
  token_id: int              (-1 for root)
  freq: uint32_t             (observation count)
  n_children / cap_children
  best_child_idx: uint32_t   (cached index of highest-freq child; UINT32_MAX = invalid)
  children: *ds4_suffix_node (sorted by token_id)

ds4_suffix_tree
  root: ds4_suffix_node
  node_count / node_budget   (bounded memory)
  total_bytes                (estimated memory)
  max_depth                  (default 32)
  telemetry counters         (query_count, query_hits, draft_tokens_produced,
                              draft_tokens_accepted, draft_score_total)
```

### 2.2 Lifecycle

| Event | Suffix tree action |
|-------|--------------------|
| `ds4_session_create` | Allocate tree (if `--suffix-decoding`) |
| `ds4_session_sync` | Reset + re-seed from full prompt |
| `ds4_session_load_payload` | Reset + re-seed from restored checkpoint |
| `ds4_session_rewind` | Reset + re-seed from truncated checkpoint |
| `ds4_session_eval` (normal decode) | Incremental learn via `suffix_learned_len` — only new tokens appended since last learn step |
| Speculative accept | Incremental learn from accepted tokens |
| `ds4_session_free` | Free tree and all nodes |

### 2.3 Draft Selection Flow

```
ds4_session_eval_speculative_argmax()
  |
  +-- target token committed, logits on hand
  |
  +-- suffix_available? -----> draft_from_suffix_tree()
  |    |                         |
  |    |                         +-- Phase 1: ds4_suffix_tree_match_depth()
  |    |                         |   Find longest matching suffix depth p
  |    |                         |   Require p >= 2
  |    |                         |
  |    |                         +-- Phase 2: compute adaptive cap
  |    |                         |   cap = p * suffix_spec_factor + suffix_spec_offset
  |    |                         |   Clamped to [1, draft_cap]
  |    |                         |
  |    |                         +-- Phase 3: ds4_suffix_tree_query()
  |    |                         |   Follow highest-freq continuation path
  |    |                         |   Probability estimation: prob *= child_freq / parent_freq
  |    |                         |   Stop if prob < suffix_min_prob
  |    |                         |   Return score = sum of per-token probs
  |    |                         |
  |    |                         +-- Require: suffix_n > 0 AND score > 0.0
  |    |                         +-- else: return, try MTP
  |    |
  |    +-- suffix hit? --------> drafts[0..n-1] = suffix tree proposals
  |    |                           skip MTP recursive loop entirely
  |    |                           log p, draft_n, score (if DS4_SUFFIX_SPEC_LOG)
  |    |
  |    +-- no suffix hit -------> MTP available? -> MTP recursive draft
  |                               MTP unavailable? -> emit only target token
  |
  +-- first draft verified free (logits already on hand)
  |
  +-- can_batch_verify?
  |    (s->graph.spec_logits != NULL)
  |    |   Allocated when either MTP or suffix decoding enabled
  |    |
  |    +-- yes -> microbatch / exact decode verifier
  |    +-- no  -> sequential verification fallback
  |
  +-- all verified drafts: DS4_SUFFIX_NOTE_ACCEPTED()
       +-- learn from checkpoint (incremental)
       +-- track draft_tokens_accepted
```

### 2.4 Query Semantics

**Two-phase query** separates matching from drafting:

1. **`ds4_suffix_tree_match_depth()`** — finds the longest suffix in the trie that completely matches the prefix and has continuations. Returns match depth `p` without generating drafts.

2. **`ds4_suffix_tree_query()`** — given a known match depth, follows the highest-frequency continuation path to propose drafts. At each step:
   - Estimates probability: `prob *= child_freq / parent_freq`
   - Filters by `min_prob`: stops if `prob < min_prob`
   - Accumulates `score += prob` for telemetry
   - Uses cached `best_child_idx` for O(1) child selection (lazy repopulated on query)

Match conditions (three must all hold):
1. `j == prefix_len` — the entire prefix suffix was consumed (no partial match)
2. `depth > match_depth` — this match is longer than any previous one
3. `cur->n_children > 0` — the matched node has continuations (not a terminal leaf)

### 2.5 Probability Estimation and Confidence Filtering

Ported from ArcticInference's `_speculate_path()`. Each draft token's conditional probability is estimated as:

```
P(next_token | context) ≈ best_child_freq / parent_freq
```

The cumulative product `prob` is tracked across the continuation walk. If `prob < min_prob` (configurable via `--suffix-min-prob`), the walk stops early, preventing low-confidence drafts from being proposed.

A cumulative `score` (sum of per-step probs) is returned for logging and can be used to compare suffix confidence against other draft sources.

Default: `--suffix-min-prob 0.0` (disabled — all drafts pass, backward compatible).

### 2.6 Adaptive Draft Cap

Ported from ArcticInference's `speculate()`. Instead of the original hardcoded `alpha=1` cap (draft at most `p` tokens for a `p`-token match):

```
adaptive_cap = p * suffix_spec_factor + suffix_spec_offset
```

Defaults: `--suffix-spec-factor 1.0`, `--suffix-spec-offset 0.0` (reproduces the original `alpha=1` behavior).

Higher factor values allow more aggressive speculation for longer matches; offset provides a floor for short matches.

### 2.7 Cached Best Child

Each `ds4_suffix_node` stores `best_child_idx` — the index of its highest-frequency child. This avoids scanning all children on every continuation step during query.

**Maintenance:**
- **Invalidated** (`UINT32_MAX`) on structural changes: `insert_child()` (memmove shifts indices), `remove_zero_leaves()` (pruning)
- **Updated** eagerly on freq increment in `insert()`/`append()` via `update_best_child()`
- **Lazy repopulated** on query: if `best_child_idx >= n_children`, a scan runs once and caches the result

### 2.8 Memory Management

- **Budget**: byte budget converted to node budget at alloc time (`byte_budget / (1.5 * sizeof(node))`)
- **Insert-time pruning**: during bulk insert (e.g., seed from long prompt), pruning triggers incrementally when node count exceeds `budget + slack`, avoiding memory spikes
- **Multi-round pruning**: `prune_toward_budget()` runs up to 16 rounds of (decrement-all-freqs + remove-zero-leaves) to fully converge to budget
- **Pruning strategy**: frequency aging (decrement all by 1, clamp at 0) then remove zero-frequency leaves. Preserves frequently-used patterns, discards one-off sequences.

## 3. Files Changed

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `ds4_suffix_tree.h` | New | 112 | Suffix trie API, `ds4_suffix_stats` telemetry type, `best_child_idx`, `match_depth()` |
| `ds4_suffix_tree.c` | New | 488 | Sorted-array trie with prob estimation, cached best child, adaptive pruning |
| `tests/suffix_tree_test.c` | New | 293 | 8 unit tests: continuation, terminal skip, partial prefix, prune/reset, probability, min_prob, best child cache, append |
| `ds4.h` | Modified | +17 | 6 engine options, forward-declared `ds4_suffix_stats`, `draft_score_total` |
| `ds4.c` | Modified | ~260 diff | Session integration, incremental learning, two-phase draft selection, score gating |
| `ds4_cli.c` | Modified | +32 | `--suffix-decoding` + 5 flags; `cli_speculative_decode_enabled()` |
| `ds4_bench.c` | Modified | +80 | `spec_steps` column, suffix telemetry CSV, 3 new config options, backend guard |
| `Makefile` | Modified | +12 | `suffix-tree-test` target, build rules, clean |
| `README.md` | Modified | +27 | `--suffix-decoding` documentation |
| `CONTRIBUTING.md` | Modified | +6 | Suffix telemetry column guidance |
| `speed-bench/README.md` | Modified | +26 | MTP and suffix bench sweep examples |

### API Surface

**Public (`ds4.h`)**:
- `ds4_engine_options.suffix_decoding` (bool)
- `ds4_engine_options.suffix_max_depth` (uint32_t, default 32)
- `ds4_engine_options.suffix_memory_budget` (uint64_t bytes, default 64MB)
- `ds4_engine_options.suffix_spec_factor` (float, default 1.0)
- `ds4_engine_options.suffix_spec_offset` (float, default 0.0)
- `ds4_engine_options.suffix_min_prob` (float, default 0.0)
- `ds4_suffix_stats` telemetry snapshot struct (includes `draft_score_total`)
- `ds4_session_suffix_stats()` query function

**Internal (`ds4_suffix_tree.h`, not exposed to downstream)**:
- Full trie API: `alloc`, `free`, `insert`, `append`, `query`, `match_depth`, `prune`, `reset`, `stats`

### CLI Flags

```
--suffix-decoding              Enable suffix tree speculative decoding
--suffix-max-depth N           Max sequence depth (default 32)
--suffix-memory-budget MB      Max tree memory in MB (default 64)
--suffix-spec-factor F         Draft cap multiplier: cap = p * F + offset (default 1.0)
--suffix-spec-offset F         Draft cap offset (default 0.0)
--suffix-min-prob F            Min conditional prob to continue drafting (default 0.0, disabled)
```

### Benchmark CSV Columns

```
spec_steps                     Total speculative decode attempts (MTP or suffix)
suffix_tree_nodes              Current tree node count
suffix_tree_bytes              Estimated tree memory
suffix_draft_attempts          Tree query count
suffix_draft_hits              Queries that returned candidates
suffix_accepted_tokens         Draft tokens accepted by target verifier
suffix_avg_draft_len           Average drafts per successful hit
```

### Environment Variables

```
DS4_SUFFIX_SPEC_LOG=1          Log suffix spec hit/miss with score to stderr
```

## 4. Verification Modes and Acceleration

| Mode | Draft source | Verifier | Acceleration |
|------|-------------|----------|-------------|
| MTP only | MTP recursive | batch verify (`spec_logits` available) | Can accelerate |
| Suffix + MTP | suffix tree (priority), MTP fallback | batch verify (`spec_logits` available) | Can accelerate |
| Suffix only | suffix tree | batch verify (`spec_logits` allocated independently) | Structurally supported, pending benchmark |
| CPU | N/A (exits early) | N/A | N/A |

### Suffix-only batch verification

`spec_logits` and spec frontier buffers are allocated when `enable_spec_verify = e->mtp_ready || e->suffix_decoding` is true (in `metal_graph_alloc()`). This means suffix-only mode allocates verification buffers without requiring MTP weights or graph state.

The structural prerequisite is in place. Actual acceleration measurement requires benchmarking with a real DeepSeek V4 model to confirm the batch verification kernel performs correctly without MTP-specific state.

## 5. Safety Properties

### Correctness guarantees

- **Target verifier gates all output**: regardless of draft source, every proposed token is verified against the target model's logits before being committed. Incorrect drafts are rejected and speculative state is rolled back via `spec_frontier_snapshot` / `spec_frontier_restore`.
- **No output distribution change**: accepted tokens are identical to what the target model would produce via greedy decode. The suffix tree only proposes drafts; the verifier decides.
- **MTP SWA counters protected**: `DS4_MTP_KEEP_ACCEPTED()` is gated by `using_mtp`, so suffix-only drafts never corrupt MTP raw SWA state.
- **Memory bounded**: hard node budget with multi-round pruning. Insert-time incremental pruning prevents transient memory spikes.
- **Independent verifier scratch**: `spec_logits` is allocated when either MTP or suffix decoding is enabled (`enable_spec_verify = e->mtp_ready || e->suffix_decoding`), so suffix-only mode has batch verification capability without depending on MTP.

### Edge cases handled

| Case | Behavior |
|------|----------|
| Empty tree (first tokens) | Query returns `p=0`, falls back to MTP or single-token |
| All drafts rejected | Verification loop breaks early, only target token emitted |
| EOS in draft sequence | Draft truncated at EOS |
| `ds4_session_rewind` | Tree re-seeded from truncated checkpoint |
| `ds4_session_sync` with new prompt | Tree reset and re-seeded from new prompt |
| Allocation failure during insert | Silent skip (tree continues with partial data) |
| CPU backend with `--suffix-decoding` | Backend guard in CLI/bench prevents speculative path entry |
| `min_prob` filters all drafts | Query returns `score ≤ 0`, caller falls back to MTP |

## 6. Testing

### Unit tests (`make suffix-tree-test`)

| Test | Validates |
|------|-----------|
| `test_repeated_continuation` | Most-frequent continuation from repeated pattern; telemetry counters; `match_depth` correctness |
| `test_skip_terminal_longest_match` | Terminal suffix (no children) is skipped in favor of non-terminal match |
| `test_no_partial_prefix_match` | Prefix `{20,30,99}` does not match tree path `{20,30,...}` — requires `j == prefix_len` |
| `test_prune_and_reset` | Tree prunes to budget under adversarial input; reset clears all nodes |
| `test_append_does_not_reinflate_prefix` | `append()` only adds new tail edge, does not re-bump existing prefix frequencies |
| `test_probability_estimation_and_score` | Score > 0 and ≤ draft_n; `draft_score_total` telemetry updated |
| `test_min_prob_cutoff` | `min_prob=0.99` rejects all drafts when P(best_token) ≈ 0.83 |
| `test_best_child_cache` | Query populates `best_child_idx`; second query uses cached value |

### Build verification

```sh
make suffix-tree-test NATIVE_CPU_FLAG=   # unit tests
make cpu NATIVE_CPU_FLAG=                 # all 5 CPU binaries
make ds4-bench NATIVE_CPU_FLAG=           # Metal binary
./ds4-eval --self-test-extractors         # extractor self-tests
```

### What is NOT tested

- End-to-end speculative decode with a real DeepSeek V4 model (requires 80GB+ GPU)
- Actual throughput speedup measurement
- Logprob distribution regression (requires model + reference outputs)
- Memory behavior under production-scale agentic workloads
- Suffix-only batch verification correctness (structural prerequisite in place, pending real model test)

## 7. Known Limitations and Follow-ups

| Limitation | Impact | Follow-up |
|------------|--------|-----------|
| No real-model benchmark yet | Cannot confirm actual speedup | Run with real DeepSeek V4 on 80GB+ GPU |
| Sync reset+re-seed on large trees | Brief pause during sync | Lazy seed or background rebuild |
| `drafts[16]` buffer limit | Max 15 suffix drafts per step | Expand buffer if deeper speculation validated |
| No logprob quality regression test | Cannot claim output quality unchanged | Run full model benchmark with reference outputs |
| Default parameters = no behavioral change | `--suffix-min-prob 0.0`, `--suffix-spec-factor 1.0`, `--suffix-spec-offset 0.0` reproduce original behavior | Tune parameters on real workloads |
| CSV column reordering | Breaks downstream CSV parsers | Document as breaking change in PR notes |

## 8. Resume Bullet

> Integrated an opt-in SuffixDecoding-style model-free speculative decoding path into DwarfStar (antirez/ds4), a DeepSeek V4-specific C/Metal/CUDA inference engine. Implemented a bounded CPU-resident suffix trie in pure C (~490 lines), seeded from prompt/checkpoint tokens and updated incrementally from accepted generation tokens, then wired it as an additional draft source before the existing MTP fallback and target verification pipeline. Ported probability estimation, adaptive draft caps, and confidence filtering from Snowflake ArcticInference. Added benchmark telemetry, score-based draft gating, and 8 unit tests — all without requiring an extra draft model or GPU kernels.

## 9. References

- SuffixDecoding paper: https://arxiv.org/abs/2411.04975
- SuffixDecoding project page: https://suffix-decoding.github.io/
- ArcticInference (Snowflake): https://github.com/snowflakedb/ArcticInference
- vLLM suffix decoding: https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/
- ds4 repository: https://github.com/antirez/ds4
