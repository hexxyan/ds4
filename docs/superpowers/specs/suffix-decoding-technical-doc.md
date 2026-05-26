# SuffixDecoding Integration: Technical Document

> Branch: `codex/ds4-sota-audit` on [antirez/ds4](https://github.com/antirez/ds4)
>
> Reference: SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications (arXiv:2411.04975)
>
> Reference implementations: [Snowflake ArcticInference](https://github.com/snowflakedb/ArcticInference), [vLLM suffix decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/)

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
  children: *ds4_suffix_node (sorted by token_id)

ds4_suffix_tree
  root: ds4_suffix_node
  node_count / node_budget   (bounded memory)
  total_bytes                (estimated memory)
  max_depth                  (default 32)
  telemetry counters
```

### 2.2 Lifecycle

| Event | Suffix tree action |
|-------|--------------------|
| `ds4_session_create` | Allocate tree (if `--suffix-decoding`) |
| `ds4_session_sync` | Reset + re-seed from full prompt |
| `ds4_session_load_payload` | Reset + re-seed from restored checkpoint |
| `ds4_session_rewind` | Reset + re-seed from truncated checkpoint |
| `ds4_session_eval` (normal decode) | Learn from last N checkpoint tokens |
| Speculative accept | Learn from updated checkpoint |
| `ds4_session_free` | Free tree and all nodes |

### 2.3 Draft Selection Flow

```
ds4_session_eval_speculative_argmax()
  |
  +-- target token committed, logits on hand
  |
  +-- suffix_available? -----> draft_from_suffix_tree()
  |    |                         |
  |    |                         +-- query trie with last N context tokens
  |    |                         +-- require complete prefix match (j == prefix_len)
  |    |                         +-- require match has continuation (n_children > 0)
  |    |                         +-- adaptive cap: draft at most p tokens (alpha=1)
  |    |                         +-- if p >= 2 and drafts > 0: use suffix drafts
  |    |                         +-- else: return, try MTP
  |    |
  |    +-- suffix hit? --------> drafts[0..n-1] = suffix tree proposals
  |    |                           skip MTP recursive loop entirely
  |    |
  |    +-- no suffix hit -------> MTP available? -> MTP recursive draft
  |                               MTP unavailable? -> emit only target token
  |
  +-- first draft verified free (logits already on hand)
  |
  +-- can_batch_verify?
  |    (s->graph.spec_logits != NULL)
  |    |
  |    +-- yes -> microbatch / exact decode verifier
  |    +-- no  -> sequential verification fallback
  |
  +-- all verified drafts: DS4_SUFFIX_NOTE_ACCEPTED()
       +-- learn from checkpoint
       +-- track draft_tokens_accepted
```

### 2.4 Query Semantics

`ds4_suffix_tree_query()` finds the longest prefix suffix that **completely matches** the tree and has at least one continuation child. Three conditions must all hold:

1. `j == prefix_len` — the entire prefix suffix was consumed (no partial match)
2. `depth > match_depth` — this match is longer than any previous one
3. `cur->n_children > 0` — the matched node has continuations (not a terminal leaf)

If no match satisfies all three, returns `p = 0` and the caller falls back to MTP or single-token decode.

### 2.5 Memory Management

- **Budget**: byte budget converted to node budget at alloc time (`byte_budget / (1.5 * sizeof(node))`)
- **Insert-time pruning**: during bulk insert (e.g., seed from long prompt), pruning triggers incrementally when node count exceeds `budget + slack`, avoiding memory spikes
- **Multi-round pruning**: `prune_toward_budget()` runs up to 16 rounds of (decrement-all-freqs + remove-zero-leaves) to fully converge to budget
- **Pruning strategy**: frequency aging (decrement all by 1, clamp at 0) then remove zero-frequency leaves. Preserves frequently-used patterns, discards one-off sequences.

## 3. Files Changed

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `ds4_suffix_tree.h` | New | 92 | Suffix trie API, `ds4_suffix_stats` telemetry type |
| `ds4_suffix_tree.c` | New | 361 | Sorted-array trie implementation |
| `tests/suffix_tree_test.c` | New | 114 | Unit tests: continuation, terminal skip, partial prefix, prune/reset |
| `ds4.h` | Modified | +13 | 3 engine options, forward-declared `ds4_suffix_stats` |
| `ds4.c` | Modified | ~210 diff | Session integration, draft selection, verification gating |
| `ds4_cli.c` | Modified | +20 | `--suffix-decoding`, `--suffix-max-depth`, `--suffix-memory-budget`; `cli_speculative_decode_enabled()` |
| `ds4_bench.c` | Modified | +38 | `spec_steps` column, suffix telemetry CSV columns, backend guard |
| `Makefile` | Modified | +12 | `suffix-tree-test` target, build rules, clean |
| `README.md` | Modified | +27 | `--suffix-decoding` documentation |
| `CONTRIBUTING.md` | Modified | +6 | Suffix telemetry column guidance |
| `speed-bench/README.md` | Modified | +26 | MTP and suffix bench sweep examples |

### API Surface

**Public (`ds4.h`)**:
- `ds4_engine_options.suffix_decoding` (bool)
- `ds4_engine_options.suffix_max_depth` (uint32_t, default 32)
- `ds4_engine_options.suffix_memory_budget` (uint64_t bytes, default 64MB)
- `ds4_suffix_stats` telemetry snapshot struct
- `ds4_session_suffix_stats()` query function

**Internal (`ds4_suffix_tree.h`, not exposed to downstream)**:
- Full trie API: `alloc`, `free`, `insert`, `query`, `prune`, `reset`, `stats`

### CLI Flags

```
--suffix-decoding              Enable suffix tree speculative decoding
--suffix-max-depth N           Max sequence depth (default 32)
--suffix-memory-budget MB      Max tree memory in MB (default 64)
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

## 4. Verification Modes and Acceleration

| Mode | Draft source | Verifier | Acceleration |
|------|-------------|----------|-------------|
| MTP only | MTP recursive | batch verify (`spec_logits` available) | Can accelerate |
| Suffix + MTP | suffix tree (priority), MTP fallback | batch verify (`spec_logits` available) | Can accelerate |
| Suffix only | suffix tree | sequential fallback (`spec_logits == NULL`) | No acceleration |
| CPU | N/A (exits early) | N/A | N/A |

### Why suffix-only cannot accelerate (current implementation)

`spec_logits` and spec frontier buffers are allocated during MTP graph construction. Without MTP, `s->graph.spec_logits` is NULL, so `can_batch_verify` is false. The code safely skips batch/exact verifiers and falls through to sequential verification, which runs one forward pass per draft token — same cost as baseline decode.

**To enable suffix-only acceleration**, a follow-up change would need to:
1. Allocate `spec_logits` and frontier buffers when `--suffix-decoding` is set (without requiring MTP weights)
2. Validate that `metal_graph_verify_suffix_tops()` works without MTP-specific graph state
3. Benchmark with a real DeepSeek V4 model

This is an architectural follow-up, not a code change to the suffix trie itself.

## 5. Safety Properties

### Correctness guarantees

- **Target verifier gates all output**: regardless of draft source, every proposed token is verified against the target model's logits before being committed. Incorrect drafts are rejected and speculative state is rolled back via `spec_frontier_snapshot` / `spec_frontier_restore`.
- **No output distribution change**: accepted tokens are identical to what the target model would produce via greedy decode. The suffix tree only proposes drafts; the verifier decides.
- **MTP SWA counters protected**: `DS4_MTP_KEEP_ACCEPTED()` is gated by `using_mtp`, so suffix-only drafts never corrupt MTP raw SWA state.
- **Memory bounded**: hard node budget with multi-round pruning. Insert-time incremental pruning prevents transient memory spikes.

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

## 6. Testing

### Unit tests (`make suffix-tree-test`)

| Test | Validates |
|------|-----------|
| `test_repeated_continuation` | Most-frequent continuation from repeated pattern; telemetry counters |
| `test_skip_terminal_longest_match` | Terminal suffix (no children) is skipped in favor of non-terminal match |
| `test_no_partial_prefix_match` | Prefix `{20,30,99}` does not match tree path `{20,30,...}` — requires `j == prefix_len` |
| `test_prune_and_reset` | Tree prunes to budget under adversarial input; reset clears all nodes |

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

## 7. Known Limitations and Follow-ups

| Limitation | Impact | Follow-up |
|------------|--------|-----------|
| Suffix-only cannot batch verify | No acceleration without MTP | Allocate `spec_logits` independently of MTP |
| `learn_checkpoint` re-inserts N-1 old tokens per step | Freq bias toward older patterns | Differential insert (only new suffixes) |
| Sync reset+re-seed on large trees | Brief pause during sync | Lazy seed or background rebuild |
| `drafts[16]` buffer limit | Max 15 suffix drafts per step | Expand buffer if deeper speculation validated |
| No logprob quality regression test | Cannot claim output quality unchanged | Run full model benchmark with reference outputs |
| CSV column reordering | Breaks downstream CSV parsers | Document as breaking change in PR notes |

## 8. Resume Bullet

> Integrated an opt-in SuffixDecoding-style model-free speculative decoding path into DwarfStar (antirez/ds4), a DeepSeek V4-specific C/Metal/CUDA inference engine. Implemented a bounded CPU-resident suffix trie in pure C (~360 lines), seeded from prompt/checkpoint tokens and updated from accepted generation tokens, then wired it as an additional draft source before the existing MTP fallback and target verification pipeline. Added benchmark telemetry and lightweight unit coverage without requiring an extra draft model or GPU kernels.

## 9. References

- SuffixDecoding paper: https://arxiv.org/abs/2411.04975
- SuffixDecoding project page: https://suffix-decoding.github.io/
- ArcticInference (Snowflake): https://github.com/snowflakedb/ArcticInference
- vLLM suffix decoding: https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/
- ds4 repository: https://github.com/antirez/ds4
