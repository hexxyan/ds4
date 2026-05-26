/* ds4_suffix_tree.c -- Suffix trie for model-free speculative decoding.
 *
 * Reference: SuffixDecoding (NeurIPS 2025 Spotlight, arxiv:2411.04975).
 *
 * This is a compact trie (not a full suffix tree) where:
 * - Insert adds all suffixes of the token sequence up to max_depth long.
 * - Query traverses the trie from the root using the last N context tokens,
 *   then follows the most frequent continuation path to propose drafts.
 * - Prune removes low-frequency leaf nodes to stay within memory budget.
 *
 * The trie uses sorted arrays for children (binary search lookup),
 * which is cache-friendly and allocation-efficient for the small fanout
 * typical of LLM token sequences.
 */

#include "ds4_suffix_tree.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------- internal helpers ---------- */

#define ST_MALLOC(sz) malloc(sz)
#define ST_CALLOC(n, sz) calloc((n), (sz))
#define ST_REALLOC(ptr, sz) realloc((ptr), (sz))
#define ST_FREE(ptr) free(ptr)

/* Binary-search a sorted children array for token_id.
 * Returns the index if found, or -1. */
static int find_child(const ds4_suffix_node *node, int token_id) {
    uint32_t lo = 0, hi = node->n_children;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (node->children[mid].token_id == token_id) return (int)mid;
        if (node->children[mid].token_id < token_id) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

/* Insert a child into the sorted array, growing if needed.
 * Returns a pointer to the new child node (inside the parent's array). */
static ds4_suffix_node *insert_child(ds4_suffix_node *parent, int token_id) {
    if (parent->n_children >= parent->cap_children) {
        uint32_t new_cap = parent->cap_children == 0 ? 4 :
                           parent->cap_children * 2;
        ds4_suffix_node *new_arr = (ds4_suffix_node *)ST_REALLOC(
            parent->children, (size_t)new_cap * sizeof(ds4_suffix_node));
        if (!new_arr) return NULL;
        parent->children = new_arr;
        parent->cap_children = new_cap;
    }

    /* Find insertion point to keep sorted order. */
    uint32_t pos = parent->n_children;
    for (uint32_t i = 0; i < parent->n_children; i++) {
        if (parent->children[i].token_id > token_id) {
            pos = i;
            break;
        }
    }

    /* Shift elements right. */
    memmove(parent->children + pos + 1,
            parent->children + pos,
            (size_t)(parent->n_children - pos) * sizeof(ds4_suffix_node));

    /* Initialize new child. */
    ds4_suffix_node *child = &parent->children[pos];
    memset(child, 0, sizeof(*child));
    child->token_id = token_id;
    child->freq = 0;
    child->n_children = 0;
    child->cap_children = 0;
    child->children = NULL;
    parent->n_children++;
    return child;
}

/* Find or create a child node for token_id in parent. */
static ds4_suffix_node *ensure_child(ds4_suffix_node *parent, int token_id,
                                      int *created) {
    int idx = find_child(parent, token_id);
    if (idx >= 0) {
        *created = 0;
        return &parent->children[idx];
    }
    *created = 1;
    return insert_child(parent, token_id);
}

/* Recursively free all children. */
static void free_node_recursive(ds4_suffix_node *node) {
    if (!node) return;
    for (uint32_t i = 0; i < node->n_children; i++) {
        free_node_recursive(&node->children[i]);
    }
    ST_FREE(node->children);
    node->children = NULL;
    node->n_children = 0;
    node->cap_children = 0;
}

/* Decrement all node frequencies by 1 (clamp at 0) via iterative DFS.
 * Uses a dynamically-allocated stack to handle wide trees without
 * silently dropping nodes. */
static void decrement_all_freqs(ds4_suffix_node *root) {
    /* Start with a reasonable on-stack buffer; malloc a larger one if needed. */
    uint32_t stack_cap = 1024;
    ds4_suffix_node **stack = (ds4_suffix_node **)ST_MALLOC(
        (size_t)stack_cap * sizeof(ds4_suffix_node *));
    if (!stack) return;  /* OOM: skip aging (not fatal) */

    int sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
        sp--;
        ds4_suffix_node *node = stack[sp];
        /* Decrement freq, clamp at 0. */
        if (node->freq > 0) node->freq--;

        /* Grow stack if needed to fit all children. */
        if (node->n_children > 0 &&
            (uint32_t)(sp + (int)node->n_children) > stack_cap) {
            uint32_t new_cap = stack_cap * 2;
            if (new_cap < stack_cap + node->n_children)
                new_cap = stack_cap + node->n_children;
            ds4_suffix_node **new_stack = (ds4_suffix_node **)ST_REALLOC(
                stack, (size_t)new_cap * sizeof(ds4_suffix_node *));
            if (!new_stack) break;  /* OOM: stop early */
            stack = new_stack;
            stack_cap = new_cap;
        }

        /* Push children onto stack. */
        for (uint32_t i = 0; i < node->n_children; i++) {
            if (sp < (int)stack_cap) {
                stack[sp++] = &node->children[i];
            }
        }
    }

    ST_FREE(stack);
}

/* Recursively remove leaf nodes with freq==0.
 * Returns the number of nodes removed. */
static uint32_t remove_zero_leaves(ds4_suffix_node *node) {
    if (!node) return 0;
    uint32_t removed = 0;

    /* Process children back-to-front. */
    uint32_t i = node->n_children;
    while (i > 0) {
        i--;
        ds4_suffix_node *child = &node->children[i];

        /* Recurse first. */
        removed += remove_zero_leaves(child);

        /* After recursing, if this child is now a leaf with freq==0, remove. */
        if (child->n_children == 0 && child->freq == 0) {
            ST_FREE(child->children);
            child->children = NULL;
            /* Shift remaining children left. */
            if (i < node->n_children - 1) {
                memmove(node->children + i,
                        node->children + i + 1,
                        (size_t)(node->n_children - i - 1) *
                        sizeof(ds4_suffix_node));
            }
            node->n_children--;
            removed++;
        }
    }
    return removed;
}

/* ---------- public API ---------- */

ds4_suffix_tree *ds4_suffix_tree_alloc(uint64_t byte_budget,
                                         uint32_t max_depth) {
    ds4_suffix_tree *tree = (ds4_suffix_tree *)ST_CALLOC(1, sizeof(*tree));
    if (!tree) return NULL;
    memset(&tree->root, 0, sizeof(tree->root));
    tree->root.token_id = -1;
    tree->root.freq = 0;
    tree->node_count = 0;
    /* Convert byte budget to node budget.  Each node uses ~sizeof(node) plus
     * average ~50% children-array overhead.  We use a conservative 1.5x. */
    uint64_t node_size_est = (uint64_t)(sizeof(ds4_suffix_node) * 3 / 2);
    tree->node_budget = byte_budget / (node_size_est > 0 ? node_size_est : 1);
    if (tree->node_budget < 1024) tree->node_budget = 1024;  /* floor */
    tree->total_bytes = sizeof(*tree);
    tree->max_depth = max_depth;
    tree->query_count = 0;
    tree->query_hits = 0;
    tree->draft_tokens_produced = 0;
    tree->draft_tokens_accepted = 0;
    return tree;
}

void ds4_suffix_tree_free(ds4_suffix_tree *tree) {
    if (!tree) return;
    free_node_recursive(&tree->root);
    ST_FREE(tree);
}

uint32_t ds4_suffix_tree_insert(ds4_suffix_tree *tree,
                                 const int *tokens, uint32_t len) {
    if (!tree || !tokens || len == 0) return 0;
    uint32_t created = 0;
    uint32_t max_d = tree->max_depth;

    /* Insert all suffixes starting at each position.
     * Each suffix is truncated to max_depth tokens. */
    for (uint32_t start = 0; start < len; start++) {
        uint32_t suffix_len = len - start;
        if (suffix_len > max_d) suffix_len = max_d;

        ds4_suffix_node *cur = &tree->root;
        for (uint32_t j = 0; j < suffix_len; j++) {
            int tok = tokens[start + j];
            int did_create = 0;
            ds4_suffix_node *child = ensure_child(cur, tok, &did_create);
            if (!child) break;  /* allocation failure, skip rest */
            child->freq++;
            if (did_create) {
                tree->node_count++;
                tree->total_bytes += sizeof(ds4_suffix_node);
                created++;
            }
            cur = child;
        }
    }

    /* Auto-prune if we exceeded budget. */
    if (tree->node_count > tree->node_budget) {
        ds4_suffix_tree_prune(tree);
    }

    return created;
}

uint32_t ds4_suffix_tree_query(ds4_suffix_tree *tree,
                                const int *prefix, uint32_t prefix_len,
                                int *drafts, uint32_t max_drafts,
                                uint32_t *draft_n) {
    if (!tree || !prefix || !drafts || !draft_n) {
        if (draft_n) *draft_n = 0;
        return 0;
    }
    *draft_n = 0;
    tree->query_count++;

    if (prefix_len == 0 || max_drafts == 0) return 0;

    /* Walk the tree from the root using the prefix tokens.
     * We try matching from different prefix start positions:
     * first try the full prefix, then progressively shorter suffixes,
     * until we find a match. This handles the case where only a
     * suffix of the current context matches a stored sequence. */
    ds4_suffix_node *match = NULL;
    uint32_t match_depth = 0;

    for (uint32_t start = prefix_len > tree->max_depth ?
                          prefix_len - tree->max_depth : 0;
         start < prefix_len; start++) {
        ds4_suffix_node *cur = &tree->root;
        uint32_t depth = 0;
        for (uint32_t j = start; j < prefix_len; j++) {
            int idx = find_child(cur, prefix[j]);
            if (idx < 0) break;
            cur = &cur->children[idx];
            depth++;
        }
        if (depth > match_depth) {
            match = cur;
            match_depth = depth;
        }
    }

    if (!match || match_depth == 0) return 0;

    /* Now follow the most frequent continuation from the match node. */
    uint32_t d = 0;
    ds4_suffix_node *cur = match;
    while (d < max_drafts && cur->n_children > 0) {
        /* Pick the child with highest frequency. */
        uint32_t best = 0;
        uint32_t best_freq = cur->children[0].freq;
        for (uint32_t i = 1; i < cur->n_children; i++) {
            if (cur->children[i].freq > best_freq) {
                best_freq = cur->children[i].freq;
                best = i;
            }
        }
        drafts[d++] = cur->children[best].token_id;
        cur = &cur->children[best];
    }

    *draft_n = d;
    tree->query_hits++;
    tree->draft_tokens_produced += d;
    return match_depth;
}

uint32_t ds4_suffix_tree_prune(ds4_suffix_tree *tree) {
    if (!tree) return 0;

    /* Phase 1: decrement all frequencies by 1 (aging). */
    decrement_all_freqs(&tree->root);

    /* Phase 2: remove leaf nodes with freq==0. */
    uint32_t removed = remove_zero_leaves(&tree->root);

    tree->node_count -= removed;
    tree->total_bytes -= (uint64_t)removed * sizeof(ds4_suffix_node);
    return removed;
}

void ds4_suffix_tree_reset(ds4_suffix_tree *tree) {
    if (!tree) return;
    free_node_recursive(&tree->root);
    memset(&tree->root, 0, sizeof(tree->root));
    tree->root.token_id = -1;
    tree->node_count = 0;
    tree->total_bytes = sizeof(*tree);
    /* Keep telemetry counters intact for reporting. */
}

void ds4_suffix_tree_stats(const ds4_suffix_tree *tree,
                             ds4_suffix_stats *out) {
    if (!out) return;
    if (!tree) { memset(out, 0, sizeof(*out)); return; }
    out->node_count = tree->node_count;
    out->total_bytes = tree->total_bytes;
    out->query_count = tree->query_count;
    out->query_hits = tree->query_hits;
    out->draft_tokens_produced = tree->draft_tokens_produced;
    out->draft_tokens_accepted = tree->draft_tokens_accepted;
}
