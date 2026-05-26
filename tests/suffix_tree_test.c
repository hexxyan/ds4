#include "../ds4_suffix_tree.h"

#include <stdio.h>

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "suffix_tree_test: %s\n", msg);
        return 1;
    }
    return 0;
}

static ds4_suffix_node *lookup_child(ds4_suffix_node *node, int token_id) {
    for (uint32_t i = 0; node && i < node->n_children; i++) {
        if (node->children[i].token_id == token_id) return &node->children[i];
    }
    return NULL;
}

static int test_repeated_continuation(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    const int seq[] = {
        1, 2, 3, 4, 9,
        1, 2, 3, 4, 9,
        1, 2, 3, 5, 8,
    };
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    int drafts[4] = {0};
    uint32_t draft_n = 0;
    const int prefix[] = {1, 2, 3};
    uint32_t p = ds4_suffix_tree_query(tree, prefix, 3, drafts, 4, &draft_n);

    int rc = 0;
    rc |= expect(p == 3, "expected full 1,2,3 match");
    rc |= expect(draft_n >= 2, "expected continuation after 1,2,3");
    rc |= expect(drafts[0] == 4, "expected most frequent continuation token 4");
    rc |= expect(drafts[1] == 9, "expected most frequent second continuation token 9");

    ds4_suffix_stats st;
    ds4_suffix_tree_stats(tree, &st);
    rc |= expect(st.query_count == 1, "query_count should be 1");
    rc |= expect(st.query_hits == 1, "query_hits should be 1");
    rc |= expect(st.draft_tokens_produced == draft_n,
                 "draft_tokens_produced should match first query");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_append_does_not_reinflate_prefix(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 4);
    if (!tree) return expect(0, "alloc failed");

    const int seq1[] = {1, 2, 3};
    const int seq2[] = {1, 2, 3, 4};
    ds4_suffix_tree_insert(tree, seq1, 3);

    ds4_suffix_node *n2 = lookup_child(&tree->root, 2);
    int rc = 0;
    rc |= expect(n2 != NULL, "expected suffix starting at token 2");
    rc |= expect(n2 && n2->freq == 1,
                 "initial token 2 suffix frequency should be 1");

    ds4_suffix_tree_append(tree, seq2, 4);
    n2 = lookup_child(&tree->root, 2);
    ds4_suffix_node *n23 = lookup_child(n2, 3);
    ds4_suffix_node *n234 = lookup_child(n23, 4);
    ds4_suffix_node *n4 = lookup_child(&tree->root, 4);

    rc |= expect(n2 && n2->freq == 1,
                 "append should not reinflate existing token 2 prefix");
    rc |= expect(n234 && n234->freq == 1,
                 "append should extend existing 2,3 suffix with token 4");
    rc |= expect(n4 && n4->freq == 1,
                 "append should add one-token suffix for new token");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_skip_terminal_longest_match(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    const int seq[] = {1, 2, 3, 4, 7, 2, 3, 4};
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    int drafts[4] = {0};
    uint32_t draft_n = 0;
    const int prefix[] = {2, 3, 4};
    uint32_t p = ds4_suffix_tree_query(tree, prefix, 3, drafts, 4, &draft_n);

    int rc = 0;
    rc |= expect(p == 3, "expected longest match with continuation, not terminal suffix");
    rc |= expect(draft_n > 0, "expected a draft from non-terminal match");
    rc |= expect(drafts[0] == 7, "expected continuation token 7");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_no_partial_prefix_match(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    const int seq[] = {10, 20, 30, 40, 50};
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    int drafts[4] = {0};
    uint32_t draft_n = 123;
    const int prefix[] = {20, 30, 99};
    uint32_t p = ds4_suffix_tree_query(tree, prefix, 3, drafts, 4, &draft_n);

    int rc = 0;
    rc |= expect(p == 0, "partial prefix should not count as a suffix match");
    rc |= expect(draft_n == 0, "partial prefix should not produce drafts");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_prune_and_reset(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1, 8);
    if (!tree) return expect(0, "alloc failed");

    int rc = 0;
    for (int i = 0; i < 2048; i++) {
        int seq[] = {i, i + 4096};
        ds4_suffix_tree_insert(tree, seq, 2);
    }
    rc |= expect(tree->node_count <= tree->node_budget,
                 "tree should prune back to its node budget");

    ds4_suffix_tree_reset(tree);
    rc |= expect(tree->node_count == 0, "reset should clear all nodes");
    rc |= expect(tree->root.n_children == 0, "reset should clear root children");

    ds4_suffix_tree_free(tree);
    return rc;
}

int main(void) {
    int rc = 0;
    rc |= test_repeated_continuation();
    rc |= test_append_does_not_reinflate_prefix();
    rc |= test_skip_terminal_longest_match();
    rc |= test_no_partial_prefix_match();
    rc |= test_prune_and_reset();
    return rc ? 1 : 0;
}
