from collections import Counter

import numpy as np
import pytest

from microns_connectivity import degree_preserving_shuffle, proximity_matched_shuffle


def random_digraph(rng, n_nodes=200, n_pre=20, n_edges=600):
    pres = rng.choice(n_nodes, n_pre, replace=False)
    edges = set()
    while len(edges) < n_edges:
        a = rng.choice(pres)
        b = rng.integers(n_nodes)
        if a != b:
            edges.add((int(a), int(b)))
    e = np.array(sorted(edges))
    return e[:, 0], e[:, 1]


@pytest.mark.parametrize("seed", range(5))
def test_degree_shuffle_preserves_degrees_and_simplicity(seed):
    rng = np.random.default_rng(seed)
    pre, post = random_digraph(rng)
    new_post = degree_preserving_shuffle(pre, post, rng)
    # out-degree preserved trivially (pre unchanged); in-degree preserved
    assert Counter(new_post.tolist()) == Counter(post.tolist())
    assert np.array_equal(pre, pre)
    # no self-loops, no duplicate edges
    assert not np.any(pre == new_post)
    assert len(set(zip(pre.tolist(), new_post.tolist()))) == len(pre)


def test_degree_shuffle_actually_randomises():
    rng = np.random.default_rng(0)
    pre, post = random_digraph(rng)
    new_post = degree_preserving_shuffle(pre, post, rng)
    overlap = len(set(zip(pre, post)) & set(zip(pre, new_post))) / len(pre)
    # each presynaptic neuron targets ~15% of nodes, so chance overlap is ~0.15;
    # an unrandomised graph would give 1.0
    assert overlap < 0.3


def test_degree_shuffle_is_reproducible():
    pre, post = random_digraph(np.random.default_rng(3))
    a = degree_preserving_shuffle(pre, post, np.random.default_rng(42))
    b = degree_preserving_shuffle(pre, post, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_degree_shuffle_handles_reciprocal_edges():
    # pre neurons that also receive input (pre->pre edges) must not create self-loops
    pre = np.array([0, 0, 1, 1, 2, 2, 3])
    post = np.array([1, 2, 0, 2, 0, 1, 0])
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = degree_preserving_shuffle(pre, post, rng)
        assert not np.any(pre == p)
        assert len(set(zip(pre.tolist(), p.tolist()))) == len(pre)


def test_proximity_shuffle_draws_only_from_candidates():
    rng = np.random.default_rng(0)
    pre = np.array([0, 0, 0, 1, 1])
    cands = {0: (np.array([10, 11, 12, 13, 14]), np.ones(5)),
             1: (np.array([20, 21, 22]), np.array([1.0, 1.0, 0.0]))}
    for _ in range(100):
        out = proximity_matched_shuffle(pre, cands, rng)
        assert set(out[:3]) <= {10, 11, 12, 13, 14} and len(set(out[:3])) == 3
        assert set(out[3:]) == {20, 21}  # zero-weight candidate never drawn


def test_proximity_shuffle_respects_weights():
    rng = np.random.default_rng(1)
    pre = np.array([0])
    cands = {0: (np.array([1, 2]), np.array([9.0, 1.0]))}
    draws = [proximity_matched_shuffle(pre, cands, rng)[0] for _ in range(4000)]
    assert np.mean(np.array(draws) == 1) == pytest.approx(0.9, abs=0.03)
