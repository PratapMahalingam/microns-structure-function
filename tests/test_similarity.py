import numpy as np
import pytest

from microns_connectivity import pair_similarity, zscore_rows, cliffs_delta_from_u


def test_pair_similarity_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 120))
    z = zscore_rows(x)
    i = rng.integers(0, 30, 200)
    j = rng.integers(0, 30, 200)
    expected = np.array([np.corrcoef(x[a], x[b])[0, 1] for a, b in zip(i, j)])
    np.testing.assert_allclose(pair_similarity(z, i, j), expected, atol=1e-12)


def test_similarity_bounds_and_identities():
    x = np.array([[1.0, 2, 3, 4], [2.0, 4, 6, 8], [4.0, 3, 2, 1]])
    z = zscore_rows(x)
    r = pair_similarity(z, np.array([0, 0, 1]), np.array([1, 2, 1]))
    np.testing.assert_allclose(r, [1.0, -1.0, 1.0])


def test_scale_and_offset_invariance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 50))
    y = np.vstack([x[0], 5 * x[1] + 3])
    r1 = pair_similarity(zscore_rows(x), np.array([0]), np.array([1]))
    r2 = pair_similarity(zscore_rows(y), np.array([0]), np.array([1]))
    np.testing.assert_allclose(r1, r2)


def test_constant_rows_give_nan_not_zero():
    x = np.array([[1.0, 1, 1, 1], [1.0, 2, 3, 4]])
    r = pair_similarity(zscore_rows(x), np.array([0]), np.array([1]))
    assert np.isnan(r[0])


@pytest.mark.parametrize("u,n1,n2,expected", [(0, 4, 5, -1.0), (20, 4, 5, 1.0), (10, 4, 5, 0.0)])
def test_cliffs_delta(u, n1, n2, expected):
    assert cliffs_delta_from_u(u, n1, n2) == pytest.approx(expected)
