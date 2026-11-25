"""Tests for conin.examples.markov_network.random_markov_network."""

import itertools
import math

import pytest

from conin.algorithms.inference import map_query
from conin.examples.markov_network import random_markov_network
from conin.graphical_models import DiscreteMarkovNetwork


def _num(node):
    """The number of node ``"x<number>"``."""
    return int(node[1:])


def _scopes(pgm):
    """The sorted factor scopes, as tuples of node numbers."""
    return sorted(tuple(_num(n) for n in f.nodes) for f in pgm.factors)


def _factors_as_tuples(pgm):
    return [(tuple(f.nodes), dict(f.values)) for f in pgm.factors]


def _brute_force_map(pgm):
    """The largest product of the factor values, over all assignments."""
    nodes = list(pgm.nodes)
    best = 0.0
    for assignment in itertools.product(*[pgm.states_of(n) for n in nodes]):
        x = dict(zip(nodes, assignment))
        value = 1.0
        for f in pgm.factors:
            key = tuple(x[n] for n in f.nodes)
            key = key[0] if len(key) == 1 else key
            value *= f.values.get(key, f.default_value)
        best = max(best, value)
    return best


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("N,M,k,max_vars", [(3, 2, 2, 2), (4, 3, 3, 3), (5, 4, 2, 3)])
def test_structure(seed, N, M, k, max_vars):
    pgm = random_markov_network(seed, N=N, M=M, k=k, max_vars=max_vars)
    assert isinstance(pgm, DiscreteMarkovNetwork)
    assert set(pgm.nodes) == {f"x{r}" for r in range(N)}
    assert all(list(pgm.states_of(n)) == list(range(k)) for n in pgm.nodes)

    random_factors = pgm.factors[:M]
    # The nodes of a factor are ordered by number, and no two factors have
    # the same nodes
    assert len({tuple(f.nodes) for f in random_factors}) == M
    for f in random_factors:
        assert 2 <= len(f.nodes) <= max_vars
        nums = [_num(n) for n in f.nodes]
        assert nums == sorted(set(nums))
        # density=1.0: every state combination has a weight
        assert len(f.values) == k ** len(f.nodes)
        assert all(math.exp(-5) <= w <= 1 for w in f.values.values())

    # The other factors are uniform unary factors for the unused nodes
    used = {r for f in random_factors for r in f.nodes}
    extra = pgm.factors[M:]
    assert {f.nodes[0] for f in extra} == set(pgm.nodes) - used
    for f in extra:
        assert len(f.nodes) == 1
        assert list(f.values.values()) == [1] * k


def test_every_scope_used():
    # There are exactly 3 pairs of 3 nodes, so every pair gets a factor
    pgm = random_markov_network(0, N=3, M=3, k=2, max_vars=2)
    assert _scopes(pgm) == [(0, 1), (0, 2), (1, 2)]


def test_too_many_factors():
    with pytest.raises(ValueError, match="only 3 sets of 2 to 2 of the 3 nodes"):
        random_markov_network(0, N=3, M=4, max_vars=2)


def test_max_vars_larger_than_N():
    pgm = random_markov_network(0, N=3, M=4, k=2, max_vars=5)
    assert _scopes(pgm) == [
        (0, 1),
        (0, 1, 2),
        (0, 2),
        (1, 2),
    ]


def test_unused_nodes_get_unary_factors():
    # One pairwise factor cannot cover five nodes
    pgm = random_markov_network(0, N=5, M=1, k=2, max_vars=2)
    assert set(pgm.nodes) == {"x0", "x1", "x2", "x3", "x4"}
    assert len(pgm.factors) == 4
    assert [len(f.nodes) for f in pgm.factors] == [2, 1, 1, 1]


@pytest.mark.parametrize("seed", range(5))
def test_unique_factor_names(seed):
    """The factor names of the pyomo backend (nodes joined by "_") are unique."""
    pgm = random_markov_network(seed, N=4, M=8, k=2, max_vars=3)
    assert not any("_" in node for node in pgm.nodes)
    names = ["_".join(f.nodes) for f in pgm.factors]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("density", [0.0, 0.3, 0.5])
def test_density(density):
    pgm = random_markov_network(1, N=4, M=3, k=3, max_vars=3, density=density)
    for f in pgm.factors[:3]:
        n = 3 ** len(f.nodes)
        assert len(f.values) == max(2, int(n * density))
        assert f.default_value == 0


def test_seed():
    a = random_markov_network(7, N=4, M=3, k=2)
    b = random_markov_network(7, N=4, M=3, k=2)
    c = random_markov_network(8, N=4, M=3, k=2)
    assert _factors_as_tuples(a) == _factors_as_tuples(b)
    assert _factors_as_tuples(a) != _factors_as_tuples(c)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("density", [1.0, 0.7])
def test_map_query_matches_brute_force(seed, density):
    pgm = random_markov_network(seed, N=4, M=4, k=2, max_vars=3, density=density)
    best = _brute_force_map(pgm)
    if best == 0:
        pytest.skip("every assignment has weight 0")
    results = map_query(pgm, method="variable_elimination")
    x = results.solution.states

    value = 1.0
    for f in pgm.factors:
        key = tuple(x[n] for n in f.nodes)
        key = key[0] if len(key) == 1 else key
        value *= f.values.get(key, f.default_value)
    assert value == pytest.approx(best)
