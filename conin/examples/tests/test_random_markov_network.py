"""Tests for conin.examples.markov_network.random_markov_network."""

import itertools
import math

import pytest

from conin.algorithms.inference import map_query
from conin.examples.markov_network import random_markov_network
from conin.graphical_models import DiscreteMarkovNetwork


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
    assert list(pgm.nodes) == list(range(N))
    assert all(list(pgm.states_of(r)) == list(range(k)) for r in range(N))

    random_factors = pgm.factors[:M]
    for f in random_factors:
        assert 2 <= len(f.nodes) <= max_vars
        assert len(set(f.nodes)) == len(f.nodes)
        # density=1.0: every state combination has a weight
        assert len(f.values) == k ** len(f.nodes)
        assert all(math.exp(-5) <= w <= 1 for w in f.values.values())

    # The other factors are uniform unary factors for the unused nodes
    used = {r for f in random_factors for r in f.nodes}
    extra = pgm.factors[M:]
    assert sorted(f.nodes[0] for f in extra) == sorted(set(range(N)) - used)
    for f in extra:
        assert len(f.nodes) == 1
        assert list(f.values.values()) == [1] * k


def test_unused_nodes_get_unary_factors():
    # One pairwise factor cannot cover five nodes
    pgm = random_markov_network(0, N=5, M=1, k=2, max_vars=2)
    assert list(pgm.nodes) == list(range(5))
    assert len(pgm.factors) == 4
    assert [len(f.nodes) for f in pgm.factors] == [2, 1, 1, 1]


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
