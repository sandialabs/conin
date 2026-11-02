"""Tests of conin.algorithms.util.variable_elimination against brute force."""

import itertools
import math
import random

import pytest

from conin.algorithms.util.variable_elimination import log_weight, max_product
from conin.exceptions import InvalidInputError
from conin.graphical_models import DiscreteFactor


def _weight(factors, assignment):
    w = 1.0
    for f in factors:
        key = (
            assignment[f.nodes[0]]
            if len(f.nodes) == 1
            else tuple(assignment[n] for n in f.nodes)
        )
        w *= f.values.get(key, f.default_value)
    return w


def _brute_force(factors, states, variables, evidence):
    """``{assignment of variables: summed weight}`` and the log partition function."""
    free = [n for n in states if n not in evidence]
    marginal = {}
    for combo in itertools.product(*(states[n] for n in free)):
        a = {**dict(zip(free, combo)), **evidence}
        key = tuple(a[v] for v in variables)
        marginal[key] = marginal.get(key, 0.0) + _weight(factors, a)
    return marginal


def _log(x):
    return math.log(x) if x > 0 else -math.inf


def _random_model(rng):
    n = rng.randint(1, 6)
    states = {f"x{i}": list(range(rng.randint(1, 3))) for i in range(n)}
    factors = []
    for _ in range(rng.randint(1, 6)):
        scope = rng.sample(list(states), rng.randint(1, min(3, n)))
        values = {}
        for combo in itertools.product(*(states[s] for s in scope)):
            key = combo[0] if len(scope) == 1 else combo
            values[key] = rng.choice([0, 0.5, 1, 2, rng.random()])
        factors.append(DiscreteFactor(nodes=scope, values=values))
    return factors, states


@pytest.mark.parametrize("seed", range(200))
def test_random_models_match_brute_force(seed):
    rng = random.Random(seed)
    factors, states = _random_model(rng)
    nodes = list(states)
    evidence = {
        n: rng.choice(states[n])
        for n in rng.sample(nodes, rng.randint(0, len(nodes) - 1))
    }
    rest = [n for n in nodes if n not in evidence]
    variables = rng.sample(rest, rng.randint(1, len(rest)))
    marginal = _brute_force(factors, states, variables, evidence)

    value, assignment = max_product(
        factors, states, variables=variables, evidence=evidence
    )
    assert value == pytest.approx(_log(max(marginal.values())), abs=1e-9)
    assert list(assignment) == variables
    if value > -math.inf:
        key = tuple(assignment[v] for v in variables)
        assert _log(marginal[key]) == pytest.approx(value, abs=1e-9)

    expected = _log(sum(marginal.values()))
    assert log_weight(factors, states, evidence=evidence) == pytest.approx(
        expected, abs=1e-9
    )


def _chain(length, card=2):
    states = {i: list(range(card)) for i in range(length)}
    values = {(a, b): (2 if a == b else 1) for a in range(card) for b in range(card)}
    factors = [
        DiscreteFactor(nodes=[i, i + 1], values=values) for i in range(length - 1)
    ]
    return factors, states


def test_full_map_by_default():
    factors, states = _chain(4)
    value, assignment = max_product(factors, states)
    assert value == pytest.approx(3 * math.log(2))
    assert len(set(assignment.values())) == 1  # a constant chain


def test_evidence_is_not_in_the_assignment():
    factors, states = _chain(3)
    value, assignment = max_product(factors, states, evidence={1: 1})
    assert assignment == {0: 1, 2: 1}
    assert value == pytest.approx(2 * math.log(2))


def test_a_node_in_no_factor_is_summed_over_its_states():
    factors = [DiscreteFactor(nodes=["A"], values={0: 1, 1: 3})]
    states = {"A": [0, 1], "B": ["x", "y", "z"]}
    assert log_weight(factors, states) == pytest.approx(math.log(4 * 3))
    value, assignment = max_product(factors, states)
    assert assignment == {"A": 1, "B": "x"}
    assert value == pytest.approx(math.log(3))
    assert max_product(factors, states, variables=["A"])[0] == pytest.approx(
        math.log(9)
    )


def test_missing_entries_use_the_default_value():
    factors = [DiscreteFactor(nodes=["A", "B"], values={(0, 0): 5}, default_value=1)]
    states = {"A": [0, 1], "B": [0, 1]}
    assert log_weight(factors, states) == pytest.approx(math.log(5 + 3))


def test_zero_weight_everywhere():
    factors = [DiscreteFactor(nodes=["A"], values={0: 0, 1: 0})]
    states = {"A": [0, 1]}
    assert max_product(factors, states)[0] == -math.inf
    assert log_weight(factors, states) == -math.inf


def test_long_chains_do_not_underflow():
    # 2000 factors of 1e-200 each: the weight is far below the smallest float.
    states = {i: [0] for i in range(2001)}
    factors = [
        DiscreteFactor(nodes=[i, i + 1], values={(0, 0): 1e-200}) for i in range(2000)
    ]
    expected = 2000 * math.log(1e-200)
    assert log_weight(factors, states) == pytest.approx(expected)
    assert max_product(factors, states)[0] == pytest.approx(expected)


def test_states_need_not_be_integers():
    factors = [DiscreteFactor(nodes=[("H", 0)], values={"a": 1, "b": 2})]
    states = {("H", 0): ["a", "b"]}
    assert max_product(factors, states)[1] == {("H", 0): "b"}


@pytest.mark.parametrize(
    "kwds,match",
    [
        (dict(evidence={"X": 0}), "Unknown evidence variable 'X'"),
        (dict(evidence={"A": 7}), "Unknown state 7 of the evidence variable 'A'"),
        (dict(variables=["X"]), "Unknown query variable 'X'"),
        (
            dict(variables=["A"], evidence={"A": 0}),
            "both a query variable and evidence",
        ),
    ],
)
def test_bad_queries_raise(kwds, match):
    factors = [DiscreteFactor(nodes=["A"], values={0: 1, 1: 1})]
    with pytest.raises(InvalidInputError, match=match):
        max_product(factors, {"A": [0, 1]}, **kwds)
