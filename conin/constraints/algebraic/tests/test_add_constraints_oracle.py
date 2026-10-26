"""Tests for create_oracle_constraints_from_algebraic.

It compiles algebraic (smoek) constraints into one FactorOracleConstraint by
generating Python code; it is used by the feasibility checks and by variable
elimination.  The oracle must accept exactly the assignments that satisfy every
algebraic constraint.
"""

import importlib
import itertools

import munch
import pytest

from conin.constrained_models.feasibility import StateSpace
from conin.constraints import (
    FactorOracleConstraint,
    algebraic_constraint_fn,
    pyomo_constraint_fn,
)
from conin.util import try_import

with try_import() as smoek_available:
    import smoek  # noqa: F401

pytestmark = pytest.mark.skipif(not smoek_available, reason="smoek not installed")

module = importlib.import_module("conin.constraints.algebraic.add_constraints_oracle")
create = module.create_oracle_constraints_from_algebraic

STATES = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]}


def _oracle(constraints, states=STATES, data=None, name=None):
    oracles = create(
        pgm=StateSpace(states), constraints=constraints, data=data, name=name
    )
    assert len(oracles) == 1
    assert isinstance(oracles[0], FactorOracleConstraint)
    return oracles[0]


def _accepted(oracle, states=STATES, data=None):
    """The assignments of the oracle's scope that it accepts."""
    nodes = oracle.nodes
    accepted = set()
    for values in itertools.product(*(states[n] for n in nodes)):
        if oracle(dict(zip(nodes, values)), data):
            accepted.add(values)
    return accepted


def _all(nodes, states=STATES):
    return set(itertools.product(*(states[n] for n in nodes)))


@algebraic_constraint_fn()
def not_both_zero(model):
    return model.V("A", 0) + model.V("B", 0) <= 1


@algebraic_constraint_fn()
def c_is_two(model):
    return model.V("C", 2) == 1


@algebraic_constraint_fn()
def a_or_b_is_one(model):
    return model.V("A", 1) + model.V("B", 1) >= 1


def test_less_or_equal():
    oracle = _oracle([not_both_zero])
    assert oracle.nodes == ["A", "B"]
    assert _accepted(oracle) == _all(["A", "B"]) - {(0, 0)}


def test_equality():
    oracle = _oracle([c_is_two])
    assert oracle.nodes == ["C"]
    assert _accepted(oracle) == {(2,)}


def test_greater_or_equal():
    oracle = _oracle([a_or_b_is_one])
    assert _accepted(oracle) == {ab for ab in _all(["A", "B"]) if 1 in ab}


def test_several_constraints_must_all_hold():
    oracle = _oracle([not_both_zero, c_is_two])
    assert oracle.nodes == ["A", "B", "C"]
    expected = {(a, b, 2) for a, b in _all(["A", "B"]) - {(0, 0)}}
    assert _accepted(oracle) == expected


def test_a_constraint_returning_a_list():
    @algebraic_constraint_fn()
    def all_different_at_zero(model):
        return [model.V(n, 0) + model.V(m, 0) <= 1 for n, m in ["AB", "BC", "AC"]]

    oracle = _oracle([all_different_at_zero])
    expected = {abc for abc in _all(["A", "B", "C"]) if list(abc).count(0) <= 1}
    assert _accepted(oracle) == expected


def test_default_and_given_names():
    assert _oracle([not_both_zero, c_is_two]).name == (
        "composite_not_both_zero_c_is_two"
    )
    assert _oracle([not_both_zero], name="mine").name == "mine"


def test_no_constraints():
    assert create(pgm=StateSpace(STATES), constraints=[], data=None) == []


def test_string_states():
    states = {"X": ["lo", "hi"], "Y": ["lo", "hi"]}

    @algebraic_constraint_fn()
    def not_both_hi(model):
        return model.V("X", "hi") + model.V("Y", "hi") <= 1

    oracle = _oracle([not_both_hi], states=states)
    assert _accepted(oracle, states) == _all(["X", "Y"], states) - {("hi", "hi")}


def test_integer_nodes():
    states = {0: [0, 1], 1: [0, 1]}

    @algebraic_constraint_fn()
    def exactly_one_set(model):
        return model.V(0, 1) + model.V(1, 1) == 1

    oracle = _oracle([exactly_one_set], states=states)
    assert _accepted(oracle, states) == {(0, 1), (1, 0)}


@pytest.mark.parametrize("name", ["H", 7])
def test_time_indexed_nodes(name):
    # Tuple nodes (name, t), as in an unrolled DBN or HMM: V(name, t, state).
    states = {(name, t): ["a", "b"] for t in range(3)}

    @algebraic_constraint_fn()
    def at_most_one_b(model):
        return sum(model.V(name, t, "b") for t in range(3)) <= 1

    oracle = _oracle([at_most_one_b], states=states)
    assert oracle.nodes == [(name, 0), (name, 1), (name, 2)]
    expected = {seq for seq in _all(oracle.nodes, states) if seq.count("b") <= 1}
    assert _accepted(oracle, states) == expected


def test_constraints_that_read_data():
    @algebraic_constraint_fn()
    def at_most_k_zeros(model, data):
        return model.V("A", 0) + model.V("B", 0) + model.V("C", 0) <= data.k

    data = munch.Munch(k=1)
    oracle = _oracle([at_most_k_zeros], data=data)
    assert oracle.num_args == 2
    expected = {abc for abc in _all(["A", "B", "C"]) if abc.count(0) <= 1}
    assert _accepted(oracle, data=data) == expected


def test_the_oracle_rejects_an_assignment_without_its_nodes():
    oracle = _oracle([not_both_zero])
    with pytest.raises(KeyError, match="Node B not found"):
        oracle({"A": 0}, None)


def test_non_algebraic_constraints_are_rejected():
    @pyomo_constraint_fn()
    def pyomo(model):
        return model

    with pytest.raises(TypeError, match="Element 1 is not an AlgebraicConstraint"):
        create(pgm=StateSpace(STATES), constraints=[not_both_zero, pyomo], data=None)


def test_smoek_is_required(monkeypatch):
    decorators = importlib.import_module("conin.constraints.algebraic.decorators")
    monkeypatch.setattr(decorators, "smoek_available", False)
    with pytest.raises(ImportError, match="smoek package must be installed"):
        create(pgm=StateSpace(STATES), constraints=[not_both_zero], data=None)


def test_code_that_does_not_compile_raises(monkeypatch):
    monkeypatch.setattr(
        module,
        "_generate_constraint_function_code",
        lambda strings, num_args: "def constraint_function(args):\n    return (",
    )
    with pytest.raises(RuntimeError, match="(?s)Failed to compile.*Generated code"):
        create(pgm=StateSpace(STATES), constraints=[not_both_zero], data=None)


@pytest.mark.parametrize(
    "num_args,signature", [(1, "(args):"), (2, "(args, data=None):")]
)
def test_generated_code_signature(num_args, signature):
    code = module._generate_constraint_function_code(["x <= 1", "y == 0"], num_args)
    assert code.startswith(f"def constraint_function{signature}")
    assert "return (x <= 1) and" in code and "(y == 0)" in code
