import pytest

from conin.exceptions import InvalidInputError
from conin.constraints import (
    Constraint,
    PyomoConstraint,
    Toulbar2Constraint,
    OracleConstraint,
    FactorOracleConstraint,
    MVRConstraint,
    AlgebraicConstraint,
    oracle_constraint_fn,
)


@pytest.mark.parametrize(
    "cls, family",
    [
        (PyomoConstraint, "pyomo"),
        (Toulbar2Constraint, "toulbar2"),
        (OracleConstraint, "oracle"),
        (FactorOracleConstraint, "factor_oracle"),
        (MVRConstraint, "mvr"),
        (AlgebraicConstraint, "algebraic"),
    ],
)
def test_family(cls, family):
    assert cls.family == family
    assert issubclass(cls, Constraint)


def test_base_family_is_none():
    assert Constraint.family is None


def test_instance_family():
    def f(model, data):
        return model

    assert PyomoConstraint(f).family == "pyomo"
    assert Toulbar2Constraint(f).family == "toulbar2"
    assert MVRConstraint(f).family == "mvr"
    # Constructing an AlgebraicConstraint does not need smoek
    assert AlgebraicConstraint(f).family == "algebraic"


def test_oracle_constraint_fn_without_nodes():
    @oracle_constraint_fn()
    def c(states):
        return True

    assert type(c) is OracleConstraint
    assert not isinstance(c, FactorOracleConstraint)
    assert not hasattr(c, "nodes")


def test_oracle_constraint_fn_with_node_list():
    @oracle_constraint_fn(nodes=["A", "B"], name="AB")
    def c(states):
        return True

    assert type(c) is FactorOracleConstraint
    assert not isinstance(c, OracleConstraint)
    assert c.nodes == ["A", "B"]
    assert c.name == "AB"


def test_oracle_constraint_fn_with_node_callable():
    def nodes(data):
        yield from data

    @oracle_constraint_fn(nodes=nodes)
    def c(states, data):
        return True

    assert type(c) is FactorOracleConstraint
    assert c.nodes is nodes
    assert c.name == "c"
    assert c.num_args == 2


def test_oracle_constraint_fn_nodes_and_partial_raises():
    with pytest.raises(InvalidInputError):
        oracle_constraint_fn(nodes=["A"], same_partial_as_func=True)


def test_factor_oracle_call_one_arg():
    c = FactorOracleConstraint(func=lambda s: s["A"] == 1, nodes=["A"])
    assert c({"A": 1}) is True
    assert c({"A": 0}) is False


def test_factor_oracle_call_two_args():
    c = FactorOracleConstraint(func=lambda s, data: s["A"] == data, nodes=["A"])
    assert c({"A": 2}, 2) is True
    assert c({"A": 2}, data=3) is False


def test_factor_oracle_requires_nodes():
    with pytest.raises(InvalidInputError):
        FactorOracleConstraint(func=lambda s: True, nodes=None)


def test_oracle_has_no_nodes():
    c = OracleConstraint(func=lambda s: True)
    assert not hasattr(c, "nodes")
    assert c({"A": 0}) is True
