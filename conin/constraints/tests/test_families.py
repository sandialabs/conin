import pytest

from conin.exceptions import InvalidInputError
from conin.constraints import (
    Constraint,
    PyomoConstraint,
    Toulbar2Constraint,
    OracleConstraint,
    FactorOracleConstraint,
    MVRConstraint,
    RecursiveConstraint,
    SmoekConstraint,
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
        (SmoekConstraint, "smoek"),
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
    # Constructing an SmoekConstraint does not need smoek
    assert SmoekConstraint(f).family == "smoek"


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


@pytest.mark.parametrize(
    "cls,family",
    [
        (PyomoConstraint, "Pyomo"),
        (Toulbar2Constraint, "Toulbar2"),
        (SmoekConstraint, "Smoek"),
    ],
)
def test_more_than_two_arguments_rejected(cls, family):
    def three(model, data, extra):
        return model

    with pytest.raises(
        InvalidInputError, match=f"{family} constraint defined with more than 2"
    ):
        cls(three)

    def two(model, data):
        return model

    assert cls(two).num_args == 2


def _identity(*args):
    return None


@pytest.mark.parametrize("argument", ["initial", "update", "feasible"])
def test_recursive_constraint_functions_must_be_callable(argument):
    kwargs = dict(initial=_identity, update=_identity, feasible=_identity)
    kwargs[argument] = 3
    with pytest.raises(
        InvalidInputError,
        match=f"RecursiveConstraint: {argument} must be callable, got int",
    ):
        RecursiveConstraint(**kwargs)


def test_recursive_constraint_feasible_partial_must_be_callable():
    with pytest.raises(
        InvalidInputError, match="feasible_partial must be callable, got str"
    ):
        RecursiveConstraint(
            initial=_identity,
            update=_identity,
            feasible=_identity,
            feasible_partial="x",
        )
    assert (
        RecursiveConstraint(
            initial=_identity, update=_identity, feasible=_identity
        ).family
        == "recursive"
    )
