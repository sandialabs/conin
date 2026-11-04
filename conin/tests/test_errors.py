"""The error policy of ``conin.exceptions`` at the call sites that used to
raise a bare ``ValueError`` or ``RuntimeError``."""

import pytest

from conin import ConstrainedDiscreteMarkovNetwork
from conin.algorithms.learning import supervised_learning
from conin.constraints import MVRConstraint, PyomoConstraint, Toulbar2Constraint
from conin.constraints.smoek.decorators import ConinV, SmoekConstraint
from conin.examples import markov_network
from conin.exceptions import InvalidInputError
from conin.representations.toulbar2.common import Toulbar2VarWrapper


def _three_arguments(model, data, extra):
    return model


@pytest.mark.parametrize(
    "cls", [PyomoConstraint, Toulbar2Constraint, SmoekConstraint, MVRConstraint]
)
def test_constraint_functions_take_at_most_two_arguments(cls):
    with pytest.raises(InvalidInputError, match="more than 2 arguments"):
        cls(_three_arguments)


@pytest.mark.parametrize(
    "V",
    [
        lambda: ConinV(),
        lambda: Toulbar2VarWrapper(markov_network.ABC_conin().pgm),
    ],
    ids=["smoek", "toulbar2"],
)
def test_V_takes_two_or_three_arguments(V):
    with pytest.raises(TypeError, match="2 or 3 arguments, got 1"):
        V()("A")


def test_a_constrained_model_needs_a_model_of_its_type():
    with pytest.raises(TypeError, match="expects a DiscreteMarkovNetwork"):
        ConstrainedDiscreteMarkovNetwork("not a model")


def test_supervised_learning_without_data_or_start_tolerance():
    with pytest.raises(InvalidInputError, match="no simulations"):
        supervised_learning(
            simulations=[],
            hidden_states=["h0"],
            observable_states=["o0"],
            start_tolerance=0,
        )
