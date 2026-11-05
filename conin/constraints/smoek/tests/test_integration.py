"""
Integration tests for the smoek algebraic modeling extension.

These tests verify end-to-end functionality by creating constrained models,
checking the MAP solutions against brute force and comparing results with the
traditional constraint syntax.
"""

import itertools
from types import SimpleNamespace

import pytest
from conin.util import try_import

with try_import() as smoek_available:
    import smoek  # noqa: F401

if not smoek_available:
    pytestmark = pytest.mark.skip(reason="Smoek not installed")


import pyomo.environ as pyo

from conin.config import default_mip_solver
from conin.examples.bayesian_network import cancer1_BN_conin
from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.algorithms.inference import map_query
from conin.algorithms.scoring import log_potential
from conin.exceptions import InvalidInputError
from conin import pyomo_constraint_fn, smoek_constraint_fn
from conin.util import try_import

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)
needs_mip = pytest.mark.skipif(default_mip_solver is None, reason="no MIP solver")


EVIDENCE = {"Pollution": 0, "Smoker": 1}


def V(states, node, value):
    """The value of the indicator variable V(node, value) in an assignment."""
    return int(states[node] == value)


# (smoek constraint expressions, the same constraints on an assignment, data).
# Without constraints the MAP solution is Cancer=1, Dyspnoea=1, Xray=1, so
# every case but the last excludes it.
CASES = {
    "addition": (
        lambda m, d: m.V("Dyspnoea", 1) + m.V("Xray", 1) <= 1,
        lambda s, d: V(s, "Dyspnoea", 1) + V(s, "Xray", 1) <= 1,
        None,
    ),
    "subtraction": (
        lambda m, d: m.V("Xray", 1) - m.V("Dyspnoea", 1) >= 1,
        lambda s, d: V(s, "Xray", 1) - V(s, "Dyspnoea", 1) >= 1,
        None,
    ),
    "multiplication_by_constant": (
        lambda m, d: 2 * m.V("Dyspnoea", 1) + m.V("Xray", 1) <= 2,
        lambda s, d: 2 * V(s, "Dyspnoea", 1) + V(s, "Xray", 1) <= 2,
        None,
    ),
    "complex_expression": (
        lambda m, d: 2 * m.V("Dyspnoea", 1) + 3 * m.V("Xray", 1) - m.V("Cancer", 0)
        <= 3,
        lambda s, d: 2 * V(s, "Dyspnoea", 1) + 3 * V(s, "Xray", 1) - V(s, "Cancer", 0)
        <= 3,
        None,
    ),
    "greater_than_or_equal": (
        lambda m, d: m.V("Dyspnoea", 0) + m.V("Xray", 0) >= 1,
        lambda s, d: V(s, "Dyspnoea", 0) + V(s, "Xray", 0) >= 1,
        None,
    ),
    "equality": (
        lambda m, d: m.V("Dyspnoea", 1) == 0,
        lambda s, d: V(s, "Dyspnoea", 1) == 0,
        None,
    ),
    "data": (
        lambda m, d: m.V("Dyspnoea", 1) + m.V("Xray", 1) <= d.limit,
        lambda s, d: V(s, "Dyspnoea", 1) + V(s, "Xray", 1) <= d.limit,
        SimpleNamespace(limit=0),
    ),
    "multiple_constraints": (
        lambda m, d: [
            m.V("Dyspnoea", 1) + m.V("Xray", 1) <= 1,
            m.V("Dyspnoea", 0) + m.V("Xray", 0) <= 1,
        ],
        lambda s, d: V(s, "Dyspnoea", 1) + V(s, "Xray", 1) <= 1
        and V(s, "Dyspnoea", 0) + V(s, "Xray", 0) <= 1,
        None,
    ),
    "non_binding": (
        lambda m, d: [
            m.V("Dyspnoea", 1) - m.V("Xray", 1) >= -1,
            m.V("Dyspnoea", 0) + m.V("Xray", 0) >= 0,
        ],
        lambda s, d: True,
        None,
    ),
}

METHODS = [
    pytest.param("integer_program", marks=needs_mip),
    pytest.param("toulbar2", marks=skipif_toulbar2_not_available),
    "variable_elimination",
]


def brute_force_map(pgm, accepts, data):
    free = [node for node in pgm.nodes if node not in EVIDENCE]
    best = None
    for values in itertools.product(*(pgm.states_of(node) for node in free)):
        states = dict(zip(free, values), **EVIDENCE)
        if accepts(states, data):
            score = log_potential(pgm, states)
            if best is None or score > best[0]:
                best = (score, states)
    return best


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("case", list(CASES))
def test_smoek_constraints_match_brute_force(case, method):
    expression, accepts, data = CASES[case]
    pgm = cancer1_BN_conin().pgm

    @smoek_constraint_fn()
    def constraint(model, data):
        return expression(model, data)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraint, data=data)
    result = map_query(cpgm, method=method, evidence=EVIDENCE)

    states = dict(result.solution.states, **EVIDENCE)
    assert accepts(states, data)
    score, expected = brute_force_map(pgm, accepts, data)
    assert states == expected
    assert result.solution.log_potential == pytest.approx(score)
    if case != "non_binding":
        unconstrained = map_query(pgm, method="variable_elimination", evidence=EVIDENCE)
        assert result.solution.states != unconstrained.solution.states


class TestBasicSmoekConstraints:
    """Test basic smoek constraint functionality."""

    @needs_mip
    def test_comparison_with_traditional_syntax(self):
        """Test that algebraic and traditional syntax produce same results."""
        pgm = cancer1_BN_conin().pgm

        # Traditional syntax
        @pyomo_constraint_fn()
        def traditional_constraint(model):
            model.c = pyo.ConstraintList()
            model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
            model.c.add(model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1)

        # Algebraic syntax
        @smoek_constraint_fn()
        def smoek_constraint(model, data):
            return [
                model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
                model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
            ]

        # Test both
        evidence = {"Pollution": 0, "Smoker": 1}

        cpgm_traditional = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=traditional_constraint
        )
        result_traditional = map_query(
            cpgm_traditional, method="integer_program", evidence=evidence
        )

        cpgm_smoek = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=smoek_constraint
        )
        result_smoek = map_query(
            cpgm_smoek, method="integer_program", evidence=evidence
        )

        # Results should be identical
        assert result_traditional.solution.states == result_smoek.solution.states

    def test_mixing_constraint_styles(self):
        """Test that smoek and traditional constraints cannot be mixed.

        A constrained model accepts constraints from one family only, so the
        constructor rejects the mix with an InvalidInputError.
        """
        pgm = cancer1_BN_conin().pgm

        @pyomo_constraint_fn()
        def traditional_constraint(model):
            model.c1 = pyo.ConstraintList()
            model.c1.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)

        @smoek_constraint_fn()
        def smoek_constraint(model, data):
            return model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1

        # Mix both styles
        with pytest.raises(InvalidInputError, match="one constraint family"):
            ConstrainedDiscreteBayesianNetwork(
                pgm, constraints=[traditional_constraint, smoek_constraint]
            )
