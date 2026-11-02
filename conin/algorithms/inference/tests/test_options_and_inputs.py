"""Every method rejects the options it does not take, the HMM functions reject
empty observations, and every backend reads the ``default_value`` of a sparse
factor."""

import pytest

import conin.examples.bayesian_network as bex
import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.config import default_mip_solver
from conin.exceptions import InvalidInputError
from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
from conin.algorithms.inference import map_query
from conin.algorithms.inference.oracle import a_star
from conin.algorithms.inference.pyomo import ip_inference, lp_inference
from conin.algorithms.inference.recursive import recursive_a_star
from conin.algorithms.inference.unconstrained import viterbi
from conin.algorithms.scoring import log_probability_query
from conin.util import try_import

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

with try_import() as torch_available:
    import torch  # noqa: F401

needs_mip_solver = pytest.mark.skipif(
    default_mip_solver is None, reason="No MIP solver installed"
)
needs_toulbar2 = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)
needs_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")

OBSERVED = ["o0", "o0", "o1", "o0", "o0"]
MIP = dict(solver=default_mip_solver)

# ---------------------------------------------------------------------
# Unknown options
# ---------------------------------------------------------------------


def _mn():
    return mex.ABC_conin().pgm


def _bn():
    return bex.cancer1_BN_conin().pgm


def _dbn():
    return dex.simple1_DDBN_conin().pgm


@pytest.mark.parametrize(
    "method,make,kwds",
    [
        pytest.param("integer_program", _mn, MIP, marks=needs_mip_solver, id="ip-mn"),
        pytest.param(
            "integer_program",
            hex_.create_hmm1,
            dict(ip_formulation="network_flow", evidence=OBSERVED, **MIP),
            marks=needs_mip_solver,
            id="ip-network-flow",
        ),
        pytest.param(
            "integer_program",
            hex_.create_chmm1_pyomo,
            dict(ip_formulation="network_flow", evidence=["o0"] * 12, **MIP),
            marks=needs_mip_solver,
            id="ip-network-flow-constrained",
        ),
        pytest.param("toulbar2", _mn, {}, marks=needs_toulbar2, id="toulbar2-mn"),
        pytest.param("variable_elimination", _mn, {}, id="ve-mn"),
        pytest.param("variable_elimination", _bn, {}, id="ve-bn"),
        pytest.param("variable_elimination", _dbn, dict(stop=1), id="ve-dbn"),
        pytest.param(
            "variable_elimination",
            hex_.create_hmm1,
            dict(evidence=OBSERVED),
            id="ve-hmm",
        ),
        pytest.param("a_star", hex_.create_hmm1, dict(evidence=OBSERVED), id="a-star"),
        pytest.param(
            "a_star",
            hex_.create_chmm1_oracle,
            dict(evidence=["o0"] * 12),
            id="a-star-oracle",
        ),
        pytest.param(
            "a_star",
            hex_.create_chmm1_recursive,
            dict(evidence=["o0"] * 12),
            id="a-star-recursive",
        ),
        pytest.param(
            "viterbi", hex_.create_hmm1, dict(evidence=OBSERVED), id="viterbi"
        ),
        pytest.param(
            "mvr_viterbi",
            hex_.create_hmm1,
            dict(evidence=OBSERVED),
            marks=needs_torch,
            id="mvr-viterbi",
        ),
    ],
)
def test_unknown_option_raises_type_error(method, make, kwds):
    # variable_elimination and viterbi used to ignore unknown options.
    with pytest.raises(TypeError, match="typo_option"):
        map_query(make(), method=method, typo_option=1, **kwds)


def test_log_probability_query_unknown_option_raises_type_error():
    with pytest.raises(TypeError, match="typo_option"):
        log_probability_query(
            _bn(), {"Cancer": 1}, method="variable_elimination", typo_option=1
        )


@pytest.mark.parametrize("option", ["solution_with_evidence", "solution_with_fixed"])
def test_variable_elimination_solution_with_fixed_is_a_synonym(option):
    mn = _mn()
    plain = map_query(mn, method="variable_elimination", evidence={"A": 0})
    assert "A" not in plain.solution.states
    results = map_query(
        mn, method="variable_elimination", evidence={"A": 0}, **{option: True}
    )
    assert results.solution.states == {**plain.solution.states, "A": 0}


# ---------------------------------------------------------------------
# Empty observations
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "search,make",
    [
        (viterbi, hex_.create_hmm1),
        (a_star, hex_.create_hmm1),
        (recursive_a_star, hex_.create_chmm1_recursive),
        (lp_inference, hex_.create_hmm1),
        (ip_inference, hex_.create_chmm1_pyomo),
    ],
    ids=["viterbi", "a_star", "recursive_a_star", "lp_inference", "ip_inference"],
)
def test_hmm_functions_reject_empty_observations(search, make):
    # viterbi, a_star and recursive_a_star raised IndexError.
    with pytest.raises(InvalidInputError, match="must not be empty"):
        search(hmm=make(), observed=[])


# ---------------------------------------------------------------------
# Sparse factors with a default_value
# ---------------------------------------------------------------------


def _sparse_mn():
    """A Markov network whose factors list only some assignments.

    The missing assignments have the weight ``default_value``, so the MAP
    assignment ``A=1, B=0`` (weight ``3 * 5``) uses defaults of both factors.
    """
    return DiscreteMarkovNetwork(
        states={"A": [0, 1, 2], "B": [0, 1, 2]},
        factors=[
            DiscreteFactor(nodes=["A"], values={0: 1}, default_value=3),
            DiscreteFactor(nodes=["A", "B"], values={(1, 0): 5}, default_value=1),
        ],
    )


@pytest.mark.parametrize(
    "method,kwds",
    [
        pytest.param("integer_program", MIP, marks=needs_mip_solver),
        pytest.param("toulbar2", {}, marks=needs_toulbar2),
        pytest.param("variable_elimination", {}),
    ],
)
def test_sparse_factors_use_their_default_value(method, kwds):
    # The integer program dropped the missing assignments (no solution), and
    # toulbar2 raised KeyError while writing its UAI file.
    results = map_query(_sparse_mn(), method=method, **kwds)
    assert results.solution.states == {"A": 1, "B": 0}
