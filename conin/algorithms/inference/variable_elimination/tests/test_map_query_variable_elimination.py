"""map_query(method="variable_elimination") on every model type: the options
that only variable elimination has, and the HMM time range."""

import munch
import pytest

import conin.examples.bayesian_network as bex
import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.algorithms.feasibility import is_feasible
from conin.exceptions import InvalidInputError
from conin.algorithms.inference import map_query


def _ve(pgm, **kwds):
    return map_query(pgm, method="variable_elimination", **kwds)


def _states(results):
    return results.solution.states


# ---------------------------------------------------------------------
# Markov and Bayesian networks
# ---------------------------------------------------------------------


def test_mn_solution():
    example = mex.ABC_conin()
    results = _ve(example.pgm)
    assert _states(results) == example.solutions[0].states
    assert "solvetime" in results
    assert isinstance(results, munch.Munch)


def test_mn_partial_query_and_evidence_in_the_solution():
    results = _ve(
        mex.ABC_conin().pgm,
        variables=["A"],
        evidence={"B": 0},
        solution_with_evidence=True,
    )
    assert _states(results) == {"A": 2, "B": 0}


def test_bn_solution():
    example = bex.cancer1_BN_constrained_oracle_conin()
    assert _states(_ve(example.pgm)) == example.solutions[0].states


def test_unknown_options_are_ignored():
    example = mex.ABC_conin()
    assert _states(_ve(example.pgm, solver="glpk")) == example.solutions[0].states


def test_mn_writes_the_factor_model(tmp_path):
    filename = tmp_path / "model.uai"
    _ve(mex.ABC_constrained_oracle_conin().pgm, write_uai_file=str(filename))
    assert filename.read_text().startswith("MARKOV")


# ---------------------------------------------------------------------
# Dynamic Bayesian networks
# ---------------------------------------------------------------------


def test_dbn_stop_none_means_one():
    example = dex.simple1_DDBN_constrained_oracle_conin()
    results = _ve(example.pgm, stop=None)
    assert _states(results) == example.solutions[0].states
    assert _states(results) == _states(_ve(example.pgm, stop=1))


# ---------------------------------------------------------------------
# HMMs
# ---------------------------------------------------------------------

OBSERVED = ["o0", "o1", "o1", "o0", "o1"]


def test_hmm_shapes_the_hidden_states():
    hmm = hex_.create_hmm1()
    as_list = _ve(hmm, evidence=OBSERVED)
    as_dict = _ve(hmm, evidence=dict(enumerate(OBSERVED)))
    assert isinstance(_states(as_list), list) and len(_states(as_list)) == 5
    assert _states(as_dict) == dict(enumerate(_states(as_list)))


def test_hmm_start_and_stop():
    results = _ve(hex_.create_hmm1(), evidence=OBSERVED, start=0, stop=2)
    assert len(_states(results)) == 3


@pytest.mark.parametrize(
    "evidence,match",
    [(None, "must be a list, a tuple or a dict"), (OBSERVED, "stop")],
)
def test_hmm_bad_observations(evidence, match):
    kwargs = dict(stop=10) if evidence is not None else {}
    with pytest.raises(InvalidInputError, match=match):
        _ve(hex_.create_hmm1(), evidence=evidence, **kwargs)


def test_hmm_timing(capsys):
    _ve(hex_.create_hmm1(), evidence=OBSERVED, timing=True)
    out = capsys.readouterr().out
    assert "map_query - START" in out and "map_query - STOP" in out


# ---------------------------------------------------------------------
# Constraints over one node
# ---------------------------------------------------------------------
#
# The materialised factor of a constraint over one node is keyed by the
# node's states, not by 1-tuples (with pgmpy, 1-tuples raised KeyError for
# every such constraint, feasible or not).


def _one_node_constraints():
    from conin.constraints import algebraic_constraint_fn, oracle_constraint_fn

    @algebraic_constraint_fn()
    def a_is_0(model):
        return [model.V("A", 0) == 1]

    @oracle_constraint_fn(nodes=["A"])
    def a_is_not_2(assignment):
        return assignment["A"] != 2

    return [a_is_0, a_is_not_2]


@pytest.mark.parametrize("index", [0, 1], ids=["algebraic", "factor_oracle"])
def test_mn_constraint_over_one_node(index):
    from conin.constrained_models import ConstrainedDiscreteMarkovNetwork

    constraint = _one_node_constraints()[index]
    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm, constraint)
    states = map_query(cmodel, method="variable_elimination").solution.states
    assert is_feasible(cmodel, {k: states[k] for k in ("A", "B", "C")})
    # The unconstrained MAP state of A is 2
    assert mex.ABC_conin().solutions[0].states["A"] == 2 and states["A"] != 2


def _hmm_never_x_at_time_1():
    from conin.constrained_models import ConstrainedHiddenMarkovModel
    from conin.constraints import algebraic_constraint_fn
    from conin.graphical_models import HiddenMarkovModel

    hmm = HiddenMarkovModel(
        start_probs={"x": 0.6, "y": 0.4},
        transition_probs={
            ("x", "x"): 0.7,
            ("x", "y"): 0.3,
            ("y", "x"): 0.4,
            ("y", "y"): 0.6,
        },
        emission_probs={
            ("x", "a"): 0.8,
            ("x", "b"): 0.2,
            ("y", "a"): 0.3,
            ("y", "b"): 0.7,
        },
    )

    @algebraic_constraint_fn()
    def never_x_at_time_1(model, data):
        return [model.V("H", 1, "x") == 0]

    return ConstrainedHiddenMarkovModel(hmm, never_x_at_time_1)


def test_hmm_constraint_over_one_node():
    cmodel = _hmm_never_x_at_time_1()
    results = map_query(cmodel, method="variable_elimination", evidence=["a", "a", "a"])
    assert results.solution.states[1] == "y"
    # The same MAP assignment as the unconstrained Viterbi path with y at time 1
    unconstrained = map_query(cmodel.model, method="viterbi", evidence=["a"] * 3)
    assert unconstrained.solution.states == ["x", "x", "x"]
    assert results.solution.states == ["x", "y", "x"]
