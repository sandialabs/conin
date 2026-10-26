"""The MAP result shape (conin.inference.results) is the same for every method.

Every method returns ``solution``, ``solutions``, ``termination_condition``
and ``solvetime``, and every solution has ``states``, ``log_potential`` and
``solver_objective``.  ``log_potential`` is the score of the assignment with
the evidence (:func:`conin.scoring.log_potential` on the model without its
constraints), so every method gives the same value for the same assignment.
Before, the HMM methods returned ``log_likelihood``, the integer program
``log_factor_sum`` (its objective: -10.21 instead of -3.97 for the same HMM
path) and toulbar2 ``primal_bound``.
"""

import math

import munch
import pytest

import conin.examples.bayesian_network as bex
import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.config import default_mip_solver
from conin.inference import map_query
from conin.inference.oracle import a_star
from conin.inference.pyomo import ip_inference, lp_inference
from conin.inference.recursive import recursive_a_star
from conin.inference.results import (
    map_results,
    map_solution,
    score_hmm_solutions,
    score_solutions,
)
from conin.inference.unconstrained import viterbi
from conin.scoring import log_potential
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

RESULT_KEYS = {"solution", "solutions", "termination_condition", "solvetime"}
SOLUTION_KEYS = {"states", "log_potential", "solver_objective"}
OBSERVED = ["o0", "o0", "o1", "o0", "o0"]

needs_mip = pytest.mark.skipif(default_mip_solver is None, reason="no MIP solver")
needs_toulbar2 = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)
needs_pgmpy = pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")

# (method, options, marks) of the solver methods
SOLVERS = [
    pytest.param("integer_program", dict(solver=default_mip_solver), marks=needs_mip),
    pytest.param("toulbar2", {}, marks=needs_toulbar2),
    pytest.param("variable_elimination", {}, marks=needs_pgmpy),
]


def _check_shape(results, extra_solution_keys=()):
    assert set(results) == RESULT_KEYS
    assert results.solution is results.solutions[0]
    assert isinstance(results.solvetime, float) and results.solvetime >= 0
    for soln in results.solutions:
        assert set(soln) == SOLUTION_KEYS | set(extra_solution_keys)


# ---------------------------------------------------------------------
# HMMs: every method, the same keys and the same log_potential
# ---------------------------------------------------------------------

HMM_METHODS = [
    pytest.param("viterbi", {}, None, id="viterbi"),
    pytest.param("a_star", {}, None, id="a_star"),
    pytest.param(
        "integer_program",
        dict(solver=default_mip_solver),
        "float",
        marks=needs_mip,
        id="integer_program",
    ),
    pytest.param(
        "integer_program",
        dict(solver=default_mip_solver, ip_formulation="network_flow"),
        "float",
        marks=needs_mip,
        id="network_flow",
    ),
    pytest.param("toulbar2", {}, "float", marks=needs_toulbar2, id="toulbar2"),
    pytest.param(
        "variable_elimination", {}, None, marks=needs_pgmpy, id="variable_elimination"
    ),
]


@pytest.mark.parametrize("method,options,objective", HMM_METHODS)
@pytest.mark.parametrize("dict_evidence", [False, True])
def test_hmm_results_have_the_same_shape_and_score(
    method, options, objective, dict_evidence
):
    hmm = hex_.create_hmm1()
    evidence = dict(enumerate(OBSERVED)) if dict_evidence else OBSERVED
    results = map_query(hmm, method=method, evidence=evidence, **options)
    # (The network-flow integer program adds "variables" only with
    # constraints, where it solves the IP instead of the LP.)
    _check_shape(results)
    assert results.termination_condition == "ok"
    soln = results.solution
    expected = ["h0"] * 5
    assert soln.states == (dict(enumerate(expected)) if dict_evidence else expected)
    # log P(hidden, observed), whatever the method
    assert soln.log_potential == pytest.approx(hmm.log_probability(OBSERVED, expected))
    assert soln.log_potential == pytest.approx(-3.968405374586326)
    if objective is None:
        assert soln.solver_objective is None
    else:
        assert isinstance(soln.solver_objective, float)
    assert "hidden" not in soln and "log_likelihood" not in soln


@needs_mip
def test_hmm_integer_program_score_is_not_its_objective():
    # The integer program's objective normalises each factor of the unrolled
    # DBN, so it is not log P(hidden, observed).
    results = map_query(
        hex_.create_hmm1(),
        method="integer_program",
        evidence=OBSERVED,
        solver=default_mip_solver,
    )
    assert results.solution.log_potential == pytest.approx(-3.968405374586326)
    assert results.solution.solver_objective == pytest.approx(-10.206729999625834)


@pytest.mark.parametrize(
    "method,options",
    [
        pytest.param(
            "integer_program", dict(solver=default_mip_solver), marks=needs_mip
        ),
        pytest.param("toulbar2", {}, marks=needs_toulbar2),
    ],
)
def test_hmm_score_covers_start_stop(method, options):
    # The solution covers start..stop; it is scored over 0..stop.
    hmm = hex_.create_hmm1()
    results = map_query(hmm, method=method, evidence=OBSERVED, stop=2, **options)
    assert results.solution.states == ["h0"] * 3
    assert results.solution.log_potential == pytest.approx(
        hmm.log_probability(OBSERVED[:3], ["h0"] * 3)
    )


def test_constrained_hmm_score_is_not_renormalised():
    # Recursive and oracle A* on equivalent constraints: the same path and
    # the same unconstrained score log P(hidden, observed).
    observed = ["o0"] * 15
    rec = map_query(hex_.create_chmm1_recursive(), method="a_star", evidence=observed)
    ora = map_query(hex_.create_chmm1_oracle(), method="a_star", evidence=observed)
    _check_shape(rec)
    _check_shape(ora)
    path = ["h1"] * 3 + ["h0"] * 12
    assert rec.solution.states == ora.solution.states == path
    hmm = hex_.create_hmm1()
    assert rec.solution.log_potential == pytest.approx(
        log_potential(hmm, path, observed=observed)
    )
    assert ora.solution.log_potential == pytest.approx(rec.solution.log_potential)


@needs_mip
def test_constrained_hmm_integer_program_and_network_flow_agree():
    observed = ["o0"] * 15
    cmodel = hex_.create_chmm1_pyomo()
    ip = map_query(
        cmodel, method="integer_program", evidence=observed, solver=default_mip_solver
    )
    nf = map_query(
        cmodel,
        method="integer_program",
        ip_formulation="network_flow",
        evidence=observed,
        solver=default_mip_solver,
    )
    assert ip.solution.states == nf.solution.states
    assert ip.solution.log_potential == pytest.approx(nf.solution.log_potential)
    assert ip.solution.log_potential == pytest.approx(
        log_potential(cmodel.model, ip.solution.states, observed=observed)
    )


def test_a_star_k_best_scores_are_sorted():
    results = map_query(
        hex_.create_hmm1(), method="a_star", evidence=OBSERVED, num_solutions=4
    )
    _check_shape(results)
    scores = [s.log_potential for s in results.solutions]
    assert scores == sorted(scores, reverse=True)
    hmm = hex_.create_hmm1()
    for s in results.solutions:
        assert s.log_potential == pytest.approx(hmm.log_probability(OBSERVED, s.states))


def test_no_solution_has_the_same_keys():
    results = map_query(
        hex_.create_chmm1_recursive(), method="a_star", evidence=["o0"] * 5
    )
    assert set(results) == RESULT_KEYS
    assert results.solution is None and results.solutions == []


# ---------------------------------------------------------------------
# The HMM functions called directly have the same shape
# ---------------------------------------------------------------------


def test_hmm_functions_have_the_same_shape():
    hmm = hex_.create_hmm1()
    expected = hmm.log_probability(OBSERVED, ["h0"] * 5)
    for results in [
        viterbi(observed=OBSERVED, hmm=hmm),
        a_star(observed=OBSERVED, hmm=hmm),
        recursive_a_star(hmm=hmm, observed=OBSERVED),
    ]:
        _check_shape(results)
        assert results.solution.states == ["h0"] * 5
        assert results.solution.log_potential == pytest.approx(expected)


@needs_mip
def test_network_flow_functions_have_the_same_shape():
    hmm = hex_.create_hmm1()
    expected = hmm.log_probability(OBSERVED, ["h0"] * 5)
    lp = lp_inference(hmm=hmm, observed=OBSERVED, solver=default_mip_solver)
    _check_shape(lp)
    ip = ip_inference(
        hmm=hex_.create_chmm1_pyomo(), observed=["o0"] * 15, solver=default_mip_solver
    )
    _check_shape(ip, ("variables",))
    assert lp.solution.log_potential == pytest.approx(expected)
    # Without constraints the LP objective is log P(hidden, observed).
    assert lp.solution.solver_objective == pytest.approx(expected)


# ---------------------------------------------------------------------
# MN, BN and DBN: the solver methods agree on log_potential
# ---------------------------------------------------------------------


def _static_cases():
    return {
        "mn": (mex.ABC_conin().pgm, {}),
        "mn_evidence": (mex.ABC_conin().pgm, dict(evidence={"A": 0})),
        "bn": (bex.cancer1_BN_conin().pgm, {}),
        "bn_evidence": (bex.cancer1_BN_conin().pgm, dict(evidence={"Cancer": 1})),
        "dbn": (dex.simple1_DDBN_conin().pgm, dict(stop=2)),
    }


@pytest.mark.parametrize("method,options", SOLVERS)
@pytest.mark.parametrize("case", list(_static_cases()))
def test_static_results_are_scored_with_the_evidence(method, options, case):
    pgm, kwds = _static_cases()[case]
    if method == "variable_elimination" and case == "dbn":
        pytest.skip("covered by the HMM case (VE on DBNs is slow here)")
    results = map_query(pgm, method=method, **kwds, **options)
    _check_shape(results)
    soln = results.solution
    # log_potential is the score of the assignment with the evidence; for a
    # Markov network the sum of the log factors.
    assignment = dict(kwds.get("evidence", {}))
    assignment.update(soln.states)
    if case == "dbn":
        unrolled = {k: v for k, v in assignment.items() if k[1] <= 2}
        assert soln.log_potential == pytest.approx(log_potential(pgm, unrolled))
    else:
        assert soln.log_potential == pytest.approx(log_potential(pgm, assignment))
    if method == "variable_elimination":
        assert soln.solver_objective is None
    else:
        assert isinstance(soln.solver_objective, float)


@pytest.mark.parametrize("case", list(_static_cases()))
def test_solver_methods_agree_on_log_potential(case):
    pgm, kwds = _static_cases()[case]
    scores = {}
    if default_mip_solver is not None:
        scores["ip"] = map_query(
            pgm, method="integer_program", solver=default_mip_solver, **kwds
        ).solution.log_potential
    if pytoulbar2_available:
        scores["tb2"] = map_query(pgm, method="toulbar2", **kwds).solution.log_potential
    if len(scores) < 2:
        pytest.skip("needs a MIP solver and pytoulbar2")
    assert scores["ip"] == pytest.approx(scores["tb2"])


@pytest.mark.parametrize("method,options", SOLVERS)
def test_constrained_static_score_is_not_renormalised(method, options):
    # A constrained BN: log_potential is the unconstrained log P(x) of the
    # (feasible) MAP assignment, the same as log_potential of the
    # constrained model (which is -inf only for an infeasible assignment).
    if method == "variable_elimination":
        example = bex.cancer1_BN_constrained_oracle_conin()
    elif method == "toulbar2":
        example = bex.cancer1_BN_constrained_toulbar2_conin()
    else:
        example = bex.cancer1_BN_constrained_pyomo_conin()
    cmodel = example.pgm
    results = map_query(cmodel, method=method, **options)
    states = results.solution.states
    assert results.solution.log_potential == pytest.approx(
        log_potential(cmodel.model, states)
    )


@needs_pgmpy
def test_partial_map_query_has_no_log_potential():
    pgm = bex.cancer1_BN_conin().pgm
    results = map_query(pgm, method="variable_elimination", variables=["Cancer"])
    _check_shape(results)
    assert set(results.solution.states) == {"Cancer"}
    assert results.solution.log_potential is None


def test_toulbar2_without_pytoulbar2_raises_import_error(monkeypatch):
    import conin.inference.toulbar2.solve as mod

    monkeypatch.setattr(mod, "pytoulbar2_available", False)
    with pytest.raises(ImportError, match="needs pytoulbar2"):
        map_query(mex.ABC_conin().pgm, method="toulbar2")


# ---------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------


def test_map_results_and_map_solution():
    soln = map_solution({"A": 1}, solver_objective=2.0, variables={})
    assert soln == munch.Munch(
        states={"A": 1}, log_potential=None, solver_objective=2.0, variables={}
    )
    results = map_results([soln], solvetime=0.5)
    assert results == munch.Munch(
        solution=soln, solutions=[soln], termination_condition="ok", solvetime=0.5
    )
    empty = map_results([], solvetime=0.0, termination_condition="max_time: 1")
    assert empty.solution is None and empty.solutions == []


def test_score_solutions_restricts_to_the_nodes_and_adds_the_evidence():
    pgm = mex.ABC_conin().pgm
    results = map_results(
        [map_solution({"B": 2, "C": 1}), map_solution({"B": 2})], solvetime=0.0
    )
    score_solutions(results, pgm, evidence={"A": 2, "Z": 0}, nodes=("A", "B", "C"))
    assert results.solutions[0].log_potential == pytest.approx(
        log_potential(pgm, {"A": 2, "B": 2, "C": 1})
    )
    assert results.solutions[1].log_potential is None


def test_score_hmm_solutions_sparse_observations():
    # A {time: label} dict: an unobserved time contributes its transition only.
    hmm = hex_.create_hmm1()
    hidden = ["h0", "h1", "h1"]
    sparse = map_results([map_solution(hidden)], solvetime=0.0)
    score_hmm_solutions(sparse, hmm, {0: "o0", 2: "o1"})
    dense = map_results([map_solution(hidden)], solvetime=0.0)
    score_hmm_solutions(dense, hmm, ["o0", "o1", "o1"])
    full = hmm.log_probability(["o0", "o1", "o1"], hidden)
    h1, o1 = hmm.hidden_to_internal["h1"], hmm.observed_to_internal["o1"]
    emission_at_1 = math.log(hmm.emission_mat[h1][o1])
    assert dense.solution.log_potential == pytest.approx(full)
    assert sparse.solution.log_potential == pytest.approx(full - emission_at_1)
