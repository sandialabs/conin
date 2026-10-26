"""Tests of the map_query dispatch: normalisation, method -> representation kind,
one solver per representation class, and the errors."""

import importlib

import munch
import pytest

import conin.examples.bayesian_network as bex
import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.config import default_mip_solver
from conin.constrained_models import (
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constrained_models.normalise import as_constrained_model
from conin.constraints import OracleConstraint
from conin.exceptions import InvalidInputError
from conin.graphical_models.hmm_observations import as_sequence
from conin.inference import map_query
from conin.inference.oracle import a_star
from conin.inference.recursive import recursive_a_star
from conin.inference.unconstrained import viterbi
from conin.representations.pgmpy import (
    PgmpyBNRepresentation,
    PgmpyDBNRepresentation,
    PgmpyHMMRepresentation,
    PgmpyMNRepresentation,
    PgmpyRepresentation,
)
from conin.representations.matvec import MatVecHMMRepresentation
from conin.representations.mvr import MVRHMMRepresentation
from conin.representations.oracle import OracleHMMRepresentation
from conin.representations.pyomo import (
    PyomoBNRepresentation,
    PyomoFactorRepresentation,
    PyomoDBNRepresentation,
    PyomoHMMRepresentation,
    PyomoMNRepresentation,
    PyomoNetworkFlowHMMRepresentation,
)
from conin.representations.recursive import RecursiveHMMRepresentation
from conin.representations.toulbar2 import (
    Toulbar2BNRepresentation,
    Toulbar2FactorRepresentation,
    Toulbar2DBNRepresentation,
    Toulbar2HMMRepresentation,
    Toulbar2MNRepresentation,
)
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pytoulbar2_available:
    import pytoulbar2

# conin.inference.map_query is also the name of the function the package exports.
mq = importlib.import_module("conin.inference.map_query")

skipif_no_mip_solver = pytest.mark.skipif(
    default_mip_solver is None, reason="No MIP solver installed"
)
skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)

OBSERVED = ["o0", "o0", "o1", "o0", "o0"]


def _cmodel(kind):
    return {
        "mn": lambda: ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm),
        "bn": lambda: ConstrainedDiscreteBayesianNetwork(bex.cancer1_BN_conin().pgm),
        "dbn": lambda: ConstrainedDynamicDiscreteBayesianNetwork(
            dex.simple1_DDBN_conin().pgm
        ),
        "hmm": lambda: ConstrainedHiddenMarkovModel(hex_.create_hmm1()),
    }[kind]()


# ---------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,cls",
    [
        (lambda: mex.ABC_conin().pgm, ConstrainedDiscreteMarkovNetwork),
        (lambda: bex.cancer1_BN_conin().pgm, ConstrainedDiscreteBayesianNetwork),
        (
            lambda: dex.simple1_DDBN_conin().pgm,
            ConstrainedDynamicDiscreteBayesianNetwork,
        ),
        (hex_.create_hmm1, ConstrainedHiddenMarkovModel),
    ],
    ids=["mn", "bn", "dbn", "hmm"],
)
def test_bare_models_are_wrapped_with_no_constraints(model, cls):
    cmodel = as_constrained_model(model())
    assert type(cmodel) is cls
    assert cmodel.constraints == ()


def test_constrained_models_are_used_as_they_are():
    cmodel = hex_.create_chmm1_oracle()
    assert as_constrained_model(cmodel) is cmodel


@skipif_pgmpy_not_available
@pytest.mark.parametrize(
    "example,cls",
    [
        (mex.ABC_pgmpy, ConstrainedDiscreteMarkovNetwork),
        (bex.cancer1_BN_pgmpy, ConstrainedDiscreteBayesianNetwork),
        (dex.simple1_DDBN_pgmpy, ConstrainedDynamicDiscreteBayesianNetwork),
    ],
    ids=["mn", "bn", "dbn"],
)
def test_pgmpy_models_are_converted(example, cls):
    pgm = example().pgm
    cmodel = as_constrained_model(getattr(pgm, "pgm", pgm))
    assert type(cmodel) is cls
    assert cmodel.constraints == ()


@pytest.mark.parametrize("model", ["not_a_model", 3, object()])
@pytest.mark.parametrize("method", list(mq.METHODS))
def test_unsupported_model_type_raises_type_error(model, method):
    with pytest.raises(TypeError, match="Unsupported model type"):
        map_query(model, method=method, evidence=["o0"])


def test_unknown_method_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported inference type: bogus"):
        map_query(hex_.create_hmm1(), method="bogus", evidence=OBSERVED)


# ---------------------------------------------------------------------
# method -> representation kind
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,options,kind",
    [
        ("integer_program", {}, "pyomo"),
        ("integer_program", {"ip_formulation": None}, "pyomo"),
        ("integer_program", {"ip_formulation": "markov_network"}, "pyomo"),
        ("integer_program", {"ip_formulation": "network_flow"}, "pyomo_network_flow"),
        ("toulbar2", {}, "toulbar2"),
        ("variable_elimination", {}, "pgmpy"),
        ("a_star", {}, "oracle"),
        ("viterbi", {}, "matvec"),
        ("mvr_viterbi", {}, "mvr"),
    ],
)
def test_method_to_representation_kind(method, options, kind):
    options = dict(options)
    assert mq._representation_kind(method, _cmodel("hmm"), options) == kind
    # ip_formulation is consumed: it never reaches the solver.
    assert "ip_formulation" not in options


@pytest.mark.parametrize(
    "cmodel,kind",
    [
        (hex_.create_chmm1_oracle, "oracle"),
        (lambda: ConstrainedHiddenMarkovModel(hex_.create_hmm1()), "oracle"),
        (hex_.create_chmm1_recursive, "recursive"),
    ],
    ids=["oracle", "none", "recursive"],
)
def test_a_star_kind_depends_on_the_constraint_family(cmodel, kind):
    assert mq._representation_kind("a_star", cmodel(), {}) == kind


def test_unknown_ip_formulation_raises_value_error():
    with pytest.raises(ValueError, match="Unexpected ip_formulation value: 'bogus'"):
        map_query(
            hex_.create_hmm1(),
            method="integer_program",
            evidence=OBSERVED,
            ip_formulation="bogus",
        )


@pytest.mark.parametrize(
    "model,kind,kwds,params",
    [
        ("mn", "pyomo", {}, {}),
        ("bn", "toulbar2", {}, {}),
        ("dbn", "pyomo", {}, dict(start=0, stop=1)),
        ("dbn", "toulbar2", dict(stop=3), dict(start=0, stop=3)),
        ("dbn", "pgmpy", dict(start=1, stop=3), dict(start=1, stop=3)),
        ("hmm", "pyomo", dict(evidence=OBSERVED), dict(start=0, stop=4)),
        ("hmm", "toulbar2", dict(evidence=OBSERVED, stop=2), dict(start=0, stop=2)),
        ("hmm", "pgmpy", dict(evidence=OBSERVED, stop=2), dict(start=0, stop=2)),
        ("hmm", "pgmpy", dict(evidence=OBSERVED), dict(start=0, stop=4)),
        ("hmm", "pyomo_network_flow", dict(evidence=OBSERVED), {}),
        ("hmm", "oracle", dict(evidence=OBSERVED, stop=2), {}),
        ("hmm", "matvec", dict(evidence=OBSERVED), {}),
        ("hmm", "mvr", dict(evidence=OBSERVED, stop=2), {}),
    ],
)
def test_structure_params(model, kind, kwds, params):
    kwds = dict(dict(evidence=None, start=0, stop=None), **kwds)
    kwds["observed"] = kwds.pop("evidence")
    assert mq._structure_params(kind, _cmodel(model), **kwds) == params


@pytest.mark.parametrize("kind", ["pyomo", "toulbar2", "pgmpy"])
def test_structure_params_stop_past_the_observations(kind):
    with pytest.raises(InvalidInputError, match="past the last observation"):
        mq._structure_params(
            kind, _cmodel("hmm"), observed=OBSERVED, start=0, stop=len(OBSERVED)
        )


# ---------------------------------------------------------------------
# One solver per backend, and one per HMM algorithm
# ---------------------------------------------------------------------

# Each representation class and the class of the overload that solves it:
# the MN/BN/DBN representations of a factor backend share its base class's
# overload; the HMM representations have their own.
OVERLOADS = {
    PyomoMNRepresentation: PyomoFactorRepresentation,
    PyomoBNRepresentation: PyomoFactorRepresentation,
    PyomoDBNRepresentation: PyomoFactorRepresentation,
    PyomoHMMRepresentation: PyomoHMMRepresentation,
    PyomoNetworkFlowHMMRepresentation: PyomoNetworkFlowHMMRepresentation,
    Toulbar2MNRepresentation: Toulbar2FactorRepresentation,
    Toulbar2BNRepresentation: Toulbar2FactorRepresentation,
    Toulbar2DBNRepresentation: Toulbar2FactorRepresentation,
    Toulbar2HMMRepresentation: Toulbar2HMMRepresentation,
    PgmpyMNRepresentation: PgmpyRepresentation,
    PgmpyBNRepresentation: PgmpyRepresentation,
    PgmpyDBNRepresentation: PgmpyRepresentation,
    PgmpyHMMRepresentation: PgmpyHMMRepresentation,
    MatVecHMMRepresentation: MatVecHMMRepresentation,
    OracleHMMRepresentation: OracleHMMRepresentation,
    RecursiveHMMRepresentation: RecursiveHMMRepresentation,
    MVRHMMRepresentation: MVRHMMRepresentation,
}


@pytest.mark.parametrize("cls", list(OVERLOADS), ids=lambda cls: cls.__name__)
def test_each_representation_class_has_the_overload_of_its_backend(cls):
    method = mq._solve_map_query.resolve(cls, munch.Munch)
    assert method.__annotations__["repn"] is OVERLOADS[cls]


def test_no_overload_for_other_objects():
    with pytest.raises(TypeError):
        mq._solve_map_query.resolve(object, munch.Munch)


@pytest.mark.parametrize(
    "model,kind,kwds,cls",
    [
        ("mn", "pyomo", {}, PyomoMNRepresentation),
        ("bn", "pyomo", {}, PyomoBNRepresentation),
        ("dbn", "pyomo", {}, PyomoDBNRepresentation),
        ("hmm", "pyomo", dict(stop=4), PyomoHMMRepresentation),
        ("hmm", "pyomo_network_flow", {}, PyomoNetworkFlowHMMRepresentation),
        ("mn", "toulbar2", {}, Toulbar2MNRepresentation),
        ("bn", "toulbar2", {}, Toulbar2BNRepresentation),
        ("dbn", "toulbar2", {}, Toulbar2DBNRepresentation),
        ("hmm", "toulbar2", dict(stop=4), Toulbar2HMMRepresentation),
        ("mn", "pgmpy", {}, PgmpyMNRepresentation),
        ("bn", "pgmpy", {}, PgmpyBNRepresentation),
        ("dbn", "pgmpy", {}, PgmpyDBNRepresentation),
        ("hmm", "pgmpy", dict(stop=4), PgmpyHMMRepresentation),
        ("hmm", "matvec", {}, MatVecHMMRepresentation),
        ("hmm", "oracle", {}, OracleHMMRepresentation),
        ("hmm", "recursive", {}, RecursiveHMMRepresentation),
        ("hmm", "mvr", {}, MVRHMMRepresentation),
    ],
)
def test_each_representation_kind_builds_its_class(model, kind, kwds, cls):
    repn = _cmodel(model).representation(kind, **kwds)
    assert type(repn) is cls
    assert cls in OVERLOADS


def _query(**kwds):
    q = munch.Munch(
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
        options={},
    )
    q.update(kwds)
    q.setdefault("model", None)
    # map_query also passes the HMM observations as a list.
    if isinstance(q.evidence, (list, tuple, dict)) and "observed" not in kwds:
        q.observed = as_sequence(q.evidence)
    return q


@skipif_no_mip_solver
@pytest.mark.parametrize(
    "model,kwds,expected",
    [
        ("mn", {}, {"A": 2, "B": 2, "C": 1}),
        ("dbn", {}, None),
    ],
)
def test_pyomo_overloads_solve(model, kwds, expected):
    cmodel = _cmodel(model)
    repn = cmodel.representation("pyomo")
    results = mq._solve_map_query(
        repn, _query(options=dict(solver=default_mip_solver), **kwds)
    )
    reference = map_query(cmodel, method="integer_program", solver=default_mip_solver)
    assert results.solution.states == reference.solution.states
    if expected is not None:
        assert results.solution.states == expected


@skipif_toulbar2_not_available
def test_toulbar2_hmm_overload_solves():
    cmodel = _cmodel("hmm")
    repn = cmodel.representation("toulbar2", start=0, stop=4)
    results = mq._solve_map_query(repn, _query(model=cmodel.model, evidence=OBSERVED))
    assert results.solution.states == ["h0"] * 5


def test_matvec_overload_is_viterbi():
    hmm = hex_.create_hmm1()
    results = mq._solve_map_query(
        ConstrainedHiddenMarkovModel(hmm).representation("matvec"),
        _query(evidence=OBSERVED),
    )
    assert results.solutions == viterbi(observed=OBSERVED, hmm=hmm).solutions


def test_oracle_overload_is_a_star():
    cmodel = hex_.create_chmm1_oracle()
    observed = ["o0"] * 12
    results = mq._solve_map_query(
        cmodel.representation("oracle"),
        _query(evidence=observed, options=dict(num_solutions=2)),
    )
    reference = a_star(observed=observed, hmm=cmodel, num_solutions=2)
    assert results.solutions == reference.solutions


def test_recursive_overload_is_recursive_a_star():
    cmodel = hex_.create_chmm1_recursive()
    observed = ["o0"] * 12
    results = mq._solve_map_query(
        cmodel.representation("recursive"), _query(evidence=observed)
    )
    reference = recursive_a_star(hmm=cmodel, observed=observed)
    assert [s.states for s in results.solutions] == [
        s.states for s in reference.solutions
    ]


# ---------------------------------------------------------------------
# Errors from the registry (method does not support the model or family)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("method", ["viterbi", "a_star", "mvr_viterbi"])
@pytest.mark.parametrize(
    "model",
    [
        lambda: mex.ABC_conin().pgm,
        lambda: bex.cancer1_BN_conin().pgm,
        lambda: dex.simple1_DDBN_conin().pgm,
    ],
    ids=["mn", "bn", "dbn"],
)
def test_hmm_only_methods_on_other_models_raise_invalid_input(method, model):
    with pytest.raises(InvalidInputError, match="No .* representation is available"):
        map_query(model(), method=method)


@pytest.mark.parametrize(
    "method,cmodel",
    [
        ("viterbi", hex_.create_chmm1_oracle),
        ("a_star", hex_.create_chmm1_pyomo),
        ("toulbar2", hex_.create_chmm1_oracle),
        ("integer_program", hex_.create_chmm1_oracle),
        ("variable_elimination", hex_.create_chmm1_pyomo),
        ("mvr_viterbi", hex_.create_chmm1_oracle),
    ],
)
def test_unsupported_constraint_family_raises_invalid_input(method, cmodel):
    with pytest.raises(InvalidInputError, match="No .* representation is available"):
        map_query(cmodel(), method=method, evidence=["o0"] * 12)


def test_network_flow_on_a_markov_network_raises_invalid_input():
    with pytest.raises(InvalidInputError, match="'pyomo_network_flow'"):
        map_query(
            mex.ABC_conin().pgm,
            method="integer_program",
            ip_formulation="network_flow",
        )


@skipif_no_mip_solver
def test_ip_formulation_markov_network_is_accepted_for_every_model():
    # ip_formulation used to reach the solver (TypeError) on MN/BN/DBN.
    results = map_query(
        mex.ABC_conin().pgm,
        method="integer_program",
        ip_formulation="markov_network",
        solver=default_mip_solver,
    )
    assert results.solution.states == {"A": 2, "B": 2, "C": 1}


# ---------------------------------------------------------------------
# HMM evidence
# ---------------------------------------------------------------------


@pytest.mark.parametrize("evidence", [None, "o0o0", {"o0", "o1"}])
@pytest.mark.parametrize("method", list(mq.METHODS))
def test_hmm_evidence_must_be_a_list_tuple_or_dict(method, evidence):
    # Viterbi and A* used to return None, the solvers raised AssertionError
    # or TypeError.
    with pytest.raises(InvalidInputError, match="a list, a tuple or a dict"):
        map_query(hex_.create_hmm1(), method=method, evidence=evidence)


@pytest.mark.parametrize(
    "evidence,match",
    [
        ([], "must not be empty"),
        ({}, "must not be empty"),
        ({0: "o0", 2: "o1"}, "keyed by the times 0..T-1"),
        ({1: "o0", 2: "o1"}, "keyed by the times 0..T-1"),
    ],
)
@pytest.mark.parametrize("method", [m for m in mq.METHODS if m != "mvr_viterbi"])
def test_hmm_evidence_dict_must_be_keyed_by_every_time(method, evidence, match):
    # A sparse dict raised KeyError, an empty one IndexError or AssertionError.
    # (mvr_viterbi reads a sparse {time: label} dict itself.)
    with pytest.raises(InvalidInputError, match=match):
        map_query(hex_.create_hmm1(), method=method, evidence=evidence)


def _hmm_method_params():
    return [
        pytest.param("viterbi", {}),
        pytest.param("a_star", {}),
        pytest.param(
            "integer_program",
            dict(solver=default_mip_solver),
            marks=skipif_no_mip_solver,
        ),
        pytest.param("toulbar2", {}, marks=skipif_toulbar2_not_available),
        pytest.param("variable_elimination", {}, marks=skipif_pgmpy_not_available),
    ]


@pytest.mark.parametrize("method,solver", _hmm_method_params())
def test_hmm_tuple_evidence_is_read_like_a_list(method, solver):
    hmm = hex_.create_hmm1()
    as_list = map_query(hmm, method=method, evidence=OBSERVED, **solver)
    as_tuple = map_query(hmm, method=method, evidence=tuple(OBSERVED), **solver)
    assert as_tuple.solution.states == as_list.solution.states
    assert isinstance(as_tuple.solution.states, list)


@pytest.mark.parametrize(
    "method,solver",
    [p for p in _hmm_method_params() if p.values[0] not in ("viterbi", "a_star")],
)
@pytest.mark.parametrize("dict_evidence", [False, True])
def test_hmm_stop_limits_the_query(method, solver, dict_evidence):
    # variable_elimination ignored stop and always used every observation.
    hmm = hex_.create_hmm1()
    observed = ["o0", "o0", "o1", "o1", "o1", "o0"]
    evidence = dict(enumerate(observed)) if dict_evidence else observed
    short = map_query(hmm, method=method, evidence=observed[:3], **solver)
    results = map_query(hmm, method=method, evidence=evidence, stop=2, **solver)
    expected = short.solution.states
    if dict_evidence:
        expected = dict(enumerate(expected))
    assert results.solution.states == expected


@pytest.mark.parametrize(
    "method,solver",
    [p for p in _hmm_method_params() if p.values[0] not in ("viterbi", "a_star")],
)
def test_hmm_stop_past_the_last_observation(method, solver):
    # It raised KeyError (integer_program, toulbar2) or was ignored
    # (variable_elimination).
    with pytest.raises(InvalidInputError, match="past the last observation"):
        map_query(
            hex_.create_hmm1(),
            method=method,
            evidence=OBSERVED,
            stop=len(OBSERVED),
            **solver,
        )


# ---------------------------------------------------------------------
# Behaviour of the uniform dispatch
# ---------------------------------------------------------------------


@pytest.mark.parametrize("evidence", [OBSERVED, dict(enumerate(OBSERVED))])
def test_viterbi_on_an_unconstrained_constrained_hmm(evidence):
    # It raised TypeError (there was no overload for constrained HMMs).
    hmm = hex_.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm)
    results = map_query(cmodel, method="viterbi", evidence=evidence)
    reference = map_query(hmm, method="viterbi", evidence=evidence)
    assert results.solutions == reference.solutions


@pytest.mark.parametrize("dict_evidence", [False, True])
def test_a_star_with_recursive_constraints_runs_recursive_a_star(dict_evidence):
    # It raised InvalidInputError (no "oracle" representation).
    cmodel = hex_.create_chmm1_recursive()
    observed = ["o0"] * 15
    evidence = dict(enumerate(observed)) if dict_evidence else observed
    results = map_query(cmodel, method="a_star", evidence=evidence, num_solutions=2)
    reference = recursive_a_star(hmm=cmodel, observed=observed, num_solutions=2)
    hidden = [s.states for s in reference.solutions]
    if dict_evidence:
        hidden = [dict(enumerate(h)) for h in hidden]
    assert [s.states for s in results.solutions] == hidden
    assert results.solution is results.solutions[0]
    assert results.termination_condition == reference.termination_condition
    assert results.solution.states == (
        dict(enumerate(["h1"] * 3 + ["h0"] * 12))
        if dict_evidence
        else ["h1"] * 3 + ["h0"] * 12
    )
    # The same answer as A* with the equivalent oracle constraints
    oracle = map_query(hex_.create_chmm1_oracle(), method="a_star", evidence=evidence)
    assert oracle.solution.states == results.solution.states


def test_a_star_with_recursive_constraints_and_no_solution():
    cmodel = hex_.create_chmm1_recursive()
    results = map_query(cmodel, method="a_star", evidence=["o0"] * 5)
    assert results.solution is None
    assert results.solutions == []
    assert results.termination_condition == "error: no feasible solutions"


@skipif_no_mip_solver
@pytest.mark.parametrize(
    "model",
    [hex_.create_hmm1, lambda: ConstrainedHiddenMarkovModel(hex_.create_hmm1())],
    ids=["bare", "constrained-empty"],
)
def test_network_flow_without_constraints_uses_the_lp_path(model):
    # A constrained HMM with no constraints used the IP path (which also
    # returns solution.variables); the empty wrapper is now transparent.
    results = map_query(
        model(),
        method="integer_program",
        evidence=OBSERVED,
        ip_formulation="network_flow",
        solver=default_mip_solver,
    )
    assert results.solution.states == ["h0"] * 5
    assert "variables" not in results.solution


@skipif_no_mip_solver
def test_network_flow_with_constraints_uses_the_ip_path():
    results = map_query(
        hex_.create_chmm1_pyomo(),
        method="integer_program",
        evidence=["o0"] * 15,
        ip_formulation="network_flow",
        solver=default_mip_solver,
    )
    assert results.solution.states.count("h0") in (10, 11, 12)
    assert "variables" in results.solution


@pytest.mark.parametrize(
    "method,solver",
    [
        pytest.param(
            "integer_program",
            dict(solver=default_mip_solver),
            marks=skipif_no_mip_solver,
        ),
        pytest.param("toulbar2", {}, marks=skipif_toulbar2_not_available),
        pytest.param("variable_elimination", {}, marks=skipif_pgmpy_not_available),
    ],
)
def test_dbn_stop_defaults_to_one(method, solver):
    # integer_program and toulbar2 raised TypeError for stop=None.
    dbn = dex.simple1_DDBN_conin().pgm
    results = map_query(dbn, method=method, **solver)
    explicit = map_query(dbn, method=method, stop=1, **solver)
    assert results.solution.states == explicit.solution.states
    assert {t for _, t in results.solution.states} == {0, 1}


@pytest.mark.parametrize("method,solver", _hmm_method_params())
@pytest.mark.parametrize("observation", ["o0", "o1"])
@pytest.mark.parametrize("dict_evidence", [False, True])
def test_hmm_with_one_observation(method, solver, observation, dict_evidence):
    # integer_program, toulbar2 and variable_elimination raised a bare
    # AssertionError from create_bn_from_dbn (stop == start == 0).
    hmm = hex_.create_hmm1()
    evidence = {0: observation} if dict_evidence else [observation]
    results = map_query(hmm, method=method, evidence=evidence, **solver)
    expected = viterbi(observed=[observation], hmm=hmm).solution.states
    if dict_evidence:
        expected = dict(enumerate(expected))
    assert results.solution.states == expected


@pytest.mark.parametrize(
    "method,solver",
    [p for p in _hmm_method_params() if p.values[0] not in ("viterbi", "a_star")],
)
def test_dbn_with_one_time_step(method, solver):
    dbn = dex.simple1_DDBN_conin().pgm
    results = map_query(dbn, method=method, stop=0, **solver)
    assert sorted(results.solution.states) == [("A", 0), ("B", 0)]
    # P(A0 = 0) = 0.9, P(B0 = 1 | A0 = 0) = 0.8
    assert results.solution.states == {("A", 0): 0, ("B", 0): 1}


@pytest.mark.parametrize("kind", ["pyomo", "toulbar2", "pgmpy"])
@pytest.mark.parametrize("model", ["dbn", "hmm"])
def test_unrolling_from_a_later_start_is_refused(kind, model):
    # create_bn_from_dbn(start > 0) failed with an AttributeError for every
    # DBN with a transition CPD.
    with pytest.raises(InvalidInputError, match="only be unrolled from start=0"):
        _cmodel(model).representation(kind, start=1, stop=3)
