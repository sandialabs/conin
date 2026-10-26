"""Regression tests: the constrained model's ``data`` reaches every backend's
constraint functions, and the integer-program and toulbar2 queries share the
same query checks."""

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
from conin.constraints import (
    FactorOracleConstraint,
    OracleConstraint,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.inference import map_query
from conin.inference.oracle import a_star
import pyomo.environ as pyo

from conin.sampling.oracle import rejection_sample_hidden
from conin.util import try_import

with try_import() as pytoulbar2_available:
    import pytoulbar2

with try_import() as pgmpy_available:
    import pgmpy

skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)

_CLASSES = {
    "mn": (ConstrainedDiscreteMarkovNetwork, lambda: mex.ABC_conin().pgm, {}),
    "bn": (ConstrainedDiscreteBayesianNetwork, lambda: bex.cancer1_BN_conin().pgm, {}),
    "dbn": (
        ConstrainedDynamicDiscreteBayesianNetwork,
        lambda: dex.weather_conin().pgm,
        dict(start=0, stop=2),
    ),
    "hmm": (ConstrainedHiddenMarkovModel, hex_.create_hmm1, dict(stop=1)),
}

# The fields each representation adds (what the functions saw before, when
# the application data was dropped).
_FIELDS = {
    ("pyomo", "mn"): dict(variables=None, evidence=None),
    ("pyomo", "dbn"): dict(start=0, stop=2, variables=None, evidence=None, T=[0, 1, 2]),
    ("pyomo", "hmm"): dict(start=0, stop=1, variables=None, evidence=None),
    ("toulbar2", "bn"): dict(variables=None, evidence=None),
    ("toulbar2", "hmm"): dict(start=0, stop=1, variables=None, evidence=None),
}


def _recorder(kind, seen):
    decorator = pyomo_constraint_fn if kind == "pyomo" else toulbar2_constraint_fn

    @decorator()
    def record(M, D):
        seen.append(D)

    return record


# ---------------------------------------------------------------------
# Regression: the "pyomo", "toulbar2" and "pgmpy" representations (and
# is_feasible and the oracle representation) never passed the constrained
# model's data to constraint functions; only network flow did
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,model",
    [
        ("pyomo", "mn"),
        ("pyomo", "dbn"),
        ("pyomo", "hmm"),
        pytest.param("toulbar2", "bn", marks=skipif_toulbar2_not_available),
        pytest.param("toulbar2", "hmm", marks=skipif_toulbar2_not_available),
    ],
)
def test_application_data_reaches_the_constraint_functions(kind, model):
    cls, make, params = _CLASSES[model]
    seen = []
    app = munch.Munch(k=3)
    cmodel = cls(make(), _recorder(kind, seen), data=app)
    cmodel.representation(kind, **params).create_model()
    (data,) = seen
    assert data.k == 3
    for key, value in _FIELDS[kind, model].items():
        assert data[key] == value
    # The representation sets its fields on a copy: the user's object is
    # unchanged.
    assert app == munch.Munch(k=3)


@pytest.mark.parametrize("kind", ["pyomo", "toulbar2"])
def test_without_application_data_the_fields_are_unchanged(kind):
    if kind == "toulbar2" and not pytoulbar2_available:
        pytest.skip("pytoulbar2 not installed")
    seen = []
    cmodel = ConstrainedDiscreteMarkovNetwork(
        mex.ABC_conin().pgm, _recorder(kind, seen)
    )
    cmodel.representation(kind).create_model(evidence={"B": 0})
    assert seen == [munch.Munch(variables=None, evidence={"B": 0})]


def test_a_representation_field_replaces_an_application_field():
    seen = []
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(
        dex.weather_conin().pgm,
        _recorder("pyomo", seen),
        data=munch.Munch(T="user", other=1),
    )
    cmodel.representation("pyomo", start=0, stop=1).create_model()
    assert seen[0].T == [0, 1] and seen[0].other == 1


def test_attribute_style_application_data():
    class AppData:
        k = 3

    seen = []
    cmodel = ConstrainedHiddenMarkovModel(
        hex_.create_hmm1(), _recorder("pyomo", seen), data=AppData()
    )
    cmodel.representation("pyomo", stop=1).create_model()
    assert seen[0].k == 3 and seen[0].hmm.T == [0, 1]
    assert not hasattr(cmodel.data, "hmm")


def _count_h0_at_most(data_key):
    """A 2-argument predicate: at most ``data[data_key]`` "h0" states."""
    return (
        lambda states, data: sum(v == "h0" for v in states.values()) <= data[data_key]
    )


def test_factor_representation_passes_the_application_data():
    seen = []

    def nodes(data):
        seen.append(data)
        return [("H", t) for t in data.hmm.T]

    constraint = FactorOracleConstraint(
        func=lambda states, data: sum(v == "h0" for v in states.values()) <= data.k,
        nodes=nodes,
        name="at_most_k",
    )
    cmodel = ConstrainedHiddenMarkovModel(
        hex_.create_hmm1(), constraint, data=munch.Munch(k=1)
    )
    cmodel.representation("pgmpy", start=0, stop=2)
    assert seen and all(d.k == 1 and d.hmm.T == [0, 1, 2] for d in seen)
    assert cmodel.data == munch.Munch(k=1)


@skipif_pgmpy_not_available
def test_variable_elimination_uses_the_application_data():
    constraint = FactorOracleConstraint(
        func=lambda states, data: sum(v == "h0" for v in states.values()) <= data.k,
        nodes=lambda data: [("H", t) for t in data.hmm.T],
        name="at_most_k",
    )
    cmodel = ConstrainedHiddenMarkovModel(
        hex_.create_hmm1(), constraint, data=munch.Munch(k=1)
    )
    results = map_query(cmodel, method="variable_elimination", evidence=["o0"] * 4)
    assert results.solution.states.count("h0") <= 1


def test_oracle_predicates_see_the_application_data():
    constraint = OracleConstraint(func=_count_h0_at_most("k"))
    cmodel = ConstrainedHiddenMarkovModel(
        hex_.create_hmm1(), constraint, data=munch.Munch(k=2)
    )
    observed = ["o0"] * 6
    results = a_star(observed=observed, hmm=cmodel)
    assert results.solution.states.count("h0") <= 2
    # The unconstrained optimum has more "h0" states.
    assert a_star(observed=observed, hmm=cmodel.model).solution.states.count("h0") > 2
    assert cmodel.is_feasible(results.solution.states)
    assert not cmodel.is_feasible(["h0"] * 6)
    cmodel.model.set_seed(1)
    assert rejection_sample_hidden(cmodel, 6).count("h0") <= 2
    assert cmodel.data == munch.Munch(k=2)


def test_is_feasible_passes_the_application_data():
    def at_most_k(states, data):
        return sum(v for v in states.values()) <= data.k

    mn = mex.ABC_conin().pgm
    constraint = FactorOracleConstraint(func=at_most_k, nodes=["A", "B"], name="c")
    cmodel = ConstrainedDiscreteMarkovNetwork(mn, constraint, data=munch.Munch(k=1))
    assert cmodel.is_feasible({"A": 1, "B": 0, "C": 2})
    assert not cmodel.is_feasible({"A": 1, "B": 1, "C": 0})


def test_is_feasible_dbn_sets_T_on_a_copy_of_the_application_data():
    seen = []

    def func(states, data):
        seen.append(data)
        return True

    dbn = dex.simple1_DDBN_conin().pgm
    constraint = FactorOracleConstraint(func=func, nodes=[("A", 0)], name="c")
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(
        dbn, constraint, data=munch.Munch(k=1)
    )
    cmodel.is_feasible({("A", 0): 0, ("A", 1): 0, ("B", 0): 0, ("B", 1): 0})
    assert seen == [munch.Munch(k=1, start=0, stop=1, T=[0, 1])]
    assert cmodel.data == munch.Munch(k=1)


@pytest.mark.skipif(default_mip_solver is None, reason="No MIP solver installed")
def test_integer_program_solves_with_the_application_data():
    @pyomo_constraint_fn()
    def at_most_k_zeros(M, D):
        M.c = pyo.Constraint(expr=sum(M.V(("H", t), "h0") for t in D.hmm.T) <= D.k)

    cmodel = ConstrainedHiddenMarkovModel(
        hex_.create_hmm1(), at_most_k_zeros, data=munch.Munch(k=2)
    )
    results = map_query(
        cmodel,
        method="integer_program",
        evidence=["o0"] * 6,
        solver=default_mip_solver,
    )
    assert results.solution.states.count("h0") <= 2


# ---------------------------------------------------------------------
# Regression: the toulbar2 queries ignored `variables` (they returned every
# variable) and raised TypeError for prune_network/create_MN, while the
# integer-program queries raise RuntimeError
# ---------------------------------------------------------------------


@skipif_toulbar2_not_available
@pytest.mark.parametrize(
    "make,kwds,variables",
    [
        (lambda: mex.ABC_conin().pgm, {}, ["A"]),
        (lambda: bex.cancer1_BN_conin().pgm, {}, ["Cancer"]),
        (lambda: dex.simple1_DDBN_conin().pgm, dict(stop=1), [("A", 0)]),
        (hex_.create_hmm1, dict(evidence=["o0", "o1"]), [("H", 0)]),
    ],
    ids=["mn", "bn", "dbn", "hmm"],
)
def test_toulbar2_partial_variables_raise_runtime_error(make, kwds, variables):
    with pytest.raises(RuntimeError):
        map_query(make(), method="toulbar2", variables=variables, **kwds)


@skipif_toulbar2_not_available
@pytest.mark.skipif(default_mip_solver is None, reason="No MIP solver installed")
@pytest.mark.parametrize(
    "make,kwds,variables",
    [
        (lambda: mex.ABC_conin().pgm, {}, ["A"]),
        (lambda: bex.cancer1_BN_conin().pgm, {}, ["Cancer"]),
        (lambda: dex.simple1_DDBN_conin().pgm, dict(stop=1), [("A", 0)]),
        (hex_.create_hmm1, dict(evidence=["o0", "o1"]), [("H", 0)]),
    ],
    ids=["mn", "bn", "dbn", "hmm"],
)
def test_partial_variables_raise_the_same_error_on_both_backends(make, kwds, variables):
    messages = []
    for method, solver in [
        ("toulbar2", {}),
        ("integer_program", dict(solver=default_mip_solver)),
    ]:
        with pytest.raises(RuntimeError) as exc:
            map_query(make(), method=method, variables=variables, **solver, **kwds)
        messages.append(str(exc.value))
    assert messages[0] == messages[1]


@skipif_toulbar2_not_available
def test_toulbar2_all_variables_is_a_full_query():
    bn = bex.cancer1_BN_conin().pgm
    full = map_query(bn, method="toulbar2")
    named = map_query(bn, method="toulbar2", variables=list(bn.nodes))
    assert named.solution.states == full.solution.states


@skipif_toulbar2_not_available
@pytest.mark.parametrize("option", ["prune_network", "create_MN"])
@pytest.mark.parametrize(
    "make,kwds",
    [
        (lambda: bex.cancer1_BN_conin().pgm, {}),
        (lambda: dex.simple1_DDBN_conin().pgm, dict(stop=1)),
        (hex_.create_hmm1, dict(evidence=["o0", "o1"])),
    ],
    ids=["bn", "dbn", "hmm"],
)
def test_toulbar2_prune_network_and_create_MN(option, make, kwds):
    pgm = make()
    expected = map_query(pgm, method="toulbar2", **kwds).solution.states
    # False is accepted (it raised TypeError) ...
    results = map_query(pgm, method="toulbar2", **{option: False}, **kwds)
    assert results.solution.states == expected
    # ... and True raises RuntimeError, as for the integer program.
    with pytest.raises(RuntimeError):
        map_query(pgm, method="toulbar2", **{option: True}, **kwds)


# ---------------------------------------------------------------------
# Regression: one solver per backend, so the options mean the same for every
# model type
# ---------------------------------------------------------------------


def _backend_params():
    return [
        pytest.param(
            "integer_program",
            dict(solver=default_mip_solver),
            marks=pytest.mark.skipif(
                default_mip_solver is None, reason="No MIP solver installed"
            ),
        ),
        pytest.param("toulbar2", {}, marks=skipif_toulbar2_not_available),
    ]


@pytest.mark.parametrize("method,solver", _backend_params())
@pytest.mark.parametrize("option", ["prune_network", "create_MN"])
def test_mn_prune_network_and_create_MN(method, solver, option):
    # On a Markov network these options reached the solver (TypeError).
    mn = mex.ABC_conin().pgm
    results = map_query(mn, method=method, **{option: False}, **solver)
    assert results.solution.states == {"A": 2, "B": 2, "C": 1}
    with pytest.raises(RuntimeError, match=f"{option}=True is not supported"):
        map_query(mn, method=method, **{option: True}, **solver)


@pytest.mark.skipif(default_mip_solver is None, reason="No MIP solver installed")
@pytest.mark.parametrize("evidence", [["o0", "o1"], ("o0", "o1")])
def test_hmm_integer_program_solution_with_evidence(evidence):
    # With list evidence this raised AttributeError; the hidden states are
    # returned as for solution_with_evidence=False.
    hmm = hex_.create_hmm1()
    expected = map_query(
        hmm, method="integer_program", evidence=evidence, solver=default_mip_solver
    )
    results = map_query(
        hmm,
        method="integer_program",
        evidence=evidence,
        solution_with_evidence=True,
        solver=default_mip_solver,
    )
    assert results.solution.states == expected.solution.states


@pytest.mark.skipif(default_mip_solver is None, reason="No MIP solver installed")
@pytest.mark.parametrize(
    "make,kwds",
    [
        (lambda: dex.simple1_DDBN_conin().pgm, {}),
        (hex_.create_hmm1, dict(evidence=["o0", "o1"])),
    ],
    ids=["dbn", "hmm"],
)
def test_integer_program_timing_on_dbns_and_hmms(capsys, make, kwds):
    # timing=True was ignored by the DBN and HMM integer programs.
    map_query(
        make(), method="integer_program", timing=True, solver=default_mip_solver, **kwds
    )
    assert "optimize_map_query_model - START" in capsys.readouterr().out
