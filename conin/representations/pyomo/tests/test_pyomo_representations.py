"""Tests for the "pyomo" and "pyomo_network_flow" representations."""

import munch
import pyomo.environ as pyo
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
from conin.constraints import pyomo_constraint_fn, toulbar2_constraint_fn
from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    HiddenMarkovModel,
)
from conin.inference.pyomo import inference_pyomo_map_query_HMM
from conin.representations import supported_representations
from conin.representations.pyomo import (
    PyomoBNRepresentation,
    PyomoDBNRepresentation,
    PyomoHMMRepresentation,
    PyomoMNRepresentation,
    PyomoNetworkFlowHMMRepresentation,
    PyomoVarWrapper,
)
from conin.util import State

skipif_no_mip_solver = pytest.mark.skipif(
    not default_mip_solver, reason="No mip solver installed"
)

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type",
    [
        DiscreteMarkovNetwork,
        DiscreteBayesianNetwork,
        DynamicDiscreteBayesianNetwork,
        HiddenMarkovModel,
    ],
)
def test_pyomo_registered_families(model_type):
    pairs = supported_representations(model_type)
    for family in (None, "pyomo", "algebraic"):
        assert ("pyomo", family) in pairs
    assert ("pyomo", "toulbar2") not in pairs


def test_pyomo_network_flow_registered_families():
    pairs = supported_representations(HiddenMarkovModel)
    for family in (None, "pyomo", "algebraic"):
        assert ("pyomo_network_flow", family) in pairs
    assert ("pyomo_network_flow", "toulbar2") not in pairs


def test_pyomo_rejects_toulbar2_constraints():
    @toulbar2_constraint_fn()
    def noop(M, D):
        pass

    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm, noop)
    with pytest.raises(InvalidInputError, match="No 'pyomo' representation"):
        cmodel.representation("pyomo")


@pytest.mark.parametrize(
    "cmodel, params, cls",
    [
        (
            ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm),
            {},
            PyomoMNRepresentation,
        ),
        (
            ConstrainedDiscreteBayesianNetwork(bex.cancer1_BN_conin().pgm),
            {},
            PyomoBNRepresentation,
        ),
        (
            ConstrainedDynamicDiscreteBayesianNetwork(dex.weather_conin().pgm),
            dict(stop=4),
            PyomoDBNRepresentation,
        ),
        (
            ConstrainedHiddenMarkovModel(hex_.create_hmm1()),
            dict(stop=4),
            PyomoHMMRepresentation,
        ),
    ],
)
def test_pyomo_representation_classes(cmodel, params, cls):
    repn = cmodel.representation("pyomo", **params)
    assert type(repn) is cls
    assert repn.kind == "pyomo"


# ---------------------------------------------------------------------
# One representation serves several queries; structure params
# ---------------------------------------------------------------------


def test_mutating_a_returned_model_does_not_affect_the_next_one():
    repn = mex.ABC_constrained_pyomo_conin().pgm.representation("pyomo")
    M1 = repn.create_model()
    n_constraints = M1.nconstraints()
    M1.extra = pyo.Constraint(expr=M1.V("A", 0) == 1)
    M1.V("B", 1).fix(1)
    M1.del_component(M1.c1)

    M2 = repn.create_model()
    assert M2 is not M1
    assert M2.nconstraints() == n_constraints
    assert not hasattr(M2, "extra")
    assert not any(var.is_fixed() for var in M2.x.values())


def test_evidence_does_not_change_the_representation():
    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm)
    repn = cmodel.representation("pyomo")
    S_before = {k: list(v) for k, v in repn.S.items()}
    M1 = repn.create_model(evidence={"B": 0})
    assert M1.V("B", 0).is_fixed()
    M2 = repn.create_model()
    assert not any(var.is_fixed() for var in M2.x.values())
    assert repn.S == S_before


def test_omit_c5_is_a_create_model_option():
    repn = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm).representation("pyomo")
    assert hasattr(repn.create_model(), "c5")
    assert not hasattr(repn.create_model(omit_c5=True), "c5")


def test_dbn_structure_params_start_and_stop():
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(dex.weather_conin().pgm)
    r4 = cmodel.representation("pyomo", stop=4)
    r5 = cmodel.representation("pyomo", stop=5)
    assert (r4.start, r4.stop, r5.stop) == (0, 4, 5)
    assert len(r5.nodes) > len(r4.nodes)


def test_hmm_structure_params_start_and_stop():
    cmodel = ConstrainedHiddenMarkovModel(hex_.create_hmm1())
    r4 = cmodel.representation("pyomo", stop=4)
    r5 = cmodel.representation("pyomo", stop=5)
    assert (r4.start, r4.stop, r5.stop) == (0, 4, 5)
    assert len(r5.nodes) > len(r4.nodes)


def test_network_flow_structure_params_variable_domains():
    cmodel = ConstrainedHiddenMarkovModel(hex_.create_hmm1())
    default = cmodel.representation("pyomo_network_flow")
    assert (default.x_binary, default.y_binary) == (True, False)
    relaxed = cmodel.representation("pyomo_network_flow", x_binary=False)
    M, _ = relaxed.create_model(observed=["o0", "o1"])
    assert not M.hmm.x[0, 0].is_binary()
    M, _ = default.create_model(observed=["o0", "o1"])
    assert M.hmm.x[0, 0].is_binary()


# ---------------------------------------------------------------------
# Constraint data seen by user functions
# ---------------------------------------------------------------------


def _recording_constraint(seen):
    @pyomo_constraint_fn()
    def record(M, D):
        seen.append(D)

    return record


def test_constraint_data_mn():
    seen = []
    cmodel = ConstrainedDiscreteMarkovNetwork(
        mex.ABC_conin().pgm, _recording_constraint(seen)
    )
    cmodel.representation("pyomo").create_model(evidence={"B": 0})
    assert seen == [munch.Munch(variables=None, evidence={"B": 0})]


def test_constraint_data_dbn():
    seen = []
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(
        dex.weather_conin().pgm, _recording_constraint(seen)
    )
    cmodel.representation("pyomo", start=0, stop=2).create_model()
    assert seen == [
        munch.Munch(start=0, stop=2, variables=None, evidence=None, T=[0, 1, 2])
    ]


def test_constraint_data_hmm():
    seen = []
    hmm = hex_.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm, _recording_constraint(seen))
    evidence = {("E", 0): "o0", ("E", 1): "o1"}
    cmodel.representation("pyomo", stop=1).create_model(evidence=evidence)
    assert seen == [
        munch.Munch(
            start=0,
            stop=1,
            variables=None,
            evidence=evidence,
            hmm=munch.Munch(T=[0, 1], hidden_to_internal=hmm.hidden_to_internal),
        )
    ]


# ---------------------------------------------------------------------
# Regression: the HMM network_flow branch could return None
# ---------------------------------------------------------------------


@pytest.mark.parametrize("evidence", [None, "o0o1", {0: "o0", 2: "o1"}])
def test_network_flow_rejects_unsupported_evidence(evidence):
    with pytest.raises(InvalidInputError, match="network_flow"):
        inference_pyomo_map_query_HMM(
            pgm=hex_.create_hmm1(), evidence=evidence, ip_formulation="network_flow"
        )


def test_network_flow_rejects_unsupported_model():
    with pytest.raises(InvalidInputError, match="network_flow"):
        inference_pyomo_map_query_HMM(
            pgm=mex.ABC_conin().pgm,
            evidence=["o0", "o1"],
            ip_formulation="network_flow",
        )


# ---------------------------------------------------------------------
# Regression: V lookups returned (rather than raised) KeyError
# ---------------------------------------------------------------------


def test_network_flow_V_lookups():
    repn = ConstrainedHiddenMarkovModel(hex_.create_hmm1()).representation(
        "pyomo_network_flow"
    )
    M, _ = repn.create_model(observed=["o0", "o1"])
    assert isinstance(M.V, PyomoVarWrapper)
    assert M.V("H", 1, "h0") is M.hmm.x[1, 0]
    assert M.V(("H", 1), "h1") is M.hmm.x[1, 1]
    assert M.V[("H", 0), State("h1")] is M.hmm.x[0, 1]


def test_var_wrapper_is_one_class():
    # PyomoVarWrapper lives in conin.util.var_wrapper (so conin.constraints.algebraic
    # can use it without an import cycle) and is re-exported here.
    import conin.representations.pyomo.common as pyomo_common
    from conin.util.var_wrapper import PyomoVarWrapper as util_wrapper

    assert PyomoVarWrapper is util_wrapper is pyomo_common.PyomoVarWrapper


def test_network_flow_V_bad_arity_raises():
    repn = ConstrainedHiddenMarkovModel(hex_.create_hmm1()).representation(
        "pyomo_network_flow"
    )
    M, _ = repn.create_model(observed=["o0", "o1"])
    with pytest.raises(ValueError, match="2 or 3 arguments"):
        M.V("H")
    with pytest.raises(ValueError, match="2 or 3 arguments"):
        M.V("H", 0, "h0", "extra")


def test_network_flow_V_missing_key_raises():
    repn = ConstrainedHiddenMarkovModel(hex_.create_hmm1()).representation(
        "pyomo_network_flow"
    )
    M, _ = repn.create_model(observed=["o0", "o1"])
    with pytest.raises(KeyError):
        M.V("H", 5, "h0")
    with pytest.raises(KeyError):
        M.V("H", 0, "no-such-state")


# ---------------------------------------------------------------------
# Network flow with algebraic constraints (newly registered)
# ---------------------------------------------------------------------


@skipif_no_mip_solver
def test_network_flow_algebraic_constraints():
    pytest.importorskip("smoek")
    results = inference_pyomo_map_query_HMM(
        pgm=hex_.create_chmm1_algebraic(),
        evidence=["o0"] * 15,
        ip_formulation="network_flow",
        solver=default_mip_solver,
    )
    assert results.solution.states == ["h1"] * 3 + ["h0"] * 12


# ---------------------------------------------------------------------
# BN build options
# ---------------------------------------------------------------------


@skipif_no_mip_solver
def test_bn_prune_network_false_is_accepted():
    from conin.inference.pyomo import inference_pyomo_map_query_BN

    example = bex.cancer1_BN_conin()
    results = inference_pyomo_map_query_BN(
        pgm=example.pgm,
        solver=default_mip_solver,
        prune_network=False,
        create_MN=False,
    )
    assert results.solution.states == example.solutions[0].states


@pytest.mark.parametrize("option", ["prune_network", "create_MN"])
def test_bn_unsupported_build_options_raise(option):
    from conin.inference.pyomo import inference_pyomo_map_query_BN

    with pytest.raises(RuntimeError):
        inference_pyomo_map_query_BN(pgm=bex.cancer1_BN_conin().pgm, **{option: True})


# ---------------------------------------------------------------------
# inference_pyomo_map_query_HMM: the ip_formulation branches
# ---------------------------------------------------------------------


def test_hmm_unknown_ip_formulation_raises():
    with pytest.raises(ValueError, match="Unexpected ip_formulation value: 'bogus'"):
        inference_pyomo_map_query_HMM(
            pgm=hex_.create_hmm1(), evidence=["o0", "o1"], ip_formulation="bogus"
        )


@pytest.mark.skipif(not default_mip_solver, reason="No mip solver installed")
@pytest.mark.parametrize("ip_formulation", [None, "markov_network", "network_flow"])
@pytest.mark.parametrize("evidence", [["o0", "o1", "o1"], {0: "o0", 1: "o1", 2: "o1"}])
def test_hmm_ip_formulations_agree(ip_formulation, evidence):
    results = inference_pyomo_map_query_HMM(
        pgm=hex_.create_hmm1(),
        evidence=evidence,
        ip_formulation=ip_formulation,
        solver=default_mip_solver,
    )
    expected = ["h1", "h1", "h1"]
    if isinstance(evidence, dict):
        expected = dict(enumerate(expected))
    assert results.solution.states == expected


@pytest.mark.skipif(not default_mip_solver, reason="No mip solver installed")
def test_network_flow_warns_about_cut_generation(caplog):
    with caplog.at_level("WARNING", logger="conin"):
        inference_pyomo_map_query_HMM(
            pgm=hex_.create_hmm1(),
            evidence=["o0", "o1"],
            ip_formulation="network_flow",
            use_cut_generation=True,
            solver=default_mip_solver,
        )
    assert "Cut generation is not supported" in caplog.text
