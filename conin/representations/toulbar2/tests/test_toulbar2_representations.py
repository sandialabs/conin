"""Tests for the "toulbar2" representations (PR 4b)."""

import os

import munch
import pytest

import conin.examples.bayesian_network as bex
import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
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
    create_bn_from_dbn,
)
from conin.inference import map_query
from conin.inference.toulbar2 import solve_toulbar2_map_query_model
from conin.io import save_model
from conin.representations import supported_representations
from conin.representations.toulbar2 import (
    Toulbar2BNRepresentation,
    Toulbar2DBNRepresentation,
    Toulbar2HMMRepresentation,
    Toulbar2MNRepresentation,
)
from conin.util import try_import

with try_import() as pytoulbar2_available:
    import pytoulbar2

skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
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
def test_toulbar2_registered_families(model_type):
    pairs = supported_representations(model_type)
    for family in (None, "toulbar2", "algebraic"):
        assert ("toulbar2", family) in pairs
    assert ("toulbar2", "pyomo") not in pairs


def test_toulbar2_rejects_pyomo_constraints():
    @pyomo_constraint_fn()
    def noop(M, D):
        pass

    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm, noop)
    with pytest.raises(InvalidInputError, match="No 'toulbar2' representation"):
        cmodel.representation("toulbar2")


@pytest.mark.parametrize(
    "cmodel,params,cls",
    [
        (
            ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm),
            {},
            Toulbar2MNRepresentation,
        ),
        (
            ConstrainedDiscreteBayesianNetwork(bex.cancer1_BN_conin().pgm),
            {},
            Toulbar2BNRepresentation,
        ),
        (
            ConstrainedDynamicDiscreteBayesianNetwork(dex.weather_conin().pgm),
            dict(stop=4),
            Toulbar2DBNRepresentation,
        ),
        (
            ConstrainedHiddenMarkovModel(hex_.create_hmm1()),
            dict(stop=4),
            Toulbar2HMMRepresentation,
        ),
    ],
)
def test_toulbar2_representation_classes(cmodel, params, cls):
    repn = cmodel.representation("toulbar2", **params)
    assert type(repn) is cls
    assert repn.kind == "toulbar2"
    assert repn.get_copy() is repn


# ---------------------------------------------------------------------
# The cached UAI text is what save_model writes (parity with the old path)
# ---------------------------------------------------------------------


def test_uai_text_matches_save_model(tmp_path):
    dbn = dex.weather_conin().pgm
    cases = [
        (ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm), {}, None),
        (ConstrainedDiscreteBayesianNetwork(bex.cancer1_BN_conin().pgm), {}, None),
        (
            ConstrainedDynamicDiscreteBayesianNetwork(dbn),
            dict(start=0, stop=3),
            create_bn_from_dbn(dbn=dbn, start=0, stop=3),
        ),
    ]
    for cmodel, params, unrolled in cases:
        repn = cmodel.representation("toulbar2", **params)
        filename = str(tmp_path / "model.uai")
        save_model(cmodel.model if unrolled is None else unrolled, filename)
        with open(filename) as f:
            assert repn.uai_text == f.read()


def test_var_wrapper_follows_the_state_order():
    pgm = bex.cancer1_BN_conin().pgm
    repn = ConstrainedDiscreteBayesianNetwork(pgm).representation("toulbar2")
    from conin.representations.toulbar2 import Toulbar2VarWrapper

    V = Toulbar2VarWrapper(repn.uai_model)
    for i, node in enumerate(pgm.nodes):
        assert V[node] == i
        for j, state in enumerate(pgm.states_of(node)):
            assert V(node, state, coef=3) == (i, j, 3)


# ---------------------------------------------------------------------
# get_copy isolation, evidence isolation and caching (§4.4, §8)
# ---------------------------------------------------------------------


@skipif_toulbar2_not_available
def test_two_gets_give_independent_models():
    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm)
    expected = mex.ABC_conin().solutions[0].states

    M1 = cmodel.representation("toulbar2").create_model()
    M1.Assign(M1.V["A"], 0)
    M1.AddGeneralizedLinearConstraint([M1.V("B", 0)], ">=", 1)

    M2 = cmodel.representation("toulbar2").create_model()
    assert M2 is not M1
    assert M2.V is not M1.V
    assert M2.V_evidence == set()
    assert solve_toulbar2_map_query_model(M2).solution.states == expected


@skipif_toulbar2_not_available
def test_evidence_never_reaches_the_cached_representation():
    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm)
    repn = cmodel.representation("toulbar2")
    text = repn.uai_text
    M1 = repn.create_model(evidence={"B": 0})
    assert M1.V_evidence == {"B"}
    M2 = cmodel.representation("toulbar2").create_model()
    assert M2.V_evidence == set()
    assert repn.uai_text == text
    assert cmodel.representation("toulbar2") is repn


def test_dbn_cache_is_keyed_by_start_and_stop():
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(dex.weather_conin().pgm)
    r4 = cmodel.representation("toulbar2", stop=4)
    assert cmodel.representation("toulbar2", stop=4) is r4
    r5 = cmodel.representation("toulbar2", stop=5)
    assert r5 is not r4
    assert (r4.start, r4.stop, r5.stop) == (0, 4, 5)
    assert len(r5.nodes) > len(r4.nodes)


def test_hmm_cache_is_keyed_by_start_and_stop():
    cmodel = ConstrainedHiddenMarkovModel(hex_.create_hmm1())
    r = cmodel.representation("toulbar2", stop=4)
    assert cmodel.representation("toulbar2", stop=4) is r
    assert cmodel.representation("toulbar2", stop=5) is not r


@skipif_toulbar2_not_available
def test_verbose_and_write_uai_file_are_not_cache_keys(tmp_path):
    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm)
    repn = cmodel.representation("toulbar2")
    filename = str(tmp_path / "out.uai")
    repn.create_model(verbose=-1, write_uai_file=filename)
    with open(filename) as f:
        assert f.read() == repn.uai_text
    assert cmodel.representation("toulbar2") is repn


@skipif_toulbar2_not_available
def test_write_uai_file_uses_the_exact_path(tmp_path):
    # save_model_uai writes to name.split(".")[0] + ".uai", which is wrong for
    # a dotted directory; the toulbar2 path writes the given file.
    d = tmp_path / "a.b"
    d.mkdir()
    filename = str(d / "out.uai")
    results = map_query(mex.ABC_conin().pgm, method="toulbar2", write_uai_file=filename)
    assert os.path.exists(filename)
    assert results.solution.states == mex.ABC_conin().solutions[0].states


# ---------------------------------------------------------------------
# Constraint data seen by user functions (unchanged by PR 4b)
# ---------------------------------------------------------------------


def _recording_constraint(seen):
    @toulbar2_constraint_fn()
    def record(M, D):
        seen.append(D)

    return record


@skipif_toulbar2_not_available
def test_constraint_data_mn():
    seen = []
    cmodel = ConstrainedDiscreteMarkovNetwork(
        mex.ABC_conin().pgm, _recording_constraint(seen)
    )
    cmodel.representation("toulbar2").create_model(evidence={"B": 0})
    assert seen == [munch.Munch(variables=None, evidence={"B": 0})]


@skipif_toulbar2_not_available
def test_constraint_data_dbn():
    seen = []
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(
        dex.weather_conin().pgm, _recording_constraint(seen)
    )
    cmodel.representation("toulbar2", start=0, stop=2).create_model()
    assert seen == [
        munch.Munch(start=0, stop=2, variables=None, evidence=None, T=[0, 1, 2])
    ]


@skipif_toulbar2_not_available
def test_constraint_data_hmm():
    seen = []
    hmm = hex_.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm, _recording_constraint(seen))
    evidence = {("E", 0): "o0", ("E", 1): "o1"}
    cmodel.representation("toulbar2", stop=1).create_model(evidence=evidence)
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
# §9: solve_toulbar2_map_query_model rejected extra **options (e.g. verbose)
# ---------------------------------------------------------------------


_VERBOSE_CASES = [
    (mex.ABC_conin, {}),
    (bex.cancer1_BN_conin, {}),
    (dex.weather_conin, dict(stop=2)),
    (hex_.create_hmm1, dict(evidence=["o0", "o1", "o1"])),
]


@skipif_toulbar2_not_available
@pytest.mark.parametrize("example,kwds", _VERBOSE_CASES)
def test_verbose_option_is_accepted(example, kwds):
    pgm = example()
    pgm = getattr(pgm, "pgm", pgm)
    expected = map_query(pgm, method="toulbar2", **kwds).solution.states
    results = map_query(pgm, method="toulbar2", verbose=-1, **kwds)
    assert results.solution.states == expected


@skipif_toulbar2_not_available
def test_unknown_options_still_raise():
    with pytest.raises(TypeError):
        map_query(mex.ABC_conin().pgm, method="toulbar2", no_such_option=1)


# ---------------------------------------------------------------------
# §9: the DBN/HMM toulbar2 paths didn't check that pytoulbar2 is available
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "module,example,kwds",
    [
        ("solve", mex.ABC_conin, {}),
        ("bn", bex.cancer1_BN_conin, {}),
        ("dbn", dex.weather_conin, dict(stop=2)),
        ("hmm", hex_.create_hmm1, dict(evidence=["o0", "o1"])),
    ],
)
def test_pytoulbar2_not_available(monkeypatch, module, example, kwds):
    import importlib

    mod = importlib.import_module(f"conin.inference.toulbar2.{module}")
    monkeypatch.setattr(mod, "pytoulbar2_available", False)
    pgm = example()
    pgm = getattr(pgm, "pgm", pgm)
    results = map_query(pgm, method="toulbar2", **kwds)
    assert results.termination_condition == "pytoulbar2 not available"
    assert results.solution is None and results.solutions == []
