import pytest
import os.path
import importlib

from conin.util import try_import
from conin.io import load_model

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pgmpy_readwrite_available:
    import pgmpy.readwrite
    from conin.common.pgmpy import convert_pgmpy_to_conin


cwd = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "io", "tests"
)


def test_convert_pgmpy_to_conin_requires_pgmpy(monkeypatch):
    module = importlib.import_module("conin.common.pgmpy.to_conin")
    monkeypatch.setattr(module, "pgmpy_available", False)

    with pytest.raises(ImportError, match="pgmpy"):
        module.convert_pgmpy_to_conin(object())


#
# cancer
#


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_convert_bn_from_pgmpy_cancer():
    pgmpy_pgm = load_model(os.path.join(cwd, "cancer_bn.bif"), model_type="pgmpy")
    pgmpy_pgm.check_model()
    pgm = convert_pgmpy_to_conin(pgmpy_pgm)

    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    assert cpds["Cancer"].values == {
        ("0", "0"): {"0": 0.03, "1": 0.97},
        ("0", "1"): {"0": 0.05, "1": 0.95},
        ("1", "0"): {"0": 0.001, "1": 0.999},
        ("1", "1"): {"0": 0.02, "1": 0.98},
    }
    assert cpds["Dyspnoea"].values == {
        "0": {"0": 0.65, "1": 0.35},
        "1": {"0": 0.3, "1": 0.7},
    }
    assert cpds["Pollution"].values == {"0": 0.9, "1": 0.1}
    assert cpds["Smoker"].values == {"0": 0.3, "1": 0.7}
    assert cpds["Xray"].values == {"0": {"0": 0.9, "1": 0.1}, "1": {"0": 0.2, "1": 0.8}}


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_convert_mn_from_pgmpy_cancer():
    pgmpy_pgm = load_model(os.path.join(cwd, "cancer_mn.uai"), model_type="pgmpy")
    pgmpy_pgm.check_model()
    pgm = convert_pgmpy_to_conin(pgmpy_pgm)

    factors = {
        tuple(f.nodes) if len(f.nodes) > 1 else f.nodes[0]: f.values
        for f in pgm.factors
    }
    assert factors["var_0", "var_1"] == {
        (0, 0): 0.65,
        (0, 1): 0.35,
        (1, 0): 0.3,
        (1, 1): 0.7,
    }
    assert factors["var_2"] == {0: 0.9, 1: 0.1}
    assert factors["var_3"] == {0: 0.3, 1: 0.7}
    assert factors["var_0", "var_4"] == {
        (0, 0): 0.9,
        (0, 1): 0.1,
        (1, 0): 0.2,
        (1, 1): 0.8,
    }
    assert factors["var_2", "var_3", "var_0"] == {
        (0, 0, 0): 0.03,
        (0, 0, 1): 0.97,
        (0, 1, 0): 0.05,
        (0, 1, 1): 0.95,
        (1, 0, 0): 0.001,
        (1, 0, 1): 0.999,
        (1, 1, 0): 0.02,
        (1, 1, 1): 0.98,
    }


#
# Dynamic Bayesian networks
#


def _unrolled_cpd_counts(pgmpy_pgm, stop):
    from collections import Counter
    from conin.graphical_models import create_bn_from_dbn

    dbn = convert_pgmpy_to_conin(pgmpy_pgm)
    bn = create_bn_from_dbn(dbn=dbn, start=0, stop=stop)
    return Counter(cpd.node for cpd in bn.cpds)


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_convert_dbn_keeps_one_cpd_per_template():
    # pgmpy holds the intra-slice CPD of B for slices 0 and 1
    # (initialize_initial_state copies it); both used to become a template CPD of
    # (B, t), so the unrolled network had two CPDs for each (B, t).
    from conin.examples.dynamic_bayesian_network import (
        simple1_DDBN_conin,
        simple1_DDBN_pgmpy,
    )

    counts = _unrolled_cpd_counts(simple1_DDBN_pgmpy().pgm, stop=3)
    assert set(counts.values()) == {1}
    assert sorted(counts) == [(v, t) for v in "AB" for t in range(4)]

    dbn = convert_pgmpy_to_conin(simple1_DDBN_pgmpy().pgm)
    expected = simple1_DDBN_conin().pgm
    assert len(dbn.cpds) == len(expected.cpds) == 3


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_convert_dbn_refuses_conflicting_slice_cpds():
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DynamicBayesianNetwork

    from conin.exceptions import InvalidInputError

    G = DynamicBayesianNetwork()
    G.add_edges_from([(("A", 0), ("B", 0)), (("A", 1), ("B", 1)), (("A", 0), ("A", 1))])
    G.add_cpds(
        TabularCPD(("A", 0), 2, [[0.9], [0.1]]),
        TabularCPD(("A", 1), 2, [[0.2, 0.9], [0.8, 0.1]], [("A", 0)], [2]),
        TabularCPD(("B", 0), 2, [[0.2, 0.9], [0.8, 0.1]], [("A", 0)], [2]),
        TabularCPD(("B", 1), 2, [[0.3, 0.9], [0.7, 0.1]], [("A", 1)], [2]),
    )
    with pytest.raises(InvalidInputError, match="same dynamic CPD template"):
        convert_pgmpy_to_conin(G)


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_convert_pgmpy_issue_1177_example():
    # The example fitted with estimator=MaximumLikelihoodEstimator (a class),
    # which newer pgmpy versions refuse ("Pass an initialized estimator").
    from conin.examples.bayesian_network import pgmpy_issue_1177_pgmpy

    model = pgmpy_issue_1177_pgmpy()
    assert model.check_model()
    assert len(model.get_cpds()) == 17
    pgm = convert_pgmpy_to_conin(model)
    assert len(pgm.nodes) == 17
