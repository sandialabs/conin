import pytest
import os.path

import conin.io
from conin.io import load_model
from conin.exceptions import InvalidInputError

cwd = os.path.dirname(__file__)

#
# errors
#


def test_load_model_error1():
    with pytest.raises(FileNotFoundError):
        load_model(os.path.join(cwd, "unknown.uai"))


def test_load_model_error2():
    with pytest.raises(InvalidInputError, match="unknown file suffix"):
        load_model(os.path.join(cwd, "test_load_model.py"))


def test_load_model_error3():
    with pytest.raises(InvalidInputError, match="Unexpected model type"):
        load_model(os.path.join(cwd, "asia.uai"), model_type="unknown")


def test_load_model_missing_bif_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing file"):
        load_model(str(tmp_path / "missing.bif"))


def test_load_model_documented_io_namespace():
    pgm = conin.io.load_model(os.path.join(cwd, "asia.uai"), model_type="conin")
    assert pgm.nodes


#
# asia.uai
#


def test_load_model_asia_uai1_conin():
    load_model(os.path.join(cwd, "asia.uai"))


def test_load_model_asia_uai2_conin():
    load_model(os.path.join(cwd, "asia_compressed.uai.gz"))


#
# asia.bif
#


def test_load_model_asia_bif1_conin():
    load_model(os.path.join(cwd, "asia.bif"))


def test_load_model_asia_bif2_conin():
    load_model(os.path.join(cwd, "asia_compressed.bif.gz"))


#
# barley.uai
#


def test_load_model_barley1_conin():
    load_model(os.path.join(cwd, "barley.uai"))


def test_load_model_barley2_conin():
    load_model(os.path.join(cwd, "barley_compressed.uai.gz"))


#
# deer.uai
#


def test_load_model_deer1_conin():
    load_model(os.path.join(cwd, "deer.uai"))


def test_load_model_deer2_conin():
    load_model(os.path.join(cwd, "deer_compressed.uai.gz"))


#
# cancer
#


def test_load_bn_model_cancer_conin():
    pgm = load_model(os.path.join(cwd, "cancer_bn.uai"))
    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    assert cpds["var0"].values == {
        (0, 0): {0: 0.03, 1: 0.97},
        (0, 1): {0: 0.05, 1: 0.95},
        (1, 0): {0: 0.001, 1: 0.999},
        (1, 1): {0: 0.02, 1: 0.98},
    }
    assert cpds["var1"].values == {0: {0: 0.65, 1: 0.35}, 1: {0: 0.3, 1: 0.7}}
    assert cpds["var2"].values == {0: 0.9, 1: 0.1}
    assert cpds["var3"].values == {0: 0.3, 1: 0.7}
    assert cpds["var4"].values == {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.8}}


def test_load_mn_model_cancer_conin():
    pgm = load_model(os.path.join(cwd, "cancer_mn.uai"))
    factors = {
        tuple(f.nodes) if len(f.nodes) > 1 else f.nodes[0]: f.values
        for f in pgm.factors
    }

    assert factors["var0", "var1"] == {
        (0, 0): 0.65,
        (0, 1): 0.35,
        (1, 0): 0.3,
        (1, 1): 0.7,
    }
    assert factors["var2"] == {0: 0.9, 1: 0.1}
    assert factors["var3"] == {0: 0.3, 1: 0.7}
    assert factors["var0", "var4"] == {
        (0, 0): 0.9,
        (0, 1): 0.1,
        (1, 0): 0.2,
        (1, 1): 0.8,
    }
    assert factors["var2", "var3", "var0"] == {
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
# toulbar2
#


def test_load_bn_model_toulbar2_conin():
    pgm = load_model(os.path.join(cwd, "toulbar2_bn.uai"))
    cpds = {cpd.node: cpd for cpd in pgm.cpds}

    assert cpds["var0"].values == {0: 0.436, 1: 0.564}
    assert cpds["var1"].values == {0: {0: 0.128, 1: 0.872}, 1: {0: 0.92, 1: 0.08}}
    assert cpds["var2"].values == {
        0: {0: 0.21, 1: 0.333, 2: 0.457},
        1: {0: 0.811, 1: 0.0, 2: 0.189},
    }


def test_uai_markov_network_edges_join_every_pair_of_factor_nodes():
    # The UAI reader joined only the last node of each factor to the others.
    import itertools

    pgm = load_model(os.path.join(cwd, "cancer_mn.uai"))
    edges = {frozenset(e) for e in pgm.edges}
    for factor in pgm.factors:
        for a, b in itertools.combinations(factor.nodes, 2):
            assert frozenset((a, b)) in edges
