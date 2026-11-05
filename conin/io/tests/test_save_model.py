import pytest
import os.path
import filecmp

import conin.io
from conin.io import save_model, load_model
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD
from conin.exceptions import InvalidInputError

cwd = os.path.dirname(__file__)


@pytest.fixture
def asia_uai():
    return load_model(os.path.join(cwd, "asia.uai"))


@pytest.fixture
def barley_uai():
    return load_model(os.path.join(cwd, "barley.uai"))


@pytest.fixture
def cancer_bn_uai():
    return load_model(os.path.join(cwd, "cancer_bn.uai"))


@pytest.fixture
def cancer_mn_uai():
    return load_model(os.path.join(cwd, "cancer_mn.uai"))


@pytest.fixture
def deer_uai():
    return load_model(os.path.join(cwd, "deer.uai"))


@pytest.fixture
def toulbar2_bn_uai():
    return load_model(os.path.join(cwd, "toulbar2_bn.uai"))


@pytest.fixture
def tiny_bn_orig():
    values = {
        (0, 0): {0: 0.31, 1: 0.69},
        (0, 1): {0: 0.23, 1: 0.77},
        (1, 0): {0: 0.27, 1: 0.73},
        (1, 1): {0: 0.56, 1: 0.44},
    }
    pgm = DiscreteBayesianNetwork(
        states={"node_%i" % i: [0, 1] for i in range(3)},
        cpds=[
            DiscreteCPD(node="node_0", values=[0.4, 0.6]),
            DiscreteCPD(node="node_1", values=[0.3, 0.7]),
            DiscreteCPD(node="node_2", parents=["node_0", "node_1"], values=values),
        ],
    )
    return pgm


@pytest.fixture
def tiny_bn_permuted():
    values = {
        (0, 0): {0: 0.31, 1: 0.69},
        (1, 0): {0: 0.27, 1: 0.73},
        (1, 1): {0: 0.56, 1: 0.44},
        (0, 1): {0: 0.23, 1: 0.77},
    }
    pgm = DiscreteBayesianNetwork(
        states={"node_%i" % i: [0, 1] for i in range(3)},
        cpds=[
            DiscreteCPD(node="node_0", values=[0.4, 0.6]),
            DiscreteCPD(node="node_1", values=[0.3, 0.7]),
            DiscreteCPD(node="node_2", parents=["node_0", "node_1"], values=values),
        ],
    )
    return pgm


@pytest.fixture
def asia_bif():
    return load_model(os.path.join(cwd, "asia.bif"))


#
# errors
#


def Xtest_save_model_error1():
    with pytest.raises(RuntimeError):
        save_model(os.path.join(cwd, "unknown.uai"))


def test_save_model_error2(asia_uai):
    with pytest.raises(InvalidInputError, match="Unknown format"):
        save_model(asia_uai, os.path.join(cwd, "test_save_model.py"))


def test_save_model_error3(asia_uai):
    with pytest.raises(InvalidInputError, match="Unexpected model type"):
        save_model(asia_uai, os.path.join(cwd, "asia.uai"), model_type="unknown")


def test_save_model_documented_io_namespace(asia_uai, tmp_path):
    fname = str(tmp_path / "asia_io_namespace_test.uai")
    conin.io.save_model(asia_uai, fname, model_type="conin")
    loaded = conin.io.load_model(fname, model_type="conin")
    assert loaded.nodes


#
# asia.uai
#


def test_save_model_asia_uai1_uai(asia_uai, tmp_path):
    fname = str(tmp_path / "asia_test.uai")
    save_model(asia_uai, fname)


def test_save_model_asia_uai1_bif(asia_bif, tmp_path):
    fname = str(tmp_path / "asia_test.uai")
    save_model(asia_bif, fname)


#
# barley.uai
#


def test_save_model_barley_uai1_uai(barley_uai, tmp_path):
    fname = str(tmp_path / "barley_test.uai")
    save_model(barley_uai, fname)


#
# cancer_bn.uai
#


def test_save_model_cancer_bn_uai(cancer_bn_uai, tmp_path):
    fname = str(tmp_path / "cancer_bn_test.uai")
    save_model(cancer_bn_uai, fname)

    orig = os.path.join(cwd, "cancer_bn.uai")
    assert filecmp.cmp(orig, fname) == True


#
# cancer_mn.uai
#


def test_save_model_cancer_mn_uai(cancer_mn_uai, tmp_path):
    fname = str(tmp_path / "cancer_mn_test.uai")
    save_model(cancer_mn_uai, fname)

    orig = os.path.join(cwd, "cancer_mn.uai")
    assert filecmp.cmp(orig, fname) == True


#
# deer.uai
#


def test_save_model_deer1_conin(deer_uai, tmp_path):
    fname = str(tmp_path / "deer_test.uai")
    save_model(deer_uai, fname)


#
# toulbar2_bn.uai
#


def test_save_model_toulbar2_conin(toulbar2_bn_uai, tmp_path):
    fname = str(tmp_path / "toulbar2_bn_test.uai")
    save_model(toulbar2_bn_uai, fname)

    orig = os.path.join(cwd, "toulbar2_bn.uai")
    assert filecmp.cmp(orig, fname) == True


#
# tiny_bn_orig and tiny_bn_permuted
#


def test_save_model_tiny_bn(tiny_bn_orig, tiny_bn_permuted, tmp_path):
    fname_orig = str(tmp_path / "tiny_bn_orig.uai")
    save_model(tiny_bn_orig, fname_orig)

    fname_permuted = str(tmp_path / "tiny_bn_permuted.uai")
    save_model(tiny_bn_permuted, fname_permuted)

    assert filecmp.cmp(fname_orig, fname_permuted)


#
# Paths with dots
#


@pytest.mark.parametrize(
    "subdir,name",
    [("a.b", "out.uai"), ("a.b", "out"), ("", "./out.uai"), ("", "out.v1.uai")],
)
def test_save_model_uai_paths_with_dots(asia_uai, tmp_path, monkeypatch, subdir, name):
    # save_uai used to write to name.split(".")[0] + ".uai", so a dot in a
    # directory name or a "./" prefix wrote another file (e.g. "./out.uai" wrote
    # ".uai").
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / subdir
    directory.mkdir(exist_ok=True)
    fname = os.path.join(str(directory), name) if subdir else name
    conin.io.save_uai(asia_uai, fname)
    expected = os.path.splitext(fname)[0] + ".uai"
    assert sorted(os.listdir(directory)) == [os.path.basename(expected)]
    assert load_model(expected).nodes == asia_uai.nodes


#
# BIF
#


def test_save_conin_bn_as_bif(cancer_bn_uai, tmp_path):
    # save_model wrote the .bif file of a CONIN model but did not return, so it
    # then raised "Unknown format specified by filename suffix".
    fname = str(tmp_path / "cancer.bif")
    save_model(cancer_bn_uai, fname)
    loaded = load_model(fname)
    assert sorted(loaded.nodes) == sorted(cancer_bn_uai.nodes)


# ---------------------------------------------------------------------
# save_conin_model: the .bif paths
# ---------------------------------------------------------------------


def test_save_conin_mn_as_bif_raises(cancer_mn_uai, tmp_path):
    with pytest.raises(TypeError, match="BIF.*DiscreteMarkovNetwork"):
        save_model(cancer_mn_uai, str(tmp_path / "cancer.bif"))


def test_save_conin_hmm_as_bif_raises(tmp_path):
    import conin.examples.hidden_markov_model as hex_

    with pytest.raises(TypeError, match="BIF.*HiddenMarkovModel"):
        save_model(hex_.create_hmm1(), str(tmp_path / "hmm.bif"))


@pytest.mark.parametrize("suffix", [".bif", ".bif.gz"])
def test_save_conin_bif(asia_bif, tmp_path, suffix):
    fname = str(tmp_path / f"asia{suffix}")
    save_model(asia_bif, fname)
    loaded = load_model(fname)
    assert loaded.states == asia_bif.states
    assert sorted(loaded.cpds, key=str) == sorted(asia_bif.cpds, key=str)


#
# sparse factors
#


def test_save_model_sparse_factors_round_trip(tmp_path):
    # save_uai raised KeyError for the assignments a factor's dict omits.
    from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork

    pgm = DiscreteMarkovNetwork(
        states={"A": [0, 1, 2], "B": [0, 1, 2]},
        factors=[
            DiscreteFactor(nodes=["A"], values={0: 1}, default_value=3),
            DiscreteFactor(nodes=["A", "B"], values={(1, 0): 5}, default_value=1),
        ],
    )
    fname = str(tmp_path / "sparse.uai")
    save_model(pgm, fname)
    loaded = load_model(fname)
    assert [dict(f.values) for f in loaded.factors] == [
        {0: 1.0, 1: 3.0, 2: 3.0},
        {(a, b): 5.0 if (a, b) == (1, 0) else 1.0 for a in range(3) for b in range(3)},
    ]
