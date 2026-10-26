import pytest
import os.path
import filecmp

from conin.util import try_import
import conin.io
from conin.io import save_model, load_model
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pgmpy_readwrite_available:
    import pgmpy.readwrite

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
        cpds=[DiscreteCPD(node="node_2", parents=["node_0", "node_1"], values=values)],
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
        cpds=[DiscreteCPD(node="node_2", parents=["node_0", "node_1"], values=values)],
    )
    return pgm


if pgmpy_readwrite_available:

    @pytest.fixture
    def asia_bif():
        return load_model(os.path.join(cwd, "asia.bif"))

    @pytest.fixture
    def deer_bif():
        return load_model(os.path.join(cwd, "deer.bif"))


#
# errors
#


def Xtest_save_model_error1():
    with pytest.raises(RuntimeError):
        pgm = save_model(os.path.join(cwd, "unknown.uai"))


def test_save_model_error2(asia_uai):
    with pytest.raises(RuntimeError):
        pgm = save_model(asia_uai, os.path.join(cwd, "test_save_model.py"))


def test_save_model_error3(asia_uai):
    with pytest.raises(RuntimeError):
        pgm = save_model(asia_uai, os.path.join(cwd, "asia.uai"), model_type="unknown")


def test_save_model_documented_io_namespace(asia_uai):
    fname = os.path.join(cwd, "asia_io_namespace_test.uai")
    conin.io.save_model(asia_uai, fname, model_type="conin")
    try:
        loaded = conin.io.load_model(fname, model_type="conin")
        assert loaded.nodes
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.skipif(
    pgmpy_available, reason="Testing an error when pgmpy is not installed"
)
def test_save_model_error4(asia_uai):
    with pytest.raises(ImportError):
        pgm = save_model(asia_uai, os.path.join(cwd, "asia.uai"), model_type="pgmpy")


#
# asia.uai
#


def test_save_model_asia_uai1_uai(asia_uai):
    fname = os.path.join(cwd, "asia_test.uai")
    pgm = save_model(asia_uai, fname)
    os.remove(fname)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_save_model_asia_uai1_bif(asia_bif):
    fname = os.path.join(cwd, "asia_test.uai")
    pgm = save_model(asia_bif, fname)
    os.remove(fname)


#
# barley.uai
#


def test_save_model_barley_uai1_uai(barley_uai):
    fname = os.path.join(cwd, "barley_test.uai")
    pgm = save_model(barley_uai, fname)
    os.remove(fname)


#
# cancer_bn.uai
#


def test_save_model_cancer_bn_uai(cancer_bn_uai):
    fname = os.path.join(cwd, "cancer_bn_test.uai")
    pgm = save_model(cancer_bn_uai, fname)

    orig = os.path.join(cwd, "cancer_bn.uai")
    assert filecmp.cmp(orig, fname) == True
    os.remove(fname)


#
# cancer_mn.uai
#


def test_save_model_cancer_mn_uai(cancer_mn_uai):
    fname = os.path.join(cwd, "cancer_mn_test.uai")
    pgm = save_model(cancer_mn_uai, fname)

    orig = os.path.join(cwd, "cancer_mn.uai")
    assert filecmp.cmp(orig, fname) == True
    os.remove(fname)


#
# deer.uai
#


def test_save_model_deer1_conin(deer_uai):
    fname = os.path.join(cwd, "deer_test.uai")
    pgm = save_model(deer_uai, fname)
    os.remove(fname)


#
# toulbar2_bn.uai
#


def test_save_model_toulbar2_conin(toulbar2_bn_uai):
    fname = os.path.join(cwd, "toulbar2_bn_test.uai")
    pgm = save_model(toulbar2_bn_uai, fname)

    orig = os.path.join(cwd, "toulbar2_bn.uai")
    assert filecmp.cmp(orig, fname) == True
    os.remove(fname)


#
# tiny_bn_orig and tiny_bn_permuted
#


def test_save_model_tiny_bn(tiny_bn_orig, tiny_bn_permuted):
    fname_orig = os.path.join(cwd, "tiny_bn_orig.uai")
    pgm_orig = save_model(tiny_bn_orig, fname_orig)

    fname_permuted = os.path.join(cwd, "tiny_bn_permuted.uai")
    pgm_permuted = save_model(tiny_bn_permuted, fname_permuted)

    assert filecmp.cmp(fname_orig, fname_permuted)
    os.remove(fname_orig)
    os.remove(fname_permuted)


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


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_save_conin_bn_as_bif(cancer_bn_uai, tmp_path):
    # save_model wrote the .bif file of a CONIN model but did not return, so it
    # then raised "Unknown format specified by filename suffix".
    fname = str(tmp_path / "cancer.bif")
    save_model(cancer_bn_uai, fname)
    loaded = load_model(fname)
    assert sorted(loaded.nodes) == sorted(cancer_bn_uai.nodes)


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy is not installed")
def test_pgmpy_save_model_unknown_suffix_raises(tmp_path):
    # It used to return without writing anything.
    pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")
    with pytest.raises(RuntimeError, match="Unknown format.*model.txt"):
        save_model(pgm, str(tmp_path / "model.txt"), model_type="pgmpy")


# ---------------------------------------------------------------------
# save_conin_model: the .bif paths and the pgmpy checks
# ---------------------------------------------------------------------


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_save_conin_mn_as_bif_is_converted_by_pgmpy(
    cancer_mn_uai, tmp_path, monkeypatch
):
    # A Markov network goes through convert_conin_to_pgmpy_mn (pgmpy's
    # BIFWriter then decides what it can write).  cancer_mn.uai has a factor
    # over three nodes: the UAI reader gave it the wrong edges, so pgmpy
    # rejected the converted model ("DiscreteFactor inconsistent with the
    # model").
    import importlib

    pgmpy_files = importlib.import_module("conin.io.pgmpy_files")
    calls = []
    monkeypatch.setattr(
        pgmpy_files,
        "save_model",
        lambda pgm, name, quiet: calls.append((type(pgm).__name__, name)),
    )
    save_model(cancer_mn_uai, str(tmp_path / "cancer.bif"))
    assert calls == [("DiscreteMarkovNetwork", str(tmp_path / "cancer.bif"))]


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy is not installed")
def test_save_conin_hmm_as_bif_raises(tmp_path):
    import conin.examples.hidden_markov_model as hex_

    with pytest.raises(RuntimeError, match="Unexpected conin model type"):
        save_model(hex_.create_hmm1(), str(tmp_path / "hmm.bif"))


def test_save_conin_bif_without_pgmpy_raises(asia_uai, tmp_path, monkeypatch):
    import importlib

    save_module = importlib.import_module("conin.io.save_model")
    monkeypatch.setattr(save_module, "pgmpy_available", False)
    with pytest.raises(RuntimeError, match="Unknown format.*asia.bif"):
        save_model(asia_uai, str(tmp_path / "asia.bif"))


def test_save_pgmpy_model_without_pgmpy_raises(asia_uai, tmp_path, monkeypatch):
    import importlib

    save_module = importlib.import_module("conin.io.save_model")
    monkeypatch.setattr(save_module, "pgmpy_available", False)
    with pytest.raises(ImportError, match="pgmpy"):
        save_model(asia_uai, str(tmp_path / "asia.uai"), model_type="pgmpy")


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy is not installed")
@pytest.mark.parametrize("suffix", ["bif", "uai"])
def test_pgmpy_save_model_without_readwrite_raises(tmp_path, monkeypatch, suffix):
    import importlib

    module = importlib.import_module("conin.io.pgmpy_files")
    pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")
    monkeypatch.setattr(module, "pgmpy_readwrite_available", False)
    with pytest.raises(RuntimeError, match=f"{suffix.upper()} format"):
        module.save_model(pgm, str(tmp_path / f"asia.{suffix}"))
