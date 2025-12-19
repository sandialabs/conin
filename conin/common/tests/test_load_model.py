import pytest
import os.path

from conin.util import try_import
from conin.common.unified import load_model

with try_import() as pgmpy_available:
    import pgmpy

#
# Note that pgmpy is needed to read BIF files by conin.
#
# These tests check if pgmpy.readwrite is available, which isn't true of all
# installations.
#
with try_import() as pgmpy_readwrite_available:
    import pgmpy.readwrite

with try_import() as pomegranate_available:
    import pomegranate

cwd = os.path.dirname(__file__)

#
# errors
#


def test_load_model_error1():
    with pytest.raises(RuntimeError):
        pgm = load_model(os.path.join(cwd, "unknown.uai"))


def test_load_model_error2():
    with pytest.raises(RuntimeError):
        pgm = load_model(os.path.join(cwd, "test_load_model.py"))


def test_load_model_error3():
    with pytest.raises(RuntimeError):
        pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="unknown")


@pytest.mark.skipif(
    pgmpy_available, reason="Testing an error when pgmpy is not installed"
)
def test_load_model_error4():
    with pytest.raises(ImportError):
        pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")


#
# asia.uai
#


def test_load_model_asia_uai1_conin():
    pgm = load_model(os.path.join(cwd, "asia.uai"))


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_load_model_asia_uai1_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")


@pytest.mark.skipif(not pomegranate_available, reason="pomegranate not installed")
def test_load_model_asia_uai1_pomegranate():
    pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pomegranate")


def test_load_model_asia_uai2_conin():
    pgm = load_model(os.path.join(cwd, "asia_compressed.uai.gz"))


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_load_model_asia_uai2_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia_compressed.uai.gz"), model_type="pgmpy")


@pytest.mark.skipif(not pomegranate_available, reason="pomegranate not installed")
def test_load_model_asia_uai2_pomegranate():
    pgm = load_model(
        os.path.join(cwd, "asia_compressed.uai.gz"), model_type="pomegranate"
    )


#
# asia.bif
#


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_load_model_asia_bif1_conin():
    pgm = load_model(os.path.join(cwd, "asia.bif"))


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_load_model_asia_bif1_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia.bif"), model_type="pgmpy")


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_load_model_asia_bif2_conin():
    pgm = load_model(os.path.join(cwd, "asia_compressed.bif.gz"))


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_load_model_asia_bif2_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia_compressed.bif.gz"), model_type="pgmpy")


#
# barley.uai
#


def test_load_model_barley1_conin():
    pgm = load_model(os.path.join(cwd, "barley.uai"))


def test_load_model_barley2_conin():
    pgm = load_model(os.path.join(cwd, "barley_compressed.uai.gz"))


#
# deer.uai
#


def test_load_model_deer1_conin():
    pgm = load_model(os.path.join(cwd, "deer.uai"))


def test_load_model_deer2_conin():
    pgm = load_model(os.path.join(cwd, "deer_compressed.uai.gz"))


#
# cancer
#


def test_load_bn_model_cancer_conin():
    pgm = load_model(os.path.join(cwd, "cancer_bn.uai"))
    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    assert cpds["var0"].values == {
        (0, 0): {0: 0.03, 1: 0.97},
        (1, 0): {0: 0.05, 1: 0.95},
        (0, 1): {0: 0.001, 1: 0.999},
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
        (0, 1): 0.3,
        (1, 0): 0.35,
        (1, 1): 0.7,
    }
    assert factors["var2"] == {0: 0.9, 1: 0.1}
    assert factors["var3"] == {0: 0.3, 1: 0.7}
    assert factors["var0", "var4"] == {
        (0, 0): 0.9,
        (0, 1): 0.2,
        (1, 0): 0.1,
        (1, 1): 0.8,
    }
    assert factors["var2", "var3", "var0"] == {
        (0, 0, 0): 0.03,
        (0, 0, 1): 0.05,
        (0, 1, 0): 0.001,
        (0, 1, 1): 0.02,
        (1, 0, 0): 0.97,
        (1, 0, 1): 0.95,
        (1, 1, 0): 0.999,
        (1, 1, 1): 0.98,
    }
