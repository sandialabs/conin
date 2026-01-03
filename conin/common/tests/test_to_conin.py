import pytest
import os.path

from conin.util import try_import
from conin.common.unified import load_model

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pgmpy_readwrite_available:
    import pgmpy.readwrite
    from conin.common.pgmpy import convert_pgmpy_to_conin


cwd = os.path.dirname(__file__)

#
# cancer
#


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_convert_bn_from_pgmpy_cancer():
    pgmpy_pgm = load_model(os.path.join(cwd, "cancer_bn.uai"), model_type="pgmpy")
    pgmpy_pgm.check_model()
    pgm = convert_pgmpy_to_conin(pgmpy_pgm)

    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    assert cpds["var_0"].values == {
        (0, 0): {0: 0.03, 1: 0.97},
        (0, 1): {0: 0.05, 1: 0.95},
        (1, 0): {0: 0.001, 1: 0.999},
        (1, 1): {0: 0.02, 1: 0.98},
    }
    assert cpds["var_1"].values == {0: {0: 0.65, 1: 0.35}, 1: {0: 0.3, 1: 0.7}}
    assert cpds["var_2"].values == {0: 0.9, 1: 0.1}
    assert cpds["var_3"].values == {0: 0.3, 1: 0.7}
    assert cpds["var_4"].values == {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.8}}


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
# toulbar2
#


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_convert_bn_from_pgmpy_toulbar2():
    pgmpy_pgm = load_model(os.path.join(cwd, "toulbar2_bn.uai"), model_type="pgmpy")
    pgmpy_pgm.check_model()
    pgm = convert_pgmpy_to_conin(pgmpy_pgm)

    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    assert cpds["var_0"].values == {
        (0, 0): {0: 0.03, 1: 0.97},
        (0, 1): {0: 0.05, 1: 0.95},
        (1, 0): {0: 0.001, 1: 0.999},
        (1, 1): {0: 0.02, 1: 0.98},
    }
    assert cpds["var_1"].values == {0: {0: 0.65, 1: 0.35}, 1: {0: 0.3, 1: 0.7}}
    assert cpds["var_2"].values == {0: 0.9, 1: 0.1}
    assert cpds["var_3"].values == {0: 0.3, 1: 0.7}
    assert cpds["var_4"].values == {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.8}}
