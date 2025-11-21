import pytest
import os.path

from conin.util import try_import
from conin.common import load_model

with try_import() as pgmpy_available:
    import pgmpy
    from conin.common.pgmpy import convert_pgmpy_to_conin


cwd = os.path.dirname(__file__)

#
# cancer.uai
#

def test_convert_from_pgmpy_cancer():
    pgmpy_pgm = load_model(os.path.join(cwd, "cancer.uai"), model_type="pgmpy")
    pgm = convert_pgmpy_to_conin(pgmpy_pgm)

    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    assert cpds["var_0"].values == {
        (0, 0): {0: 0.03, 1: 0.97},
        (1, 0): {0: 0.05, 1: 0.95},
        (0, 1): {0: 0.001, 1: 0.999},
        (1, 1): {0: 0.02, 1: 0.98},
    }
    assert cpds["var_1"].values == {0: {0: 0.65, 1: 0.35}, 1: {0: 0.3, 1: 0.7}}
    assert cpds["var_2"].values == {0: 0.9, 1: 0.1}
    assert cpds["var_3"].values == {0: 0.3, 1: 0.7}
    assert cpds["var_4"].values == {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.8}}
