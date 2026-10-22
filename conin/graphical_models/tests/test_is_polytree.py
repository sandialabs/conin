import pytest
import os.path

from conin.util import try_import
from conin.graphical_models import is_polytree
from conin.io import load_model
import conin.pgmpy

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pgmpy_readwrite_available:
    import pgmpy.readwrite
    from conin.pgmpy import convert_pgmpy_to_conin


cwd = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "io", "tests"
)

#
# asia.uai
#


def test_is_polytree_conin_asia():
    pgm = load_model(os.path.join(cwd, "asia.uai"))
    assert is_polytree(pgm) == False


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_is_polytree_pgmpy_asia():
    pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")
    assert conin.pgmpy.is_polytree(pgm) == False

    conin_pgm = convert_pgmpy_to_conin(pgm)
    assert is_polytree(conin_pgm) == False

    pgm = load_model(os.path.join(cwd, "asia.bif"), model_type="pgmpy")
    assert conin.pgmpy.is_polytree(pgm) == False

    conin_pgm = convert_pgmpy_to_conin(pgm)
    assert is_polytree(conin_pgm) == False


#
# barley.uai
#


def test_is_polytree_conin_barley():
    pgm = load_model(os.path.join(cwd, "barley.uai"))
    assert is_polytree(pgm) == False


#
# cancer.uai
#


def test_is_polytree_conin_cancer():
    pgm = load_model(os.path.join(cwd, "cancer_bn.uai"))
    assert is_polytree(pgm) == True


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_is_polytree_pgmpy_cancer():
    pgm = load_model(os.path.join(cwd, "cancer_bn.uai"), model_type="pgmpy")
    assert conin.pgmpy.is_polytree(pgm) == True

    conin_pgm = convert_pgmpy_to_conin(pgm)
    assert is_polytree(conin_pgm) == True

    pgm = load_model(os.path.join(cwd, "cancer_bn.bif"), model_type="pgmpy")
    assert conin.pgmpy.is_polytree(pgm) == True

    conin_pgm = convert_pgmpy_to_conin(pgm)
    assert is_polytree(conin_pgm) == True


#
# deer.uai
#


def test_is_polytree_conin_deer():
    with pytest.raises(TypeError):
        # deer.uai defines a Markov Network
        pgm = load_model(os.path.join(cwd, "deer.uai"))
        is_polytree(pgm)
