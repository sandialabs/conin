import pytest
import os.path

from conin.graphical_models import is_polytree
from conin.io import load_model
from conin.tests.optional import pgmpy_installed

pgmpy_readwrite_available = pgmpy_installed


cwd = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "io", "tests"
)

#
# asia.uai
#


def test_is_polytree_conin_asia():
    pgm = load_model(os.path.join(cwd, "asia.uai"))
    assert not is_polytree(pgm)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_is_polytree_bif_asia():
    pgm = load_model(os.path.join(cwd, "asia.bif"))
    assert not is_polytree(pgm)


#
# barley.uai
#


def test_is_polytree_conin_barley():
    pgm = load_model(os.path.join(cwd, "barley.uai"))
    assert not is_polytree(pgm)


#
# cancer.uai
#


def test_is_polytree_conin_cancer():
    pgm = load_model(os.path.join(cwd, "cancer_bn.uai"))
    assert is_polytree(pgm)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_is_polytree_bif_cancer():
    pgm = load_model(os.path.join(cwd, "cancer_bn.bif"))
    assert is_polytree(pgm)


#
# deer.uai
#


def test_is_polytree_conin_deer():
    with pytest.raises(TypeError):
        # deer.uai defines a Markov Network
        pgm = load_model(os.path.join(cwd, "deer.uai"))
        is_polytree(pgm)
