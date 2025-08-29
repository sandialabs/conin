import pytest
import os.path

from conin.util import try_import
from conin.common import load_model

with try_import() as pgmpy_available:
    import pgmpy

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
    with pytest.raises(RuntimeError):
        pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")


#
# asia.uai
#


def test_load_model_asia_uai1_conin():
    pgm = load_model(os.path.join(cwd, "asia.uai"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_asia_uai1_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")


def test_load_model_asia_uai2_conin():
    pgm = load_model(os.path.join(cwd, "asia_compressed.uai.gz"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_asia_uai2_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia_compressed.uai.gz"), model_type="pgmpy")


#
# asia.bif
#
# Note that pgmpy is needed to read BIF files by conin.
#


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_asia_bif1_conin():
    pgm = load_model(os.path.join(cwd, "asia.bif"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_asia_bif1_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia.bif"), model_type="pgmpy")


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_asia_bif2_conin():
    pgm = load_model(os.path.join(cwd, "asia_compressed.bif.gz"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
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
