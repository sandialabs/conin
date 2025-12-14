import pytest
import os.path

from conin.util import try_import
from conin.common import save_model, load_model

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pgmpy_readwrite_available:
    import pgmpy.readwrite

cwd = os.path.dirname(__file__)


@pytest.fixture
def asia_uai():
    return load_model(os.path.join(cwd, "asia.uai"))


@pytest.fixture
def deer_uai():
    return load_model(os.path.join(cwd, "deer.uai"))


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


def Xtest_save_model_error2():
    with pytest.raises(RuntimeError):
        pgm = save_model(os.path.join(cwd, "test_save_model.py"))


def Xtest_save_model_error3():
    with pytest.raises(RuntimeError):
        pgm = save_model(os.path.join(cwd, "asia.uai"), model_type="unknown")


@pytest.mark.skipif(
    pgmpy_available, reason="Testing an error when pgmpy is not installed"
)
def Xtest_save_model_error4():
    with pytest.raises(ImportError):
        pgm = save_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")


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


def Xtest_save_model_asia_uai2_uai(asia_uai):
    pgm = save_model(os.path.join(cwd, "asia_compressed_test.uai.gz"))


#
# barley.uai
#


def Xtest_save_model_barley1_conin():
    pgm = save_model(os.path.join(cwd, "barley.uai"))


def Xtest_save_model_barley2_conin():
    pgm = save_model(os.path.join(cwd, "barley_compressed.uai.gz"))


#
# deer.uai
#


def Xtest_save_model_deer1_conin():
    pgm = save_model(os.path.join(cwd, "deer.uai"))


def Xtest_save_model_deer2_conin():
    pgm = save_model(os.path.join(cwd, "deer_compressed.uai.gz"))
