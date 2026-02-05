import pytest
import os.path
import filecmp

from conin.util import try_import
from conin.common.unified import save_model, load_model
from conin.bayesian_network import DiscreteBayesianNetwork, DiscreteCPD

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
    pgm = DiscreteBayesianNetwork()
    pgm.states = {'node_%i'%i:[0,1] for i in range(3)}
    values = {
        (0,0): {0:.31, 1:.69},
        (0,1): {0:.23, 1:.77},
        (1,0): {0:.27, 1:.73},
        (1,1): {0:.56, 1:.44}
    }
    pgm.cpds = []    
    pgm.cpds.append(DiscreteCPD(node='node_2', parents=['node_0', 'node_1'], values=values))
    return pgm
    

@pytest.fixture
def tiny_bn_permuted():
    pgm = DiscreteBayesianNetwork()
    pgm.states = {'node_%i'%i:[0,1] for i in range(3)}
    values = {
        (0,0): {0:.31, 1:.69},
        (1,0): {0:.27, 1:.73},
        (1,1): {0:.56, 1:.44},
        (0,1): {0:.23, 1:.77}
    }
    pgm.cpds = []    
    pgm.cpds.append(DiscreteCPD(node='node_2', parents=['node_0', 'node_1'], values=values))
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
    fname_orig = os.path.join(cwd, 'tiny_bn_orig.uai')
    pgm_orig = save_model(tiny_bn_orig, fname_orig)

    fname_permuted = os.path.join(cwd, 'tiny_bn_permuted.uai')
    pgm_permuted = save_model(tiny_bn_permuted, fname_permuted)

    assert filecmp.cmp(fname_orig, fname_permuted)
    os.remove(fname_orig)
    os.remove(fname_permuted)