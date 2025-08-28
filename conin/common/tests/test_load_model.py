import pytest
import os.path

from conin.util import try_import
from conin.common import load_model

with try_import() as pgmpy_available:
    import pgmpy

cwd = os.path.dirname(__file__)

#
# asia
#


def test_load_model_asia1_conin():
    pgm = load_model(os.path.join(cwd, "asia.uai"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_asia1_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")


def test_load_model_asia2_conin():
    pgm = load_model(os.path.join(cwd, "asia_compressed.uai.gz"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_asia2_pgmpy():
    pgm = load_model(os.path.join(cwd, "asia_compressed.uai.gz"), model_type="pgmpy")


#
# barley
#


def test_load_model_barley1_conin():
    pgm = load_model(os.path.join(cwd, "barley.uai"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_barley1_pgmpy():
    pgm = load_model(os.path.join(cwd, "barley.uai"), model_type="pgmpy")


def test_load_model_barley2_conin():
    pgm = load_model(os.path.join(cwd, "barley_compressed.uai.gz"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_barley2_pgmpy():
    pgm = load_model(os.path.join(cwd, "barley_compressed.uai.gz"), model_type="pgmpy")


#
# deer
#


def test_load_model_deer1_conin():
    pgm = load_model(os.path.join(cwd, "deer.uai"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_deer1_pgmpy():
    pgm = load_model(os.path.join(cwd, "deer.uai"), model_type="pgmpy")


def test_load_model_deer2_conin():
    pgm = load_model(os.path.join(cwd, "deer_compressed.uai.gz"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_load_model_deer2_pgmpy():
    pgm = load_model(os.path.join(cwd, "deer_compressed.uai.gz"), model_type="pgmpy")
