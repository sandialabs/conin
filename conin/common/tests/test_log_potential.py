import pytest

from conin.util import try_import
import conin.common.pgmpy
from conin.common import log_potential
from conin.common.log_potential import log_potential_conin

import conin.examples.markov_network

pgm = conin.examples.markov_network.ABC_conin()
import conin.examples.bayesian_network

with try_import() as pgmpy_available:
    import pgmpy

#
# MarkovNetwork tests
#


def test_log_potential_ABC_conin():
    pgm = conin.examples.markov_network.ABC_conin().pgm
    variables = {"A": 2, "B": 2, "C": 1}
    assert log_potential_conin(pgm, variables) == pytest.approx(2.4849066497880004)
    assert log_potential(pgm, variables) == pytest.approx(2.4849066497880004)


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_log_potential_ABC_pgmpy():
    pgm = conin.examples.markov_network.ABC_pgmpy().pgm
    variables = {"A": 2, "B": 2, "C": 1}
    assert conin.common.pgmpy.log_potential(pgm, variables) == pytest.approx(
        2.4849066497880004
    )
    assert log_potential(pgm, variables) == pytest.approx(2.4849066497880004)
