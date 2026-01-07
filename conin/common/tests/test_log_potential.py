import pytest

from conin.util import try_import
import conin.common.pgmpy
import conin.common.conin
from conin.common.unified import log_potential

import conin.markov_network.examples

pgm = conin.markov_network.examples.ABC_conin()
import conin.bayesian_network.examples

with try_import() as pgmpy_available:
    import pgmpy

#
# MarkovNetwork tests
#


def test_log_potential_ABC_conin():
    pgm = conin.markov_network.examples.ABC_conin().pgm
    variables = {"A": 2, "B": 2, "C": 1}
    assert conin.common.conin.log_potential(pgm, variables) == pytest.approx(
        2.4849066497880004
    )
    assert log_potential(pgm, variables) == pytest.approx(2.4849066497880004)


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_log_potential_ABC_pgmpy():
    pgm = conin.markov_network.examples.ABC_pgmpy().pgm
    variables = {"A": 2, "B": 2, "C": 1}
    assert conin.common.pgmpy.log_potential(pgm, variables) == pytest.approx(
        2.4849066497880004
    )
    assert log_potential(pgm, variables) == pytest.approx(2.4849066497880004)
