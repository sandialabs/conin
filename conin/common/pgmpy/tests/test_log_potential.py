import pytest

from conin.util import try_import
from conin.common.pgmpy.log_potential import log_potential

import conin.markov_network.tests.examples
import conin.bayesian_network.tests.examples

with try_import() as pgmpy_available:
    import pgmpy

#
# MarkovNetwork tests
#

@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_log_potential_ABC():
    pgm = conin.markov_network.tests.examples.ABC_pgmpy()
    variables = {"A": 2, "B": 2, "C": 1}
    assert log_potential(pgm, variables) == pytest.approx(2.4849066497880004)

