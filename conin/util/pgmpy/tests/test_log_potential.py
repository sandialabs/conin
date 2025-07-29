import pytest

from conin.util.pgmpy.log_potential import log_potential

import conin.markov_network.tests.examples
import conin.bayesian_network.tests.examples

try:
    import pgmpy
    pgmpy_available = True
except Exception as e:
    pgmpy_available = False

#
# MarkovNetwork tests
#

@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_log_potential_ABC():
    pgm = conin.markov_network.tests.examples.ABC()
    variables = {"A": 2, "B": 2, "C": 1}
    assert log_potential(pgm, variables) == pytest.approx(2.4849066497880004)

