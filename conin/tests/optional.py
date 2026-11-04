"""Skip markers for the optional dependencies, decided without importing them.

pytest-xdist imports every test module in every worker, and importing pgmpy
(with pandas, scipy and scikit-learn) or torch takes seconds, so a test
module should not import them at module level only to decide whether to
skip its tests: these markers just look the packages up
(``importlib.util.find_spec``).  The tests import what they use.  (A
package that is installed but broken is found, so its tests fail instead of
being skipped.)
"""

import importlib.util

import pytest


def installed(name):
    """Whether the top-level package ``name`` is installed (without importing it)."""
    return importlib.util.find_spec(name) is not None


pgmpy_installed = installed("pgmpy")
torch_installed = installed("torch")

needs_pgmpy = pytest.mark.skipif(not pgmpy_installed, reason="pgmpy not installed")
needs_torch = pytest.mark.skipif(not torch_installed, reason="torch not installed")
