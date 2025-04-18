# clio.__init__.py

import os.path
from . import config
import importlib
import inspect


imports = ["clio_pybind11"]

__using_pybind11__ = False

for import_ in imports:
    if import_ == "clio_pybind11":
        try:
            import clio_pybind11

            __using_pybind11__ = True
        except ImportError:
            if config.clio_home is not None:
                sys.path.insert(0, os.path.join(config.clio_home, "lib"))
                sys.path.insert(0, os.path.join(config.clio_home, "lib64"))
            try:
                import clio_pybind11

                __using_pybind11__ = True
            except ImportError:
                pass
            pass
        if __using_pybind11__:
            break

from . import __about__

#
# Import clio symbols
#

if __using_pybind11__:
    __doc__ = "pybind11"
    print("<clio %s using clio_lib built with pybind11>" % __about__.__version__)
    # TODO - Add pybind11 specific logic in a module
    # from clio.clio_pybind11 import *

else:
    # raise ImportError("No clio_lib interface installed!")
    print("WARNING: No clio_lib interface installed!")


# File specific
from .exceptions import *
from .constraint import Constraint
from .common_constraints import *
from . import hmm
from . import bayesian_network
from . import dynamic_bayesian_network
from . import markov_network
