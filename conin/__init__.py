# conin.__init__.py

import os.path
from . import config
import importlib
import inspect
import sys


imports = ["conin_pybind11"]

__using_pybind11__ = False

for import_ in imports:
    if import_ == "conin_pybind11":
        try:
            import conin_pybind11

            __using_pybind11__ = True
        except ImportError:
            if config.conin_home is not None:
                sys.path.insert(0, os.path.join(config.conin_home, "lib"))
                sys.path.insert(0, os.path.join(config.conin_home, "lib64"))
            try:
                import conin_pybind11

                __using_pybind11__ = True
            except ImportError:
                pass
            pass
        if __using_pybind11__:
            break

from . import __about__

#
# Import conin symbols
#

if __using_pybind11__:
    __doc__ = "pybind11"
    print("<conin %s using conin_lib built with pybind11>" % __about__.__version__)
    # TODO - Add pybind11 specific logic in a module
    # from conin.conin_pybind11 import *

else:
    # Don't generate a warning, since this import is now optional
    # raise ImportError("No conin_lib interface installed!")
    # print("WARNING: No clio_lib interface installed!")
    pass


# File specific
from .exceptions import InvalidInputError, InsufficientSolutionsError
from .constraint import Constraint
from . import hmm
from . import bayesian_network
from . import dynamic_bayesian_network
from . import markov_network
from .common_constraints import *
from .inference import *
