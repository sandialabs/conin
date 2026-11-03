"""conin: constrained inference on discrete probabilistic graphical models.

The names in ``__all__`` are the public API for everyday use: the model
classes, the constraint helpers, inference, scoring and the feasibility
check, model files and the exceptions.  The rest is in the subpackages
(:mod:`conin.graphical_models`, :mod:`conin.constrained_models`,
:mod:`conin.constraints`, :mod:`conin.algorithms`, :mod:`conin.io`,
:mod:`conin.representations` and :mod:`conin.examples`).
"""

__version__ = "2.0.0"

from . import __about__
from .exceptions import InsufficientSolutionsError, InvalidInputError, ModelFileError
from .graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    HiddenMarkovModel,
)
from .constraints import (
    Constraint,
    mvr_constraint_fn,
    oracle_constraint_fn,
    pyomo_constraint_fn,
    smoek_constraint_fn,
    toulbar2_constraint_fn,
)
from .constrained_models import (
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from .io import load_model, save_model
from .algorithms.feasibility import is_feasible
from .algorithms.scoring import log_potential, log_probability, log_probability_query
from .algorithms.inference import map_query
from . import algorithms, config, representations

__all__ = [
    # Models
    "DiscreteFactor",
    "DiscreteMarkovNetwork",
    "DiscreteCPD",
    "DiscreteBayesianNetwork",
    "DynamicDiscreteBayesianNetwork",
    "HiddenMarkovModel",
    # Constrained models
    "ConstrainedDiscreteMarkovNetwork",
    "ConstrainedDiscreteBayesianNetwork",
    "ConstrainedDynamicDiscreteBayesianNetwork",
    "ConstrainedHiddenMarkovModel",
    # Constraints
    "Constraint",
    "mvr_constraint_fn",
    "oracle_constraint_fn",
    "pyomo_constraint_fn",
    "smoek_constraint_fn",
    "toulbar2_constraint_fn",
    # Inference, scoring and feasibility
    "map_query",
    "log_probability_query",
    "log_probability",
    "log_potential",
    "is_feasible",
    # Model files
    "load_model",
    "save_model",
    # Exceptions
    "InvalidInputError",
    "InsufficientSolutionsError",
    "ModelFileError",
]
