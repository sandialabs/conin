"""Normalise the model argument of a query to a constrained CONIN model.

Internal: the query entry points (e.g. ``map_query``) call
:func:`as_constrained_model` first, so that every query runs on a constrained
model (a bare model is wrapped with no constraints).
"""

from conin.util import try_import

from .base import ConstrainedGraphicalModel
from .bayesian_network import ConstrainedDiscreteBayesianNetwork
from .dynamic_bayesian_network import ConstrainedDynamicDiscreteBayesianNetwork
from .hidden_markov_model import ConstrainedHiddenMarkovModel
from .markov_network import ConstrainedDiscreteMarkovNetwork

with try_import() as pgmpy_available:
    import pgmpy.models
    from conin.common.pgmpy import convert_pgmpy_to_conin

CONSTRAINED_CLASSES = (
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)

PGMPY_CLASSES = (
    (
        pgmpy.models.DiscreteMarkovNetwork,
        pgmpy.models.DiscreteBayesianNetwork,
        pgmpy.models.DynamicBayesianNetwork,
    )
    if pgmpy_available
    else ()
)


def as_constrained_model(pgm):
    """Return ``pgm`` as a constrained CONIN model.

    A constrained model is returned as it is, pgmpy models are converted to
    CONIN models, and bare CONIN models are wrapped with no constraints.

    Raises
    ------
    TypeError
        If ``pgm`` is not a supported model.
    """
    if isinstance(pgm, ConstrainedGraphicalModel):
        return pgm
    if isinstance(pgm, PGMPY_CLASSES):
        pgm = convert_pgmpy_to_conin(pgm)
    for cls in CONSTRAINED_CLASSES:
        if isinstance(pgm, cls.model_type):
            return cls(pgm)
    expected = []
    for cls in CONSTRAINED_CLASSES:
        expected += [cls.model_type.__name__, cls.__name__]
    expected += [f"pgmpy.models.{cls.__name__}" for cls in PGMPY_CLASSES]
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. Expected one of: {', '.join(expected)}."
    )
