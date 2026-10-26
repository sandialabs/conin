"""The ``"pyomo"`` representation of a (constrained) HMM (``ip_formulation="markov_network"``)."""

from conin.graphical_models import HiddenMarkovModel, create_dbn_from_hmm
from conin.representations.base import register_representation
from conin.representations.unrolled import UnrolledHMMMixin, hmm_time_range

from .common import PyomoFactorRepresentation
from .dbn import build_pyomo_dbn_representation


class PyomoHMMRepresentation(UnrolledHMMMixin, PyomoFactorRepresentation):
    """Factor-repn IP of a constrained HMM unrolled (as a DBN) over ``start..stop``.

    Structure params: ``start`` and ``stop``.  The hidden
    nodes are ``("H", t)`` and the observed nodes ``("E", t)``; evidence must
    be keyed by ``("E", t)``.  Constraint functions see
    ``data = Munch(start, stop, variables, evidence,
    hmm=Munch(T, hidden_to_internal))``.  ``pgm`` is the HMM's DBN.
    """


@register_representation(
    "pyomo", HiddenMarkovModel, families=(None, "pyomo", "algebraic")
)
def _build(cmodel, *, start=0, stop=None):
    start, stop = hmm_time_range(start, stop)
    hmm = cmodel.model
    dbn = create_dbn_from_hmm(hmm=hmm)
    return build_pyomo_dbn_representation(
        dbn,
        cmodel.constraints,
        start=start,
        stop=stop,
        cls=PyomoHMMRepresentation,
        pgm=dbn,
        hidden_to_internal=hmm.hidden_to_internal,
        data=cmodel.data,
    )
