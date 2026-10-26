"""The ``"toulbar2"`` representation of a (constrained) HMM."""

from conin.graphical_models import HiddenMarkovModel, create_dbn_from_hmm
from conin.representations.base import register_representation
from conin.representations.unrolled import UnrolledHMMMixin, hmm_time_range

from .common import Toulbar2FactorRepresentation
from .dbn import build_toulbar2_dbn_representation


class Toulbar2HMMRepresentation(UnrolledHMMMixin, Toulbar2FactorRepresentation):
    """toulbar2 CFN of a constrained HMM unrolled (as a DBN) over ``start..stop``.

    Structure params: ``start`` and ``stop``.  The hidden
    nodes are ``("H", t)`` and the observed nodes ``("E", t)``; evidence must
    be keyed by ``("E", t)``.  Constraint functions see
    ``data = Munch(start, stop, variables, evidence,
    hmm=Munch(T, hidden_to_internal))``.  ``pgm`` is the HMM's DBN and
    ``uai_model`` its unrolled BN.
    """


@register_representation(
    "toulbar2", HiddenMarkovModel, families=(None, "toulbar2", "algebraic")
)
def _build(cmodel, *, start=0, stop=None):
    start, stop = hmm_time_range(start, stop)
    hmm = cmodel.model
    dbn = create_dbn_from_hmm(hmm=hmm)
    return build_toulbar2_dbn_representation(
        dbn,
        cmodel.constraints,
        start=start,
        stop=stop,
        cls=Toulbar2HMMRepresentation,
        pgm=dbn,
        hidden_to_internal=hmm.hidden_to_internal,
        data=cmodel.data,
    )
