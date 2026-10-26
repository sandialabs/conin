"""The ``"pgmpy"`` representation of a (constrained) HMM."""

from conin.graphical_models import HiddenMarkovModel, create_dbn_from_hmm
from conin.representations.base import register_representation
from conin.representations.unrolled import UnrolledHMMMixin, hmm_time_range

from .common import PgmpyRepresentation
from .dbn import build_pgmpy_dbn_representation


class PgmpyHMMRepresentation(UnrolledHMMMixin, PgmpyRepresentation):
    """Factor model of a constrained HMM unrolled (as a DBN) over ``start..stop``.

    Structure params: ``start`` and ``stop``.  The hidden nodes are
    ``("H", t)`` and the observed nodes ``("E", t)``; evidence must be keyed by
    ``("E", t)``.  ``pgm`` is the unrolled BN.  Constraint functions see
    ``data = Munch(start, stop, hmm=Munch(T, hidden_to_internal))`` with
    ``T = list(range(start, stop + 1))``.
    """


@register_representation(
    "pgmpy", HiddenMarkovModel, families=(None, "factor_oracle", "algebraic")
)
def _build(cmodel, *, start=0, stop=None):
    start, stop = hmm_time_range(start, stop)
    hmm = cmodel.model
    return build_pgmpy_dbn_representation(
        create_dbn_from_hmm(hmm=hmm),
        cmodel.constraints,
        start=start,
        stop=stop,
        cls=PgmpyHMMRepresentation,
        hidden_to_internal=hmm.hidden_to_internal,
        data=cmodel.data,
    )
