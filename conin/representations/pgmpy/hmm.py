"""The ``"pgmpy"`` representation of a (constrained) HMM."""

import munch

from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel, create_dbn_from_hmm
from conin.representations.base import register_representation

from .dbn import PgmpyDBNRepresentation, build_pgmpy_dbn_representation


class PgmpyHMMRepresentation(PgmpyDBNRepresentation):
    """Factor model of a constrained HMM unrolled (as a DBN) over ``start..stop``.

    Structure params: ``start`` and ``stop``.  The hidden nodes are
    ``("H", t)`` and the observed nodes ``("E", t)``; evidence must be keyed by
    ``("E", t)``.  ``pgm`` is the unrolled BN.  Constraint functions see
    ``data = Munch(hmm=Munch(T))`` with ``T = list(range(start, stop + 1))``.
    """

    def constraint_data(self):
        return munch.Munch(hmm=munch.Munch(T=list(range(self.start, self.stop + 1))))


@register_representation(
    "pgmpy", HiddenMarkovModel, families=(None, "factor_oracle", "algebraic")
)
def _build(cmodel, *, start=0, stop=None):
    if stop is None:
        raise InvalidInputError(
            "The HMM representation needs a stop time (the last observation)"
        )
    dbn = create_dbn_from_hmm(cmodel.model)
    return build_pgmpy_dbn_representation(
        dbn,
        cmodel.constraints,
        start=start,
        stop=stop,
        cls=PgmpyHMMRepresentation,
        data=cmodel.data,
    )
