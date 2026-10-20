"""The ``"factor"`` representation of a (constrained) HMM."""

import munch

from conin.graphical_models import HiddenMarkovModel, create_dbn_from_hmm
from conin.representations.base import register_representation

from .dbn import FactorDBNRepresentation, build_factor_dbn_representation


class FactorHMMRepresentation(FactorDBNRepresentation):
    """Factor model of a constrained HMM unrolled (as a DBN) over ``start..stop``.

    Structure params: ``start`` and ``stop``.  The hidden nodes are
    ``("H", t)`` and the observed nodes ``("E", t)``; evidence must be keyed by
    ``("E", t)``.  ``pgm`` is the unrolled BN.  Constraint functions see
    ``data = Munch(hmm=Munch(T))`` with ``T = list(range(start, stop + 1))``.
    """

    def constraint_data(self):
        return munch.Munch(hmm=munch.Munch(T=list(range(self.start, self.stop + 1))))


@register_representation(
    "factor", HiddenMarkovModel, families=(None, "factor_oracle", "algebraic")
)
def _build(cmodel, *, start=0, stop=None):
    assert stop is not None, "Expecting evidence (observations) for an HMM"
    dbn = create_dbn_from_hmm(cmodel.model)
    return build_factor_dbn_representation(
        dbn,
        cmodel.constraints,
        start=start,
        stop=stop,
        cls=FactorHMMRepresentation,
    )
