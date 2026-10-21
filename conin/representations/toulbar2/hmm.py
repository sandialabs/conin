"""The ``"toulbar2"`` representation of a (constrained) HMM."""

import munch

from conin.graphical_models import HiddenMarkovModel, create_dbn_from_hmm
from conin.representations.base import register_representation

from .common import Toulbar2FactorRepresentation
from .dbn import build_toulbar2_dbn_representation


class Toulbar2HMMRepresentation(Toulbar2FactorRepresentation):
    """toulbar2 CFN of a constrained HMM unrolled (as a DBN) over ``start..stop``.

    Structure params: ``start`` and ``stop``.  The hidden
    nodes are ``("H", t)`` and the observed nodes ``("E", t)``; evidence must
    be keyed by ``("E", t)``.  Constraint functions see
    ``data = Munch(start, stop, variables, evidence,
    hmm=Munch(T, hidden_to_internal))``.  ``pgm`` is the HMM's DBN and
    ``uai_model`` its unrolled BN.
    """

    def __init__(self, *, start, stop, hidden_to_internal, **kwds):
        super().__init__(**kwds)
        self.start = start
        self.stop = stop
        self.hidden_to_internal = hidden_to_internal

    def constraint_data(self, *, evidence=None, variables=None):
        return munch.Munch(
            start=self.start,
            stop=self.stop,
            variables=variables,
            evidence=evidence,
            hmm=munch.Munch(
                T=list(range(self.start, self.stop + 1)),
                hidden_to_internal=dict(self.hidden_to_internal),
            ),
        )


@register_representation(
    "toulbar2", HiddenMarkovModel, families=(None, "toulbar2", "algebraic")
)
def _build(cmodel, *, start=0, stop=None):
    assert stop is not None, "Expecting evidence (observations) for an HMM"
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
