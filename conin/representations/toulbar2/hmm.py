"""The ``"toulbar2"`` representation of a (constrained) HMM."""

from conin.graphical_models import HiddenMarkovModel, create_dbn_from_hmm
from conin.representations.base import register_representation
from conin.representations.matvec import MatVecHMMRepresentation
from conin.representations.unrolled import (
    ObservationsEvidenceMixin,
    UnrolledHMMMixin,
    hmm_time_range,
)

from .common import Toulbar2FactorRepresentation
from .dbn import build_toulbar2_dbn_representation


class Toulbar2HMMRepresentation(
    ObservationsEvidenceMixin, UnrolledHMMMixin, Toulbar2FactorRepresentation
):
    """toulbar2 CFN of a constrained HMM unrolled (as a DBN) over ``start..stop``.

    Structure params: ``start`` and ``stop``.  The hidden nodes are
    ``("H", t)`` and the observed nodes ``("E", t)``; the evidence of
    ``create_model`` is keyed by ``("E", t)``, or is the list of observations
    (:class:`~conin.representations.unrolled.ObservationsEvidenceMixin`).
    Constraint functions see
    ``data = Munch(start, stop, variables, evidence,
    hmm=Munch(T, hidden_to_internal))``.  ``pgm`` is the HMM's DBN and
    ``uai_model`` its unrolled BN.
    """


@register_representation(
    "toulbar2", HiddenMarkovModel, families=(None, "toulbar2", "smoek")
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
        hidden_to_internal=MatVecHMMRepresentation(hmm).hidden_to_internal,
        data=cmodel.data,
    )
