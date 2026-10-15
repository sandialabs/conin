"""The ``"pyomo_network_flow"`` representation of a constrained HMM.

It wraps :class:`conin.hidden_markov_model.chmm_algebraic.PyomoAlgebraic_CHMM`
(the ``ip_formulation="network_flow"`` IP). The create/solve split and the
shared Pyomo helpers come in PR 4a.
"""

import copy

from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation


class PyomoNetworkFlowHMMRepresentation(Representation):
    """Network-flow IP formulation of a constrained HMM.

    Attributes
    ----------
    chmm : PyomoAlgebraic_CHMM
        The wrapped object.

    Notes
    -----
    ``PyomoAlgebraic_CHMM`` writes per-query state when a model is generated
    (``data.hmm``, the index sets of the last observation sequence, and
    ``observed``). So the wrapper holds a *shallow copy* of the constrained
    model's ``data`` (the user's object never gets a ``.hmm`` attribute), and
    :meth:`get_copy` returns a fresh wrapper with its own shallow copy.
    """

    kind = "pyomo_network_flow"

    def __init__(self, hidden_markov_model, constraints, data):
        from conin.hidden_markov_model.chmm_algebraic import PyomoAlgebraic_CHMM

        self._hidden_markov_model = hidden_markov_model
        self._constraints = tuple(constraints)
        self._data = data
        self.chmm = PyomoAlgebraic_CHMM(
            hidden_markov_model=hidden_markov_model,
            constraints=list(self._constraints),
            data=None if data is None else copy.copy(data),
        )

    @property
    def data(self):
        """The data passed to the constraints (``data.hmm`` holds the index sets)."""
        return self.chmm.data

    def generate_unconstrained_model(self, *, observed, Tmax=None):
        """Build the network-flow Pyomo model without the user constraints."""
        return self.chmm.generate_unconstrained_model(observed=observed, Tmax=Tmax)

    def generate_model(self, *, observed):
        """Build the network-flow Pyomo model and apply the user constraints."""
        return self.chmm.generate_model(observed=observed)

    def get_copy(self):
        """Return a fresh wrapper (cheap: the Pyomo model is built per query)."""
        return PyomoNetworkFlowHMMRepresentation(
            self._hidden_markov_model, self._constraints, self._data
        )


@register_representation(
    "pyomo_network_flow", HiddenMarkovModel, families=(None, "pyomo")
)
def build_pyomo_network_flow_hmm_representation(cmodel):
    return PyomoNetworkFlowHMMRepresentation(
        cmodel.model, cmodel.constraints, cmodel.data
    )
