"""The ``"pyomo_network_flow"`` representation of a constrained HMM.

This is the ``ip_formulation="network_flow"`` integer program: a shortest-path
(network-flow) formulation over the time-expanded hidden-state graph.

The model depends on the observations (the index sets prune the infeasible
transitions and hold the emission log-probabilities), so it is built per query
by :meth:`PyomoNetworkFlowHMMRepresentation.create_model`.  The representation
only holds the HMM, the constraints, the application data and the variable
domains.
"""

import math

import munch
import numpy as np
import pyomo.environ as pyo

from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation
from conin.util import State
from conin.util.constraint_data import merge_constraint_data

from .common import PyomoVarWrapper, apply_pyomo_constraints


def _create_index_sets(*, hmm, observed, Tmax=None):
    # N - Number of hidden states
    # start_probs[i] - map from i=1..N to a probability value in 0..1
    # emission_probes[i][k] - probability that output k is generated when in hidden state i
    # trans_mat[i][j] - probability of transitioning from hidden state i to
    # hidden state j

    """Create index sets and log-probability data for algebraic HMM models.

    Parameters
    ----------
    hmm : HiddenMarkovModel
        Hidden Markov model supplying probabilities and label mappings.
    observed : sequence
        Observed-state labels used to restrict feasible hidden states.
    Tmax : int, optional
        Number of time steps to model. If omitted, use ``len(observed)``.

    Returns
    -------
    munch.Munch
        Collection of index sets, feasible transitions, and cached log
        probabilities for algebraic formulations.
    """
    N = hmm.num_hidden_states
    obs = [hmm.observed_to_internal[o] for o in observed]
    start_probs = hmm.start_vec
    emission_probs = hmm.emission_mat
    trans_mat = np.array(hmm.transition_mat)

    if Tmax is None:
        Tmax = len(observed)
    T = list(range(Tmax))

    A = list(range(N))

    # F: (a,b) in transition matrix
    F = set()
    # FF: (t,a,b) such that transition probability == 0
    FF = set()
    # Gt: (t,a,b) ->  log(transition probability from a to b at time t)
    Gt = {}
    # Ge: (t,b) ->  log(emission probability for b at time t)
    Ge = {}
    states = {}
    latest = None
    for t in T:
        if t == 0:
            curr = set()
            for i in A:
                if start_probs[i] > 0 and emission_probs[i][obs[t]] > 0:
                    Gt[0, -1, i] = math.log(start_probs[i])
                    Ge[0, i] = math.log(emission_probs[i][obs[t]])
                    curr.add(i)
                else:
                    FF.add((0, -1, i))
        else:
            curr = set()
            for a, row in enumerate(trans_mat):
                if a not in latest:
                    continue
                for b, val in enumerate(row):
                    F.add((a, b))
                    if val > 0 and emission_probs[b][obs[t]] > 0:
                        Gt[t, a, b] = math.log(val)
                        Ge[t, b] = math.log(emission_probs[b][obs[t]])
                        curr.add(b)
                    else:
                        FF.add((t, a, b))
        latest = curr
        states[t] = latest

        if not latest:
            raise InvalidInputError(
                f"No feasible transitions to hidden states at time {t}: the "
                "observations have probability zero"
            )

    # print("XXX", [len(s) for _,s in states.items()])
    # print("XXX", statistics.mean([len(s) for _,s in states.items()]))
    # pprint.pprint(emission_probs)

    # for k,v in sorted(G.items()):
    #    print("G",k,v)
    # for k in sorted(FF):
    #    print("FF",k)

    GG = {(Tmax, i, -2) for i in latest}

    # The keys in Gt are in sorted order, but not the elements of GG
    E = list(Gt.keys()) + list(sorted(GG))
    #           (t,a,b) where
    #           (t, -1,  i) when t = 0
    #           (t,  a,  b) when t = 1..Tmax
    #           (t,  i, -2) when t = Tmax
    # for i in A:
    #    E.append((-1, -1, i))
    # for t in range(Tmax - 1):
    #    for g in F:
    #        E.append(tuple([t] + list(g)))
    # E = list(G.keys())
    # for i in latest:
    #    E.append((Tmax - 1, i, -2))

    index_sets = munch.Munch()
    index_sets.Tmax = Tmax
    index_sets.N = N
    index_sets.A = A
    index_sets.T = T
    index_sets.E = E
    index_sets.F = F  # Not used
    index_sets.FF = FF  # Not used
    index_sets.Gt = Gt
    index_sets.Ge = Ge
    index_sets.GG = GG
    index_sets.states = states
    index_sets.observed_index = obs
    index_sets.hidden_to_internal = hmm.hidden_to_internal

    return index_sets


def _create_network_flow_model(D, *, hidden_to_external, x_binary, y_binary):
    """Build the unconstrained network-flow model from the index sets ``D``."""
    M = pyo.ConcreteModel()

    M.hmm = pyo.Block()

    # Variables

    if y_binary:
        M.hmm.y = pyo.Var(D.E, bounds=(0, 1), within=pyo.Boolean)
    else:
        M.hmm.y = pyo.Var(D.E, bounds=(0, 1))

    if x_binary:
        M.hmm.x = pyo.Var(D.T, D.A, bounds=(0, 1), within=pyo.Boolean)
    else:
        M.hmm.x = pyo.Var(D.T, D.A, bounds=(0, 1))

    M.V = PyomoVarWrapper(
        ((("H", t), State(hidden_to_external[s])), M.hmm.x[t, s]) for t, s in M.hmm.x
    )

    # Hidden states

    def hidden_(m, t, b):
        if b not in D.states[t]:
            return m.x[t, b] == 0.0
        elif t == 0:
            return m.x[t, b] == m.y[t, -1, b]
        else:
            return m.x[t, b] == sum(m.y[t, aa, b] for aa in D.A if (t, aa, b) in D.Gt)

    M.hmm.hidden = pyo.Constraint(D.T, D.A, rule=hidden_)

    # Shortest path constraints

    def flow_(m, t, b):
        if b not in D.states[t]:
            return pyo.Constraint.Skip
        elif t == 0:
            return m.y[t, -1, b] == sum(
                m.y[t + 1, b, a] for a in D.A if (t + 1, b, a) in D.Gt
            )
        elif t == D.Tmax - 1:
            return (
                sum(m.y[t, a, b] for a in D.A if (t, a, b) in D.Gt) == m.y[t + 1, b, -2]
            )
        else:
            return sum(m.y[t, a, b] for a in D.A if (t, a, b) in D.Gt) == sum(
                m.y[t + 1, b, a] for a in D.A if (t + 1, b, a) in D.Gt
            )

    M.hmm.flow = pyo.Constraint(D.T, D.A, rule=flow_)

    def flow_start_(m):
        return sum(m.y[0, -1, b] for b in D.A if (0, -1, b) in D.Gt) == 1

    M.hmm.flow_start = pyo.Constraint(rule=flow_start_)

    def flow_end_(m):
        return sum(m.y[D.Tmax, a, -2] for a in D.A if (D.Tmax, a, -2) in D.GG) == 1

    M.hmm.flow_end = pyo.Constraint(rule=flow_end_)

    # Should o_val be defined with an Expression?  If so, how can we provide the value to the user?
    M.hmm.o_val = pyo.Var(D.Gt, bounds=(None, 0))

    def o_val_c_(m, t, a, b):
        return M.hmm.o_val[t, a, b] == (D.Gt[t, a, b] + D.Ge[t, b]) * M.hmm.y[t, a, b]

    M.hmm.o_val_c = pyo.Constraint(D.Gt, rule=o_val_c_)

    M.hmm.o = pyo.Objective(
        expr=sum(M.hmm.o_val[t, a, b] for t, a, b in D.Gt), sense=pyo.maximize
    )

    return M


class PyomoNetworkFlowHMMRepresentation(Representation):
    """Network-flow IP formulation of a constrained HMM.

    Parameters
    ----------
    hidden_markov_model : HiddenMarkovModel
    constraints : sequence of PyomoConstraint or AlgebraicConstraint
    data : object, optional
        Application data.  Each query gets a shallow copy with ``.hmm`` set to
        the index sets (``data.hmm.T``, ``data.hmm.A``,
        ``data.hmm.hidden_to_internal``, ...); the user's object is never
        modified.
    x_binary : bool, optional
        Hidden-state variables ``M.hmm.x`` are binary (default ``True``).
    y_binary : bool, optional
        Transition variables ``M.hmm.y`` are binary (default ``False``).

    Notes
    -----
    Structure params: ``x_binary`` and ``y_binary``.
    Constraint functions see ``M.V("H", t, h)`` (a :class:`PyomoVarWrapper`
    over ``M.hmm.x``) and ``M.hmm``.  Nothing is mutated after construction,
    so one representation can serve several queries.
    """

    kind = "pyomo_network_flow"

    def __init__(
        self,
        hidden_markov_model,
        constraints=(),
        data=None,
        *,
        x_binary=True,
        y_binary=False,
    ):
        self.hidden_markov_model = hidden_markov_model
        self.constraints = tuple(constraints)
        self._data = data
        self.x_binary = x_binary
        self.y_binary = y_binary

    def create_model(self, *, observed, Tmax=None):
        """Build the network-flow Pyomo model for ``observed`` and apply the constraints.

        Parameters
        ----------
        observed : sequence
            Observed-state labels.
        Tmax : int, optional
            Number of time steps (default ``len(observed)``).

        Returns
        -------
        (pyomo.environ.ConcreteModel, object)
            The model and the per-query ``data`` passed to the constraints:
            a shallow copy of the application data (see
            :func:`~conin.util.constraint_data.merge_constraint_data`) with
            ``start=0``, ``stop=Tmax-1`` and ``hmm`` (the index sets,
            including ``T`` and ``hidden_to_internal``) set.
        """
        hmm = self.hidden_markov_model
        D = _create_index_sets(hmm=hmm, observed=observed, Tmax=Tmax)
        data = merge_constraint_data(
            self._data, munch.Munch(start=0, stop=D.Tmax - 1, hmm=D)
        )

        M = _create_network_flow_model(
            D,
            hidden_to_external=hmm.hidden_to_external,
            x_binary=self.x_binary,
            y_binary=self.y_binary,
        )
        if self.constraints:
            apply_pyomo_constraints(
                pgm=hmm, constraints=self.constraints, model=M, data=data
            )
        return M, data


@register_representation(
    "pyomo_network_flow", HiddenMarkovModel, families=(None, "pyomo", "algebraic")
)
def build_pyomo_network_flow_hmm_representation(
    cmodel, *, x_binary=True, y_binary=False
):
    return PyomoNetworkFlowHMMRepresentation(
        cmodel.model,
        cmodel.constraints,
        cmodel.data,
        x_binary=x_binary,
        y_binary=y_binary,
    )
