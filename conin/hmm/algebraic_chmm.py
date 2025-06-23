import weakref
import pprint
import math
import numpy as np
import munch

from conin.exceptions import InvalidInputError
from conin.hmm import HMM, Statistical_Model
from . import chmm_base

import pyomo.environ as pyo


def _create_index_sets(*, hmm, observations):
    # N - Number of hidden states
    # start_probs[i] - map from i=1..N to a probability value in 0..1
    # emission_probes[i][k] - probability that output k is generated when in hidden state i
    # trans_mat[i][j] - probability of transitioning from hidden state i to hidden state j

    N = hmm.num_hidden_states
    obs = [hmm.observed_to_internal[o] for o in observations]
    start_probs = hmm.start_vec
    emission_probs = hmm.emission_mat
    trans_mat = np.array(hmm.transition_mat)

    Tmax = len(observations)
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

        assert len(
            latest) > 0, f"No feasible transitions to hidden states at time {t}"

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
    index_sets.observations_index = obs

    return index_sets


class Algebraic_CHMM(chmm_base.CHMM_Base):
    """
    A class to represent a Hidden Markov Model (HMM) with optimization equations.
    """

    def __init__(self, *, hmm=None, cache_indices=None, app=None):
        """
        Constructor.

        Parameters:
            hmm (HMM, optional): An instance of the HMM class (default is None, which initializes a new HMM instance).
        """
        super().__init__(hmm=hmm)
        self.cache_indices = True if cache_indices is None else cache_indices
        # An empty Munch object for index data
        self.data = munch.Munch()
        self._app = None if app is None else weakref.ref(app)

    def load_model(
        self,
        *,
        start_probs=None,
        transition_probs=None,
        emission_probs=None,
        hmm=None,
    ):
        """
        Loads the HMM model with the given parameters.
        Either give all three dictionaries or hmm, but not both

        Parameters:
            start_probs (dict, optional): A dictionary representing the start probabilities.
            transition_probs (dict, optional): A dictionary representing the transition probabilities.
            emission_probs (dict, optional: A dictionary representing the emission probabilities.
            hmm (HMM, optional): The HMM we wish to load with.

        Raises:
            InvalidInputError: If we supply too much or not enough information
        """
        if (
            hmm is not None
            and start_probs is None
            and transition_probs is None
            and emission_probs is None
        ):
            # If an HMM object is provided, load it directly
            self.hmm = hmm
        elif (
            start_probs is not None
            and transition_probs is not None
            and emission_probs is not None
        ):
            # If dictionaries are provided, create an HMM object and load it
            hmm = HMM()
            hmm.load_model(
                start_probs=start_probs,
                transition_probs=transition_probs,
                emission_probs=emission_probs,
            )
            self.hmm = hmm
        else:
            raise InvalidInputError(
                "You must provide either an HMM object or all three dictionaries, and not both."
            )

    def generate_model(self, *, observations):
        M = self.generate_unconstrained_model(observations=observations)
        if self._app is None:
            return M
        return self._generate_application_constraints(M)

    def _generate_application_constraints(self, M):  # pragma: nocover
        raise NotImplementedError(
            "Algebraic_CHMM.generate_application_constraints() is not implemented"
        )

    def generate_unconstrained_model(self, *, observations):  # pragma: nocover
        raise NotImplementedError(
            "Algebraic_CHMM.generate_unconstrained_model() is not implemented"
        )


class PyomoAlgebraic_CHMM(Algebraic_CHMM):

    def __init__(
        self,
        *,
        hmm=None,
        cache_indices=None,
        y_binary=False,
        x_binary=True,
        solver=None,
        solver_options=None,
        app=None,
    ):
        super().__init__(hmm=hmm, cache_indices=cache_indices, app=app)

        # Generate models with binary y-variables
        self.y_binary = y_binary
        self.x_binary = x_binary

        # Default configuration
        self.solver = "gurobi" if solver is None else solver
        self.solver_options = {} if solver_options is None else solver_options

    def _generate_application_constraints(self, M):
        return self.generate_pyomo_constraints(M)

    def generate_pyomo_constraints(self, M):
        return self._app().generate_pyomo_constraints(M=M)

    def generate_unconstrained_model(self, *, observations):
        self.observations = observations
        D = _create_index_sets(hmm=self.hmm, observations=observations)
        if self.cache_indices:
            self.data = D

        M = pyo.ConcreteModel()

        M.hmm = pyo.Block()

        # Variables

        if self.y_binary:
            M.hmm.y = pyo.Var(D.E, bounds=(0, 1), within=pyo.Boolean)
        else:
            M.hmm.y = pyo.Var(D.E, bounds=(0, 1))

        if self.x_binary:
            M.hmm.x = pyo.Var(D.T, D.A, bounds=(0, 1), within=pyo.Boolean)
        else:
            M.hmm.x = pyo.Var(D.T, D.A, bounds=(0, 1))

        # Hidden states

        def hidden_(m, t, b):
            if b not in D.states[t]:
                return m.x[t, b] == 0.0
            elif t == 0:
                return m.x[t, b] == m.y[t, -1, b]
            else:
                return m.x[t, b] == sum(
                    m.y[t, aa, b] for aa in D.A if (t, aa, b) in D.Gt
                )

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
                    sum(m.y[t, a, b] for a in D.A if (t, a, b) in D.Gt)
                    == m.y[t + 1, b, -2]
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

        M.hmm.o = pyo.Objective(
            expr=sum(
                (D.Gt[t, a, b] + D.Ge[t, b]) * M.hmm.y[t, a, b] for t, a, b in D.Gt
            ),
            sense=pyo.maximize,
        )

        return M

    def generate_hidden(self, *, observations, solver=None, solver_options=None):
        """
        This should probably be called something different

        Randomly generate hidden states using the HMM parameters
        """
        # TODO is this right?
        if "quiet" in solver_options:
            quiet = solver_options["quiet"]
        else:
            quiet = True
        hidden = self.hmm.generate_hidden_conditioned_on_observations(
            observations)
        T = len(observations)

        # Find the closest feasible point
        old_x_binary, old_y_binary, old_cache_indices = (
            self.x_binary,
            self.y_binary,
            self.cache_indices,
        )
        self.x_binary = True
        self.y_binary = False
        self.cache_indices = True
        M = self.generate_algebraic_constraints(observations=observations)
        M.hmm.o.deactivate()

        M.closest_point = pyo.Objective(
            expr=sum(
                M.hmm.x[t, self.hmm.external_to_hidden[hidden[i]]]
                for t in range(T)
                for i in range(
                    len(hidden)
                )  # CLM: I have no idea if this is right, but i wasn't defined before this
            ),
            sense=pyo.maximize,
        )

        if solver is None:
            solver = self.solver
        if solver_options is None:
            solver_options = self.solver_options
        opt = pyo.SolverFactory(solver)
        res = opt.solve(M, tee=not quiet, solver_options=solver_options)

        feasible_hidden = ["__UNKNOWN__"] * T
        for t in range(T):
            for a in self.data.A:
                if pyo.value(M.hmm.x[t, a]) > 0.5:
                    feasible_hidden[t] = self.hmm.hidden_to_external[a]
            assert (
                hidden[t] != "__UNKNOWN__"
            ), f"ERROR: Unexpected missing hidden state at time step {t}"

        # Reset application configuration
        self.x_binary, self.y_binary, self.cache_indices = (
            old_x_binary,
            old_y_binary,
            old_cache_indices,
        )

        return feasible_hidden


def create_algebraic_chmm(aml, **kwds):
    if aml == "pyomo":
        return PyomoAlgebraic_CHMM(**kwds)
    raise NotImplementedError(  # pragma: nocover
        f"Cannot construct an algebraic HMM with unknown AML: {aml}"
    )
