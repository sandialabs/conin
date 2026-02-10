import pyomo.environ as pe
import conin.hmm
from conin import pyomo_constraint_fn

import munch
import random

"""
This script has a collection of HMM test cases that can be used to
test conin capabilities.
"""


def create_hmm0():
    start_probs = {"h0": 1, "h1": 0}
    transition_probs = {
        ("h0", "h0"): 0,
        ("h0", "h1"): 1,
        ("h1", "h0"): 0,
        ("h1", "h1"): 1,
    }
    emission_probs = {
        ("h0", "o0"): 1,
        ("h0", "o1"): 0,
        ("h1", "o0"): 0,
        ("h1", "o1"): 1,
    }
    hmm = conin.hmm.HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm1():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.9,
        ("h0", "h1"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.8,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
    }
    hmm = conin.hmm.HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm1_aos():
    """
    Unconstrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.

    For all observed state Lengths > 0, the follow behavior occurs:
    best sol should be h0*Length
    second best sol should be h1*Length
    all other sols should have prob 0
    """

    start_probs = {"h0": 0.51, "h1": 0.49}
    transition_probs = {
        ("h0", "h0"): 1.0,
        ("h1", "h1"): 1.0,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.7,
        ("h1", "o1"): 0.3,
    }
    hmm = conin.hmm.HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm2():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.8,
        ("h0", "h1"): 0.1,
        ("h0", "h2"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.7,
        ("h1", "h2"): 0.1,
        ("h2", "h2"): 1.0,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
        ("h2", "o2"): 1.0,
    }
    hmm = conin.hmm.HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm2_aos():
    """
    Unconstrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.

    For all observed state Lengths > 0, the follow behavior occurs:
    best sol is h0*length
    second best is h1*length
    third best is h2, h0*(length-1)
    fourth best is h1,h2, h0*(length-2)
    """
    start_probs = {"h0": 0.6, "h1": 0.3, "h2": 0.1}
    transition_probs = {
        ("h0", "h0"): 1.0,
        ("h1", "h1"): 0.99,
        ("h1", "h2"): 0.01,
        ("h2", "h0"): 1.0,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.2,
        ("h0", "o2"): 0.1,
        ("h1", "o0"): 0.7,
        ("h1", "o1"): 0.2,
        ("h1", "o2"): 0.1,
        ("h2", "o0"): 0.7,
        ("h2", "o1"): 0.2,
        ("h2", "o2"): 0.1,
    }
    hmm = conin.hmm.HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_chmm1_oracle():
    hmm = create_hmm1()

    num_zeros_greater_than_nine = conin.hmm.OracleConstraint(
        func=lambda seq: seq.count("h0") > 9,
        partial_func=lambda T, seq: T - len(seq) + seq.count("h0") >= 10,
    )
    num_zeros_less_than_thirteen = conin.hmm.OracleConstraint(
        func=lambda seq: seq.count("h0") < 13,
        partial_func=lambda T, seq: seq.count("h0") < 13,
    )
    constraints = [num_zeros_greater_than_nine, num_zeros_less_than_thirteen]

    chmm = conin.hmm.ConstrainedHiddenMarkovModel(hmm=hmm, constraints=constraints)
    chmm.initialize_chmm()
    return chmm


def create_chmm1_pyomo():
    hmm = create_hmm1()

    @conin.pyomo_constraint_fn()
    def num_zeros_greater_than_nine(M, D):
        h0 = D.hmm.hidden_to_internal["h0"]
        M.h0_lower = pe.Constraint(expr=sum(M.hmm.x[t, h0] for t in D.hmm.T) >= 10)

    @conin.pyomo_constraint_fn()
    def num_zeros_less_than_thirteen(M, D):
        h0 = D.hmm.hidden_to_internal["h0"]
        M.h0_upper = pe.Constraint(expr=sum(M.hmm.x[t, h0] for t in D.hmm.T) <= 12)

    constraints = [num_zeros_greater_than_nine, num_zeros_less_than_thirteen]

    chmm = conin.hmm.ConstrainedHiddenMarkovModel(hmm=hmm, constraints=constraints)
    chmm.initialize_chmm()
    return chmm


def create_chmm1_pyomo_aos():
    """
    Constrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.
    Modification of create_hmm1_aos to add constraints

    Based on observed state Lengths > 0, the follow behavior occurs:
    if length > 12, only sol with prob > 0 is h1*length
    if length <= 12, best sol h0*length, second best h1*length
    """
    hmm = create_hmm1_aos()

    @conin.pyomo_constraint_fn()
    def num_zeros_less_than_thirteen(M, D):
        h0 = D.hmm.hidden_to_internal["h0"]
        M.h0_upper = pe.Constraint(expr=sum(M.hmm.x[t, h0] for t in D.hmm.T) <= 12)

    constraints = [num_zeros_less_than_thirteen]

    chmm = conin.hmm.ConstrainedHiddenMarkovModel(hmm=hmm, constraints=constraints)
    chmm.initialize_chmm()
    return chmm


def create_chmm2_pyomo_aos():
    """
    Constrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.
    Modification of create_hmm2_aos to add constraints

    Based on observed state Lengths, the follow behavior occurs:
    if length > 12, second best can't happen, move sols up one
    best sol is h0*length
    second best is h1*length,
    third best is h2, h0*(length-1)
    fourth best is h1,h2, h0*(length-2)
    """
    hmm = create_hmm2_aos()

    @conin.pyomo_constraint_fn()
    def num_ones_less_than_thirteen(M, D):
        h1 = D.hmm.hidden_to_internal["h1"]
        M.h0_upper = pe.Constraint(expr=sum(M.hmm.x[t, h1] for t in D.hmm.T) <= 12)

    constraints = [num_ones_less_than_thirteen]

    chmm = conin.hmm.ConstrainedHiddenMarkovModel(hmm=hmm, constraints=constraints)
    chmm.initialize_chmm()
    return chmm


class Num_Zeros(conin.hmm.HMMApplication):

    def __init__(self):
        self.num_zeros = None
        super().__init__(self.__class__.__name__)

    def initialize(
        self,
        *,
        hmm,
        oracle_constraints,
        lb,
        ub,
    ):
        self.hmm = hmm
        self._oracle_constraints = oracle_constraints
        self.lb = lb
        self.ub = ub

    def run_simulations(
        self,
        *,
        num=1,
        debug=False,
        seed=None,
        with_observations=False,
        time_steps,
    ):
        if seed is not None:
            random.seed(seed)
        output = []
        for n in range(num):
            res = munch.Munch()
            hidden = self.oracle.generate_hidden(time_steps)
            if with_observations:
                observed = self.oracle.generate_observed_from_hidden(hidden)
            res = munch.Munch(hidden=hidden, index=n)
            if with_observations:
                res.observed = observed
            output.append(res)
        return output

    def get_oracle_constraints(self):
        return self._oracle_constraints

    def get_pyomo_constraints(self):
        @pyomo_constraint_fn()
        def constraint(M, D):
            h0 = D.hmm.hidden_to_internal["h0"]
            M.h0_lower = pe.Constraint(
                expr=sum(M.hmm.x[t, h0] for t in D.hmm.T) >= self.lb
            )
            M.h0_upper = pe.Constraint(
                expr=sum(M.hmm.x[t, h0] for t in D.hmm.T) <= self.ub
            )

        return [constraint]

    #
    # The following methods are used by recursive A*
    #

    def initialize_constraint_data(self, hidden_state):
        if hidden_state == "h0":
            return 1
        else:
            return 0

    def constraint_data_feasible_partial(self, *, constraint_data, t, time_steps):
        return (
            constraint_data + (time_steps - t) >= self.lb
        ) and constraint_data <= self.ub

    def constraint_data_feasible(self, constraint_data):
        return constraint_data >= self.lb and constraint_data <= self.ub

    def update_constraint_data(self, *, hidden_state, constraint_data):
        if hidden_state == "h0":
            return constraint_data + 1
        else:
            return constraint_data
