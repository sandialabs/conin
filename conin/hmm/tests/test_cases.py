import pyomo.environ as pe
import conin.hmm

import munch
import random
import math

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


def create_chmm1():
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
    num_zeros_greater_than_nine = conin.hmm.Constraint(
        func=lambda seq: seq.count("h0") > 9,
        partial_func=lambda T, seq: T - len(seq) + seq.count("h0") >= 10,
    )
    num_zeros_less_than_thirteen = conin.hmm.Constraint(
        func=lambda seq: seq.count("h0") < 13,
        partial_func=lambda T, seq: seq.count("h0") < 13,
    )
    hmm = conin.hmm.HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    chmm = conin.hmm.ConstrainedHiddenMarkovModel(hmm=hmm)
    chmm.add_constraint(num_zeros_greater_than_nine)
    chmm.add_constraint(num_zeros_less_than_thirteen)
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
        constraints,
        lb,
        ub,
    ):
        self.hmm = hmm
        self.constraints = constraints
        self.oracle.set_constraints(constraints)
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

    def generate_pyomo_constraints(self, *, M):
        # Data used to construct the base HMM formulation
        D = self.algebraic.data

        h0 = self.hmm.hidden_to_internal["h0"]
        M.h0_lower = pe.Constraint(expr=sum(M.hmm.x[t, h0] for t in D.T) >= self.lb)
        M.h0_upper = pe.Constraint(expr=sum(M.hmm.x[t, h0] for t in D.T) <= self.ub)

        return M
