import pytest
import math
import random
import munch
import copy

import pyomo.environ as pyo

from conin.hmm import *
from conin.oracle_constraints import *
from conin.constraint import pyomo_constraint_fn
from conin.hmm.inference import viterbi, a_star, lp_inference, ip_inference


#
# The **Num_Zeros** class provides the base definition for the num_zeros application.
#
class Num_Zeros(HMMApplication):

    def __init__(self):
        self.num_zeros = None
        super().__init__(self.__class__.__name__)

    def initialize(
        self,
        *,
        hmm=None,
        prob_stay_in_same_state=None,
        prob_error=None,
        zero_start_prob=0.5,
        num_zeros,
        time
    ):
        if hmm is None:
            start_probs = {0: zero_start_prob, 1: 1 - zero_start_prob}
            transition_probs = {
                (0, 0): prob_stay_in_same_state,
                (0, 1): 1 - prob_stay_in_same_state,
                (1, 0): 1 - prob_stay_in_same_state,
                (1, 1): prob_stay_in_same_state,
            }
            emission_probs = {
                (0, 0): 1 - prob_error,
                (0, 1): prob_error,
                (1, 0): prob_error,
                (1, 1): 1 - prob_error,
            }
            hmm = HiddenMarkovModel()
            hmm.load_model(
                start_probs=start_probs,
                transition_probs=transition_probs,
                emission_probs=emission_probs,
            )
            self.hidden_markov_model = hmm
        else:
            self.hidden_markov_model = hmm
        self.num_zeros = num_zeros
        self._hidden_states = {0, 1}
        self._observable_states = {0, 1}
        self.time = time

    def run_simulations(
        self, *, num=1, debug=False, seed=None, with_observations=False
    ):
        if seed is not None:
            random.seed(seed)
        output = []
        oracle_chmm = self.create_chmm("oracle")

        for n in range(num):
            res = munch.Munch()
            hidden = oracle_chmm.generate_hidden(self.time)
            if with_observations:
                observed = oracle_chmm.generate_observed_from_hidden(hidden)
            res = munch.Munch(hidden=hidden, index=n)
            if with_observations:
                res.observed = observed
            output.append(res)
        return output

    def get_oracle_constraints(self):
        constraint = has_exact_number_of_occurences_constraint(
            val=0, count=self.num_zeros
        )
        return [constraint]

    def get_pyomo_constraints(self):
        @pyomo_constraint_fn()
        def constraint(M, data):
            index_sets = list(M.hmm.x.index_set().subsets())
            T = list(index_sets[0])

            M.num_zeros = pyo.Constraint(
                expr=sum(M.hmm.x[t, 0] for t in T) == data.num_zeros
            )

        return [constraint]


@pytest.fixture
def app():
    # 1/(1-prob_stay_in_same_state) = expected number of iterations of the
    # same state
    prob_stay_in_same_state = 0.6
    prob_error = 0.3  # Probability that hidden state h has an observed state that does not match it
    num_zeros = 10  # Number of zeros
    time = 20
    app = Num_Zeros()
    app.initialize(
        prob_stay_in_same_state=prob_stay_in_same_state,
        prob_error=prob_error,
        num_zeros=num_zeros,
        time=time,
    )
    return app


@pytest.fixture
def oracle_chmm(app):
    return app.create_chmm("oracle")


class Test_Application_CHMM:
    def test_hmm(self, app):
        assert app.hidden_markov_model.transition_mat == [[0.6, 0.4], [0.4, 0.6]]
        assert app.hidden_markov_model.emission_mat == [[0.7, 0.3], [0.3, 0.7]]
        assert app.hidden_markov_model.start_vec == [0.5, 0.5]
        assert app._hidden_states == {0, 1}
        assert app._observable_states == {0, 1}
        assert app.time == 20

    def test_oracle_type(self, app):
        chmm = app.create_chmm("oracle")
        assert isinstance(chmm.chmm, Oracle_CHMM)
        assert app.hidden_markov_model == chmm.hidden_markov_model

    def test_algebraic_type(self, app):
        chmm = app.create_chmm("pyomo")
        assert isinstance(chmm.chmm, PyomoAlgebraic_CHMM)
        assert app.hidden_markov_model == chmm.hidden_markov_model

    def test_hmm_equality_setter(self, app):
        hmm = HiddenMarkovModel()
        hmm.load_model(
            emission_probs={
                (0, 0): 0.6,
                (0, 1): 0.4,
                (1, 0): 0.4,
                (1, 1): 0.6,
            },
            start_probs={0: 0.5, 1: 0.5},
            transition_probs={
                (0, 0): 0.6,
                (0, 1): 0.4,
                (1, 0): 0.4,
                (1, 1): 0.6,
            },
        )
        app.hidden_markov_model = hmm
        chmm = app.create_chmm("oracle")
        assert app.hidden_markov_model == chmm.hidden_markov_model
        chmm = app.create_chmm("pyomo")
        assert app.hidden_markov_model == chmm.hidden_markov_model

    def test_get_internal_hmm(self, app):
        repn = app.hidden_markov_model.repn
        assert repn.transition_mat == [
            [0.6, 0.4],
            [0.4, 0.6],
        ]
        assert repn.emission_mat == [[0.7, 0.3], [0.3, 0.7]]
        assert repn.start_vec == [0.5, 0.5]

    def test_run_simulations(self, app):
        seed = 1
        num_simulations = 5
        simulations = app.run_simulations(
            num=num_simulations, with_observations=True, seed=seed
        )

        oracle_chmm = app.create_chmm("oracle")
        assert len(simulations) == num_simulations
        for i in range(num_simulations):
            assert len(simulations[i].hidden) == app.time
            assert len(simulations[i].observed) == app.time
            assert oracle_chmm.is_feasible(simulations[i].hidden)

    def test_oracle_is_feasible(self, oracle_chmm):
        seq1 = [0] * 9
        seq2 = [0] * 10
        seq3 = [0] * 11
        assert not oracle_chmm.is_feasible(seq1)
        assert oracle_chmm.is_feasible(seq2)
        assert not oracle_chmm.is_feasible(seq3)

    # This assumes that the internal logic is correct, and is really just
    def test_initalize_hmm_from_simulations(self, app):
        simulations = app.run_simulations(with_observations=True)
        app.initialize_hmm_from_simulations(simulations=simulations)
        repn = app.hidden_markov_model.repn
        assert repn.transition_mat != [
            [0.6, 0.4],
            [0.4, 0.6],
        ]  # Just checks that it's updated
