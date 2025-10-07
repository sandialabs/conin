import math
import numpy as np
import munch

from . import learning
from .constrained_hmm import ConstrainedHiddenMarkovModel
from .chmm_algebraic import create_algebraic_chmm
from .chmm_oracle import Oracle_CHMM


class HMMApplication:

    def __init__(self, name="unknown"):
        # Application data
        self.data = munch.Munch()
        self.name = name

        # Oracle HMM representation
        self._hmm = None
        self.oracle = ConstrainedHiddenMarkovModel()
        self.generate_oracle_constraints()

        # Algebraic HMM representation
        self.algebraic_aml = None
        self.algebraic = None
        self.initialize_algebraic_hmm("pyomo")

        # Applicaton data used to initialize the HMM from simulations
        self._transition_prior = (None,)  # Nonzero values
        self._emission_prior = None  # Nonzero values
        self._hidden_states = None
        self._observable_states = None

    def initialize_algebraic_hmm(self, aml):
        if aml != self.algebraic_aml:
            self.algebraic_aml = aml
            self.algebraic = create_algebraic_chmm(aml, app=self)

    @property
    def hmm(self):
        return self._hmm

    @hmm.setter
    def hmm(self, hmm):
        self._hmm = hmm
        self.oracle.hmm = hmm
        self.algebraic.load_model(hmm=hmm)

    def get_hmm(self):
        return self._hmm

    def get_internal_hmm(self):
        return self._hmm.internal_hmm

    def get_constrained_hmm(self):
        return self.oracle

    def get_internal_constrained_hmm(self):
        return self.oracle.internal_constrained_hmm

    def initialize(self, *args, **kwargs):
        """
        This method is used to initialize the application.  In particular, this method
        creates an HMM that
        """
        pass

    def run_simulations(
        self, *, num=1, debug=False, with_observations=False, seed=None
    ):
        """
        This method is used to generate feasible simulations of hidden states in an HMM
        application.

        This method is defined by the application developer, and it provides a
        strategy for expressing domain knowledge regarding feasible hidden states.
        """
        return None

    def initialize_hmm_from_simulations(
        self,
        *,
        num=100,
        debug=False,
        seed=None,
        start_tolerance=None,
        transition_tolerance=None,
        emission_tolerance=None,
        simulation_args=None,
    ):
        assert (
            self._hidden_states is not None
        ), "HMMApplication.create_hmm_from_simulations must be run after the initialize() method is executed"
        if simulation_args is None:
            simulation_args = {}
        simulation_args["num"] = num
        simulation_args["debug"] = debug
        simulation_args["seed"] = seed
        simulation_args["with_observations"] = True
        simulations = self.run_simulations(**simulation_args)
        if debug:
            for sim in simulations:
                print("TSIM", sim.observations, sim.hidden)

        assert (
            simulations is not None
        ), f"HMMApplication.create_hmm_from_simulations - Method run_simulations() has not been defined for the {self.name} application"

        hmm = learning.supervised_learning(
            simulations=simulations,
            hidden_states=self._hidden_states,
            observable_states=self._observable_states,
            start_tolerance=start_tolerance,
            transition_tolerance=transition_tolerance,
            emission_tolerance=emission_tolerance,
            transition_prior=self._transition_prior,
            emission_prior=self._emission_prior,
        )
        self.hmm = hmm

    def generate_oracle_constraints(self):
        pass

    def generate_pyomo_constraints(self, M):
        return M
