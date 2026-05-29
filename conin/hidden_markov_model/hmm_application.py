import munch

from . import learning
from .constrained_hmm import ConstrainedHiddenMarkovModel


class HMMApplication:

    def __init__(self, name="unknown"):
        # Application data
        self.data = munch.Munch()
        self.name = name

        self._hidden_markov_model = None
        self._simulations = None

        # Applicaton data used to initialize the HMM from simulations
        self._transition_prior = (None,)  # Nonzero values
        self._emission_prior = None  # Nonzero values
        self._hidden_states = None
        self._observable_states = None

    @property
    def hidden_markov_model(self):
        return self._hidden_markov_model

    @hidden_markov_model.setter
    def hidden_markov_model(self, hidden_markov_model):
        self._hidden_markov_model = hidden_markov_model

    @property
    def simulations(self):
        return self._simulations

    @simulations.setter
    def simulations(self, simulations):
        self._simulations = simulations

    def create_chmm(self, constraint_type=None):
        chmm = ConstrainedHiddenMarkovModel(hmm=self.hidden_markov_model)
        if constraint_type == "oracle":
            chmm.constraints = self.get_oracle_constraints()
        elif constraint_type == "pyomo":
            chmm.constraints = self.get_pyomo_constraints()
        chmm.initialize_chmm(constraint_type)
        return chmm

    def initialize(self, *args, **kwargs):
        """
        This method is used to initialize the application.  This does not create
        or initialize the class HiddenMarkovModel instance.
        """
        pass

    # TODO - return an error if these methods are not defined
    def run_simulations(
        self, *, num=1, debug=False, with_observations=False, seed=None
    ):
        """
        This method is used to generate feasible simulations of hidden states in a
        HiddenMarkovModel application.

        This method is defined by the application developer, and it provides a
        strategy for expressing domain knowledge regarding feasible hidden states.

        This method returns the simulations generated
        """
        pass

    def initialize_hmm_from_simulations(
        self,
        *,
        start_tolerance=None,
        transition_tolerance=None,
        emission_tolerance=None,
        simulations=None,
    ):
        assert (
            self._hidden_states is not None
        ), "HMMApplication.create_hmm_from_simulations must be run after the initialize() method is executed"

        if simulations is not None:
            self.simulations = simulations
        assert (
            self.simulations is not None
        ), "HMMApplication.create_hmm_from_simulations - No simulations specified"

        self.hidden_markov_model = learning.supervised_learning(
            simulations=self.simulations,
            hidden_states=self._hidden_states,
            observable_states=self._observable_states,
            start_tolerance=start_tolerance,
            transition_tolerance=transition_tolerance,
            emission_tolerance=emission_tolerance,
            transition_prior=self._transition_prior,
            emission_prior=self._emission_prior,
        )

    # TODO - return an error if these methods are not defined
    def get_oracle_constraints(self):
        return []

    # TODO - return an error if these methods are not defined
    def get_pyomo_constraints(self):
        return []
