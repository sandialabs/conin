import munch

from . import learning
from .constrained_hmm import ConstrainedHiddenMarkovModel


class HMMApplication:
    """Base class for application-specific hidden Markov model workflows."""

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
        """Return the application's hidden Markov model.

        Returns
        -------
        HiddenMarkovModel or None
            Hidden Markov model associated with the application.
        """
        return self._hidden_markov_model

    @hidden_markov_model.setter
    def hidden_markov_model(self, hidden_markov_model):
        """Set the application's hidden Markov model.

        Parameters
        ----------
        hidden_markov_model : HiddenMarkovModel or None
            Hidden Markov model to associate with the application.
        """
        self._hidden_markov_model = hidden_markov_model

    @property
    def simulations(self):
        """Return cached simulation data for the application.

        Returns
        -------
        object
            Simulation data associated with the application.
        """
        return self._simulations

    @simulations.setter
    def simulations(self, simulations):
        """Set cached simulation data for the application.

        Parameters
        ----------
        simulations : object
            Simulation data to cache on the application.
        """
        self._simulations = simulations

    def create_chmm(self, constraint_type=None):
        """Create and initialize a constrained HMM for this application.

        Parameters
        ----------
        constraint_type : {"oracle", "pyomo"}, optional
            Constraint backend to use when configuring the constrained HMM.

        Returns
        -------
        ConstrainedHiddenMarkovModel
            Initialized constrained HMM wrapper.
        """
        chmm = ConstrainedHiddenMarkovModel(hmm=self.hidden_markov_model)
        if constraint_type == "oracle":
            chmm.constraints = self.get_oracle_constraints()
        elif constraint_type == "pyomo":
            chmm.constraints = self.get_pyomo_constraints()
        chmm.initialize_chmm(constraint_type)
        return chmm

    def initialize(self, *args, **kwargs):
        """Initialize application-specific state.

        Notes
        -----
        Subclasses should override this method. It does not create or initialize
        the ``HiddenMarkovModel`` instance by itself.
        """
        pass

    # TODO - return an error if these methods are not defined
    def run_simulations(
        self, *, num=1, debug=False, with_observations=False, seed=None
    ):
        """Generate application-specific simulations.

        Parameters
        ----------
        num : int, optional
            Number of simulations to generate.
        debug : bool, optional
            If ``True``, enable application-specific debug behavior.
        with_observations : bool, optional
            If ``True``, include observed-state sequences in the generated
            simulations.
        seed : int, optional
            Random seed for reproducible simulation.

        Returns
        -------
        object
            Simulation data defined by the application subclass.

        Notes
        -----
        Subclasses are expected to override this method with a domain-specific
        strategy for generating feasible hidden-state trajectories.
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
        """Fit a hidden Markov model from simulation data.

        Parameters
        ----------
        start_tolerance : float, optional
            Lower bound or smoothing tolerance applied to start
            probabilities.
        transition_tolerance : float, optional
            Lower bound or smoothing tolerance applied to transition
            probabilities.
        emission_tolerance : float, optional
            Lower bound or smoothing tolerance applied to emission
            probabilities.
        simulations : optional
            Simulation data to use in place of ``self.simulations``.
        """
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
        """Return oracle-style constraints for this application.

        Returns
        -------
        list
            Oracle constraint functors used by oracle-constrained HMM solvers.
        """
        return []

    # TODO - return an error if these methods are not defined
    def get_pyomo_constraints(self):
        """Return Pyomo-style constraints for this application.

        Returns
        -------
        list
            Constraint functors used by algebraic constrained HMM solvers.
        """
        return []
