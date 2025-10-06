from conin.exceptions import InvalidInputError, InsufficientSolutionsError
from conin.hmm import HiddenMarkovModel, HMM
from conin.hmm.oracle_chmm import Oracle_CHMM

from .viterbi import viterbi
from .a_star import a_star


class Inference:
    def __init__(self, *, statistical_model, num_solutions=1, oracle_based=True):
        """
        Initializes the Inference class with the specified parameters.

        Parameters:
            statistical_model (hmm or constrained hmm, optional): What we wish to run inference on
            num_solutions (int, optional): The number of solutions to generate.
                Must be a positive integer. If not provided,defaults to 1.
            oracle_based (bool, optional): A flag indicating whether to use an oracle-based approach for inference.
                Defaults to True.
        """
        self.set_statistical_model(statistical_model)
        self.set_num_solutions(num_solutions)
        self.oracle_based = oracle_based

    def set_statistical_model(self, statistical_model):
        """
        Does setting of statistical model

        Parameters:
            statistical_model (hmm or constrained hmm): What we wish to load

        Raises:
            InvalidInputError: If statistical_model is not hmm or constrained hmm
        """
        if isinstance(statistical_model, HiddenMarkovModel) or isinstance(statistical_model, Oracle_CHMM) or isinstance(statistical_model, HMM):
            self.statistical_model = statistical_model
        else:
            raise InvalidInputError(
                "In inference load_statisical_model, statistical_model should be either an "
                "HMM or Constrained HMM."
            )

    def set_num_solutions(self, num_solutions):
        """
        Sets num_solutions

        Parameters:
            num_solutions (int): What we wish to load

        Raises:
            InvalidInputError: If num_solutions is not positive
        """
        if num_solutions <= 0:
            raise InvalidInputError(
                "In inference set_num_solutions, num_solutions must be positive."
            )
        else:
            self.num_solutions = num_solutions

    def __call__(self, observed):
        """
        Performs inference using the specified observed.

        This method uses the Viterbi algorithm if the conditions are met: the inference is
        oracle-based, only one solution is required, and the statistical model is an instance
        of the HMM class or a CHMM with no constraints.

        Parameters:
            observed (list):  The sequence of observed to perform inference on.

        Returns:
            list: The most likely sequence of hidden states.

        Raises:
            InvalidInputError: If Connor hasn't implement the method you wish to use.
                               Your observed has an value not appearing in the statistical model.
        """
        # Check observed
        for o in observed:
            if not self.statistical_model.is_valid_observed_state(o):
                raise InvalidInputError(
                    "In Viterbi, observation contains invalid states."
                )

        # viterbi
        if (
            self.oracle_based
            and self.num_solutions == 1
            and isinstance(self.statistical_model, HiddenMarkovModel)
        ):
            return self._viterbi(observed)

        elif (
            self.oracle_based
            and self.num_solutions == 1
            and isinstance(self.statistical_model, Oracle_CHMM)
            and not self.statistical_model.constraints
        ):
            return self._viterbi(observed)

        # a_star
        elif self.oracle_based and isinstance(self.statistical_model, Oracle_CHMM):
            return self._a_star(observed)

        elif (
            self.oracle_based
            and isinstance(self.statistical_model, HiddenMarkovModel)
            and self.num_solutions > 1
        ):
            return self._a_star(observed)

        else:
            raise InvalidInputError(
                "Infernce is not prepared (yet!) to deal with these parameters."
            )

    def _viterbi(self, observed):
        return viterbi(observed=observed, hmm=self.statistical_model)

    def _a_star(self, observed):
        return a_star(
            statistical_model=self.statistical_model,
            num_solutions=self.num_solutions,
            observed=observed,
        )
