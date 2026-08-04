import numpy as np
import pprint
import math
import json
import ast

from conin.exceptions import InvalidInputError
from conin.util import Util


class HMM_MatVecRepn:
    """Matrix/vector representation of a hidden Markov model.

    Parameters
    ----------
    start_vec : array-like
        Starting probabilities for hidden states.
    transition_mat : array-like
        Transition probabilities between hidden states.
    emission_mat : array-like
        Emission probabilities from hidden states to observed states.
    check_errors : bool, optional
        If ``True``, validate probabilities and dimensions during
        initialization.
    """

    def __init__(self, *, start_vec, transition_mat, emission_mat, check_errors=True):
        self.load_start_vec(start_vec, check_errors=check_errors)
        self.load_transition_mat(transition_mat, check_errors=check_errors)
        self.load_emission_mat(emission_mat, check_errors=check_errors)
        if check_errors:
            self.check_dimensions()
        self.load_dimensions()

    def load_start_vec(self, start_vec, check_errors=True):
        """Load the starting probability vector.

        Parameters
        ----------
        start_vec : array-like
            Starting probabilities for hidden states.
        check_errors : bool, optional
            If ``True``, validate that entries are nonnegative and sum to 1.

        Raises
        ------
        InvalidInputError
            If ``start_vec`` contains negative entries or does not sum to 1.
        """
        if check_errors:
            # Confirm that the start_vec is non-negative
            for prob in start_vec:
                if not prob >= 0:
                    raise InvalidInputError(
                        "start_probs values must be positive floats."
                    )
            # Confirm that the start_vec sums to one
            if not np.isclose(sum(start_vec), 1):
                raise InvalidInputError("start_prob values must sum to 1.")

        self.start_vec = start_vec

    def load_transition_mat(self, transition_mat, check_errors=True):
        """Load the hidden-state transition matrix.

        Parameters
        ----------
        transition_mat : array-like
            Transition probabilities between hidden states.
        check_errors : bool, optional
            If ``True``, validate that entries are nonnegative and each row sums to 1.

        Raises
        ------
        InvalidInputError
            If ``transition_mat`` contains negative entries or rows that do not sum to
            1.
        """
        if check_errors:
            # Non-negative transition probabilities
            for h1 in range(len(transition_mat)):
                for h2 in range(len(transition_mat[h1])):
                    if not transition_mat[h1][h2] >= 0:
                        raise InvalidInputError(
                            "Transition_mat must be positive floats."
                        )
            # Rows sum to 1
            for vec in transition_mat:
                if not np.isclose(sum(vec), 1):
                    raise InvalidInputError("Transition_mat rows do not sum to 1.")

        self.transition_mat = transition_mat

    def load_emission_mat(self, emission_mat, check_errors=True):
        """Load the emission probability matrix.

        Parameters
        ----------
        emission_mat : array-like
            Emission probabilities from hidden states to observed states.
        check_errors : bool, optional
            If ``True``, validate that entries are nonnegative and each row sums to 1.

        Raises
        ------
        InvalidInputError
            If ``emission_mat`` contains negative entries or rows that do not sum to
            1.
        """
        if check_errors:
            # Non-negative emission probabilities
            for h1 in range(len(emission_mat)):
                for h2 in range(len(emission_mat[h1])):
                    if not emission_mat[h1][h2] >= 0:
                        raise InvalidInputError("Emission_mat must be positive floats.")
            # Rows sum to 1
            for vec in emission_mat:
                if not np.isclose(sum(vec), 1):
                    raise InvalidInputError(
                        f"Emission_mat rows do not sum to 1: {sum(vec)}"
                    )

        self.emission_mat = emission_mat

    def check_dimensions(self):
        """Validate that the model arrays have compatible dimensions.

        Raises
        ------
        InvalidInputError
            If the starting vector, transition matrix, and emission matrix do not have
            compatible shapes.
        """
        correct_dimension = True

        if len(self.start_vec) != len(self.transition_mat):
            correct_dimension = False  # pragma: no cover
        for vec in self.transition_mat:
            if len(self.start_vec) != len(vec):
                correct_dimension = False  # pragma: no cover

        if len(self.start_vec) != len(self.emission_mat):
            correct_dimension = False  # pragma: no cover
        for vec in self.emission_mat:
            if len(self.emission_mat[0]) != len(vec):
                correct_dimension = False  # pragma: no cover

        if not correct_dimension:
            raise InvalidInputError(
                "Dimensions do not line up correctly in check_dimensions, you shouldn't see this."  # pragma: no cover
            )

    def load_dimensions(self):
        """Update cached hidden-state and observed-state dimensions."""
        self.num_hidden_states = len(self.start_vec)
        self.num_observed_states = len(self.emission_mat[0])
        self.hidden_states = range(self.num_hidden_states)
        self.observed_states = range(self.num_observed_states)

    def generate_hidden(self, time_steps):
        """Generate a sequence of hidden-state indices.

        Parameters
        ----------
        time_steps : int
            Number of time steps for which to generate hidden states.

        Returns
        -------
        list
            Generated hidden-state indices.

        Raises
        ------
        InvalidInputError
            If ``time_steps`` is negative.
        """
        if time_steps < 0:
            raise InvalidInputError("In generate_hidden time_steps > 0.")
        hidden = []

        # Sample the first hidden state
        hidden.append(Util.sample_from_vec(self.start_vec))

        # Sample subsequent hidden states
        for t in range(time_steps - 1):
            hidden.append(Util.sample_from_vec(self.transition_mat[hidden[t]]))

        return hidden

    def generate_hidden_until_state(self, h):
        """Generate hidden-state indices until a target state is reached.

        Parameters
        ----------
        h : int
            Target hidden-state index at which sampling stops.

        Returns
        -------
        list
            Generated hidden-state indices ending in ``h``.
        """
        hidden = []

        # Sample the first hidden state
        hidden.append(Util.sample_from_vec(self.start_vec))

        # Sample until the last hidden state is h
        while hidden[-1] != h:
            hidden.append(Util.sample_from_vec(self.transition_mat[hidden[-1]]))

        return hidden

    def generate_observed_from_hidden(self, hidden):
        """Generate observed-state indices from hidden-state indices.

        Parameters
        ----------
        hidden : list
            Hidden-state indices used to sample observations.

        Returns
        -------
        list
            Generated observed-state indices.
        """
        observed = []
        time_steps = len(hidden)

        for t in range(time_steps):
            observed.append(Util.sample_from_vec(self.emission_mat[hidden[t]]))

        return observed

    def generate_observed(self, time_steps):
        """Generate observed-state indices from the model.

        Parameters
        ----------
        time_steps : int
            Number of time steps for which to generate observed states.

        Returns
        -------
        list
            Generated observed-state indices.

        Raises
        ------
        InvalidInputError
            If ``time_steps`` is negative.
        """
        if time_steps < 0:
            raise InvalidInputError("In generate_observed time_steps > 0.")
        hidden = self.generate_hidden(time_steps)
        return self.generate_observed_from_hidden(hidden)


class HiddenMarkovModel:
    """Hidden Markov model with external-label and numeric representations.

    Attributes
    ----------
    start_vec : list
        Starting probabilities for hidden states in internal index order.
    transition_mat : list
        Transition probabilities between internal hidden states.
    emission_mat : list
        Emission probabilities from internal hidden states to observed states.
    hidden_to_internal : dict
        Mapping from external hidden-state labels to internal indices.
    observed_to_internal : dict
        Mapping from external observed-state labels to internal indices.
    hidden_to_external : list
        Mapping from internal hidden-state indices to external labels.
    observed_to_external : list
        Mapping from internal observed-state indices to external labels.
    num_hidden_states : int or None
        Number of hidden states.
    num_observed_states : int or None
        Number of observed states.
    """

    def __init__(self):
        """Initialize an empty hidden Markov model."""
        self._repn = None
        self.start_vec = []
        self.transition_mat = []
        self.emission_mat = []
        self.hidden_to_internal = {}  # Maps from labels to indices
        self.observed_to_internal = {}
        self.hidden_to_external = []  # Maps from labels to indices
        self.observed_to_external = []
        self.num_hidden_states = None  # Number of hidden variables
        self.num_observed_states = None  # Number of observed variables

    def __str__(self):
        """Return a readable dictionary-style representation of the model."""
        return pprint.pformat(self.to_dict(tolerance=1e-3), indent=4, sort_dicts=True)

    @property
    def repn(self):
        """Return the numeric matrix/vector representation of the model.

        Returns
        -------
        HMM_MatVecRepn or None
            Numeric representation of the model, or ``None`` if the model has
            not yet been populated.
        """
        return self.initialize()

    @repn.setter
    def repn(self, hmm):
        """Set the numeric matrix/vector representation of the model.

        Parameters
        ----------
        hmm : HMM_MatVecRepn or None
            Numeric representation to cache on the model.
        """
        self._repn = hmm

    @property
    def hidden_states(self):
        """Return the external hidden-state labels.

        Returns
        -------
        list
            Hidden-state labels in internal index order.
        """
        return self.hidden_to_external

    @property
    def observed_states(self):
        """Return the external observed-state labels.

        Returns
        -------
        list
            Observed-state labels in internal index order.
        """
        return self.observed_to_external

    def load_model(
        self, *, start_probs, transition_probs, emission_probs, initialize=False
    ):
        """Load model probabilities from dictionary-based inputs.

        Parameters
        ----------
        start_probs : dict
            Mapping from hidden-state labels to starting probabilities.
        transition_probs : dict
            Mapping from ``(from_state, to_state)`` pairs to transition probabilities.
        emission_probs : dict
            Mapping from ``(hidden_state, observed_state)`` pairs to emission
            probabilities.
        initialize : bool, optional
            If ``True``, immediately construct the numeric representation.

        Raises
        ------
        InvalidInputError
            If the supplied probability dictionaries are inconsistent with each other.
        """
        self._repn = None
        self.start_vec = []
        self.transition_mat = []
        self.emission_mat = []
        self.hidden_to_internal = {}  # Maps from labels to indices
        self.observed_to_internal = {}
        self.hidden_to_external = []  # Maps from labels to indices
        self.observed_to_external = []
        self.num_hidden_states = None  # Number of hidden variables
        self.num_observed_states = None  # Number of observed variables

        # Setup hidden_to_internal, hidden_to_external, and num_hidden_states
        for h1, h2 in sorted(transition_probs.keys()):
            if h1 not in self.hidden_to_internal.keys():
                self.hidden_to_external.append(h1)
                self.hidden_to_internal[h1] = len(self.hidden_to_external) - 1
            if h2 not in self.hidden_to_internal.keys():
                self.hidden_to_external.append(h2)
                self.hidden_to_internal[h2] = len(self.hidden_to_external) - 1
        self.num_hidden_states = len(self.hidden_to_internal)

        if not set(start_probs.keys()).issubset(set(self.hidden_to_external)):
            raise InvalidInputError("start_prob keys match with transition keys")

        # Setup observed_to_internal, observed_to_external, and
        # num_observed_states
        for h, o in sorted(emission_probs.keys()):
            if o not in self.observed_to_internal:
                self.observed_to_external.append(o)
                self.observed_to_internal[o] = len(self.observed_to_external) - 1
        self.num_observed_states = len(self.observed_to_internal)

        # Setup start_vec
        self.start_vec = [0] * self.num_hidden_states
        for h, prob in start_probs.items():
            self.start_vec[self.hidden_to_internal[h]] = prob

        # Setup transition_mat
        self.transition_mat = [
            [0 for _ in range(self.num_hidden_states)]
            for _ in range(self.num_hidden_states)
        ]
        for (h1, h2), prob in transition_probs.items():
            # No new hidden states
            if (h1 not in self.hidden_to_internal) or (
                h2 not in self.hidden_to_internal
            ):
                raise InvalidInputError("You shouldn't see this")  # pragma: no cover
            self.transition_mat[self.hidden_to_internal[h1]][
                self.hidden_to_internal[h2]
            ] = prob

        # Setup emission_mat
        self.emission_mat = [
            [0 for _ in range(self.num_observed_states)]
            for _ in range(self.num_hidden_states)
        ]
        for (h, o), prob in emission_probs.items():
            # No new hidden states
            if h not in self.hidden_to_internal:
                raise InvalidInputError(
                    "start_probs does not contain all hidden states appearing in emission_probs, (",
                    h,
                    ", ",
                    o,
                    "): ",
                    prob,
                )
            self.emission_mat[self.hidden_to_internal[h]][
                self.observed_to_internal[o]
            ] = prob

        if initialize:
            self.initialize(True)

    def initialize(self, avoid_reinitialization=True):
        """Construct the numeric matrix/vector representation of the model.

        Parameters
        ----------
        avoid_reinitialization : bool, optional
            If ``True``, reuse an existing representation when one is already
            available.

        Returns
        -------
        HMM_MatVecRepn or None
            Numeric representation of the model, or ``None`` if the model has not yet
            been populated.
        """
        if avoid_reinitialization and self._repn is not None:
            return self._repn
        if not self.start_vec:
            return self._repn

        self._repn = HMM_MatVecRepn(
            start_vec=self.start_vec,
            transition_mat=self.transition_mat,
            emission_mat=self.emission_mat,
        )
        return self._repn

    def is_valid_observed_state(self, o):
        """Check whether an observed-state label is valid.

        Parameters
        ----------
        o : Any
            Observed-state label to validate.

        Returns
        -------
        bool
            ``True`` if ``o`` is a known observed-state label and ``False`` otherwise.
        """
        return o in self.observed_to_internal

    def is_valid_hidden_state(self, h):
        """Check whether a hidden-state label is valid.

        Parameters
        ----------
        h : Any
            Hidden-state label to validate.

        Returns
        -------
        bool
            ``True`` if ``h`` is a known hidden-state label and ``False`` otherwise.
        """
        return h in self.hidden_to_internal

    def set_seed(self, seed):
        """Set the NumPy random seed used for model sampling.

        Parameters
        ----------
        seed : int
            Seed value for the random number generator.
        """
        np.random.seed(seed)

    def get_hidden_states(self):
        """Return the external hidden-state labels.

        Returns
        -------
        list
            Hidden-state labels in internal index order.
        """
        return self.hidden_to_external

    def get_observable_states(self):
        """Return the external observed-state labels.

        Returns
        -------
        list
            Observed-state labels in internal index order.
        """
        return self.observed_to_external

    def get_start_probs(self):
        """Return the starting probabilities keyed by hidden-state label.

        Returns
        -------
        dict
            Mapping from hidden-state labels to starting probabilities.
        """
        # Same format as in load_model
        return {
            self.hidden_to_external[h]: self.start_vec[h]
            for h in range(self.num_hidden_states)
        }

    def get_transition_probs(self):
        # Same format as in load_model
        """Return the transition probabilities keyed by hidden-state pairs.

        Returns
        -------
        dict
            Mapping from ``(from_state, to_state)`` pairs to transition
            probabilities.
        """
        return {
            (
                self.hidden_to_external[h1],
                self.hidden_to_external[h2],
            ): self.transition_mat[h1][h2]
            for h1 in range(self.num_hidden_states)
            for h2 in range(self.num_hidden_states)
        }

    def get_emission_probs(self):
        """Return the emission probabilities keyed by state pairs.

        Returns
        -------
        dict
            Mapping from ``(hidden_state, observed_state)`` pairs to emission
            probabilities.
        """
        # Same format as in load_model
        return {
            (
                self.hidden_to_external[h],
                self.observed_to_external[o],
            ): self.emission_mat[h][o]
            for h in range(self.num_hidden_states)
            for o in range(self.num_observed_states)
        }

    def to_dict(self, tolerance=0.0):
        """Return a dictionary representation of the model data.

        Parameters
        ----------
        tolerance : float, optional
            Minimum probability value to include in the serialized representation.

        Returns
        -------
        dict
            Dictionary representation of the model data.
        """

        start_probs = [
            (self.hidden_to_external[i], v)
            for i, v in enumerate(self.start_vec)
            if v > tolerance
        ]
        transition_probs = [
            ((self.hidden_to_external[i], self.hidden_to_external[j]), v)
            for i, row in enumerate(self.transition_mat)
            for j, v in enumerate(row)
            if v > tolerance
        ]
        emission_probs = [
            ((self.hidden_to_external[i], self.observed_to_external[o]), v)
            for i, row in enumerate(self.emission_mat)
            for o, v in enumerate(row)
            if v > tolerance
        ]

        return dict(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            num_hidden=self.num_hidden_states,
            num_observed=self.num_observed_states,
        )

    # Generation
    def generate_hidden(self, time_steps):
        """Generate a sequence of hidden-state labels.

        Parameters
        ----------
        time_steps : int
            Number of hidden states to generate.

        Returns
        -------
        list
            Generated hidden-state labels.
        """
        internal_hidden = self.repn.generate_hidden(time_steps)
        return [self.hidden_to_external[h] for h in internal_hidden]

    def generate_hidden_until_state(self, h):
        """Generate hidden-state labels until a target state is reached.

        Parameters
        ----------
        h : Any
            Target hidden-state label at which sampling stops.

        Returns
        -------
        list
            Generated hidden-state labels ending in ``h``.
        """
        internal_hidden = self.repn.generate_hidden_until_state(
            self.hidden_to_internal[h]
        )
        return [self.hidden_to_external[h] for h in internal_hidden]

    def generate_observed_from_hidden(self, hidden):
        """Generate observed-state labels from hidden-state labels.

        Parameters
        ----------
        hidden : list
            Hidden-state labels from which to generate observations.

        Returns
        -------
        list
            Generated observed-state labels.
        """
        internal_hidden = [self.hidden_to_internal[h] for h in hidden]
        internal_observed = self.repn.generate_observed_from_hidden(internal_hidden)
        return [self.observed_to_external[o] for o in internal_observed]

    def generate_observed(self, time_steps):
        """Generate observed-state labels from the model.

        Parameters
        ----------
        time_steps : int
            Number of time steps for which to generate observations.

        Returns
        -------
        list
            Generated observed-state labels.
        """
        internal_observed = self.repn.generate_observed(time_steps)
        return [self.observed_to_external[o] for o in internal_observed]

    def log_probability(self, observed, hidden):
        """Compute the joint log-probability of aligned observed and hidden states.

        Parameters
        ----------
        observed : list
            Observed-state labels.
        hidden : list
            Hidden-state labels aligned with ``observed``.

        Returns
        -------
        float
            Log-probability of the paired hidden and observed sequences.
        """

        h = [self.hidden_to_internal[hval] for hval in hidden]
        o = [self.observed_to_internal[oval] for oval in observed]

        ans = math.log(self.start_vec[h[0]]) + math.log(self.emission_mat[h[0]][o[0]])
        for t in range(1, len(observed)):
            ans += math.log(self.transition_mat[h[t - 1]][h[t]]) + math.log(
                self.emission_mat[h[t]][o[t]]
            )

        return ans

    def make_non_zero(self, tol=1e-6):
        """Floor small probabilities and renormalize the model.

        Parameters
        ----------
        tol : float, optional
            Minimum probability assigned before renormalization.
        """
        start_probs = self.get_start_probs()
        transition_probs = self.get_transition_probs()
        emission_probs = self.get_emission_probs()

        for h, val in start_probs.items():
            if val < tol:
                start_probs[h] = tol
        for key, val in transition_probs.items():
            if val < tol:
                transition_probs[key] = tol
        for key, val in emission_probs.items():
            if val < tol:
                emission_probs[key] = tol

        start_probs = Util.normalize_dictionary(start_probs)
        transition_probs = Util.normalize_2d_dictionary(transition_probs)
        emission_probs = Util.normalize_2d_dictionary(emission_probs)

        self.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )

    def write_to_file(self, file_name):
        """Write the model parameters to a JSON file.

        Parameters
        ----------
        file_name : str or path-like
            Destination file path.
        """
        start_probs = self.get_start_probs()
        transition_probs = self.get_transition_probs()
        emission_probs = self.get_emission_probs()

        # Convert tuples to strings for JSON serialization
        transition_probs_serializable = {str(k): v for k, v in transition_probs.items()}
        emission_probs_serializable = {str(k): v for k, v in emission_probs.items()}

        # Create a dictionary to hold all the data
        data = {
            "start_probs": start_probs,
            "transition_probs": transition_probs_serializable,
            "emission_probs": emission_probs_serializable,
        }

        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def read_from_file(self, file_name):
        """Load model parameters from a JSON file.

        Parameters
        ----------
        file_name : str or path-like
            Source file path.
        """

        # Read the data from the JSON file
        with open(file_name, "r") as json_file:
            data = json.load(json_file)

        # Convert string keys back to tuples
        transition_probs = {
            ast.literal_eval(k): v for k, v in data["transition_probs"].items()
        }
        emission_probs = {
            ast.literal_eval(k): v for k, v in data["emission_probs"].items()
        }

        # Extract start probabilities
        start_probs = data["start_probs"]

        self.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )
