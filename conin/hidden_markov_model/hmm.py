import numpy as np
import pprint
import math
import json
import ast

from conin.exceptions import InvalidInputError
from conin.util import Util


def _validate_start_vec(start_vec, check_errors=True):
    """Validate (optionally) and copy a starting-probability vector.

    Parameters
    ----------
    start_vec : array-like
        Starting probabilities for hidden states.
    check_errors : bool, optional
        If ``True``, validate that entries are nonnegative and sum to 1.

    Returns
    -------
    list
        A fresh copy of ``start_vec``.

    Raises
    ------
    InvalidInputError
        If ``start_vec`` contains negative entries or does not sum to 1.
    """
    if check_errors:
        # Confirm that the start_vec is non-negative
        for prob in start_vec:
            if not prob >= 0:
                raise InvalidInputError("start_probs values must be positive floats.")
        # Confirm that the start_vec sums to one
        if not np.isclose(sum(start_vec), 1):
            raise InvalidInputError("start_prob values must sum to 1.")

    return list(start_vec)


def _validate_transition_mat(transition_mat, check_errors=True):
    """Validate (optionally) and copy a hidden-state transition matrix.

    Parameters
    ----------
    transition_mat : array-like
        Transition probabilities between hidden states.
    check_errors : bool, optional
        If ``True``, validate that entries are nonnegative and each row sums to 1.

    Returns
    -------
    list
        A fresh copy of ``transition_mat`` (list of lists).

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
                    raise InvalidInputError("Transition_mat must be positive floats.")
        # Rows sum to 1
        for vec in transition_mat:
            if not np.isclose(sum(vec), 1):
                raise InvalidInputError("Transition_mat rows do not sum to 1.")

    return [list(row) for row in transition_mat]


def _validate_emission_mat(emission_mat, check_errors=True):
    """Validate (optionally) and copy an emission probability matrix.

    Parameters
    ----------
    emission_mat : array-like
        Emission probabilities from hidden states to observed states.
    check_errors : bool, optional
        If ``True``, validate that entries are nonnegative and each row sums to 1.

    Returns
    -------
    list
        A fresh copy of ``emission_mat`` (list of lists).

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

    return [list(row) for row in emission_mat]


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
        self._start_vec = _validate_start_vec(start_vec, check_errors=check_errors)
        self._transition_mat = _validate_transition_mat(
            transition_mat, check_errors=check_errors
        )
        self._emission_mat = _validate_emission_mat(
            emission_mat, check_errors=check_errors
        )
        if check_errors:
            self._check_dimensions()
        self._load_dimensions()

    @property
    def start_vec(self):
        """Return a copy of the starting probability vector.

        Returns
        -------
        list
            Starting probabilities for hidden states.
        """
        return list(self._start_vec)

    @property
    def transition_mat(self):
        """Return a copy of the hidden-state transition matrix.

        Returns
        -------
        list
            Transition probabilities between hidden states.
        """
        return [list(row) for row in self._transition_mat]

    @property
    def emission_mat(self):
        """Return a copy of the emission probability matrix.

        Returns
        -------
        list
            Emission probabilities from hidden states to observed states.
        """
        return [list(row) for row in self._emission_mat]

    @property
    def num_hidden_states(self):
        """Return the number of hidden states."""
        return self._num_hidden_states

    @property
    def num_observed_states(self):
        """Return the number of observed states."""
        return self._num_observed_states

    @property
    def hidden_states(self):
        """Return the range of hidden-state indices."""
        return self._hidden_states

    @property
    def observed_states(self):
        """Return the range of observed-state indices."""
        return self._observed_states

    def _check_dimensions(self):
        """Validate that the model arrays have compatible dimensions.

        Raises
        ------
        InvalidInputError
            If the starting vector, transition matrix, and emission matrix do not have
            compatible shapes.
        """
        correct_dimension = True

        if len(self._start_vec) != len(self._transition_mat):
            correct_dimension = False  # pragma: no cover
        for vec in self._transition_mat:
            if len(self._start_vec) != len(vec):
                correct_dimension = False  # pragma: no cover

        if len(self._start_vec) != len(self._emission_mat):
            correct_dimension = False  # pragma: no cover
        for vec in self._emission_mat:
            if len(self._emission_mat[0]) != len(vec):
                correct_dimension = False  # pragma: no cover

        if not correct_dimension:
            raise InvalidInputError(
                "Dimensions do not line up correctly in check_dimensions, you shouldn't see this."  # pragma: no cover
            )

    def _load_dimensions(self):
        """Compute and cache hidden-state and observed-state dimensions."""
        self._num_hidden_states = len(self._start_vec)
        self._num_observed_states = len(self._emission_mat[0])
        self._hidden_states = range(self._num_hidden_states)
        self._observed_states = range(self._num_observed_states)

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
        hidden.append(Util.sample_from_vec(self._start_vec))

        # Sample subsequent hidden states
        for t in range(time_steps - 1):
            hidden.append(Util.sample_from_vec(self._transition_mat[hidden[t]]))

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
        hidden.append(Util.sample_from_vec(self._start_vec))

        # Sample until the last hidden state is h
        while hidden[-1] != h:
            hidden.append(Util.sample_from_vec(self._transition_mat[hidden[-1]]))

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
            observed.append(Util.sample_from_vec(self._emission_mat[hidden[t]]))

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

    Parameters
    ----------
    start_vec : list, optional
        Starting probabilities for hidden states in internal index order.
    transition_mat : list, optional
        Transition probabilities between internal hidden states.
    emission_mat : list, optional
        Emission probabilities from internal hidden states to observed states.
    hidden_to_internal : dict, optional
        Mapping from external hidden-state labels to internal indices.
    observed_to_internal : dict, optional
        Mapping from external observed-state labels to internal indices.
    hidden_to_external : list, optional
        Mapping from internal hidden-state indices to external labels.
    observed_to_external : list, optional
        Mapping from internal observed-state indices to external labels.

    Notes
    -----
    This is a low-level constructor that takes fully-resolved arrays and label
    maps. All arguments are optional, but either none or all of them must be
    given (an "empty" model with no args behaves identically to one built with
    all of its collections empty). Prefer :meth:`load_model` or
    :meth:`read_from_file` to build a model from dictionary-based or
    file-based inputs.
    """

    def __init__(
        self,
        *,
        start_vec=None,
        transition_mat=None,
        emission_mat=None,
        hidden_to_internal=None,
        observed_to_internal=None,
        hidden_to_external=None,
        observed_to_external=None,
    ):
        args = (
            start_vec,
            transition_mat,
            emission_mat,
            hidden_to_internal,
            observed_to_internal,
            hidden_to_external,
            observed_to_external,
        )
        given = [arg is not None for arg in args]
        if any(given) and not all(given):
            raise InvalidInputError(
                "If any of start_vec/transition_mat/emission_mat/"
                "hidden_to_internal/observed_to_internal/hidden_to_external/"
                "observed_to_external is given, all must be given."
            )

        self._repn = None
        self._start_vec = list(start_vec) if start_vec is not None else []
        self._transition_mat = (
            [list(row) for row in transition_mat]
            if transition_mat is not None
            else []
        )
        self._emission_mat = (
            [list(row) for row in emission_mat] if emission_mat is not None else []
        )
        self._hidden_to_internal = (
            dict(hidden_to_internal) if hidden_to_internal is not None else {}
        )
        self._observed_to_internal = (
            dict(observed_to_internal) if observed_to_internal is not None else {}
        )
        self._hidden_to_external = (
            list(hidden_to_external) if hidden_to_external is not None else []
        )
        self._observed_to_external = (
            list(observed_to_external) if observed_to_external is not None else []
        )
        self._num_hidden_states = (
            len(self._hidden_to_internal) if hidden_to_internal is not None else None
        )
        self._num_observed_states = (
            len(self._observed_to_internal)
            if observed_to_internal is not None
            else None
        )

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

    @property
    def start_vec(self):
        """Return a copy of the starting-probability vector."""
        return list(self._start_vec)

    @property
    def transition_mat(self):
        """Return a copy of the hidden-state transition matrix."""
        return [list(row) for row in self._transition_mat]

    @property
    def emission_mat(self):
        """Return a copy of the emission probability matrix."""
        return [list(row) for row in self._emission_mat]

    @property
    def hidden_to_internal(self):
        """Return a copy of the mapping from hidden-state labels to indices."""
        return dict(self._hidden_to_internal)

    @property
    def observed_to_internal(self):
        """Return a copy of the mapping from observed-state labels to indices."""
        return dict(self._observed_to_internal)

    @property
    def hidden_to_external(self):
        """Return a copy of the mapping from hidden-state indices to labels."""
        return list(self._hidden_to_external)

    @property
    def observed_to_external(self):
        """Return a copy of the mapping from observed-state indices to labels."""
        return list(self._observed_to_external)

    @property
    def num_hidden_states(self):
        """Return the number of hidden states, or ``None`` if unset."""
        return self._num_hidden_states

    @property
    def num_observed_states(self):
        """Return the number of observed states, or ``None`` if unset."""
        return self._num_observed_states

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

    @classmethod
    def load_model(
        cls, *, start_probs, transition_probs, emission_probs, initialize=False
    ):
        """Build a new model from dictionary-based inputs.

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

        Returns
        -------
        HiddenMarkovModel
            A new model built from the supplied probabilities.

        Raises
        ------
        InvalidInputError
            If the supplied probability dictionaries are inconsistent with each other.
        """
        hidden_to_internal = {}  # Maps from labels to indices
        hidden_to_external = []  # Maps from indices to labels
        observed_to_internal = {}
        observed_to_external = []

        # Setup hidden_to_internal, hidden_to_external, and num_hidden_states
        for h1, h2 in sorted(transition_probs.keys()):
            if h1 not in hidden_to_internal.keys():
                hidden_to_external.append(h1)
                hidden_to_internal[h1] = len(hidden_to_external) - 1
            if h2 not in hidden_to_internal.keys():
                hidden_to_external.append(h2)
                hidden_to_internal[h2] = len(hidden_to_external) - 1
        num_hidden_states = len(hidden_to_internal)

        if not set(start_probs.keys()).issubset(set(hidden_to_external)):
            raise InvalidInputError("start_prob keys match with transition keys")

        # Setup observed_to_internal, observed_to_external, and
        # num_observed_states
        for h, o in sorted(emission_probs.keys()):
            if o not in observed_to_internal:
                observed_to_external.append(o)
                observed_to_internal[o] = len(observed_to_external) - 1
        num_observed_states = len(observed_to_internal)

        # Setup start_vec
        start_vec = [0] * num_hidden_states
        for h, prob in start_probs.items():
            start_vec[hidden_to_internal[h]] = prob

        # Setup transition_mat
        transition_mat = [
            [0 for _ in range(num_hidden_states)] for _ in range(num_hidden_states)
        ]
        for (h1, h2), prob in transition_probs.items():
            # No new hidden states
            if (h1 not in hidden_to_internal) or (h2 not in hidden_to_internal):
                raise InvalidInputError("You shouldn't see this")  # pragma: no cover
            transition_mat[hidden_to_internal[h1]][hidden_to_internal[h2]] = prob

        # Setup emission_mat
        emission_mat = [
            [0 for _ in range(num_observed_states)] for _ in range(num_hidden_states)
        ]
        for (h, o), prob in emission_probs.items():
            # No new hidden states
            if h not in hidden_to_internal:
                raise InvalidInputError(
                    "start_probs does not contain all hidden states appearing in emission_probs, (",
                    h,
                    ", ",
                    o,
                    "): ",
                    prob,
                )
            emission_mat[hidden_to_internal[h]][observed_to_internal[o]] = prob

        hmm = cls(
            start_vec=start_vec,
            transition_mat=transition_mat,
            emission_mat=emission_mat,
            hidden_to_internal=hidden_to_internal,
            observed_to_internal=observed_to_internal,
            hidden_to_external=hidden_to_external,
            observed_to_external=observed_to_external,
        )

        if initialize:
            hmm.initialize(True)

        return hmm

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
        if not self._start_vec:
            return self._repn

        self._repn = HMM_MatVecRepn(
            start_vec=self._start_vec,
            transition_mat=self._transition_mat,
            emission_mat=self._emission_mat,
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
        return o in self._observed_to_internal

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
        return h in self._hidden_to_internal

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
            self._hidden_to_external[h]: self._start_vec[h]
            for h in range(self._num_hidden_states)
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
                self._hidden_to_external[h1],
                self._hidden_to_external[h2],
            ): self._transition_mat[h1][h2]
            for h1 in range(self._num_hidden_states)
            for h2 in range(self._num_hidden_states)
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
                self._hidden_to_external[h],
                self._observed_to_external[o],
            ): self._emission_mat[h][o]
            for h in range(self._num_hidden_states)
            for o in range(self._num_observed_states)
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
            (self._hidden_to_external[i], v)
            for i, v in enumerate(self._start_vec)
            if v > tolerance
        ]
        transition_probs = [
            ((self._hidden_to_external[i], self._hidden_to_external[j]), v)
            for i, row in enumerate(self._transition_mat)
            for j, v in enumerate(row)
            if v > tolerance
        ]
        emission_probs = [
            ((self._hidden_to_external[i], self._observed_to_external[o]), v)
            for i, row in enumerate(self._emission_mat)
            for o, v in enumerate(row)
            if v > tolerance
        ]

        return dict(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            num_hidden=self._num_hidden_states,
            num_observed=self._num_observed_states,
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
        return [self._hidden_to_external[h] for h in internal_hidden]

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
            self._hidden_to_internal[h]
        )
        return [self._hidden_to_external[h] for h in internal_hidden]

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
        internal_hidden = [self._hidden_to_internal[h] for h in hidden]
        internal_observed = self.repn.generate_observed_from_hidden(internal_hidden)
        return [self._observed_to_external[o] for o in internal_observed]

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
        return [self._observed_to_external[o] for o in internal_observed]

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

        h = [self._hidden_to_internal[hval] for hval in hidden]
        o = [self._observed_to_internal[oval] for oval in observed]

        ans = math.log(self._start_vec[h[0]]) + math.log(
            self._emission_mat[h[0]][o[0]]
        )
        for t in range(1, len(observed)):
            ans += math.log(self._transition_mat[h[t - 1]][h[t]]) + math.log(
                self._emission_mat[h[t]][o[t]]
            )

        return ans

    def make_non_zero(self, tol=1e-6):
        """Return a new model with small probabilities floored and renormalized.

        Parameters
        ----------
        tol : float, optional
            Minimum probability assigned before renormalization.

        Returns
        -------
        HiddenMarkovModel
            A new model with floored and renormalized probabilities; ``self``
            is not modified.
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

        return HiddenMarkovModel.load_model(
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

    @classmethod
    def read_from_file(cls, file_name):
        """Build a new model from parameters stored in a JSON file.

        Parameters
        ----------
        file_name : str or path-like
            Source file path.

        Returns
        -------
        HiddenMarkovModel
            A new model built from the file's parameters.
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

        return cls.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )
