import numpy as np
import pprint
import math
import json
import ast

from conin.exceptions import InvalidInputError
from conin.util import Util, as_rng
from conin.util.frozen import freeze
from conin.graphical_models.base import GraphicalModel


def _validate_start_vec(start_vec):
    """Check that the starting probabilities are nonnegative and sum to 1.

    Raises
    ------
    InvalidInputError
        If ``start_vec`` contains negative entries or does not sum to 1.
    """
    for prob in start_vec:
        if not prob >= 0:
            raise InvalidInputError("start_probs values must be positive floats.")
    if not np.isclose(sum(start_vec), 1):
        raise InvalidInputError("start_prob values must sum to 1.")


def _validate_transition_mat(transition_mat):
    """Check that the transition probabilities are nonnegative and each row sums to 1.

    Raises
    ------
    InvalidInputError
        If ``transition_mat`` contains negative entries or rows that do not sum to
        1.
    """
    for row in transition_mat:
        for prob in row:
            if not prob >= 0:
                raise InvalidInputError("Transition_mat must be positive floats.")
    for vec in transition_mat:
        if not np.isclose(sum(vec), 1):
            raise InvalidInputError("Transition_mat rows do not sum to 1.")


def _validate_emission_mat(emission_mat):
    """Check that the emission probabilities are nonnegative and each row sums to 1.

    Raises
    ------
    InvalidInputError
        If ``emission_mat`` contains negative entries or rows that do not sum to
        1.
    """
    for row in emission_mat:
        for prob in row:
            if not prob >= 0:
                raise InvalidInputError("Emission_mat must be positive floats.")
    for vec in emission_mat:
        if not np.isclose(sum(vec), 1):
            raise InvalidInputError(f"Emission_mat rows do not sum to 1: {sum(vec)}")


class HiddenMarkovModel(GraphicalModel):
    """Hidden Markov model over labelled hidden and observed states.

    Parameters
    ----------
    start_probs : dict
        Mapping from hidden-state labels to starting probabilities.
    transition_probs : dict
        Mapping from ``(from_state, to_state)`` pairs to transition probabilities.
    emission_probs : dict
        Mapping from ``(hidden_state, observed_state)`` pairs to emission
        probabilities.

    Raises
    ------
    InvalidInputError
        If any of the probability dictionaries is empty, if they are
        inconsistent with each other, or if a probability is negative or the
        start probabilities, a row of transition probabilities or a row of
        emission probabilities do not sum to 1.

    Notes
    -----
    The hidden states are the labels appearing in ``transition_probs``, and the
    observed states are the observed labels appearing in ``emission_probs``.
    Both are sorted (:attr:`hidden_states`, :attr:`observed_states`),
    independent of dictionary insertion order and of which entries are
    present. Missing entries are treated as zero probabilities.

    The model holds label-keyed tables only.  The parameters as arrays on
    state indices (position ``i`` of :attr:`hidden_states` is index ``i``),
    and the label-to-index maps, are in the ``"matvec"`` representation:
    ``hmm.representation("matvec")``.
    """

    def __init__(
        self,
        *,
        start_probs,
        transition_probs,
        emission_probs,
    ):
        start_probs = dict(start_probs)
        transition_probs = dict(transition_probs)
        emission_probs = dict(emission_probs)
        for name, probs in (
            ("start_probs", start_probs),
            ("transition_probs", transition_probs),
            ("emission_probs", emission_probs),
        ):
            if not probs:
                raise InvalidInputError(f"{name} must be a non-empty dictionary")

        # Internal indices follow the sorted order of the state labels
        hidden_to_external = sorted({h for key in transition_probs for h in key})
        hidden_to_internal = {h: i for i, h in enumerate(hidden_to_external)}
        observed_to_external = sorted({o for _, o in emission_probs})
        observed_to_internal = {o: i for i, o in enumerate(observed_to_external)}
        num_hidden_states = len(hidden_to_external)
        num_observed_states = len(observed_to_external)

        # Setup start_vec
        start_vec = [0] * num_hidden_states
        for h, prob in start_probs.items():
            if h not in hidden_to_internal:
                raise InvalidInputError(
                    f"start_probs contains hidden state {h!r} that does not "
                    "appear in transition_probs"
                )
            start_vec[hidden_to_internal[h]] = prob

        # Setup transition_mat
        transition_mat = [[0] * num_hidden_states for _ in range(num_hidden_states)]
        for (h1, h2), prob in transition_probs.items():
            transition_mat[hidden_to_internal[h1]][hidden_to_internal[h2]] = prob

        # Setup emission_mat
        emission_mat = [[0] * num_observed_states for _ in range(num_hidden_states)]
        for (h, o), prob in emission_probs.items():
            if h not in hidden_to_internal:
                raise InvalidInputError(
                    f"emission_probs contains hidden state {h!r} that does not "
                    f"appear in transition_probs: ({h!r}, {o!r}): {prob}"
                )
            emission_mat[hidden_to_internal[h]][observed_to_internal[o]] = prob

        _validate_start_vec(start_vec)
        _validate_transition_mat(transition_mat)
        _validate_emission_mat(emission_mat)

        # Dense label-keyed tables (zeros included), in index order
        self._hidden_states = freeze(hidden_to_external)
        self._observed_states = freeze(observed_to_external)
        self._start_probs = freeze(dict(zip(hidden_to_external, start_vec)))
        self._transition_probs = freeze(
            {
                (h1, h2): transition_mat[i][j]
                for i, h1 in enumerate(hidden_to_external)
                for j, h2 in enumerate(hidden_to_external)
            }
        )
        self._emission_probs = freeze(
            {
                (h, o): emission_mat[i][j]
                for i, h in enumerate(hidden_to_external)
                for j, o in enumerate(observed_to_external)
            }
        )

    def __str__(self):
        """Return a readable dictionary-style representation of the model."""
        return pprint.pformat(self.to_dict(tolerance=1e-3), indent=4, sort_dicts=True)

    @property
    def num_hidden_states(self):
        """Return the number of hidden states."""
        return len(self._hidden_states)

    @property
    def num_observed_states(self):
        """Return the number of observed states."""
        return len(self._observed_states)

    @property
    def hidden_states(self):
        """Return the hidden-state labels.

        Returns
        -------
        list
            Sorted hidden-state labels (read-only); position ``i`` is the
            state index of the ``"matvec"`` representation.
        """
        return self._hidden_states

    @property
    def observed_states(self):
        """Return the observed-state labels.

        Returns
        -------
        list
            Sorted observed-state labels (read-only); position ``j`` is the
            state index of the ``"matvec"`` representation.
        """
        return self._observed_states

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
        return o in self._observed_states

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
        return h in self._hidden_states

    def get_hidden_states(self):
        """Return the external hidden-state labels.

        Returns
        -------
        list
            Hidden-state labels in internal index order.
        """
        return self.hidden_states

    def get_observable_states(self):
        """Return the external observed-state labels.

        Returns
        -------
        list
            Observed-state labels in internal index order.
        """
        return self.observed_states

    def get_start_probs(self):
        """Return the starting probabilities keyed by hidden-state label.

        Returns
        -------
        dict
            Mapping from hidden-state labels to starting probabilities.
        """
        # Same format as the constructor arguments
        return dict(self._start_probs)

    def get_transition_probs(self):
        # Same format as the constructor arguments
        """Return the transition probabilities keyed by hidden-state pairs.

        Returns
        -------
        dict
            Mapping from ``(from_state, to_state)`` pairs to transition
            probabilities.
        """
        return dict(self._transition_probs)

    def get_emission_probs(self):
        """Return the emission probabilities keyed by state pairs.

        Returns
        -------
        dict
            Mapping from ``(hidden_state, observed_state)`` pairs to emission
            probabilities.
        """
        # Same format as the constructor arguments
        return dict(self._emission_probs)

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

        start_probs = [(h, v) for h, v in self._start_probs.items() if v > tolerance]
        transition_probs = [
            (key, v) for key, v in self._transition_probs.items() if v > tolerance
        ]
        emission_probs = [
            (key, v) for key, v in self._emission_probs.items() if v > tolerance
        ]

        return dict(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            num_hidden=self.num_hidden_states,
            num_observed=self.num_observed_states,
        )

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

        ans = math.log(self._start_probs[hidden[0]]) + math.log(
            self._emission_probs[hidden[0], observed[0]]
        )
        for t in range(1, len(observed)):
            ans += math.log(
                self._transition_probs[hidden[t - 1], hidden[t]]
            ) + math.log(self._emission_probs[hidden[t], observed[t]])

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

        return HiddenMarkovModel(
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

        return cls(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )


def random_hmm(*, hidden_states, observed_states, rng=None):
    """Generate a random hidden Markov model.

    Parameters
    ----------
    hidden_states : iterable
        Hidden state labels for the generated model.
    observed_states : iterable
        Observable state labels for the generated model.
    rng : None, int or numpy.random.Generator, optional
        The random generator, or a seed for a new one (see
        :func:`conin.util.as_rng`); pass a seed for a reproducible model.

    Returns
    -------
    HiddenMarkovModel
        Hidden Markov model with randomly generated start, transition, and
        emission probabilities.
    """
    rng = as_rng(rng)

    S = Util.normalize_dictionary({s: rng.random() for s in hidden_states})

    T = Util.normalize_2d_dictionary(
        {(a, b): rng.random() for a in hidden_states for b in hidden_states}
    )
    E = Util.normalize_2d_dictionary(
        {(s, o): rng.random() for s in hidden_states for o in observed_states}
    )

    hmm = HiddenMarkovModel(start_probs=S, transition_probs=T, emission_probs=E)
    return hmm
