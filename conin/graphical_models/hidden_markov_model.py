import numpy as np
import pprint
import json
import ast

from conin.exceptions import InvalidInputError
from conin.util import Util, as_rng
from conin.util.frozen import freeze
from conin.graphical_models.base import GraphicalModel


def _state_order(name, given, labels):
    """The state order: ``given`` (checked against ``labels``), or the sorted labels."""
    if given is None:
        return sorted(labels)
    if isinstance(given, (str, bytes)):
        raise InvalidInputError(f"{name} must be a list of labels, got {given!r}")
    order = list(given)
    if not order:
        raise InvalidInputError(f"{name} must not be empty")
    if len(set(order)) != len(order):
        duplicates = sorted({repr(x) for x in order if order.count(x) > 1})
        raise InvalidInputError(f"{name} has duplicate labels: {', '.join(duplicates)}")
    return order


def _check_row(name, values):
    """Raise unless ``values`` are nonnegative probabilities summing to 1."""
    for key, prob in values.items():
        if not prob >= 0:
            raise InvalidInputError(f"{name} must be nonnegative, got {key!r}: {prob}")
    total = sum(values.values())
    if not np.isclose(total, 1):
        raise InvalidInputError(f"{name} must sum to 1, got {total}")


class HiddenMarkovModel(GraphicalModel):
    """Hidden Markov model over labelled hidden and observable states.

    Parameters
    ----------
    start_probs : dict
        Mapping from hidden-state labels to starting probabilities.
    transition_probs : dict
        Mapping from ``(from_state, to_state)`` pairs to transition probabilities.
    emission_probs : dict
        Mapping from ``(hidden_state, observable_state)`` pairs to emission
        probabilities.
    hidden_states : list, optional
        The hidden-state labels, in the model's state order.  Default: the
        labels appearing in ``transition_probs``, sorted.
    observable_states : list, optional
        The observable-state labels, in the model's state order; a state
        that is never emitted may be listed.  Default: the labels appearing
        in ``emission_probs``, sorted.

    Raises
    ------
    InvalidInputError
        If any of the probability dictionaries is empty, if they mention a
        label that is not a state, if ``hidden_states`` or
        ``observable_states`` is empty or has duplicates, or if a
        probability is negative or the start probabilities, a row of
        transition probabilities or a row of emission probabilities do not
        sum to 1.

    Notes
    -----
    Missing entries of the probability dictionaries are zero probabilities.
    The model is immutable.  It holds label-keyed tables only; the
    parameters as arrays on state indices are in the ``"matvec"``
    representation (``hmm.representation("matvec")``), whose indices follow
    the state order (:attr:`hidden_states`, :attr:`observable_states`).
    """

    def __init__(
        self,
        *,
        start_probs,
        transition_probs,
        emission_probs,
        hidden_states=None,
        observable_states=None,
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

        hidden = _state_order(
            "hidden_states", hidden_states, {h for key in transition_probs for h in key}
        )
        observable = _state_order(
            "observable_states", observable_states, {o for _, o in emission_probs}
        )
        known_hidden, known_observable = set(hidden), set(observable)

        def check_labels(name, key, labels):
            for label, known, kind in labels:
                if label not in known:
                    entry = "" if key == label else f" entry {key!r}"
                    raise InvalidInputError(
                        f"{name}{entry} has {label!r}, which is not one of "
                        f"the {kind} states"
                    )

        for h in start_probs:
            check_labels("start_probs", h, [(h, known_hidden, "hidden")])
        for h1, h2 in transition_probs:
            check_labels(
                "transition_probs",
                (h1, h2),
                [(h1, known_hidden, "hidden"), (h2, known_hidden, "hidden")],
            )
        for h, o in emission_probs:
            check_labels(
                "emission_probs",
                (h, o),
                [(h, known_hidden, "hidden"), (o, known_observable, "observable")],
            )

        # Dense label-keyed tables (zeros included), in state order
        self._hidden_states = freeze(hidden)
        self._observable_states = freeze(observable)
        self._start_probs = freeze({h: start_probs.get(h, 0) for h in hidden})
        self._transition_probs = freeze(
            {(a, b): transition_probs.get((a, b), 0) for a in hidden for b in hidden}
        )
        self._emission_probs = freeze(
            {(h, o): emission_probs.get((h, o), 0) for h in hidden for o in observable}
        )
        self.check_model()

    def check_model(self):
        """Validate the probabilities of the model.

        The constructor calls this, so a model that exists is valid.

        Raises
        ------
        InvalidInputError
            If a probability is negative, or the start probabilities, a row
            of transition probabilities or a row of emission probabilities do
            not sum to 1.
        """
        _check_row("start_probs", self._start_probs)
        for a in self._hidden_states:
            _check_row(
                f"transition_probs from {a!r}",
                {(a, b): self._transition_probs[a, b] for b in self._hidden_states},
            )
        for h in self._hidden_states:
            _check_row(
                f"emission_probs of {h!r}",
                {(h, o): self._emission_probs[h, o] for o in self._observable_states},
            )

    def _key(self):
        return (
            tuple(self._hidden_states),
            tuple(self._observable_states),
            tuple(self._start_probs.items()),
            tuple(self._transition_probs.items()),
            tuple(self._emission_probs.items()),
        )

    def __eq__(self, other):
        """Models are equal if they have the same states, in the same order, and probabilities."""
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return (
            f"{type(self).__name__}(hidden_states={list(self._hidden_states)!r}, "
            f"observable_states={list(self._observable_states)!r})"
        )

    def __str__(self):
        """Return the nonzero probabilities (above 1e-3) as a readable dictionary."""

        def nonzero(table):
            return [(key, v) for key, v in table.items() if v > 1e-3]

        return pprint.pformat(
            dict(
                start_probs=nonzero(self._start_probs),
                transition_probs=nonzero(self._transition_probs),
                emission_probs=nonzero(self._emission_probs),
                num_hidden=self.num_hidden_states,
                num_observable=self.num_observable_states,
            ),
            indent=4,
            sort_dicts=True,
        )

    @property
    def num_hidden_states(self):
        """int: The number of hidden states."""
        return len(self._hidden_states)

    @property
    def num_observable_states(self):
        """int: The number of observable states."""
        return len(self._observable_states)

    @property
    def hidden_states(self):
        """list: The hidden-state labels in the model's state order (read-only)."""
        return self._hidden_states

    @property
    def observable_states(self):
        """list: The observable-state labels in the model's state order (read-only)."""
        return self._observable_states

    @property
    def start_probs(self):
        """dict: ``{hidden_state: probability}`` for every hidden state (read-only)."""
        return self._start_probs

    @property
    def transition_probs(self):
        """dict: ``{(from_state, to_state): probability}`` for every pair (read-only)."""
        return self._transition_probs

    @property
    def emission_probs(self):
        """dict: ``{(hidden_state, observable_state): probability}`` for every pair (read-only)."""
        return self._emission_probs

    def make_non_zero(self, tol=1e-6):
        """Return a new model with small probabilities floored and renormalized.

        Parameters
        ----------
        tol : float, optional
            Minimum probability assigned before renormalization.

        Returns
        -------
        HiddenMarkovModel
            A new model with floored and renormalized probabilities and the
            same state order; ``self`` is not modified.
        """

        def floored(table):
            return {key: max(val, tol) for key, val in table.items()}

        return HiddenMarkovModel(
            start_probs=Util.normalize_dictionary(floored(self._start_probs)),
            transition_probs=Util.normalize_2d_dictionary(
                floored(self._transition_probs)
            ),
            emission_probs=Util.normalize_2d_dictionary(floored(self._emission_probs)),
            hidden_states=self._hidden_states,
            observable_states=self._observable_states,
        )

    def write_to_file(self, file_name):
        """Write the model parameters to a JSON file.

        Parameters
        ----------
        file_name : str or path-like
            Destination file path.
        """
        # Convert tuples to strings for JSON serialization
        data = {
            "start_probs": dict(self._start_probs),
            "transition_probs": {str(k): v for k, v in self._transition_probs.items()},
            "emission_probs": {str(k): v for k, v in self._emission_probs.items()},
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


def random_hmm(*, hidden_states, observable_states, rng=None):
    """Generate a random hidden Markov model.

    Parameters
    ----------
    hidden_states : iterable
        Hidden state labels for the generated model, in its state order.
    observable_states : iterable
        Observable state labels for the generated model, in its state order.
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
    hidden_states, observable_states = list(hidden_states), list(observable_states)

    S = Util.normalize_dictionary({s: rng.random() for s in hidden_states})

    T = Util.normalize_2d_dictionary(
        {(a, b): rng.random() for a in hidden_states for b in hidden_states}
    )
    E = Util.normalize_2d_dictionary(
        {(s, o): rng.random() for s in hidden_states for o in observable_states}
    )

    return HiddenMarkovModel(
        start_probs=S,
        transition_probs=T,
        emission_probs=E,
        hidden_states=hidden_states,
        observable_states=observable_states,
    )
