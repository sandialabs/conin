"""CONIN's JSON model format (hidden Markov models).

A file holds one JSON object with a ``format`` tag and a ``version``::

    {
      "format": "conin.HiddenMarkovModel",
      "version": 1,
      "hidden_states": ["h0", "h1"],
      "observable_states": ["o0", "o1"],
      "start_probs": [["h0", 0.4], ["h1", 0.6]],
      "transition_probs": [[["h0", "h0"], 0.9], ...],
      "emission_probs": [[["h0", "o0"], 0.7], ...]
    }

The state lists keep the model's state order; the tables list the nonzero
entries.  State labels are JSON values: strings, numbers, booleans or
``null``.

:func:`load_json` also reads the files written by the removed
``HiddenMarkovModel.write_to_file`` (an untagged object whose table keys are
the ``str()`` of the label tuples).
"""

import ast
import json
import math

from conin.exceptions import InvalidInputError, ModelFileError
from conin.graphical_models import HiddenMarkovModel

HMM_FORMAT = "conin.HiddenMarkovModel"
HMM_VERSION = 1


def _check_label(label):
    if label is None or isinstance(label, (str, bool, int)):
        return label
    if isinstance(label, float) and math.isfinite(label):
        return label
    raise InvalidInputError(
        f"The state label {label!r} cannot be saved to JSON: labels must be "
        "strings, numbers, booleans or None"
    )


def _hmm_document(hmm):
    hidden = [_check_label(h) for h in hmm.hidden_states]
    observable = [_check_label(o) for o in hmm.observable_states]

    def nonzero(table, key):
        return [[key(k), v] for k, v in table.items() if v != 0]

    return {
        "format": HMM_FORMAT,
        "version": HMM_VERSION,
        "hidden_states": hidden,
        "observable_states": observable,
        "start_probs": nonzero(hmm.start_probs, lambda h: h),
        "transition_probs": nonzero(hmm.transition_probs, list),
        "emission_probs": nonzero(hmm.emission_probs, list),
    }


def save_json(model, filename=None):
    """Write a model in CONIN's JSON format.

    Parameters
    ----------
    model : HiddenMarkovModel
        The model.
    filename : str or path-like, optional
        The output file.  Without it, the JSON text is returned.

    Returns
    -------
    str or None
        The JSON text if ``filename`` is not given.

    Raises
    ------
    TypeError
        For a model that is not a ``HiddenMarkovModel``.
    InvalidInputError
        For a state label that is not a JSON value.
    """
    if not isinstance(model, HiddenMarkovModel):
        raise TypeError(
            "The JSON format holds a HiddenMarkovModel, got " f"{type(model).__name__}"
        )
    text = json.dumps(_hmm_document(model), indent=2) + "\n"
    if filename is None:
        return text
    with open(filename, "w") as f:
        f.write(text)


def _entries(doc, name, key_length):
    """The ``[key, probability]`` pairs of a table as ``{key: probability}``."""
    entries = doc.get(name)
    if not isinstance(entries, list):
        raise ModelFileError(f"The JSON HMM's {name} must be a list")
    table = {}
    for entry in entries:
        if not (isinstance(entry, list) and len(entry) == 2):
            raise ModelFileError(
                f"The JSON HMM's {name} entries must be [key, probability] "
                f"pairs, got {entry!r}"
            )
        key, prob = entry
        if key_length > 1:
            if not (isinstance(key, list) and len(key) == key_length):
                raise ModelFileError(
                    f"The JSON HMM's {name} keys must be pairs of labels, got {key!r}"
                )
            key = tuple(key)
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            raise ModelFileError(
                f"The JSON HMM's {name} has a probability that is not a "
                f"number: {prob!r}"
            )
        if key in table:
            raise ModelFileError(f"The JSON HMM's {name} repeats the key {key!r}")
        table[key] = prob
    return table


def _states(doc, name):
    states = doc.get(name)
    if not isinstance(states, list):
        raise ModelFileError(f"The JSON HMM's {name} must be a list")
    return states


def _hmm_from_document(doc):
    version = doc.get("version")
    if version != HMM_VERSION:
        raise ModelFileError(
            f"Unsupported version {version!r} of the {HMM_FORMAT} JSON "
            f"format (this version of CONIN reads version {HMM_VERSION})"
        )
    return HiddenMarkovModel(
        start_probs=_entries(doc, "start_probs", 1),
        transition_probs=_entries(doc, "transition_probs", 2),
        emission_probs=_entries(doc, "emission_probs", 2),
        hidden_states=_states(doc, "hidden_states"),
        observable_states=_states(doc, "observable_states"),
    )


def _legacy_hmm(doc):
    """A model from the format of the removed ``HiddenMarkovModel.write_to_file``."""

    def pairs(name):
        table = doc[name]
        if not isinstance(table, dict):
            raise ModelFileError(f"The JSON HMM's {name} must be an object")
        result = {}
        for key, prob in table.items():
            try:
                pair = ast.literal_eval(key)
            except (ValueError, SyntaxError):
                pair = None
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ModelFileError(
                    f"The JSON HMM's {name} keys must be the str() of label "
                    f"pairs, got {key!r}"
                )
            result[pair] = prob
        return result

    start_probs = doc["start_probs"]
    if not isinstance(start_probs, dict):
        raise ModelFileError("The JSON HMM's start_probs must be an object")
    return HiddenMarkovModel(
        start_probs=start_probs,
        transition_probs=pairs("transition_probs"),
        emission_probs=pairs("emission_probs"),
    )


def load_json(filename=None, string=None):
    """Read a model in CONIN's JSON format.

    Parameters
    ----------
    filename : str or path-like, optional
        The input file.
    string : str, optional
        The JSON text (used when ``filename`` is not given).

    Returns
    -------
    HiddenMarkovModel

    Raises
    ------
    FileNotFoundError
        If ``filename`` does not exist.
    InvalidInputError
        If neither ``filename`` nor ``string`` is given.
    ModelFileError
        If the data is not JSON, not a CONIN model (or an old
        ``write_to_file`` file), or does not describe a valid model.
    """
    if filename is not None:
        with open(filename) as f:
            string = f.read()
    if string is None:
        raise InvalidInputError("Either a filename or a string must be specified")
    try:
        doc = json.loads(string)
    except json.JSONDecodeError as e:
        raise ModelFileError(f"Invalid JSON model data: {e}") from e
    if not isinstance(doc, dict):
        raise ModelFileError("A JSON model file must contain an object")
    try:
        if "format" not in doc and {
            "start_probs",
            "transition_probs",
            "emission_probs",
        } <= set(doc):
            return _legacy_hmm(doc)
        if doc.get("format") != HMM_FORMAT:
            raise ModelFileError(
                f"Unknown JSON model format {doc.get('format')!r}; expected "
                f"{HMM_FORMAT!r}"
            )
        return _hmm_from_document(doc)
    except ModelFileError:
        raise
    except (InvalidInputError, TypeError) as e:
        raise ModelFileError(f"Invalid JSON HMM: {e}") from e
