"""CONIN's JSON model format (conin.io.json_format) and its use by save_model/load_model."""

import gzip
import json

import pytest

import conin.examples.hidden_markov_model as hex_
from conin.exceptions import InvalidInputError, ModelFileError
from conin.graphical_models import HiddenMarkovModel
from conin.io import load_json, load_model, save_json, save_model


def _hmm(**kwds):
    return HiddenMarkovModel(
        start_probs={"h0": 0.4, "h1": 0.6},
        transition_probs={("h0", "h0"): 0.9, ("h0", "h1"): 0.1, ("h1", "h1"): 1.0},
        emission_probs={("h0", "o0"): 0.7, ("h0", "o1"): 0.3, ("h1", "o1"): 1.0},
        **kwds,
    )


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_save_model_load_model_round_trip(tmp_path, suffix):
    hmm = hex_.create_hmm2()
    name = str(tmp_path / f"hmm{suffix}")
    save_model(hmm, name)
    assert load_model(name) == hmm


def test_gzipped_file_is_gzipped_json(tmp_path):
    name = str(tmp_path / "hmm.json.gz")
    save_model(_hmm(), name)
    with gzip.open(name, "rt") as f:
        assert json.load(f)["format"] == "conin.HiddenMarkovModel"


def test_the_document():
    doc = json.loads(save_json(_hmm()))
    assert doc == {
        "format": "conin.HiddenMarkovModel",
        "version": 1,
        "hidden_states": ["h0", "h1"],
        "observable_states": ["o0", "o1"],
        "start_probs": [["h0", 0.4], ["h1", 0.6]],
        # Only the nonzero entries
        "transition_probs": [
            [["h0", "h0"], 0.9],
            [["h0", "h1"], 0.1],
            [["h1", "h1"], 1.0],
        ],
        "emission_probs": [
            [["h0", "o0"], 0.7],
            [["h0", "o1"], 0.3],
            [["h1", "o1"], 1.0],
        ],
    }


def test_state_order_and_unemitted_states_survive():
    hmm = _hmm(hidden_states=["h1", "h0"], observable_states=["o2", "o1", "o0"])
    loaded = load_json(string=save_json(hmm))
    assert loaded == hmm
    assert loaded.hidden_states == ["h1", "h0"]
    assert loaded.observable_states == ["o2", "o1", "o0"]


def test_non_string_labels():
    hmm = HiddenMarkovModel(
        start_probs={1: 1.0},
        transition_probs={(1, 2): 1.0, (2, 1): 1.0},
        emission_probs={(1, None): 0.5, (1, 2.5): 0.5, (2, True): 1.0},
        observable_states=[None, 2.5, True],
    )
    assert load_json(string=save_json(hmm)) == hmm


@pytest.mark.parametrize("label", [("a", 1), float("nan"), object()])
def test_labels_that_are_not_json_values(label):
    hmm = HiddenMarkovModel(
        start_probs={label: 1.0},
        transition_probs={(label, label): 1.0},
        emission_probs={(label, "o"): 1.0},
        hidden_states=[label],
    )
    with pytest.raises(InvalidInputError, match="cannot be saved to JSON"):
        save_json(hmm)


def test_only_hmms(tmp_path):
    from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork

    mn = DiscreteMarkovNetwork(
        states={"A": [0, 1]}, factors=[DiscreteFactor(nodes=["A"], values=[1, 2])]
    )
    with pytest.raises(TypeError, match="holds a HiddenMarkovModel"):
        save_model(mn, str(tmp_path / "mn.json"))


def _write_to_file_text(hmm):
    # What the removed HiddenMarkovModel.write_to_file wrote.
    return json.dumps(
        {
            "start_probs": dict(hmm.start_probs),
            "transition_probs": {str(k): v for k, v in hmm.transition_probs.items()},
            "emission_probs": {str(k): v for k, v in hmm.emission_probs.items()},
        },
        indent=4,
    )


def test_reads_files_written_by_write_to_file(tmp_path):
    hmm = hex_.create_hmm1()
    name = tmp_path / "old.json"
    name.write_text(_write_to_file_text(hmm))
    assert load_model(str(name)) == hmm


def test_old_files_with_bad_keys():
    text = json.dumps(
        {
            "start_probs": {"h0": 1.0},
            "transition_probs": {"h0": 1.0},
            "emission_probs": {"('h0', 'o0')": 1.0},
        }
    )
    with pytest.raises(ModelFileError, match="str\\(\\) of label pairs"):
        load_json(string=text)


def _doc(**changes):
    doc = json.loads(save_json(_hmm()))
    doc.update(changes)
    return json.dumps(doc)


@pytest.mark.parametrize(
    "text, match",
    [
        ("{not json", "Invalid JSON model data"),
        ("[1, 2]", "must contain an object"),
        ('{"format": "something else"}', "Unknown JSON model format 'something else'"),
        (_doc(version=2), "Unsupported version 2"),
        (_doc(hidden_states="h0 h1"), "hidden_states must be a list"),
        (_doc(start_probs={"h0": 1.0}), "start_probs must be a list"),
        (_doc(start_probs=[["h0", 0.4, 0.6]]), r"\[key, probability\] pairs"),
        (_doc(start_probs=[["h0", "0.4"], ["h1", 0.6]]), "not a number: '0.4'"),
        (_doc(start_probs=[["h0", 0.4], ["h0", 0.6]]), "repeats the key 'h0'"),
        (_doc(transition_probs=[["h0", 1.0]]), "keys must be pairs of labels"),
        (_doc(hidden_states=[["h0"], "h1"]), "Invalid JSON HMM"),
        # Not a valid model: the constructor's checks
        (_doc(start_probs=[["h0", 0.5]]), "start_probs must sum to 1"),
        (_doc(hidden_states=["h0"]), "not one of the hidden states"),
    ],
)
def test_malformed_and_invalid_documents(text, match):
    with pytest.raises(ModelFileError, match=match):
        load_json(string=text)


def test_model_file_error_is_an_invalid_input_and_a_value_error():
    assert issubclass(ModelFileError, InvalidInputError)
    assert issubclass(ModelFileError, ValueError)


def test_missing_file_and_no_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        load_json(filename=str(tmp_path / "missing.json"))
    with pytest.raises(InvalidInputError, match="Either a filename or a string"):
        load_json()


def test_load_json_from_a_file(tmp_path):
    name = tmp_path / "hmm.json"
    save_json(_hmm(), name)
    assert load_json(filename=name) == _hmm()


def test_hmm_has_no_json_methods():
    assert not hasattr(HiddenMarkovModel, "write_to_file")
    assert not hasattr(HiddenMarkovModel, "read_from_file")
