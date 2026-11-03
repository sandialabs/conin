"""Error paths and less common inputs of the UAI reader/writer (conin.io.uai)."""

import pytest

from conin.exceptions import ModelFileError
from conin.io import load_model, load_uai, save_uai
from conin.io.uai import _uai_text
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
)


def test_missing_uai_file(tmp_path):
    name = str(tmp_path / "missing.uai")
    with pytest.raises(
        FileNotFoundError, match="Cannot read missing UAI file.*missing.uai"
    ):
        load_uai(filename=name)


@pytest.mark.parametrize(
    "text,n,N",
    [
        # Markov network: the pairwise table should have 2 * 3 entries
        ("MARKOV\n2\n2 3\n1\n2 0 1\n\n4\n0.1 0.2 0.3 0.4\n", 4, 6),
        # Bayesian network: the unary table should have 2 entries
        ("BAYES\n1\n2\n1\n1 0\n\n3\n0.2 0.3 0.5\n", 3, 2),
    ],
)
def test_inconsistent_function_definition(text, n, N):
    with pytest.raises(
        ValueError,
        match=f"Inconsistent function definition at line .*read {n} but expected {N}",
    ):
        load_uai(string=text)


def test_inconsistent_function_definition_reports_line():
    text = "# comment\nMARKOV\n1\n2\n1\n1 0\n\n3\n0.1 0.2 0.3\n"
    # Line numbers are 0-based; the table size is on line 7
    with pytest.raises(ModelFileError, match="at line 7:"):
        load_uai(string=text)


def test_uai_file_and_comments(tmp_path):
    name = tmp_path / "model.uai"
    name.write_text("# A comment\nMARKOV\n\n1\n3\n1\n1 0\n\n3\n\t0.5  0.25\t0.25\n")
    pgm = load_uai(filename=str(name))
    assert isinstance(pgm, DiscreteMarkovNetwork)
    assert pgm.states == {"var0": [0, 1, 2]}
    assert [f.values for f in pgm.factors] == [{0: 0.5, 1: 0.25, 2: 0.25}]


@pytest.mark.parametrize("from_file", [False, True])
def test_truncated_uai_raises_value_error(tmp_path, from_file):
    # It raised a bare StopIteration.
    text = "MARKOV\n2\n2 2\n1\n2 0 1\n4\n0.1 0.2"
    with pytest.raises(
        ModelFileError, match="Unexpected end of the UAI data after line 7"
    ):
        if from_file:
            name = tmp_path / "truncated.uai"
            name.write_text(text)
            load_uai(filename=str(name))
        else:
            load_uai(string=text)


#
# Writing
#


def test_uai_text_rejects_unknown_model_type():
    with pytest.raises(TypeError, match="A UAI file holds a DiscreteBayesianNetwork"):
        _uai_text(object())


def test_save_model_uai_rejects_unknown_model_type(tmp_path):
    name = tmp_path / "model.uai"
    with pytest.raises(TypeError, match="A UAI file holds a DiscreteBayesianNetwork"):
        save_uai("not a model", str(name))
    assert not name.exists()


def test_bayesian_network_round_trip(tmp_path):
    bn = DiscreteBayesianNetwork(
        states={"a": [0, 1], "b": [0, 1]},
        cpds=[
            DiscreteCPD(node="a", values={0: 0.25, 1: 0.75}),
            DiscreteCPD(
                node="b",
                parents=["a"],
                values={0: {0: 0.5, 1: 0.5}, 1: {0: 0.1, 1: 0.9}},
            ),
        ],
    )
    text = _uai_text(bn)
    assert text == "BAYES\n2\n2 2\n2\n1 0\n2 0 1\n\n2\n0.25 0.75\n4\n0.5 0.5 0.1 0.9\n"

    save_uai(bn, str(tmp_path / "bn.txt"))
    assert (tmp_path / "bn.uai").read_text() == text

    loaded = load_uai(filename=str(tmp_path / "bn.uai"))
    cpds = {cpd.node: cpd for cpd in loaded.cpds}
    assert cpds["var0"].values == {0: 0.25, 1: 0.75}
    assert cpds["var1"].parents == ["var0"]
    assert cpds["var1"].values == {0: {0: 0.5, 1: 0.5}, 1: {0: 0.1, 1: 0.9}}


def test_dynamic_bayesian_network_is_written_as_bayes():
    dbn = DynamicDiscreteBayesianNetwork(
        states={"a": [0, 1]},
        cpds=[DiscreteCPD(node="a", values={0: 0.25, 1: 0.75})],
    )
    text = _uai_text(dbn)
    assert text == "BAYES\n1\n2\n1\n1 0\n\n2\n0.25 0.75\n"
    assert load_uai(string=text).cpds[0].values == {
        0: 0.25,
        1: 0.75,
    }


def test_time_indexed_dynamic_bayesian_network_cannot_be_written(tmp_path):
    # It raised KeyError: ('A', 0).  A UAI file has no time-indexed templates.
    import conin.examples.dynamic_bayesian_network as examples
    from conin.exceptions import InvalidInputError

    dbn = examples.simple1_DDBN_conin().pgm
    with pytest.raises(InvalidInputError, match="unroll it first"):
        save_uai(dbn, str(tmp_path / "dbn.uai"))
    assert not (tmp_path / "dbn.uai").exists()


def test_an_unrolled_dynamic_bayesian_network_can_be_written(tmp_path):
    import conin.examples.dynamic_bayesian_network as examples
    from conin.graphical_models import create_bn_from_dbn

    bn = create_bn_from_dbn(dbn=examples.simple1_DDBN_conin().pgm, start=0, stop=2)
    save_uai(bn, str(tmp_path / "bn.uai"))
    loaded = load_uai(filename=str(tmp_path / "bn.uai"))
    assert len(loaded.nodes) == len(bn.nodes) == 6


@pytest.mark.parametrize(
    "text, match",
    [
        ("MARKOV\n2\n2", "Unexpected end of the UAI data"),
        ("MARKOV\ntwo\n2 2\n", "Invalid UAI data: invalid literal for int"),
        ("MARKOV\n1\n2\n1\n1 0\n\n2\n0.5 x\n", "Invalid UAI data: could not convert"),
        ("MARKOV\n1\n2\n1\n1 3\n\n2\n0.5 0.5\n", "Invalid UAI data"),
    ],
)
def test_malformed_uai_data_raises_model_file_error(text, match):
    # The int()/float() failures and bad variable indices used to escape as
    # bare ValueError/IndexError.
    with pytest.raises(ModelFileError, match=match):
        load_uai(string=text)


def test_corrupt_gzip_file_raises_model_file_error(tmp_path):
    name = tmp_path / "model.uai.gz"
    name.write_bytes(b"not gzip data")
    with pytest.raises(ModelFileError, match="Cannot read"):
        load_model(str(name))
