import pytest
import pyomo.opt

from conin.util import try_import
from conin.inference.map_query import map_query
import conin.hidden_markov_model.tests.examples

#
# map_query with method="viterbi" tests
#


def test_map_query_Viterbi_hmm1_list_evidence():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = ["o0", "o0", "o1", "o0", "o0"]
    results = map_query(
        pgm,
        method="viterbi",
        variables=None,
        evidence=observed,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
    )
    assert results.solution.states == ["h0", "h0", "h0", "h0", "h0"]


def test_map_query_Viterbi_hmm1_dict_evidence():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"}
    results = map_query(
        pgm,
        method="viterbi",
        variables=None,
        evidence=observed,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
    )
    assert results.solution.states == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
    }


def test_map_query_Viterbi_hmm1_test1_list():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = ["o0", "o1", "o1", "o1", "o1"]
    results = map_query(
        pgm,
        method="viterbi",
        variables=None,
        evidence=observed,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
    )
    assert results.solution.states == ["h1", "h1", "h1", "h1", "h1"]


def test_map_query_Viterbi_hmm1_test1_dict():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = {0: "o0", 1: "o1", 2: "o1", 3: "o1", 4: "o1"}
    results = map_query(
        pgm,
        method="viterbi",
        variables=None,
        evidence=observed,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
    )
    assert results.solution.states == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h1",
        4: "h1",
    }


def test_map_query_Viterbi_unsupported_model():
    class UnsupportedModel:
        pass

    pgm = UnsupportedModel()
    with pytest.raises(TypeError, match="Unsupported model type"):
        map_query(
            pgm,
            method="viterbi",
            variables=None,
            evidence=[],
            show_progress=False,
            timing=False,
            start=0,
            stop=None,
        )
