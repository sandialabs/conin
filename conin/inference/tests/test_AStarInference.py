import pytest
import pyomo.opt

import conin.hidden_markov_model.tests.examples
from conin.util import try_import
from conin.inference.map_query import map_query


#
# map_query with method="a_star" tests
#


def test_map_query_AStar_hmm1_list_evidence():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = ["o0", "o0", "o1", "o0", "o0"]
    results = map_query(
        pgm, method="a_star", variables=None, evidence=observed, show_progress=False, timing=False, start=0, stop=None
    )
    assert results.solution.states == ["h0", "h0", "h0", "h0", "h0"]


def test_map_query_AStar_hmm1_dict_evidence():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"}
    results = map_query(
        pgm, method="a_star", variables=None, evidence=observed, show_progress=False, timing=False, start=0, stop=None
    )
    assert results.solution.states == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
    }


def test_map_query_AStar_chmm1_list_evidence():
    pgm = conin.hidden_markov_model.tests.examples.create_chmm1_oracle()
    observed = ["o0"] * 15
    results = map_query(
        pgm, method="a_star", variables=None, evidence=observed, show_progress=False, timing=False, start=0, stop=None
    )
    assert results.solution.states == [
        "h1",
        "h1",
        "h1",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
    ]


def test_map_query_AStar_unsupported_model():
    class UnsupportedModel:
        pass

    pgm = UnsupportedModel()
    with pytest.raises(TypeError, match="Unsupported model type"):
        map_query(
            pgm, method="a_star", variables=None, evidence=[], show_progress=False, timing=False, start=0, stop=None
        )
