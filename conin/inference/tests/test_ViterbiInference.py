import pytest
import pyomo.opt

from conin.util import try_import
from conin.inference.ViterbiInference import (
    ViterbiInference,
)
import conin.hmm.tests.examples

#
# HiddenMarkovModel tests
#


def test_ViterbiInference_hmm1_test0():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = ViterbiInference(pgm)
    observed = ["o0", "o0", "o1", "o0", "o0"]
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == ["h0", "h0", "h0", "h0", "h0"]


def test_ViterbiInference_hmm1_test1():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = ViterbiInference(pgm)
    observed = ["o0", "o1", "o1", "o1", "o1"]
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == ["h1", "h1", "h1", "h1", "h1"]


def test_ViterbiInference_hmm1_test2():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = ViterbiInference(pgm)
    observed = {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"}
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
    }


def test_ViterbiInference_hmm1_test3():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = ViterbiInference(pgm)
    observed = {0: "o0", 1: "o1", 2: "o1", 3: "o1", 4: "o1"}
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h1",
        4: "h1",
    }
