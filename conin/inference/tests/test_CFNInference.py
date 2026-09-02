import os
from conin.util import try_import
from conin.inference.map_query import map_query
import conin.markov_network.examples
import conin.bayesian_network.examples
import conin.hidden_markov_model.tests.examples
import conin.dynamic_bayesian_network.examples

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pytoulbar2_available:
    import pytoulbar2

import pytest

skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)

cwd = os.path.dirname(__file__)
testfile_uai = os.path.join(cwd, "test.uai")


#
# DiscreteMarkovNetwork tests
#


@skipif_toulbar2_not_available
def test_CFNInference_ABC_conin():
    example = conin.markov_network.examples.ABC_conin()
    results = map_query(example.pgm, inference="toulbar2")
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(example.pgm, inference="toulbar2", write_uai_file=testfile_uai)
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_CFNInference_ABC_pgmpy():
    example = conin.markov_network.examples.ABC_pgmpy()
    results = map_query(example.pgm, inference="toulbar2")
    assert results.solution.states == example.solutions[0].states


#
# ConstrainedDiscreteMarkovNetwork tests
#


@skipif_toulbar2_not_available
def test_CFNInference_ABC_constrained_toulbar2_conin():
    example = conin.markov_network.examples.ABC_constrained_toulbar2_conin()
    results = map_query(example.pgm, inference="toulbar2")
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(example.pgm, inference="toulbar2", write_uai_file=testfile_uai)
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


#
# DiscreteBayesianNetwork tests
#


@skipif_toulbar2_not_available
def test_CFNInference_cancer1_BN_conin():
    example = conin.bayesian_network.examples.cancer1_BN_conin()
    results = map_query(example.pgm, inference="toulbar2")
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(example.pgm, inference="toulbar2", write_uai_file=testfile_uai)
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.states == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Smoker": 0,
#            "Xray": 0,
#        }
#
#    # TODO - Confirm that these marginalized results are correct
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.states == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Xray": 0,
#        }


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_CFNInference_cancer1_BN_pgmpy():
    example = conin.bayesian_network.examples.cancer1_BN_pgmpy()
    results = map_query(example.pgm, inference="toulbar2")
    assert results.solution.states == example.solutions[0].states


#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.states == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Smoker": 0,
#            "Xray": 0,
#        }
#
#    # TODO - Confirm that these marginalized results are correct
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.states == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Xray": 0,
#        }


#
# ConstrainedBayesianNetwork tests
#


@skipif_toulbar2_not_available
def test_CFNInference_cancer1_BN_constrained_toulbar2_conin():
    example = conin.bayesian_network.examples.cancer1_BN_constrained_toulbar2_conin()
    results = map_query(example.pgm, inference="toulbar2")
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(example.pgm, inference="toulbar2", write_uai_file=testfile_uai)
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.states == {
#            "Dyspnoea": 1,
#            "Pollution": 0,
#            "Xray": 0,
#        }


#
# HiddenMarkovModel tests
#


@skipif_toulbar2_not_available
def test0_CFNInference_hmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = ["o0", "o0", "o1", "o0", "o0"]
    results = map_query(pgm, inference="toulbar2", evidence=observed)
    assert results.solution.states == ["h0", "h0", "h0", "h0", "h0"]
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(
        pgm, inference="toulbar2", evidence=observed, write_uai_file=testfile_uai
    )
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


@skipif_toulbar2_not_available
def test1_CFNInference_hmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = ["o0", "o1", "o1", "o1", "o1"]
    results = map_query(pgm, inference="toulbar2", evidence=observed)
    assert results.solution.states == ["h1", "h1", "h1", "h1", "h1"]


@skipif_toulbar2_not_available
def test2_CFNInference_hmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"}
    results = map_query(pgm, inference="toulbar2", evidence=observed)
    assert results.solution.states == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
    }


@skipif_toulbar2_not_available
def test3_CFNInference_hmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = {0: "o0", 1: "o1", 2: "o1", 3: "o1", 4: "o1"}
    results = map_query(pgm, inference="toulbar2", evidence=observed)
    assert results.solution.states == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h1",
        4: "h1",
    }


@skipif_toulbar2_not_available
def test0_CFNInference_chmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_chmm1_toulbar2()
    observed = ["o0"] * 15
    results = map_query(pgm, inference="toulbar2", evidence=observed)
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
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(
        pgm, inference="toulbar2", evidence=observed, write_uai_file=testfile_uai
    )
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


@skipif_toulbar2_not_available
def test1_CFNInference_chmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_chmm1_toulbar2()
    observed = ["o0"] + ["o1"] * 14
    results = map_query(pgm, inference="toulbar2", evidence=observed)
    assert results.solution.states == [
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
        "h1",
        "h1",
        "h1",
        "h1",
        "h1",
    ]


@skipif_toulbar2_not_available
def test2_CFNInference_chmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_chmm1_toulbar2()
    observed = {i: "o0" for i in range(15)}
    results = map_query(pgm, inference="toulbar2", evidence=observed)
    assert results.solution.states == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h0",
        4: "h0",
        5: "h0",
        6: "h0",
        7: "h0",
        8: "h0",
        9: "h0",
        10: "h0",
        11: "h0",
        12: "h0",
        13: "h0",
        14: "h0",
    }


@skipif_toulbar2_not_available
def test3_CFNInference_chmm1():
    pgm = conin.hidden_markov_model.tests.examples.create_chmm1_toulbar2()
    observed = {0: "o0"}
    for i in range(14):
        observed[i + 1] = "o1"
    results = map_query(pgm, inference="toulbar2", evidence=observed)
    assert results.solution.states == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
        5: "h0",
        6: "h0",
        7: "h0",
        8: "h0",
        9: "h0",
        10: "h1",
        11: "h1",
        12: "h1",
        13: "h1",
        14: "h1",
    }


#
# DynamicBayesianNetwork tests
#


weather_evidence = {
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
}

q_unconstrained = {
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Hot",
    ("T", 2): "Mild",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Cloudy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}

q_constrained = {
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Mild",
    ("T", 2): "Cold",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Rainy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}


@skipif_toulbar2_not_available
def test_DPGM_CFNInference_weather_conin():
    example = conin.dynamic_bayesian_network.examples.weather_conin()

    # without evidence
    results = map_query(example.pgm, inference="toulbar2", stop=4)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # with evidence
    results = map_query(
        example.pgm,
        inference="toulbar2",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_unconstrained == results.solution.states

    results = map_query(
        example.pgm, inference="toulbar2", stop=4, write_uai_file=testfile_uai
    )
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_DPGM_CFNInference_weather():
    example = conin.dynamic_bayesian_network.examples.weather2_pgmpy()

    # without evidence
    results = map_query(example.pgm, inference="toulbar2", stop=4)
    assert results.solution.states == example.solutions[0].states

    # with evidence
    results = map_query(
        example.pgm,
        inference="toulbar2",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_unconstrained == results.solution.states


#
# ConstrainedDynamicBayesianNetwork tests
#


@skipif_toulbar2_not_available
def test_DPGM_CFNInference_weather_constrained_conin():
    example = (
        conin.dynamic_bayesian_network.examples.weather_constrained_toulbar2_conin()
    )

    # without evidence
    results = map_query(example.pgm, inference="toulbar2", stop=4)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # with evidence
    results = map_query(
        example.pgm,
        inference="toulbar2",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_constrained == results.solution.states

    results = map_query(
        example.pgm, inference="toulbar2", stop=4, write_uai_file=testfile_uai
    )
    assert os.path.exists(testfile_uai)
    os.remove(testfile_uai)


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_DPGM_CFNInference_weather_constrained_pgmpy():
    example = (
        conin.dynamic_bayesian_network.examples.weather_constrained_toulbar2_pgmpy()
    )

    # without evidence
    results = map_query(example.pgm, inference="toulbar2", stop=4)
    assert results.solution.states == example.solutions[0].states

    # with evidence
    results = map_query(
        example.pgm,
        inference="toulbar2",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_constrained == results.solution.states
