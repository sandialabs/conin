import pytest
import pyomo.opt

from conin.util import try_import
from conin.inference.CFNInference import (
    CFNInference,
    DDBN_CFNInference,
)
import conin.markov_network.examples
import conin.bayesian_network.examples
import conin.hmm.tests.examples
import conin.dynamic_bayesian_network.examples

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pytoulbar2_available:
    import pytoulbar2

#
# DiscreteMarkovNetwork tests
#


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_CFNInference_ABC_conin():
    example = conin.markov_network.examples.ABC_conin()
    inf = CFNInference(example.pgm)
    results = inf.map_query()
    assert results.solution.variable_value == example.solutions[0].states


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_CFNInference_ABC_pgmpy():
    example = conin.markov_network.examples.ABC_pgmpy()
    inf = CFNInference(example.pgm)
    results = inf.map_query()
    assert results.solution.variable_value == example.solutions[0].states


#
# ConstrainedDiscreteMarkovNetwork tests
#


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_ABC_constrained():
    example = conin.markov_network.examples.ABC_constrained_toulbar2_conin()
    inf = CFNInference(example.pgm)
    results = inf.map_query()
    assert results.solution.variable_value == example.solutions[0].states


#
# DiscreteBayesianNetwork tests
#


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_CFNInference_cancer1_ALL_conin():
    example = conin.bayesian_network.examples.cancer1_BN_conin()
    inf = CFNInference(example.pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
    )
    assert results.solution.variable_value == example.solution

    # TODO - Confirm that these marginalized results are correct


#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.variable_value == {
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
#        assert results.solution.variable_value == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Xray": 0,
#        }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_CFNInference_cancer1_ALL_pgmpy():
    example = conin.bayesian_network.examples.cancer1_BN_pgmpy()
    inf = CFNInference(example.pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
    )
    assert results.solution.variable_value == example.solution

    # TODO - Confirm that these marginalized results are correct


#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.variable_value == {
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
#        assert results.solution.variable_value == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Xray": 0,
#        }


#
# ConstrainedBayesianNetwork tests
#


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_cancer1_constrained_conin():
    example = conin.bayesian_network.examples.cancer1_BN_constrained_toulbar2_conin()
    inf = CFNInference(example.pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
    )
    assert results.solution.variable_value == example.solution

    # TODO - Confirm that these marginalized results are correct


#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.variable_value == {
#            "Dyspnoea": 1,
#            "Pollution": 0,
#            "Xray": 0,
#        }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_cancer1_constrained_pgmpy():
    example = conin.bayesian_network.examples.cancer1_BN_constrained_toulbar2_pgmpy()
    inf = CFNInference(example.pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
    )
    assert results.solution.variable_value == example.solution

    # TODO - Confirm that these marginalized results are correct


#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Xray"],
#            evidence={"Cancer": 0},
#            ,
#        )
#        assert results.solution.variable_value == {
#            "Dyspnoea": 1,
#            "Pollution": 0,
#            "Xray": 0,
#        }


#
# HiddenMarkovModel tests
#


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_hmm1_test0():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = CFNInference(pgm)
    observed = ["o0", "o0", "o1", "o0", "o0"]
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == ["h0", "h0", "h0", "h0", "h0"]


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_hmm1_test1():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = CFNInference(pgm)
    observed = ["o0", "o1", "o1", "o1", "o1"]
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == ["h1", "h1", "h1", "h1", "h1"]


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_hmm1_test2():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = CFNInference(pgm)
    observed = {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"}
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
    }


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_hmm1_test3():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = CFNInference(pgm)
    observed = {0: "o0", 1: "o1", 2: "o1", 3: "o1", 4: "o1"}
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h1",
        4: "h1",
    }


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_chmm1_test0():
    pgm = conin.hmm.tests.examples.create_chmm1_pyomo()
    inf = CFNInference(pgm)
    observed = ["o0"] * 15
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == [
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


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_chmm1_test1():
    pgm = conin.hmm.tests.examples.create_chmm1_pyomo()
    inf = CFNInference(pgm)
    observed = ["o0"] + ["o1"] * 14
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == [
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


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_chmm1_test2():
    pgm = conin.hmm.tests.examples.create_chmm1_pyomo()
    inf = CFNInference(pgm)
    observed = {i: "o0" for i in range(15)}
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == {
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


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_CFNInference_chmm1_test3():
    pgm = conin.hmm.tests.examples.create_chmm1_pyomo()
    inf = CFNInference(pgm)
    observed = {0: "o0"}
    for i in range(14):
        observed[i + 1] = "o1"
    results = inf.map_query(evidence=observed)
    assert results.solution.variable_value == {
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


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_DDBN_CFNInference_weather_conin():
    example = conin.dynamic_bayesian_network.examples.weather_conin()
    inf = DDBN_CFNInference(example.pgm)
    results = inf.map_query(stop=4)
    assert results.solution.variable_value == example.solution


#    evidence = {
#        ("O", 0): "Wet",
#        ("O", 1): "Wet",
#        ("O", 2): "Dry",
#        ("O", 3): "Dry",
#        ("O", 4): "Dry",
#        ("H", 0): "Medium",
#        ("H", 1): "Medium",
#        ("H", 2): "Medium",
#        ("H", 3): "Medium",
#        ("H", 4): "Medium",
#    }
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(stop=4, evidence=evidence)
#        # TODO - Confirm that this result makes sense
#        assert results.solution.variable_value == {
#            ("T", 0): "Hot",
#            ("T", 1): "Hot",
#            ("T", 2): "Hot",
#            ("T", 3): "Hot",
#            ("T", 4): "Hot",
#            ("W", 0): "Cloudy",
#            ("W", 1): "Cloudy",
#            ("W", 2): "Cloudy",
#            ("W", 3): "Cloudy",
#            ("W", 4): "Cloudy",
#        }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_DDBN_CFNInference_weather():
    example = conin.dynamic_bayesian_network.examples.weather2_pgmpy()
    inf = DDBN_CFNInference(example.pgm)
    results = inf.map_query(stop=4)
    assert results.solution.variable_value == example.solution


#    evidence = {
#        ("O", 0): "Wet",
#        ("O", 1): "Wet",
#        ("O", 2): "Dry",
#        ("O", 3): "Dry",
#        ("O", 4): "Dry",
#        ("H", 0): "Medium",
#        ("H", 1): "Medium",
#        ("H", 2): "Medium",
#        ("H", 3): "Medium",
#        ("H", 4): "Medium",
#    }
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(stop=4, evidence=evidence)
#        # TODO - Confirm that this result makes sense
#        assert results.solution.variable_value == {
#            ("T", 0): "Hot",
#            ("T", 1): "Hot",
#            ("T", 2): "Hot",
#            ("T", 3): "Hot",
#            ("T", 4): "Hot",
#            ("W", 0): "Cloudy",
#            ("W", 1): "Cloudy",
#            ("W", 2): "Cloudy",
#            ("W", 3): "Cloudy",
#            ("W", 4): "Cloudy",
#        }


#
# ConstrainedDynamicBayesianNetwork tests
#


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_DDBN_CFNInference_weather_constrained_conin():
    example = (
        conin.dynamic_bayesian_network.examples.weather_constrained_toulbar2_conin()
    )
    inf = DDBN_CFNInference(example.pgm)
    results = inf.map_query(stop=4)
    assert results.solution.variable_value == example.solution


#    evidence = {
#        ("O", 0): "Wet",
#        ("O", 1): "Wet",
#        ("O", 2): "Dry",
#        ("O", 3): "Dry",
#        ("O", 4): "Dry",
#        ("H", 0): "Medium",
#        ("H", 1): "Medium",
#        ("H", 2): "Medium",
#        ("H", 3): "Medium",
#        ("H", 4): "Medium",
#    }
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(stop=4, evidence=evidence)
#        assert results.solution.variable_value == {
#            ("T", 0): "Hot",
#            ("T", 1): "Mild",
#            ("T", 2): "Cold",
#            ("T", 3): "Hot",
#            ("T", 4): "Hot",
#            ("W", 0): "Rainy",
#            ("W", 1): "Rainy",
#            ("W", 2): "Sunny",
#            ("W", 3): "Sunny",
#            ("W", 4): "Sunny",
#        }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_DDBN_CFNInference_weather_constrained_pgmpy():
    example = (
        conin.dynamic_bayesian_network.examples.weather_constrained_toulbar2_pgmpy()
    )
    inf = DDBN_CFNInference(example.pgm)
    results = inf.map_query(stop=4)
    assert results.solution.variable_value == example.solution


#    evidence = {
#        ("O", 0): "Wet",
#        ("O", 1): "Wet",
#        ("O", 2): "Dry",
#        ("O", 3): "Dry",
#        ("O", 4): "Dry",
#        ("H", 0): "Medium",
#        ("H", 1): "Medium",
#        ("H", 2): "Medium",
#        ("H", 3): "Medium",
#        ("H", 4): "Medium",
#    }
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(stop=4, evidence=evidence)
#        assert results.solution.variable_value == {
#            ("T", 0): "Hot",
#            ("T", 1): "Mild",
#            ("T", 2): "Cold",
#            ("T", 3): "Hot",
#            ("T", 4): "Hot",
#            ("W", 0): "Rainy",
#            ("W", 1): "Rainy",
#            ("W", 2): "Sunny",
#            ("W", 3): "Sunny",
#            ("W", 4): "Sunny",
#        }
