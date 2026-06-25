import pytest
import os
import sys
import numpy as np

from conin.util import try_import
from conin.common.conin import convert_conin_to_pgmpy_mn
from conin.bayesian_network import create_mn_from_bn
import conin.markov_network.examples
import conin.bayesian_network.examples

with try_import() as pgmpy_available:
    import pgmpy
    from pgmpy.models import DiscreteMarkovNetwork as pgmpy_DiscreteMarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor as pgmpy_DiscreteFactor

# Mark tests that require pgmpy
require_pgmpy = pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")


def test_import_convert_function():
    """Test that the conversion function can be imported."""
    from conin.common.conin import convert_conin_to_pgmpy_mn

    assert callable(convert_conin_to_pgmpy_mn)


@require_pgmpy
def test_convert_invalid_input():
    """Test error handling for invalid input types."""
    from conin.common.conin import convert_conin_to_pgmpy_mn

    with pytest.raises(ValueError, match="Expected conin DiscreteMarkovNetwork"):
        convert_conin_to_pgmpy_mn("not a model")

    with pytest.raises(ValueError, match="Expected conin DiscreteMarkovNetwork"):
        convert_conin_to_pgmpy_mn(None)


def check_values(factor, values):
    for i,assign in enumerate(factor.assignment(list(range(len(values))))):
        kwargs = {k:v for k,v in assign}
        assert factor.get_value(**kwargs) == values[i]


@require_pgmpy
def test_ABC():
    conin_pgm = conin.markov_network.examples.ABC_conin().pgm

    # Convert to pgmpy
    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_pgm)

    # Verify structure
    assert isinstance(pgmpy_pgm, pgmpy_DiscreteMarkovNetwork)
    assert set(pgmpy_pgm.nodes()) == {"A", "B", "C"}
    assert len(pgmpy_pgm.edges()) == 3
    assert len(pgmpy_pgm.factors) == 6

    # Check variables
    assert pgmpy_pgm.factors[0].variables == ["A"]
    assert pgmpy_pgm.factors[1].variables == ["B"]
    assert pgmpy_pgm.factors[2].variables == ["C"]
    assert pgmpy_pgm.factors[3].variables == ["A", "B"]
    assert pgmpy_pgm.factors[4].variables == ["B", "C"]
    assert pgmpy_pgm.factors[5].variables == ["A", "C"]
   
    # Check cardinality 
    assert pgmpy_pgm.factors[0].cardinality == [3]
    assert pgmpy_pgm.factors[1].cardinality == [3]
    assert pgmpy_pgm.factors[2].cardinality == [3]
    assert list(pgmpy_pgm.factors[3].cardinality) == [3, 3]
    assert list(pgmpy_pgm.factors[4].cardinality) == [3, 3]
    assert list(pgmpy_pgm.factors[5].cardinality) == [3, 3]

    # Check values 
    check_values(pgmpy_pgm.factors[0], [1,1,2])
    check_values(pgmpy_pgm.factors[1], [1,1,3])
    check_values(pgmpy_pgm.factors[2], [1,2,1])
    check_values(pgmpy_pgm.factors[3], np.ones(9))
    check_values(pgmpy_pgm.factors[4], np.ones(9))
    check_values(pgmpy_pgm.factors[5], np.ones(9))


@require_pgmpy
def test_ABC2():
    conin_pgm = conin.markov_network.examples.ABC2_conin().pgm

    # Convert to pgmpy
    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_pgm)

    # Verify structure
    assert isinstance(pgmpy_pgm, pgmpy_DiscreteMarkovNetwork)
    assert set(pgmpy_pgm.nodes()) == {"A", "B", "C"}
    assert len(pgmpy_pgm.edges()) == 3
    assert len(pgmpy_pgm.factors) == 6

    # Check variables
    assert pgmpy_pgm.factors[0].variables == ["A"]
    assert pgmpy_pgm.factors[1].variables == ["B"]
    assert pgmpy_pgm.factors[2].variables == ["C"]
    assert pgmpy_pgm.factors[3].variables == ["A", "B"]
    assert pgmpy_pgm.factors[4].variables == ["B", "C"]
    assert pgmpy_pgm.factors[5].variables == ["A", "C"]
   
    # Check cardinality 
    assert pgmpy_pgm.factors[0].cardinality == [3]
    assert pgmpy_pgm.factors[1].cardinality == [3]
    assert pgmpy_pgm.factors[2].cardinality == [3]
    assert list(pgmpy_pgm.factors[3].cardinality) == [3, 3]
    assert list(pgmpy_pgm.factors[4].cardinality) == [3, 3]
    assert list(pgmpy_pgm.factors[5].cardinality) == [3, 3]

    # Check values 
    check_values(pgmpy_pgm.factors[0], [10,19,20])
    check_values(pgmpy_pgm.factors[1], [10,10,30])
    check_values(pgmpy_pgm.factors[2], [10,20,10])
    check_values(pgmpy_pgm.factors[3], np.ones(9))
    check_values(pgmpy_pgm.factors[4], np.ones(9))
    check_values(pgmpy_pgm.factors[5], np.ones(9))


@require_pgmpy
def Xtest_simple_root_node_conversion():
    """Test conversion of a simple model with only root nodes."""
    # Create conin model
    conin_bn = DiscreteBayesianNetwork()
    conin_bn.states = {"A": [0, 1], "B": [0, 1]}
    conin_bn.cpds = [
        DiscreteCPD(node="A", values=[0.3, 0.7]),
        DiscreteCPD(node="B", values=[0.2, 0.8]),
    ]

    # Convert to pgmpy
    conin_mn = convert_bn_to_mn(conin_bn)
    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_mn)

    # Verify structure
    assert isinstance(pgmpy_pgm, pgmpy_DiscreteBayesianNetwork)
    assert set(pgmpy_pgm.nodes()) == {"A", "B"}
    assert len(pgmpy_pgm.edges()) == 0  # No edges for root nodes

    # Verify factors
    factors = pgmpy_pgm.get_factors()
    assert len(factors) == 2

    # Find and verify each factor
    factor_a = next(factor for factor in factors if factor.variables[-1] == "A")
    factor_b = next(factor for factor in factors if factor.variables[-1] == "B")

    assert factor_a.variable == "A"
    assert factor_a.variable_card == 2
    np.testing.assert_array_almost_equal(factor_a.get_values(), [[0.3], [0.7]])

    assert factor_b.variable == "B"
    assert factor_b.variable_card == 2
    np.testing.assert_array_almost_equal(factor_b.get_values(), [[0.2], [0.8]])

    # Verify model is valid
    pgmpy_pgm.check_model()


@require_pgmpy
def Xtest_simple_parent_child_conversion():
    """Test conversion of a model with parent-child relationships."""
    # Create conin model
    conin_pgm = DiscreteBayesianNetwork()
    conin_pgm.states = {"A": [0, 1], "B": [0, 1]}
    conin_pgm.cpds = [
        DiscreteCPD(node="A", values=[0.3, 0.7]),
        DiscreteCPD(node="B", parents=["A"], values={0: [0.2, 0.8], 1: [0.9, 0.1]}),
    ]

    # Convert to pgmpy
    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_pgm)

    # Verify structure
    assert isinstance(pgmpy_pgm, pgmpy_DiscreteBayesianNetwork)
    assert set(pgmpy_pgm.nodes()) == {"A", "B"}
    assert len(pgmpy_pgm.edges()) == 1
    assert ("A", "B") in pgmpy_pgm.edges()

    # Verify CPDs
    cpds = pgmpy_pgm.get_cpds()
    assert len(cpds) == 2

    # Find and verify each CPD
    cpd_a = next(cpd for cpd in cpds if cpd.variable == "A")
    cpd_b = next(cpd for cpd in cpds if cpd.variable == "B")

    # Verify A (root node)
    assert cpd_a.variable == "A"
    assert cpd_a.variable_card == 2
    np.testing.assert_array_almost_equal(cpd_a.get_values(), [[0.3], [0.7]])

    # Verify B (child node)
    assert cpd_b.variable == "B"
    assert cpd_b.variable_card == 2
    np.testing.assert_array_almost_equal(cpd_b.get_values(), [[0.2, 0.9], [0.8, 0.1]])

    # Verify model is valid
    pgmpy_pgm.check_model()


@require_pgmpy
def Xtest_cancer_model_conversion():
    """Test conversion using the complex cancer example."""
    from conin.bayesian_network.examples import cancer1_BN_conin

    # Get conin cancer model
    cancer_data = cancer1_BN_conin()
    conin_pgm = cancer_data.pgm

    # Convert to pgmpy
    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_pgm)

    # Verify structure
    expected_nodes = {"Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"}
    assert set(pgmpy_pgm.nodes()) == expected_nodes

    # Verify edges (from cancer model structure)
    expected_edges = {
        ("Pollution", "Cancer"),
        ("Smoker", "Cancer"),
        ("Cancer", "Xray"),
        ("Cancer", "Dyspnoea"),
    }
    actual_edges = set(pgmpy_pgm.edges())
    assert actual_edges == expected_edges

    # Verify CPDs
    cpds = {cpd.variable: cpd for cpd in pgmpy_pgm.get_cpds()}

    # Verify Pollution CPD
    cpd_poll = cpds["Pollution"]
    assert cpd_poll.variable_card == 2
    np.testing.assert_array_almost_equal(cpd_poll.get_values(), [[0.9], [0.1]])

    # Verify Smoker CPD
    cpd_smoke = cpds["Smoker"]
    assert cpd_smoke.variable_card == 2
    np.testing.assert_array_almost_equal(cpd_smoke.get_values(), [[0.3], [0.7]])

    # Verify Cancer CPD (has parents: Smoker, Pollution)
    cpd_cancer = cpds["Cancer"]
    assert cpd_cancer.variable_card == 2
    expected_cancer_values = [[0.03, 0.05, 0.001, 0.02], [0.97, 0.95, 0.999, 0.98]]
    np.testing.assert_array_almost_equal(
        cpd_cancer.get_values(), expected_cancer_values
    )

    # Verify Xray CPD (has parent: Cancer)
    cpd_xray = cpds["Xray"]
    assert cpd_xray.variable_card == 2
    expected_xray_values = [[0.9, 0.2], [0.1, 0.8]]
    np.testing.assert_array_almost_equal(cpd_xray.get_values(), expected_xray_values)

    # Verify Dyspnoea CPD (has parent: Cancer)
    cpd_dysp = cpds["Dyspnoea"]
    assert cpd_dysp.variable_card == 2
    expected_dysp_values = [[0.65, 0.3], [0.35, 0.7]]
    np.testing.assert_array_almost_equal(cpd_dysp.get_values(), expected_dysp_values)

    # Verify model is valid
    pgmpy_pgm.check_model()


@require_pgmpy
def Xtest_model_equivalence():
    """Test that converted models produce equivalent results to original conin models."""
    from conin.common.pgmpy import log_potential as pgmpy_log_potential
    from conin.common.conin import log_potential as conin_log_potential

    # Create test model
    conin_pgm = DiscreteBayesianNetwork()
    conin_pgm.states = {"A": [0, 1], "B": [0, 1]}
    conin_pgm.cpds = [
        DiscreteCPD(node="A", values=[0.3, 0.7]),
        DiscreteCPD(node="B", parents=["A"], values={0: [0.2, 0.8], 1: [0.9, 0.1]}),
    ]

    # Convert to pgmpy
    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_pgm)

    # Test various assignments
    test_cases = [
        {"A": 0, "B": 0},
        {"A": 0, "B": 1},
        {"A": 1, "B": 0},
        {"A": 1, "B": 1},
    ]

    for assignment in test_cases:
        # Calculate log potentials
        conin_log_pot = conin_log_potential(conin_pgm, assignment)
        pgmpy_log_pot = pgmpy_log_potential(pgmpy_pgm, assignment)

        # They should be approximately equal
        np.testing.assert_almost_equal(
            conin_log_pot,
            pgmpy_log_pot,
            decimal=10,
            err_msg=f"Log potentials differ for assignment {assignment}",
        )


@require_pgmpy
def Xtest_single_node_model():
    """Test conversion of a model with a single node."""
    conin_pgm = DiscreteBayesianNetwork()
    conin_pgm.states = {"A": [0, 1, 2]}
    conin_pgm.cpds = [DiscreteCPD(node="A", values=[0.1, 0.2, 0.7])]

    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_pgm)

    assert len(pgmpy_pgm.nodes()) == 1
    assert len(pgmpy_pgm.edges()) == 0
    pgmpy_pgm.check_model()


@require_pgmpy
def Xtest_model_with_multiple_parents():
    """Test conversion of a model with nodes having multiple parents."""
    conin_pgm = DiscreteBayesianNetwork()
    conin_pgm.states = {"A": [0, 1], "B": [0, 1], "C": [0, 1], "D": [0, 1]}
    conin_pgm.cpds = [
        DiscreteCPD(node="A", values=[0.5, 0.5]),
        DiscreteCPD(node="B", values=[0.5, 0.5]),
        DiscreteCPD(node="C", values=[0.5, 0.5]),
        DiscreteCPD(
            node="D",
            parents=["A", "B", "C"],
            values={
                (0, 0, 0): [0.1, 0.9],
                (0, 0, 1): [0.2, 0.8],
                (0, 1, 0): [0.3, 0.7],
                (0, 1, 1): [0.4, 0.6],
                (1, 0, 0): [0.5, 0.5],
                (1, 0, 1): [0.6, 0.4],
                (1, 1, 0): [0.7, 0.3],
                (1, 1, 1): [0.8, 0.2],
            },
        ),
    ]

    pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_pgm)

    # Verify structure
    assert len(pgmpy_pgm.nodes()) == 4
    assert len(pgmpy_pgm.edges()) == 3  # A->D, B->D, C->D

    # Verify D CPD has correct structure
    cpd_d = next(cpd for cpd in pgmpy_pgm.get_cpds() if cpd.variable == "D")
    assert cpd_d.variable_card == 2

    pgmpy_pgm.check_model()


@require_pgmpy
def test_integration_with_bayesian_examples():
    """Test conversion works with various examples from the bayesian_network module."""
    from conin.bayesian_network.examples import (
        simple1_BN_conin,
        DBDA_5_1_conin,
        holmes_conin,
        tb2_BN_conin,
    )

    examples = [simple1_BN_conin(), DBDA_5_1_conin(), holmes_conin(), tb2_BN_conin()]

    for example_data in examples:
        # Convert to pgmpy
        conin_bn = example_data.pgm
        conin_mn = create_mn_from_bn(conin_bn)
        pgmpy_pgm = convert_conin_to_pgmpy_mn(conin_mn)

        # Verify basic properties
        assert len(pgmpy_pgm.nodes()) == len(conin_mn.nodes)
        assert isinstance(pgmpy_pgm, pgmpy_DiscreteMarkovNetwork)

        # Verify model validity
        pgmpy_pgm.check_model()

        # Verify all CPDs were converted
        assert len(pgmpy_pgm.get_factors()) == len(conin_mn.factors)
