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
    for i, assign in enumerate(factor.assignment(list(range(len(values))))):
        kwargs = {k: v for k, v in assign}
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
    check_values(pgmpy_pgm.factors[0], [1, 1, 2])
    check_values(pgmpy_pgm.factors[1], [1, 1, 3])
    check_values(pgmpy_pgm.factors[2], [1, 2, 1])
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
    check_values(pgmpy_pgm.factors[0], [10, 19, 20])
    check_values(pgmpy_pgm.factors[1], [10, 10, 30])
    check_values(pgmpy_pgm.factors[2], [10, 20, 10])
    check_values(pgmpy_pgm.factors[3], np.ones(9))
    check_values(pgmpy_pgm.factors[4], np.ones(9))
    check_values(pgmpy_pgm.factors[5], np.ones(9))


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

        pgmpy_pgm.check_model()
