"""Tests for ConstrainedDynamicDiscreteBayesianNetwork."""

import pytest

import conin.examples.bayesian_network
import conin.examples.dynamic_bayesian_network
from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
from conin.exceptions import InvalidInputError


def _make_dbn():
    return conin.examples.dynamic_bayesian_network.simple1_DDBN_conin().pgm


class TestConstrainedDynamicDiscreteBayesianNetwork:
    def test_wraps_model(self):
        dbn = _make_dbn()
        cdbn = ConstrainedDynamicDiscreteBayesianNetwork(dbn)
        assert cdbn.model is dbn
        assert cdbn.constraints == ()

    def test_nodes_is_a_property(self):
        # Regression test: ``nodes`` used to be a method on this class.
        dbn = _make_dbn()
        cdbn = ConstrainedDynamicDiscreteBayesianNetwork(dbn)
        assert isinstance(type(cdbn).__dict__["nodes"], property)
        assert cdbn.nodes == dbn.nodes

    def test_states_of_delegates(self):
        dbn = _make_dbn()
        cdbn = ConstrainedDynamicDiscreteBayesianNetwork(dbn)
        for node in dbn.dynamic_nodes:
            assert cdbn.states_of(node) == dbn.states_of(node)
            assert cdbn.states_of((node, 0)) == dbn.states_of((node, 0))

    def test_check_model_delegates(self):
        cdbn = ConstrainedDynamicDiscreteBayesianNetwork(_make_dbn())
        cdbn.check_model()  # should not raise

    def test_wrong_model_type_rejected(self):
        bn = conin.examples.bayesian_network.simple1_BN_conin().pgm
        with pytest.raises(InvalidInputError):
            ConstrainedDynamicDiscreteBayesianNetwork(bn)
