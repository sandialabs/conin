"""Tests for ConstrainedDiscreteBayesianNetwork."""

import pytest

import conin.examples.bayesian_network
import conin.examples.markov_network
from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.exceptions import InvalidInputError


def _make_bn():
    return conin.examples.bayesian_network.simple1_BN_conin().pgm


class TestConstrainedDiscreteBayesianNetwork:
    def test_wraps_model(self):
        bn = _make_bn()
        cbn = ConstrainedDiscreteBayesianNetwork(bn)
        assert cbn.model is bn
        assert cbn.constraints == ()

    def test_nodes_delegates_to_model(self):
        bn = _make_bn()
        cbn = ConstrainedDiscreteBayesianNetwork(bn)
        assert cbn.nodes == bn.nodes

    def test_states_of_delegates(self):
        bn = _make_bn()
        cbn = ConstrainedDiscreteBayesianNetwork(bn)
        for node in bn.nodes:
            assert cbn.states_of(node) == bn.states_of(node)

    def test_check_model_delegates(self):
        cbn = ConstrainedDiscreteBayesianNetwork(_make_bn())
        cbn.check_model()  # should not raise

    def test_wrong_model_type_rejected(self):
        mn = conin.examples.markov_network.ABC_conin().pgm
        with pytest.raises(InvalidInputError):
            ConstrainedDiscreteBayesianNetwork(mn)
