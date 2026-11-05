import pytest

from conin.graphical_models import DiscreteBayesianNetwork


def _assert_same_model(a, b):
    """Assert that two loaded models have the same nodes, states and tables."""
    assert type(a) is type(b)
    assert a.nodes == b.nodes
    assert a.states == b.states
    if isinstance(a, DiscreteBayesianNetwork):
        assert {c.node: c for c in a.cpds} == {c.node: c for c in b.cpds}
    else:
        assert list(a.factors) == list(b.factors)


@pytest.fixture
def assert_same_model():
    return _assert_same_model
