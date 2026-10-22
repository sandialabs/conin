"""Tests for conin.graphical_models.conversions."""

import pytest

from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteCPD,
    DynamicDiscreteBayesianNetwork,
    ExpressionVariable,
    create_bn_from_dbn,
)


def _z_dbn():
    t = ExpressionVariable()
    cpds = [
        DiscreteCPD(node=("Z", 0), values=[0.5, 0.5]),
        DiscreteCPD(
            node=("Z", t),
            parents=[("Z", t - 1)],
            values={0: [0.7, 0.3], 1: [0.8, 0.2]},
        ),
    ]
    return DynamicDiscreteBayesianNetwork(dynamic_states={"Z": [0, 1]}, cpds=cpds, t=t)


def _nodes(bn):
    return {cpd.node for cpd in bn.cpds}


def test_create_bn_from_dbn_does_not_mutate_dbn_t():
    dbn = _z_dbn()
    assert dbn.t.value() is None

    bn1 = create_bn_from_dbn(dbn=dbn, start=0, stop=3)
    assert dbn.t.value() is None

    bn2 = create_bn_from_dbn(dbn=dbn, start=0, stop=3)
    assert dbn.t.value() is None
    assert _nodes(bn1) == _nodes(bn2) == {("Z", i) for i in range(4)}


def test_create_bn_from_dbn_restores_preset_dbn_t():
    dbn = _z_dbn()
    dbn.t.set_value(7)

    bn = create_bn_from_dbn(dbn=dbn, start=0, stop=3)
    assert dbn.t.value() == 7
    assert _nodes(bn) == {("Z", i) for i in range(4)}


def test_create_bn_from_dbn_one_time_step():
    # stop == start used to raise a bare AssertionError, so map_query on a
    # one-observation HMM failed for integer_program, toulbar2 and VE.
    bn = create_bn_from_dbn(dbn=_z_dbn(), start=0, stop=0)
    assert _nodes(bn) == {("Z", 0)}
    assert bn.cpds[0].parents in (None, [])


@pytest.mark.parametrize("start", [1, 2])
def test_create_bn_from_dbn_refuses_a_later_start(start):
    # A later start used to fail with "AttributeError: 'float' object has no
    # attribute 'keys'": the time-start nodes kept the initial CPDs.
    with pytest.raises(InvalidInputError, match="only be unrolled from start=0"):
        create_bn_from_dbn(dbn=_z_dbn(), start=start, stop=start + 2)


def test_create_bn_from_dbn_refuses_stop_before_start():
    with pytest.raises(InvalidInputError, match="stop >= start"):
        create_bn_from_dbn(dbn=_z_dbn(), start=0, stop=-1)
