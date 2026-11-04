"""Constraints cannot be changed after construction.

Before, every attribute of a constraint could be reassigned (``c.func = ...``,
``mvr.name = ...``, ``mvr_timerange(mvr, r, inplace=True)``) and the lists
and dicts it held could be changed in place, so a constrained model, which
is meant to be immutable, could be changed through its constraints.
"""

import copy
import pickle

import numpy as np
import pytest

from conin import ConstrainedHiddenMarkovModel, HiddenMarkovModel
from conin.constraints import (
    FactorOracleConstraint,
    MVRConstraint,
    OracleConstraint,
    PyomoConstraint,
    RecursiveConstraint,
    SmoekConstraint,
    Toulbar2Constraint,
)
from conin.constraints.mvr.base import HomMVR, InhomMVR


def predicate(assignment):
    return True


def apply(model):
    return model


def initial(h):
    return 0


def update(state, h):
    return state


def feasible(state):
    return True


def _hom_mvr():
    return HomMVR(
        hidden_states=["a", "b"],
        mediation_states=[False, True],
        ini={"a": True, "b": False},
        upd={(m, h): m or h == "a" for m in [False, True] for h in ["a", "b"]},
        evl={False: False, True: True},
        time_range=[0, 3],
        name="contains_a",
    )


def _inhom_mvr():
    return InhomMVR(
        hidden_states=["a", "b"],
        mediation_states=[[False, True], [False, True]],
        ini={"a": True, "b": False},
        upd=[{(m, h): m or h == "a" for m in [False, True] for h in ["a", "b"]}],
        evl=[{False: False, True: True}, {False: False, True: True}],
    )


CONSTRAINTS = {
    "pyomo": lambda: PyomoConstraint(apply),
    "toulbar2": lambda: Toulbar2Constraint(apply),
    "smoek": lambda: SmoekConstraint(apply),
    "oracle": lambda: OracleConstraint(func=predicate),
    "factor_oracle": lambda: FactorOracleConstraint(func=predicate, nodes=["A"]),
    "recursive": lambda: RecursiveConstraint(
        initial=initial, update=update, feasible=feasible
    ),
    "mvr_constraint": lambda: MVRConstraint(func=apply),
    "hom_mvr": _hom_mvr,
    "inhom_mvr": _inhom_mvr,
}


@pytest.fixture(params=sorted(CONSTRAINTS))
def constraint(request):
    return CONSTRAINTS[request.param]()


def test_attributes_cannot_be_set_or_deleted(constraint):
    with pytest.raises(AttributeError, match="immutable"):
        constraint.name = "other"
    with pytest.raises(AttributeError, match="immutable"):
        constraint.new_attribute = 1
    with pytest.raises(AttributeError, match="immutable"):
        del constraint.name
    assert constraint.name != "other"


def test_copies_and_pickles_are_equal_and_immutable(constraint):
    for duplicate in (
        copy.copy(constraint),
        copy.deepcopy(constraint),
        pickle.loads(pickle.dumps(constraint)),
    ):
        assert type(duplicate) is type(constraint)
        assert duplicate.name == constraint.name
        with pytest.raises(AttributeError, match="immutable"):
            duplicate.name = "other"


def test_factor_oracle_nodes_are_read_only():
    nodes = ["A", "B"]
    constraint = FactorOracleConstraint(func=predicate, nodes=nodes)
    with pytest.raises(TypeError, match="read-only"):
        constraint.nodes.append("C")
    nodes.append("C")
    assert constraint.nodes == ["A", "B"]


@pytest.mark.parametrize("make", [_hom_mvr, _inhom_mvr])
def test_mvr_data_is_read_only(make):
    mvr = make()
    with pytest.raises(TypeError, match="read-only"):
        mvr.hidden_states.append("c")
    with pytest.raises(TypeError, match="read-only"):
        mvr.ini["a"] = False
    with pytest.raises(TypeError, match="read-only"):
        mvr.evl.clear()
    if mvr.time_range is not None:
        with pytest.raises(TypeError, match="read-only"):
            mvr.time_range[0] = 1


def test_mvr_inputs_are_copied():
    hidden_states = ["a", "b"]
    ini = {"a": True, "b": False}
    mvr = HomMVR(
        hidden_states=hidden_states,
        mediation_states=[False, True],
        ini=ini,
        upd={(m, h): m or h == "a" for m in [False, True] for h in ["a", "b"]},
        evl={False: False, True: True},
    )
    hidden_states.append("c")
    ini["a"] = False
    assert mvr.hidden_states == ["a", "b"]
    assert mvr.ini == {"a": True, "b": False}


@pytest.mark.parametrize("make", [_hom_mvr, _inhom_mvr])
def test_mvr_repn_is_read_only(make):
    repn = make().repn
    with pytest.raises(AttributeError, match="immutable"):
        repn.ini_array = np.zeros((2, 2))
    with pytest.raises(AttributeError, match="immutable"):
        del repn.evl_array
    with pytest.raises(ValueError, match="read-only"):
        repn.ini_array[0, 0] = 0.0
    with pytest.raises(TypeError, match="read-only"):
        repn.hidden_states.append(2)
    for duplicate in (copy.deepcopy(repn), pickle.loads(pickle.dumps(repn))):
        with pytest.raises(AttributeError, match="immutable"):
            duplicate.ini_array = np.zeros((2, 2))
        with pytest.raises(ValueError, match="read-only"):
            duplicate.ini_array[0, 0] = 0.0


def test_with_name_leaves_the_mvr_unchanged():
    mvr = _hom_mvr()
    renamed = mvr.with_name("other")
    assert renamed.name == "other"
    assert mvr.name == "contains_a"
    assert renamed.time_range == mvr.time_range


def test_a_constrained_model_cannot_be_changed_through_its_constraints():
    hmm = HiddenMarkovModel(
        start_probs={"a": 0.5, "b": 0.5},
        transition_probs={(i, j): 0.5 for i in "ab" for j in "ab"},
        emission_probs={(h, o): 0.5 for h in "ab" for o in "xy"},
    )
    mvr = _hom_mvr()
    cmodel = ConstrainedHiddenMarkovModel(hmm, mvr)
    with pytest.raises(AttributeError, match="immutable"):
        cmodel.constraints[0]._time_range = None
    with pytest.raises(TypeError, match="read-only"):
        cmodel.constraints[0].upd[(False, "b")] = True
    assert cmodel.constraints[0].time_range == [0, 3]
