import copy
import json
import pickle

import pytest

from conin.util.frozen import FrozenDict, FrozenList, freeze


def _nested():
    return freeze({"a": [1, 2, {"x": [3]}], ("k", 1): {"s": 0.5}})


def test_freeze_is_recursive_and_equal_to_the_input():
    data = {"a": [1, 2, {"x": [3]}], ("k", 1): {"s": 0.5}}
    frozen = freeze(data)
    assert frozen == data
    assert repr(frozen) == repr(data)
    assert type(frozen) is FrozenDict
    assert type(frozen["a"]) is FrozenList
    assert type(frozen["a"][2]) is FrozenDict
    assert type(frozen["a"][2]["x"]) is FrozenList


def test_freeze_copies_its_input():
    states = [0, 1]
    frozen = freeze({"A": states})
    states.append(2)
    assert frozen["A"] == [0, 1]


def test_freeze_of_frozen_data_is_the_same_object():
    frozen = _nested()
    assert freeze(frozen) is frozen


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.append(1),
        lambda v: v.extend([1]),
        lambda v: v.insert(0, 1),
        lambda v: v.pop(),
        lambda v: v.remove(1),
        lambda v: v.clear(),
        lambda v: v.sort(),
        lambda v: v.reverse(),
        lambda v: v.__setitem__(0, 9),
        lambda v: v.__delitem__(0),
        lambda v: v.__iadd__([1]),
        lambda v: v.__imul__(2),
    ],
)
def test_frozen_list_is_read_only(mutate):
    with pytest.raises(TypeError, match="read-only"):
        mutate(freeze([1, 2]))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("b", 1),
        lambda d: d.__delitem__("a"),
        lambda d: d.__ior__({"b": 1}),
        lambda d: d.pop("a"),
        lambda d: d.popitem(),
        lambda d: d.clear(),
        lambda d: d.update(b=1),
        lambda d: d.setdefault("b", 1),
    ],
)
def test_frozen_dict_is_read_only(mutate):
    with pytest.raises(TypeError, match="read-only"):
        mutate(freeze({"a": 1}))


def test_reading_and_copying_give_plain_containers():
    frozen = freeze([3, 1])
    assert sorted(frozen) == [1, 3]
    assert type(frozen + [0]) is list
    assert type(list(frozen)) is list
    assert type(dict(_nested())) is dict
    assert json.dumps(_nested()["a"]) == '[1, 2, {"x": [3]}]'


def test_pickle_and_copy_keep_the_data_read_only():
    frozen = _nested()
    unpickled = pickle.loads(pickle.dumps(frozen))
    assert unpickled == frozen
    assert type(unpickled["a"][2]) is FrozenDict
    with pytest.raises(TypeError):
        unpickled["a"].append(1)
    for copied in (copy.copy(frozen), copy.deepcopy(frozen)):
        assert copied == frozen and type(copied) is FrozenDict
        with pytest.raises(TypeError):
            copied["a"].append(1)
    # A deep copy copies mutable items too
    item = [1]
    nested = FrozenList([item])
    assert copy.deepcopy(nested)[0] is not item
