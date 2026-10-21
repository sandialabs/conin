import pytest

from conin.exceptions import InvalidInputError
from conin.graphical_models.hmm_observations import (
    as_sequence,
    require_sequence_type,
    time_range,
    unrolled_evidence,
)


@pytest.mark.parametrize(
    "seq",
    [["a", "b"], ("a", "b"), {0: "a", 1: "b"}, {1: "b", 0: "a"}],
    ids=["list", "tuple", "dict", "unordered-dict"],
)
def test_as_sequence(seq):
    assert as_sequence(seq) == ["a", "b"]


def test_as_sequence_returns_a_new_list():
    seq = ["a", "b"]
    assert as_sequence(seq) is not seq


@pytest.mark.parametrize(
    "seq,match",
    [
        (None, "list, a tuple or a dict"),
        ("ab", "list, a tuple or a dict"),
        ({"a", "b"}, "list, a tuple or a dict"),
        ([], "must not be empty"),
        ({}, "must not be empty"),
        ({0: "a", 2: "b"}, "keyed by the times 0..T-1"),
        ({1: "a"}, "keyed by the times 0..T-1"),
        ({"0": "a"}, "keyed by the times 0..T-1"),
    ],
)
def test_as_sequence_errors(seq, match):
    with pytest.raises(InvalidInputError, match=match):
        as_sequence(seq)


def test_as_sequence_names_the_sequence():
    with pytest.raises(InvalidInputError, match="The hidden sequence must not"):
        as_sequence([], "hidden sequence")


@pytest.mark.parametrize("seq", [[], (), {}, {0: "a", 5: "b"}])
def test_require_sequence_type_accepts_sparse_dicts(seq):
    require_sequence_type(seq)


@pytest.mark.parametrize("seq", [None, "ab", 3])
def test_require_sequence_type_errors(seq):
    with pytest.raises(InvalidInputError, match="list, a tuple or a dict"):
        require_sequence_type(seq)


def test_time_range():
    observed = ["a", "b", "c"]
    assert time_range(observed, 0, None) == (0, 2)
    assert time_range(observed, 0, 1) == (0, 1)
    assert time_range(observed, 1, 2) == (1, 2)
    with pytest.raises(InvalidInputError, match="stop=3 is past the last"):
        time_range(observed, 0, 3)


def test_unrolled_evidence():
    assert unrolled_evidence(["a", "b", "c"], 1, 2) == {("E", 1): "b", ("E", 2): "c"}
