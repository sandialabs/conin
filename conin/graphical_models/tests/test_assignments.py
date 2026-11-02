import pytest

from conin.exceptions import InvalidInputError
from conin.graphical_models.assignments import as_sequence, require_sequence_type


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
