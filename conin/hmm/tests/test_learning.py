# TODO we need way more here
import pytest

from conin.hmm import learning


def test_add_unknowns():
    assert learning.add_unknowns([[0, 0, 1], [2], [0, 1, 3], [4]]) == [
        [0, 0, 1],
        ["__UNKNOWN__"],
        [0, 1, "__UNKNOWN__"],
        ["__UNKNOWN__"],
    ]
    assert learning.add_unknowns([[0, 0, 1], [2], [0, 1, 3], [4]], num=2) == [
        [0, 0, "__UNKNOWN__"],
        ["__UNKNOWN__"],
        [0, "__UNKNOWN__", "__UNKNOWN__"],
        ["__UNKNOWN__"],
    ]
    assert learning.add_unknowns([[0, 0, 1], [2], [0, 1, 3], [4]], token="test") == [
        [0, 0, 1],
        ["test"],
        [0, 1, "test"],
        ["test"],
    ]
