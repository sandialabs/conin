import pytest

from conin.util import Util


class Test_Util:
    def test_normalize_dictionary(self):
        assert Util.normalize_dictionary({"0": 1, "1": 1}) == {
            "0": 0.5, "1": 0.5}
        assert Util.normalize_dictionary({"0": 1, "1": 1, "2": 1, "3": -3}) == {
            "0": 0.25,
            "1": 0.25,
            "2": 0.25,
            "3": 0.25,
        }

    def test_normalize_2d_dictionary(self):
        assert Util.normalize_2d_dictionary(
            {("0", "0"): 1, ("0", "1"): 3, ("1", "0"): 1, ("1", "1"): -1}
        ) == {("0", "0"): 0.25, ("0", "1"): 0.75, ("1", "0"): 0.5, ("1", "1"): 0.5}

    def test_sample_from_vec(self):
        vec = [0.1, 0.3, 0.6]
        for i in range(10):
            assert Util.sample_from_vec(vec) in {0, 1, 2}

    def test_normalize_vec(self):
        vec = [1, 2, 3, 4, 6]
        assert Util.normalize_vector(
            vec) == [1 / 16, 2 / 16, 3 / 16, 4 / 16, 6 / 16]
        assert Util.normalize_vector(
            [-1, 2, 0, -1]) == [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    def test_normalize_matrix(self):
        mat = [[1, 2, 3, 4, 6], [1, -1, 1], [-1, 1]]
        assert Util.normalize_matrix(mat) == [
            [1 / 16, 2 / 16, 3 / 16, 4 / 16, 6 / 16],
            [1, -1, 1],
            [1 / 2, 1 / 2],
        ]
