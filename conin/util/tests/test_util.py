import pytest
import numpy as np

from conin.util import Util


class Test_Util:
    def test_normalize_dictionary(self):
        assert Util.normalize_dictionary({"0": 1, "1": 1}) == {
            "0": 0.5,
            "1": 0.5,
        }
        assert Util.normalize_dictionary({"0": 1, "1": 1, "2": 1, "3": -3}) == {
            "0": 0.25,
            "1": 0.25,
            "2": 0.25,
            "3": 0.25,
        }

    def test_normalize_2d_dictionary(self):
        assert Util.normalize_2d_dictionary(
            {("0", "0"): 1, ("0", "1"): 3, ("1", "0"): 1, ("1", "1"): -1}
        ) == {
            ("0", "0"): 0.25,
            ("0", "1"): 0.75,
            ("1", "0"): 0.5,
            ("1", "1"): 0.5,
        }

    def test_sample_from_vec(self):
        # Inverse transform sampling: the index of the first cumulative
        # probability that reaches the uniform draw.
        class FixedDraw:
            def __init__(self, value):
                self.value = value

            def random(self):
                return self.value

        vec = [0.1, 0.3, 0.6]
        for draw, index in [(0.0, 0), (0.05, 0), (0.2, 1), (0.39, 1), (0.41, 2)]:
            assert Util.sample_from_vec(vec, FixedDraw(draw)) == index
        # Probabilities that sum to slightly less than 1 give the last index.
        assert Util.sample_from_vec([0.5, 0.4999], FixedDraw(0.99995)) == 1

        rng = np.random.default_rng(0)
        draws = [Util.sample_from_vec(vec, rng) for _ in range(20000)]
        frequencies = np.bincount(draws, minlength=3) / len(draws)
        assert frequencies == pytest.approx(vec, abs=0.015)

    def test_normalize_vec(self):
        vec = [1, 2, 3, 4, 6]
        assert Util.normalize_vector(vec) == [
            1 / 16,
            2 / 16,
            3 / 16,
            4 / 16,
            6 / 16,
        ]
        assert Util.normalize_vector([-1, 2, 0, -1]) == [
            1 / 4,
            1 / 4,
            1 / 4,
            1 / 4,
        ]
