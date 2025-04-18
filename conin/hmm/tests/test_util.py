import pytest

from conin.util import Util


class Test_Util:
    def test_normalize_dictionary(self):
        assert Util.normalize_dictionary({"0": 1, "1": 1}) == {"0": 0.5, "1": 0.5}
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
