import pytest

from conin import *
from conin.hmm import *
from conin.hmm.oracle_chmm import Oracle_CHMM

import conin.hmm.tests.test_cases as tc


@pytest.fixture
def constraint():
    def num_zeros_eq_five(seq):
        return seq.count(0) == 5

    return Constraint(func=num_zeros_eq_five, name="Test")


class Test_Constraints:
    true_seq = [0, 1, 0, 2, 3, 0, 0, 5, 0]
    false_seq1 = [0, 0, 0, 3, 3, 0, 0, 0]
    false_seq2 = [1, 1, 2, 3, 0, 0, 0]
    false_seq3 = []

    def test_call_no_func(self):
        with pytest.raises(InvalidInputError):
            _constraint = Constraint(name="No function")
            assert _constraint.name == "No function"
            _constraint(self.true_seq)

    def test_name(self, constraint):
        assert constraint.name == "Test"

    def test_no_name(self):
        _constraint = Constraint()
        assert _constraint.name == "Unnamed constraint"

    def test_call(self, constraint):
        assert constraint(self.true_seq)
        assert not constraint(self.false_seq1)
        assert not constraint(self.false_seq2)
        assert not constraint(self.false_seq3)

    def test_name_automate(self):
        def num_zeros_eq_five(seq):
            return seq.count(0) == 5  # pragma no cover

        _constraint = Constraint(func=num_zeros_eq_five)
        assert _constraint.name == "num_zeros_eq_five"


class Test_Common_Constraints:

    def test_all_dif(self):
        assert all_diff_constraint([1, 2])
        assert not all_diff_constraint([1, 1])
        assert all_diff_constraint([])

        assert all_diff_constraint.partial_func(3, [1, 2])
        assert not all_diff_constraint.partial_func(3, [1, 1])
        assert all_diff_constraint.partial_func(0, [])

    def test_always_appears_before(self):
        constraint = always_appears_before_constraint(1, 2)
        assert always_appears_before_constraint(1, 2)([1, 2])
        assert constraint([0, 1])
        assert constraint([2, 2])
        assert not constraint([2, 1])
        assert not constraint([1, 0, 2, 1])

        assert constraint.partial_func(3, [1, 2])
        assert constraint.partial_func(3, [0, 1])
        assert constraint.partial_func(3, [2, 2])
        assert not constraint.partial_func(3, [2, 1])
        assert not constraint.partial_func(4, [1, 0, 2, 1])

    def test_appears_at_least_once_before(self):
        constraint = appears_at_least_once_before_constraint(1, 2)
        assert appears_at_least_once_before_constraint(1, 2)([1, 2])
        assert constraint([0, 0, 1])
        assert not constraint([2, 2])
        assert not constraint([2, 1])
        assert constraint([0, 1, 2, 1])
        assert constraint([0, 0, 0])

        assert constraint.partial_func(3, [1, 2])
        assert constraint.partial_func(3, [0, 1])
        assert not constraint.partial_func(3, [2, 2])
        assert not constraint.partial_func(3, [2, 1])
        assert constraint.partial_func(5, [1, 0, 2, 1])

    def test_always_appears_after(self):
        constraint = always_appears_after_constraint(2, 1)
        assert always_appears_after_constraint(2, 1)([1, 2])
        assert constraint([0, 1])
        assert constraint([2, 2])
        assert not constraint([2, 1])
        assert not constraint([1, 0, 2, 1])

        assert constraint.partial_func(3, [1, 2])
        assert constraint.partial_func(3, [0, 1])
        assert constraint.partial_func(3, [2, 2])
        assert not constraint.partial_func(3, [2, 1])
        assert not constraint.partial_func(5, [1, 0, 2, 1])

    def test_appears_at_least_once_after(self):
        constraint = appears_at_least_once_after_constraint(1, 2)
        assert not appears_at_least_once_after_constraint(1, 2)([1, 2])
        assert constraint([0, 1])
        assert not constraint([2, 2])
        assert constraint([2, 1])
        assert constraint([1, 0, 2, 1])

        # Should always be true
        assert constraint.partial_func(3, [1, 2])
        assert constraint.partial_func(3, [0, 1])
        assert constraint.partial_func(3, [2, 2])
        assert constraint.partial_func(3, [2, 1])
        assert constraint.partial_func(5, [1, 0, 2, 1])

    def test_citation(self):
        constraint = citation_constraint()
        assert constraint([])
        assert constraint([1, 1, 1, 2, 2, 2, 7, 7])
        assert not constraint([1, 2, 2, 7, 7, 7, 7, 1])

        assert constraint.partial_func(2, [])
        assert constraint.partial_func(10, [1, 1, 1, 2, 2, 2, 7, 7])
        assert not constraint.partial_func(10, [1, 2, 2, 7, 7, 7, 7, 1])

    def test_has_minimum_number_of_occurences(self):
        constraint = has_minimum_number_of_occurences_constraint(
            val="h", count=2
        )
        assert not constraint([])
        assert not constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert constraint([1, 2, 1, 2, "h", "h", "h"])

        assert not constraint.partial_func(1, [])
        assert constraint.partial_func(10, [1, 2, "h", 1, 2])
        assert constraint.partial_func(11, ["h", "h"])
        assert constraint.partial_func(12, [1, 2, 1, 2, "h", "h", "h"])

    def test_has_maximum_number_of_occurences(self):
        constraint = has_maximum_number_of_occurences_constraint(
            val="h", count=2
        )
        assert constraint([])
        assert constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert not constraint([1, 2, 1, 2, "h", "h", "h"])

        assert constraint.partial_func(3, [])
        assert constraint.partial_func(6, [1, 2, "h", 1, 2])
        assert constraint.partial_func(3, ["h", "h"])
        assert not constraint.partial_func(9, [1, 2, 1, 2, "h", "h", "h"])

    def test_has_exact_number_of_occurences(self):
        constraint = has_exact_number_of_occurences_constraint(
            val="h", count=2
        )
        assert not constraint([])
        assert not constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert not constraint([1, 2, 1, 2, "h", "h", "h"])

        assert not constraint.partial_func(1, [])
        assert constraint.partial_func(10, [1, 2, "h", 1, 2])
        assert constraint.partial_func(7, ["h", "h"])
        assert not constraint.partial_func(8, [1, 2, 1, 2, "h", "h", "h"])

    def test_appears_at_least_once(self):
        constraint = appears_at_least_once_constraint(val="h")
        assert not constraint([])
        assert constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert constraint([1, 2, 1, 2, "h", "h", "h"])

        # Should always be true
        assert constraint.partial_func(10, [])
        assert constraint.partial_func(10, [1, 2, "h", 1, 2])
        assert constraint.partial_func(10, ["h", "h"])
        assert constraint.partial_func(10, [1, 2, 1, 2, "h", "h", "h"])

    def test_does_not_occur(self):
        constraint = does_not_occur_constraint(val="h")
        assert constraint([])
        assert constraint([1, 2, 1, 2])
        assert not constraint(["h", "h"])
        assert not constraint([1, 2, 1, 2, "h", "h", "h"])

        assert constraint.partial_func(4, [])
        assert constraint.partial_func(5, [1, 2, 1, 2])
        assert not constraint.partial_func(6, ["h", "h"])
        assert not constraint.partial_func(7, [1, 2, 1, 2, "h", "h", "h"])

    def test_fix_final_state(self):
        constraint = fix_final_state_constraint(val="h")
        assert not constraint([])
        assert not constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert constraint([1, 2, 1, 2, "h", "h", "h"])

        # Always true
        assert constraint.partial_func(2, [])
        assert constraint.partial_func(100, [1, 2, "h", 1, 2])
        assert constraint.partial_func(3, ["h", "h"])
        assert constraint.partial_func(8, [1, 2, 1, 2, "h", "h", "h"])

    def test_occurs_only_in_time_frame(self):
        seq = [1, 2, 3, 4, 5, 5, 5]
        assert occurs_only_in_time_frame_constraint(3, lower_t=2, upper_t=4)(
            seq
        )
        assert not occurs_only_in_time_frame_constraint(
            2, lower_t=2, upper_t=4
        )(seq)
        assert occurs_only_in_time_frame_constraint(6, lower_t=0, upper_t=5)(
            seq
        )
        assert occurs_only_in_time_frame_constraint(7)(
            seq
        )  # 1 occurs before the range
        assert not occurs_only_in_time_frame_constraint(5, upper_t=6)(
            seq
        )  # 5 is at the end
        assert not occurs_only_in_time_frame_constraint(1, lower_t=1)(
            seq
        )  # 1 is at the start
        assert occurs_only_in_time_frame_constraint(1)(
            []
        )  # No occurrences in an empty sequence

        assert occurs_only_in_time_frame_constraint(
            3, lower_t=2, upper_t=4
        ).partial_func(8, seq)
        assert not occurs_only_in_time_frame_constraint(
            2, lower_t=2, upper_t=4
        ).partial_func(8, seq)
        assert occurs_only_in_time_frame_constraint(
            6, lower_t=2, upper_t=100
        ).partial_func(8, seq)
        assert occurs_only_in_time_frame_constraint(7).partial_func(
            8, seq
        )  # 1 occurs before the range
        assert not occurs_only_in_time_frame_constraint(
            5, upper_t=6
        ).partial_func(
            8, seq
        )  # 5 is at the end
        assert not occurs_only_in_time_frame_constraint(
            1, lower_t=1
        ).partial_func(
            8, seq
        )  # 1 is at the start
        assert occurs_only_in_time_frame_constraint(1).partial_func(
            2, []
        )  # No occurrences in an empty sequence

    def test_occurs_at_least_once_in_time_frame(self):
        seq = [1, 2, 3, 4, 5, 5, 5]
        assert occurs_at_least_once_in_time_frame_constraint(
            3, lower_t=2, upper_t=4
        )(seq)
        assert not occurs_at_least_once_in_time_frame_constraint(
            2, lower_t=2, upper_t=4
        )(seq)
        assert not occurs_at_least_once_in_time_frame_constraint(
            6, lower_t=0, upper_t=5
        )(seq)
        assert not occurs_at_least_once_in_time_frame_constraint(7)(
            seq
        )  # 1 occurs before the range
        assert occurs_at_least_once_in_time_frame_constraint(5, upper_t=6)(
            seq
        )  # 5 is at the end
        assert not occurs_at_least_once_in_time_frame_constraint(1, lower_t=1)(
            seq
        )  # 1 is at the start
        assert not occurs_at_least_once_in_time_frame_constraint(1)(
            []
        )  # No occurrences in an empty sequence

        assert occurs_at_least_once_in_time_frame_constraint(
            3, lower_t=2, upper_t=4
        ).partial_func(8, seq)
        assert not occurs_at_least_once_in_time_frame_constraint(
            2, lower_t=2, upper_t=4
        ).partial_func(8, seq)
        assert occurs_at_least_once_in_time_frame_constraint(
            6, lower_t=0, upper_t=10
        ).partial_func(8, seq)
        assert not occurs_at_least_once_in_time_frame_constraint(
            7
        ).partial_func(
            8, seq
        )  # 1 occurs before the range
        assert occurs_at_least_once_in_time_frame_constraint(
            5, upper_t=6
        ).partial_func(
            8, seq
        )  # 5 is at the end
        assert not occurs_at_least_once_in_time_frame_constraint(
            1, lower_t=1
        ).partial_func(
            8, seq
        )  # 1 is at the start
        assert not occurs_at_least_once_in_time_frame_constraint(
            1
        ).partial_func(
            2, []
        )  # No occurrences in an empty sequence

    def test_or_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(
            val="h", count=3
        )
        constraint2 = has_maximum_number_of_occurences_constraint(
            val="h", count=1
        )
        constraint = or_constraints([constraint1, constraint2])
        assert constraint([])
        assert constraint([1, 2, "h", 1, 2])
        assert not constraint(["h", "h"])
        assert constraint([1, 2, 1, 2, "h", "h", "h"])

        assert constraint.partial_func(2, [])
        assert constraint.partial_func(17, [1, 2, "h", 1, 2])
        assert constraint.partial_func(3, ["h", "h"])
        assert not constraint.partial_func(2, ["h", "h"])
        assert constraint.partial_func(8, [1, 2, 1, 2, "h", "h", "h"])

    def test_not_constraint(self):
        constraint = has_exact_number_of_occurences_constraint(
            val="h", count=2
        )
        constraint = not_constraint(constraint)
        assert constraint([])
        assert constraint([1, 2, "h", 1, 2])
        assert not constraint(["h", "h"])
        assert constraint([1, 2, 1, 2, "h", "h", "h"])

        # Should all be true
        assert constraint.partial_func(10, [])
        assert constraint.partial_func(10, [1, 2, "h", 1, 2])
        assert constraint.partial_func(10, ["h", "h"])
        assert constraint.partial_func(10, [1, 2, 1, 2, "h", "h", "h"])

    def test_xor_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(
            val="h", count=2
        )
        constraint2 = has_maximum_number_of_occurences_constraint(
            val="h", count=2
        )
        constraint = xor_constraints([constraint1, constraint2])
        assert constraint([])
        assert constraint([1, 2, "h", 1, 2])
        assert not constraint(["h", "h"])
        assert constraint([1, 2, 1, 2, "h", "h", "h"])

        assert constraint.partial_func(5, [])
        assert constraint.partial_func(7, [1, 2, "h", 1, 2])
        assert constraint.partial_func(2, ["h", "h"])
        assert constraint.partial_func(3, ["h", "h"])
        assert constraint.partial_func(17, [1, 2, 1, 2, "h", "h", "h"])

        constraint1 = has_minimum_number_of_occurences_constraint(
            val="h", count=3
        )
        constraint2 = has_maximum_number_of_occurences_constraint(
            val="h", count=1
        )
        constraint = xor_constraints([constraint1, constraint2])
        assert not constraint(["h", "h"])

        assert constraint.partial_func(3, ["h", "h"])

        constraint1 = has_maximum_number_of_occurences_constraint(
            val="h", count=2
        )
        constraint2 = has_maximum_number_of_occurences_constraint(
            val="h", count=1
        )
        constraint = xor_constraints([constraint1, constraint2])
        assert not constraint.partial_func(10, ["h", "h", "h"])

    def test_and_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(
            val="h", count=2
        )
        constraint2 = has_maximum_number_of_occurences_constraint(
            val="h", count=2
        )
        constraint = and_constraints([constraint1, constraint2])
        assert not constraint([])
        assert not constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert not constraint([1, 2, 1, 2, "h", "h", "h"])

        assert not constraint.partial_func(1, [])
        assert constraint.partial_func(2, [])
        assert constraint.partial_func(9, [1, 2, "h", 1, 2])
        assert constraint.partial_func(9, ["h", "h"])
        assert not constraint.partial_func(9, [1, 2, 1, 2, "h", "h", "h"])
