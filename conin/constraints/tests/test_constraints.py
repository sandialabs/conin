import pytest

from conin.exceptions import InvalidInputError
from conin.constraints.oracle import *
from conin.hidden_markov_model import *


def _d(seq):
    """Convert a list to a dict with integer indices as keys."""
    return {i: v for i, v in enumerate(seq)}


@pytest.fixture
def constraint():
    @oracle_constraint_fn(name="Test")
    def num_zeros_eq_five(states):
        return sum(1 for v in states.values() if v == 0) == 5

    return num_zeros_eq_five


class Test_Constraints:
    true_seq = _d([0, 1, 0, 2, 3, 0, 0, 5, 0])
    false_seq1 = _d([0, 0, 0, 3, 3, 0, 0, 0])
    false_seq2 = _d([1, 1, 2, 3, 0, 0, 0])
    false_seq3 = _d([])

    def test_call_no_func(self):
        with pytest.raises(InvalidInputError):
            _constraint = OracleConstraint(name="No function")
            assert _constraint.name == "No function"
            _constraint(self.true_seq)

    def test_name(self, constraint):
        assert constraint.name == "Test"

    def test_no_name(self):
        _constraint = OracleConstraint()
        assert _constraint.name == "Unnamed constraint"

    def test_call(self, constraint):
        assert constraint(self.true_seq)
        assert not constraint(self.false_seq1)
        assert not constraint(self.false_seq2)
        assert not constraint(self.false_seq3)

    def test_name_automate(self):
        @oracle_constraint_fn()
        def num_zeros_eq_five(states):
            return sum(1 for v in states.values() if v == 0) == 5  # pragma no cover

        assert num_zeros_eq_five.name == "num_zeros_eq_five"


class Test_Common_Constraints:

    def test_all_dif(self):
        assert all_diff_constraint(_d([1, 2]))
        assert not all_diff_constraint(_d([1, 1]))
        assert all_diff_constraint(_d([]))

        assert all_diff_constraint.partial_func(3, _d([1, 2]))
        assert not all_diff_constraint.partial_func(3, _d([1, 1]))
        assert all_diff_constraint.partial_func(0, _d([]))

    def test_always_appears_before(self):
        constraint = always_appears_before_constraint(1, 2)
        assert always_appears_before_constraint(1, 2)(_d([1, 2]))
        assert constraint(_d([0, 1]))
        assert constraint(_d([2, 2]))
        assert not constraint(_d([2, 1]))
        assert not constraint(_d([1, 0, 2, 1]))

        assert constraint.partial_func(3, _d([1, 2]))
        assert constraint.partial_func(3, _d([0, 1]))
        assert constraint.partial_func(3, _d([2, 2]))
        assert not constraint.partial_func(3, _d([2, 1]))
        assert not constraint.partial_func(4, _d([1, 0, 2, 1]))

    def test_appears_at_least_once_before(self):
        constraint = appears_at_least_once_before_constraint(1, 2)
        assert appears_at_least_once_before_constraint(1, 2)(_d([1, 2]))
        assert constraint(_d([0, 0, 1]))
        assert not constraint(_d([2, 2]))
        assert not constraint(_d([2, 1]))
        assert constraint(_d([0, 1, 2, 1]))
        assert constraint(_d([0, 0, 0]))

        assert constraint.partial_func(3, _d([1, 2]))
        assert constraint.partial_func(3, _d([0, 1]))
        assert not constraint.partial_func(3, _d([2, 2]))
        assert not constraint.partial_func(3, _d([2, 1]))
        assert constraint.partial_func(5, _d([1, 0, 2, 1]))

    def test_always_appears_after(self):
        constraint = always_appears_after_constraint(2, 1)
        assert always_appears_after_constraint(2, 1)(_d([1, 2]))
        assert constraint(_d([0, 1]))
        assert constraint(_d([2, 2]))
        assert not constraint(_d([2, 1]))
        assert not constraint(_d([1, 0, 2, 1]))

        assert constraint.partial_func(3, _d([1, 2]))
        assert constraint.partial_func(3, _d([0, 1]))
        assert constraint.partial_func(3, _d([2, 2]))
        assert not constraint.partial_func(3, _d([2, 1]))
        assert not constraint.partial_func(5, _d([1, 0, 2, 1]))

    def test_appears_at_least_once_after(self):
        constraint = appears_at_least_once_after_constraint(1, 2)
        assert not appears_at_least_once_after_constraint(1, 2)(_d([1, 2]))
        assert constraint(_d([0, 1]))
        assert not constraint(_d([2, 2]))
        assert constraint(_d([2, 1]))
        assert constraint(_d([1, 0, 2, 1]))

        # Should always be true
        assert constraint.partial_func(3, _d([1, 2]))
        assert constraint.partial_func(3, _d([0, 1]))
        assert constraint.partial_func(3, _d([2, 2]))
        assert constraint.partial_func(3, _d([2, 1]))
        assert constraint.partial_func(5, _d([1, 0, 2, 1]))

    def test_citation(self):
        constraint = citation_constraint
        assert constraint(_d([]))
        assert constraint(_d([1, 1, 1, 2, 2, 2, 7, 7]))
        assert not constraint(_d([1, 2, 2, 7, 7, 7, 7, 1]))

        assert constraint.partial_func(2, _d([]))
        assert constraint.partial_func(10, _d([1, 1, 1, 2, 2, 2, 7, 7]))
        assert not constraint.partial_func(10, _d([1, 2, 2, 7, 7, 7, 7, 1]))

    def test_has_minimum_number_of_occurences(self):
        constraint = has_minimum_number_of_occurences_constraint(val="h", count=2)
        assert not constraint(_d([]))
        assert not constraint(_d([1, 2, "h", 1, 2]))
        assert constraint(_d(["h", "h"]))
        assert constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        assert not constraint.partial_func(1, _d([]))
        assert constraint.partial_func(10, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(11, _d(["h", "h"]))
        assert constraint.partial_func(12, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_has_maximum_number_of_occurences(self):
        constraint = has_maximum_number_of_occurences_constraint(val="h", count=2)
        assert constraint(_d([]))
        assert constraint(_d([1, 2, "h", 1, 2]))
        assert constraint(_d(["h", "h"]))
        assert not constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        assert constraint.partial_func(3, _d([]))
        assert constraint.partial_func(6, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(3, _d(["h", "h"]))
        assert not constraint.partial_func(9, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_has_exact_number_of_occurences(self):
        constraint = has_exact_number_of_occurences_constraint(val="h", count=2)
        assert not constraint(_d([]))
        assert not constraint(_d([1, 2, "h", 1, 2]))
        assert constraint(_d(["h", "h"]))
        assert not constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        assert not constraint.partial_func(1, _d([]))
        assert constraint.partial_func(10, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(7, _d(["h", "h"]))
        assert not constraint.partial_func(8, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_appears_at_least_once(self):
        constraint = appears_at_least_once_constraint(val="h")
        assert not constraint(_d([]))
        assert constraint(_d([1, 2, "h", 1, 2]))
        assert constraint(_d(["h", "h"]))
        assert constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        # Should always be true
        assert constraint.partial_func(10, _d([]))
        assert constraint.partial_func(10, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(10, _d(["h", "h"]))
        assert constraint.partial_func(10, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_does_not_occur(self):
        constraint = does_not_occur_constraint(val="h")
        assert constraint(_d([]))
        assert constraint(_d([1, 2, 1, 2]))
        assert not constraint(_d(["h", "h"]))
        assert not constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        assert constraint.partial_func(4, _d([]))
        assert constraint.partial_func(5, _d([1, 2, 1, 2]))
        assert not constraint.partial_func(6, _d(["h", "h"]))
        assert not constraint.partial_func(7, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_fix_final_state(self):
        constraint = fix_final_state_constraint(val="h")
        assert not constraint(_d([]))
        assert not constraint(_d([1, 2, "h", 1, 2]))
        assert constraint(_d(["h", "h"]))
        assert constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        # Always true
        assert constraint.partial_func(2, _d([]))
        assert constraint.partial_func(100, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(3, _d(["h", "h"]))
        assert constraint.partial_func(8, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_occurs_only_in_time_frame(self):
        seq = _d([1, 2, 3, 4, 5, 5, 5])
        assert occurs_only_in_time_frame_constraint(3, lower_t=2, upper_t=4)(seq)
        assert not occurs_only_in_time_frame_constraint(2, lower_t=2, upper_t=4)(seq)
        assert occurs_only_in_time_frame_constraint(6, lower_t=0, upper_t=5)(seq)
        assert occurs_only_in_time_frame_constraint(7)(seq)  # 1 occurs before the range
        assert not occurs_only_in_time_frame_constraint(5, upper_t=6)(
            seq
        )  # 5 is at the end
        assert not occurs_only_in_time_frame_constraint(1, lower_t=1)(
            seq
        )  # 1 is at the start
        assert occurs_only_in_time_frame_constraint(1)(
            _d([])
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
        assert not occurs_only_in_time_frame_constraint(5, upper_t=6).partial_func(
            8, seq
        )  # 5 is at the end
        assert not occurs_only_in_time_frame_constraint(1, lower_t=1).partial_func(
            8, seq
        )  # 1 is at the start
        assert occurs_only_in_time_frame_constraint(1).partial_func(
            2, _d([])
        )  # No occurrences in an empty sequence

    def test_occurs_at_least_once_in_time_frame(self):
        seq = _d([1, 2, 3, 4, 5, 5, 5])
        assert occurs_at_least_once_in_time_frame_constraint(3, lower_t=2, upper_t=4)(
            seq
        )
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
            _d([])
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
        assert not occurs_at_least_once_in_time_frame_constraint(7).partial_func(
            8, seq
        )  # 1 occurs before the range
        assert occurs_at_least_once_in_time_frame_constraint(5, upper_t=6).partial_func(
            8, seq
        )  # 5 is at the end
        assert not occurs_at_least_once_in_time_frame_constraint(
            1, lower_t=1
        ).partial_func(
            8, seq
        )  # 1 is at the start
        assert not occurs_at_least_once_in_time_frame_constraint(1).partial_func(
            2, _d([])
        )  # No occurrences in an empty sequence

    def test_or_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=3)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=1)
        constraint = or_constraints([constraint1, constraint2])
        assert constraint(_d([]))
        assert constraint(_d([1, 2, "h", 1, 2]))
        assert not constraint(_d(["h", "h"]))
        assert constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        assert constraint.partial_func(2, _d([]))
        assert constraint.partial_func(17, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(3, _d(["h", "h"]))
        assert not constraint.partial_func(2, _d(["h", "h"]))
        assert constraint.partial_func(8, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_not_constraint(self):
        constraint = has_exact_number_of_occurences_constraint(val="h", count=2)
        constraint = not_constraint(constraint)
        assert constraint(_d([]))
        assert constraint(_d([1, 2, "h", 1, 2]))
        assert not constraint(_d(["h", "h"]))
        assert constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        # Should all be true
        assert constraint.partial_func(10, _d([]))
        assert constraint.partial_func(10, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(10, _d(["h", "h"]))
        assert constraint.partial_func(10, _d([1, 2, 1, 2, "h", "h", "h"]))

    def test_xor_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=2)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=2)
        constraint = xor_constraints([constraint1, constraint2])
        assert constraint(_d([]))
        assert constraint(_d([1, 2, "h", 1, 2]))
        assert not constraint(_d(["h", "h"]))
        assert constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        assert constraint.partial_func(5, _d([]))
        assert constraint.partial_func(7, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(2, _d(["h", "h"]))
        assert constraint.partial_func(3, _d(["h", "h"]))
        assert constraint.partial_func(17, _d([1, 2, 1, 2, "h", "h", "h"]))

        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=3)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=1)
        constraint = xor_constraints([constraint1, constraint2])
        assert not constraint(_d(["h", "h"]))

        assert constraint.partial_func(3, _d(["h", "h"]))

        constraint1 = has_maximum_number_of_occurences_constraint(val="h", count=2)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=1)
        constraint = xor_constraints([constraint1, constraint2])
        assert not constraint.partial_func(10, _d(["h", "h", "h"]))

    def test_and_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=2)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=2)
        constraint = and_constraints([constraint1, constraint2])
        assert not constraint(_d([]))
        assert not constraint(_d([1, 2, "h", 1, 2]))
        assert constraint(_d(["h", "h"]))
        assert not constraint(_d([1, 2, 1, 2, "h", "h", "h"]))

        assert not constraint.partial_func(1, _d([]))
        assert constraint.partial_func(2, _d([]))
        assert constraint.partial_func(9, _d([1, 2, "h", 1, 2]))
        assert constraint.partial_func(9, _d(["h", "h"]))
        assert not constraint.partial_func(9, _d([1, 2, 1, 2, "h", "h", "h"]))
