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
            _constraint(self.true_seq)

    def test_name(self, constraint):
        assert constraint.name == "Test"

    def test_call(self, constraint):
        assert constraint(self.true_seq)
        assert not constraint(self.false_seq1)
        assert not constraint(self.false_seq2)
        assert not constraint(self.false_seq3)

    def test_name_automate(self):
        def num_zeros_eq_five(seq):
            return seq.count(0) == 5

        _constraint = Constraint(func=num_zeros_eq_five)
        assert _constraint.name == "num_zeros_eq_five"


class Test_Oracle_CHMM:
    T = 25

    def test_load_model(self):
        chmm = tc.create_chmm1()
        assert chmm.hmm.get_start_probs() == chmm.hmm.get_start_probs()
        assert chmm.hmm.get_emission_probs() == chmm.hmm.get_emission_probs()
        assert chmm.hmm.get_transition_probs() == chmm.hmm.get_transition_probs()

    def test_load_model2(self):
        chmm = tc.create_chmm1()
        _chmm = Oracle_CHMM()
        _hmm = HMM()
        _hmm.load_model(
            start_probs=chmm.hmm.get_start_probs(),
            emission_probs=chmm.hmm.get_emission_probs(),
            transition_probs=chmm.hmm.get_transition_probs(),
        )
        _chmm.load_model(hmm=_hmm)
        assert chmm.hmm.get_start_probs() == chmm.hmm.get_start_probs()
        assert chmm.hmm.get_emission_probs() == chmm.hmm.get_emission_probs()
        assert chmm.hmm.get_transition_probs() == chmm.hmm.get_transition_probs()

    def test_load_model_failure(self):
        with pytest.raises(InvalidInputError):
            _chmm = Oracle_CHMM()
            _chmm.load_model(start_probs={"h0": 0.4, "h1": 0.6})

    def test_load_model_failure2(self):
        _chmm = Oracle_CHMM()
        _hmm = HMM()
        with pytest.raises(InvalidInputError):
            _hmm.load_model(
                start_probs={"h0": 0.4, "h1": 0.6},
                emission_probs={"h0": 0.7, "h1": 0.3},
                transition_probs={("h0", "h0"): 0.9, ("h0", "h1"): 0.1},
            )
        with pytest.raises(InvalidInputError):
            _chmm.load_model(start_probs={"h0": 0.4, "h1": 0.6}, hmm=_hmm)

    def test_internal_is_feasible(self):
        chmm = tc.create_chmm1()
        fail_seq1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        pass_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fail_seq2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert not chmm.internal_constrained_hmm.is_feasible(fail_seq1)
        assert chmm.internal_constrained_hmm.is_feasible(pass_seq)
        assert not chmm.internal_constrained_hmm.is_feasible(fail_seq2)

    def test_is_feasible(self):
        chmm = tc.create_chmm1()
        fail_seq1 = ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"]
        pass_seq = ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"]
        fail_seq2 = [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

        assert not chmm.is_feasible(fail_seq1)
        assert chmm.is_feasible(pass_seq)
        assert not chmm.is_feasible(fail_seq2)

    def test_generate_hidden_length(self):
        chmm = tc.create_chmm1()
        assert len(chmm.generate_hidden(self.T)) == self.T
        with pytest.raises(InvalidInputError):
            chmm.generate_hidden(-1)

    def test_generate_hidden_output(self):
        chmm = tc.create_chmm1()
        for h in chmm.generate_hidden(self.T):
            assert h in {"h0", "h1"}

    def test_generate_hidden_constraint(self):
        chmm = tc.create_chmm1()
        assert chmm.is_feasible(chmm.generate_hidden(self.T))

    def test_generate_hidden_negative_time_steps(self):
        chmm = tc.create_chmm1()
        with pytest.raises(InvalidInputError):
            chmm.generate_hidden(-1)

    def test_generate_observed_from_hidden_length(self):
        chmm = tc.create_chmm1()
        hidden = chmm.generate_hidden(self.T)
        assert len(chmm.generate_observed_from_hidden(hidden)) == self.T

    def test_generate_observed_from_hidden_failure(self):
        chmm = tc.create_chmm1()
        hidden = ["h0"]
        with pytest.raises(InvalidInputError):
            chmm.generate_observed_from_hidden(hidden)

    def test_generate_observed_from_hidden_output(self):
        chmm = tc.create_chmm1()
        hidden = chmm.generate_hidden(self.T)
        for o in chmm.generate_observed_from_hidden(hidden):
            assert o in {"o0", "o1"}

    def test_generate_observed_length(self):
        chmm = tc.create_chmm1()
        assert len(chmm.generate_observed(self.T)) == self.T
        with pytest.raises(InvalidInputError):
            chmm.generate_observed(-1)

    def test_generate_observed_output(self):
        chmm = tc.create_chmm1()
        for h in chmm.generate_observed(self.T):
            assert h in {"o0", "o1"}

    def test_generate_observed_negative_time_steps(self):
        chmm = tc.create_chmm1()
        with pytest.raises(InvalidInputError):
            chmm.generate_observed(-1)


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
        assert constraint([0, 1])
        assert not constraint([2, 2])
        assert not constraint([2, 1])
        assert constraint([1, 0, 2, 1])

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
        constraint = has_minimum_number_of_occurences_constraint(val="h", count=2)
        assert not constraint([])
        assert not constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert constraint([1, 2, 1, 2, "h", "h", "h"])

        assert not constraint.partial_func(1, [])
        assert constraint.partial_func(10, [1, 2, "h", 1, 2])
        assert constraint.partial_func(11, ["h", "h"])
        assert constraint.partial_func(12, [1, 2, 1, 2, "h", "h", "h"])

    def test_has_maximum_number_of_occurences(self):
        constraint = has_maximum_number_of_occurences_constraint(val="h", count=2)
        assert constraint([])
        assert constraint([1, 2, "h", 1, 2])
        assert constraint(["h", "h"])
        assert not constraint([1, 2, 1, 2, "h", "h", "h"])

        assert constraint.partial_func(3, [])
        assert constraint.partial_func(6, [1, 2, "h", 1, 2])
        assert constraint.partial_func(3, ["h", "h"])
        assert not constraint.partial_func(9, [1, 2, 1, 2, "h", "h", "h"])

    def test_has_exact_number_of_occurences(self):
        constraint = has_exact_number_of_occurences_constraint(val="h", count=2)
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
        assert not occurs_only_in_time_frame_constraint(5, upper_t=6).partial_func(
            8, seq
        )  # 5 is at the end
        assert not occurs_only_in_time_frame_constraint(1, lower_t=1).partial_func(
            8, seq
        )  # 1 is at the start
        assert occurs_only_in_time_frame_constraint(1).partial_func(
            2, []
        )  # No occurrences in an empty sequence

    def test_occurs_at_least_once_in_time_frame(self):
        seq = [1, 2, 3, 4, 5, 5, 5]
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
            2, []
        )  # No occurrences in an empty sequence

    def test_or_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=3)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=1)
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
        constraint = has_exact_number_of_occurences_constraint(val="h", count=2)
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
        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=2)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=2)
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

        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=3)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=1)
        constraint = xor_constraints([constraint1, constraint2])
        assert not constraint(["h", "h"])

        assert constraint.partial_func(3, ["h", "h"])

        constraint1 = has_maximum_number_of_occurences_constraint(val="h", count=2)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=1)
        constraint = xor_constraints([constraint1, constraint2])
        assert not constraint.partial_func(10, ["h", "h", "h"])

    def test_and_constraints(self):
        constraint1 = has_minimum_number_of_occurences_constraint(val="h", count=2)
        constraint2 = has_maximum_number_of_occurences_constraint(val="h", count=2)
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
