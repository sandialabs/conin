import pytest

from conin import *
from conin.hmm import *
from conin.hmm.oracle_chmm import Oracle_CHMM

import conin.hmm.tests.test_cases as tc
import conin.common_constraints as cc


class Test_Oracle_CHMM:
    T = 25

    def test_load_model(self):
        chmm = tc.create_chmm1()

        assert chmm.get_start_probs() == chmm.hmm.get_start_probs()
        assert chmm.get_emission_probs() == chmm.hmm.get_emission_probs()
        assert chmm.get_transition_probs() == chmm.hmm.get_transition_probs()


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

    def test_load_model3(self):
        hmm = tc.create_hmm0()
        constraints = [cc.all_diff_constraint]
        chmm = Oracle_CHMM(hmm=hmm, constraints=constraints)
        assert chmm.hmm == hmm
        assert len(chmm.constraints) == 1

    def test_load_model_empty(self):
        chmm = Oracle_CHMM()
        assert chmm.constraints == []

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

    def test_is_valid_hidden_state(self):
        chmm = tc.create_chmm1()
        assert chmm.is_valid_hidden_state("h0")
        assert not chmm.is_valid_hidden_state("invalid")

    def test_set_seed(self):
        chmm = tc.create_chmm1()
        chmm.set_seed(1)
        assert chmm._seed == 1

    def test_generate(self):
        chmm = tc.create_chmm1()
        observed = chmm.generate_observed_from_hidden(
            [
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
        )
        assert len(observed) == 11
        T = 25
        observed2 = chmm.generate_observed(T)
        assert len(observed2) == T

        with pytest.raises(InvalidInputError):
            chmm.generate_observed(-1)

        with pytest.raises(InvalidInputError):
            chmm.generate_observed_from_hidden(["h0"])

    def test_is_feasible(self):
        chmm = tc.create_chmm1()
        fail_seq1 = ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"]
        pass_seq = [
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

    def test_partial_is_feasible(self):
        chmm = tc.create_chmm1()
        fail_seq1 = ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"]
        pass_seq = [
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

        assert not chmm.partial_is_feasible(T=len(fail_seq1), seq=fail_seq1)
        assert chmm.partial_is_feasible(T=100, seq=fail_seq1)
        assert chmm.partial_is_feasible(T=len(pass_seq), seq=pass_seq)
        assert chmm.partial_is_feasible(T=100, seq=pass_seq)
        assert not chmm.partial_is_feasible(T=len(fail_seq2), seq=fail_seq2)
        assert not chmm.partial_is_feasible(T=100, seq=fail_seq2)

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
