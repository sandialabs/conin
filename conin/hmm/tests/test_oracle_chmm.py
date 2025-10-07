import pytest

from conin import *
from conin.hmm import *
from conin.hmm.chmm_oracle import Oracle_CHMM

import conin.hmm.tests.test_cases as tc
import conin.common_constraints as cc


class Test_Oracle_CHMM:
    T = 25

    def test_load_model(self):
        cpgm = tc.create_chmm1()

        assert cpgm.chmm.hmm.start_vec == [
            cpgm.hidden_markov_model.get_start_probs()[h]
            for h in cpgm.hidden_markov_model.hidden_states
        ]

    def test_load_model2(self):
        cpgm = tc.create_chmm1()

        _hmm = HiddenMarkovModel()
        _hmm.load_model(
            start_probs=cpgm.hidden_markov_model.get_start_probs(),
            emission_probs=cpgm.hidden_markov_model.get_emission_probs(),
            transition_probs=cpgm.hidden_markov_model.get_transition_probs(),
        )
        _chmm = Oracle_CHMM(hmm=_hmm.hmm)

        assert _chmm.hmm.start_vec == [
            cpgm.hidden_markov_model.get_start_probs()[h]
            for h in cpgm.hidden_markov_model.hidden_states
        ]

    def test_load_model3(self):
        pgm = tc.create_hmm0()
        constraints = [cc.all_diff_constraint]
        chmm = Oracle_CHMM(hmm=pgm.hmm, constraints=constraints)
        assert chmm.hmm == pgm.hmm
        assert len(chmm.constraints) == 1

    def test_internal_is_feasible(self):
        cpgm = tc.create_chmm1()
        assert len(cpgm.constraints) == 2

        fail_seq1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        pass_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fail_seq2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert not cpgm.chmm.is_feasible(fail_seq1)
        assert cpgm.chmm.is_feasible(pass_seq)
        assert not cpgm.chmm.is_feasible(fail_seq2)

    def Xtest_is_valid_hidden_state(self):
        cpgm = tc.create_chmm1()
        assert cpgm.hidden_markov_model.is_valid_hidden_state("h0")
        assert not cpgm.hidden_markov_model.is_valid_hidden_state("invalid")

    def Xtest_set_seed(self):
        chmm = tc.create_chmm1()
        chmm.set_seed(1)
        assert chmm._seed == 1

    def test_generate(self):
        cpgm = tc.create_chmm1()
        observed = cpgm.generate_observed_from_hidden(
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
        observed2 = cpgm.generate_observed(T)
        assert len(observed2) == T

        with pytest.raises(InvalidInputError):
            cpgm.generate_observed(-1)

        with pytest.raises(InvalidInputError):
            cpgm.generate_observed_from_hidden(["h0"])

    def test_is_feasible(self):
        cpgm = tc.create_chmm1()
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

        assert not cpgm.is_feasible(fail_seq1)
        assert cpgm.is_feasible(pass_seq)
        assert not cpgm.is_feasible(fail_seq2)

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
