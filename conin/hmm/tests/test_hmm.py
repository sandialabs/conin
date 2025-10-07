import pytest

from conin import InvalidInputError
from conin.hmm import HiddenMarkovModel
import conin.hmm.hmm_util

import conin.hmm.tests.examples as tc
import math


class Test_HMM1:
    T = 25

    def test_HMM(self):
        hmm = tc.create_hmm1()
        assert hmm is not None

    def test_HMM_print(self):
        hmm = tc.create_hmm1()
        print(hmm)

    def test_hidden_map(self):
        hmm = tc.create_hmm1()
        assert hmm.hidden_to_internal == {"h0": 0, "h1": 1}

    def test_hidden_map_inv(self):
        hmm = tc.create_hmm1()
        assert hmm.hidden_to_external == ["h0", "h1"]

    def test_num_hidden_states(self):
        hmm = tc.create_hmm1()
        assert hmm.num_hidden_states == 2

    def test_observed_map(self):
        hmm = tc.create_hmm1()
        assert hmm.observed_to_internal == {"o0": 0, "o1": 1}

    def test_observed_map_inv(self):
        hmm = tc.create_hmm1()
        assert hmm.observed_to_external == ["o0", "o1"]

    def test_num_observed_states(self):
        hmm = tc.create_hmm1()
        assert hmm.num_observed_states == 2

    def test_start_vec(self):
        hmm = tc.create_hmm1()
        assert hmm.start_vec == [0.4, 0.6]

    def test_start_vec_negative(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.get_start_probs().copy()
            _start_probs["h0"] = -0.6
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=_start_probs,
                transition_probs=hmm.get_transition_probs(),
                emission_probs=hmm.get_emission_probs(),
            )

    def test_start_vec_new_hidden_state(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.get_start_probs().copy()
            _start_probs["h2"] = 0.1
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=_start_probs,
                transition_probs=hmm.get_transition_probs(),
                emission_probs=hmm.get_emission_probs(),
            )

    def test_start_vec_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.get_start_probs().copy()
            _start_probs["h0"] = 0.6
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=_start_probs,
                transition_probs=hmm.get_transition_probs(),
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_matrix(self):
        hmm = tc.create_hmm1()
        assert hmm.transition_mat == [[0.9, 0.1], [0.2, 0.8]]

    def test_transition_mat_negative(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h0", "h0")] = -0.1
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h0", "h0")] = 0.8
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_filled_out(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            del _transition_probs[("h0", "h0")]
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_extra_label(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h", "h0")] = 0.3
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_emission_matrix(self):
        hmm = tc.create_hmm1()
        assert hmm.emission_mat == [[0.7, 0.3], [0.4, 0.6]]

    def test_emission_mat_negative(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h0", "h0")] = -0.1
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h0", "o0")] = 0.6
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_filled_out(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            del _emission_probs[("h0", "o0")]
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_extra_label(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h", "o0")] = 1
            _hmm = HiddenMarkovModel()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_get_hidden_states(self):
        hmm = tc.create_hmm1()
        assert hmm.get_hidden_states() == ["h0", "h1"]

    def test_get_observable_states(self):
        hmm = tc.create_hmm1()
        assert hmm.get_observable_states() == ["o0", "o1"]

    def test_get_start_probs(self):
        hmm = tc.create_hmm1()
        assert hmm.get_start_probs() == {"h0": 0.4, "h1": 0.6}

    def test_get_transition_probs(self):
        hmm = tc.create_hmm1()
        assert hmm.get_transition_probs() == {
            ("h0", "h0"): 0.9,
            ("h0", "h1"): 0.1,
            ("h1", "h0"): 0.2,
            ("h1", "h1"): 0.8,
        }

    def test_get_emission_probs(self):
        hmm = tc.create_hmm1()
        assert hmm.get_emission_probs() == {
            ("h0", "o0"): 0.7,
            ("h0", "o1"): 0.3,
            ("h1", "o0"): 0.4,
            ("h1", "o1"): 0.6,
        }

    def test_generate_hidden_length(self):
        hmm = tc.create_hmm1()
        assert len(hmm.generate_hidden(self.T)) == self.T
        with pytest.raises(InvalidInputError):
            hmm.generate_hidden(-1)

    def test_generate_hidden_output(self):
        hmm = tc.create_hmm1()
        for h in hmm.generate_hidden(self.T):
            assert h in {"h0", "h1"}

    def test_generate_hidden_from_state(self):
        hmm = tc.create_hmm1()
        # Make sure that we start in state h0 so we test sequences of length at
        # least 2
        start_probs = hmm.get_start_probs()
        tranisition_probs = hmm.get_transition_probs()
        emission_probs = hmm.get_emission_probs()
        start_probs = {"h0": 1, "h1": 0}
        hmm = HiddenMarkovModel()
        hmm.load_model(
            start_probs=start_probs,
            emission_probs=emission_probs,
            transition_probs=tranisition_probs,
        )
        # This should be all h0's followed by h1
        vec = hmm.generate_hidden_until_state("h1")
        assert all(val == "h0" for val in vec[:-1])

    def test_generate_observed_from_hidden_length(self):
        hmm = tc.create_hmm1()
        hidden = hmm.generate_hidden(self.T)
        assert len(hmm.generate_observed_from_hidden(hidden)) == self.T

    def test_generate_observed_from_hidden_output(self):
        hmm = tc.create_hmm1()
        hidden = hmm.generate_hidden(self.T)
        for o in hmm.generate_observed_from_hidden(hidden):
            assert o in {"o0", "o1"}

    def test_generate_observed_length(self):
        hmm = tc.create_hmm1()
        assert len(hmm.generate_observed(self.T)) == self.T
        with pytest.raises(InvalidInputError):
            hmm.generate_observed(-1)

    def test_generate_observed_output(self):
        hmm = tc.create_hmm1()
        for h in hmm.generate_observed(self.T):
            assert h in {"o0", "o1"}

    def test_hmm_is_valid_observed_state(self):
        hmm = tc.create_hmm1()
        assert hmm.is_valid_observed_state("o0")
        assert not hmm.is_valid_observed_state("h0")

    def test_hmm_is_valid_hidden_state(self):
        hmm = tc.create_hmm1()
        assert hmm.is_valid_hidden_state("h1")
        assert not hmm.is_valid_hidden_state("o1")

    def test_hmm_make_non_zero(self):
        hmm = tc.create_hmm1()
        start_probs = {"h0": 1, "h1": 0}
        transition_probs = {
            ("h0", "h0"): 1,
            ("h0", "h1"): 0,
            ("h1", "h0"): 0.6,
            ("h1", "h1"): 0.4,
        }
        emission_probs = {
            ("h0", "o0"): 1,
            ("h0", "o1"): 0,
            ("h1", "o0"): 0.4,
            ("h1", "o1"): 0.6,
        }
        hmm = HiddenMarkovModel()
        hmm.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )
        tol = 1e-6
        hmm.make_non_zero(tol=tol)
        low_val = 1e-6 / (1 + tol)
        high_val = 1 / (1 + tol)

        assert hmm.get_start_probs() == {"h0": high_val, "h1": low_val}
        assert hmm.get_transition_probs() == {
            ("h0", "h0"): high_val,
            ("h0", "h1"): low_val,
            ("h1", "h0"): 0.6,
            ("h1", "h1"): 0.4,
        }
        assert hmm.get_emission_probs() == {
            ("h0", "o0"): high_val,
            ("h0", "o1"): low_val,
            ("h1", "o0"): 0.4,
            ("h1", "o1"): 0.6,
        }

    def test_read_and_write(self, tmp_path):
        hmm = tc.create_hmm1()
        hmm.write_to_file(tmp_path / "temp.txt")
        _hmm = HiddenMarkovModel()
        _hmm.read_from_file(tmp_path / "temp.txt")
        assert _hmm.get_start_probs() == {"h0": 0.4, "h1": 0.6}
        assert _hmm.get_transition_probs() == {
            ("h0", "h0"): 0.9,
            ("h0", "h1"): 0.1,
            ("h1", "h0"): 0.2,
            ("h1", "h1"): 0.8,
        }
        assert _hmm.get_emission_probs() == {
            ("h0", "o0"): 0.7,
            ("h0", "o1"): 0.3,
            ("h1", "o0"): 0.4,
            ("h1", "o1"): 0.6,
        }

    def test_log_probability(self):
        hmm = tc.create_hmm1()
        observation1 = ["o1"]
        hidden1 = ["h0"]
        observation2 = ["o0", "o0"]
        hidden2 = ["h1", "h0"]

        assert math.isclose(
            hmm.log_probability(observations=observation1, hidden=hidden1),
            math.log(0.4) + math.log(0.3),
        )
        assert math.isclose(
            hmm.log_probability(observations=observation2, hidden=hidden2),
            math.log(0.6) + math.log(0.4) + math.log(0.2) + math.log(0.7),
        )


class Test_HMM_Util:
    def test_random_hmm(self):
        hidden_states = {1, 2, 3, 4}
        observed_states = ["a", "b", "c"]
        seed = 1
        hmm = conin.hmm.hmm_util.random_hmm(
            hidden_states=hidden_states,
            observed_states=observed_states,
            seed=1,
        )
        assert set(hmm.get_hidden_states()) == set(hidden_states)
        assert set(hmm.get_observable_states()) == set(observed_states)
