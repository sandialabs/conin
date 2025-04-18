import pytest

from conin import InvalidInputError
from conin.hmm import HMM, Inference


@pytest.fixture
def hmm():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.9,
        ("h0", "h1"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.8,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
    }
    hmm = HMM()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


class Test_HMM:
    T = 25

    def test_HMM(self, hmm):
        assert hmm is not None

    def test_hidden_map(self, hmm):
        assert hmm.hidden_to_internal == {"h0": 0, "h1": 1}

    def test_hidden_map_inv(self, hmm):
        assert hmm.hidden_to_external == ["h0", "h1"]

    def test_num_hidden_states(self, hmm):
        assert hmm.num_hidden_states == 2

    def test_observed_map(self, hmm):
        assert hmm.observed_to_internal == {"o0": 0, "o1": 1}

    def test_observed_map_inv(self, hmm):
        assert hmm.observed_to_external == ["o0", "o1"]

    def test_num_observed_states(self, hmm):
        assert hmm.num_observed_states == 2

    def test_start_vec(self, hmm):
        assert hmm.start_vec == [0.4, 0.6]

    def test_start_vec_negative(self, hmm):
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.get_start_probs().copy()
            _start_probs["h0"] = -0.6
            _hmm = HMM()
            _hmm.load_model(
                start_probs=_start_probs,
                transition_probs=hmm.get_transition_probs(),
                emission_probs=hmm.get_emission_probs(),
            )

    def test_start_vec_sum_to_one(self, hmm):
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.get_start_probs().copy()
            _start_probs["h0"] = 0.6
            _hmm = HMM()
            _hmm.load_model(
                start_probs=_start_probs,
                transition_probs=hmm.get_transition_probs(),
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_matrix(self, hmm):
        assert hmm.transition_mat == [[0.9, 0.1], [0.2, 0.8]]

    def test_transition_mat_negative(self, hmm):
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h0", "h0")] = -0.1
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_sum_to_one(self, hmm):
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h0", "h0")] = 0.8
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_filled_out(self, hmm):
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            del _transition_probs[("h0", "h0")]
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_extra_label(self, hmm):
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h", "h0")] = 0.3
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_emission_matrix(self, hmm):
        assert hmm.emission_mat == [[0.7, 0.3], [0.4, 0.6]]

    def test_emission_mat_negative(self, hmm):
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h0", "h0")] = -0.1
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_sum_to_one(self, hmm):
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h0", "o0")] = 0.6
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_filled_out(self, hmm):
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            del _emission_probs[("h0", "o0")]
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_extra_label(self, hmm):
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h", "o0")] = 1
            _hmm = HMM()
            _hmm.load_model(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_get_hidden_states(self, hmm):
        assert hmm.get_hidden_states() == {"h0", "h1"}

    def test_get_observable_states(self, hmm):
        assert hmm.get_observable_states() == {"o0", "o1"}

    def test_get_start_probs(self, hmm):
        assert hmm.get_start_probs() == {"h0": 0.4, "h1": 0.6}

    def test_get_transition_probs(self, hmm):
        assert hmm.get_transition_probs() == {
            ("h0", "h0"): 0.9,
            ("h0", "h1"): 0.1,
            ("h1", "h0"): 0.2,
            ("h1", "h1"): 0.8,
        }

    def test_get_emission_probs(self, hmm):
        assert hmm.get_emission_probs() == {
            ("h0", "o0"): 0.7,
            ("h0", "o1"): 0.3,
            ("h1", "o0"): 0.4,
            ("h1", "o1"): 0.6,
        }

    def test_generate_hidden_length(self, hmm):
        assert len(hmm.generate_hidden(self.T)) == self.T

    def test_generate_hidden_output(self, hmm):
        for h in hmm.generate_hidden(self.T):
            assert h in {"h0", "h1"}

    def test_generate_hidden_from_state(self, hmm):
        # This should be all h0's followed by h1
        vec = hmm.generate_hidden_until_state("h1")
        assert vec[-1] == "h1"
        vec.pop()
        for val in vec:
            assert val == "h0"

    def test_generate_observed_from_hidden_length(self, hmm):
        hidden = hmm.generate_hidden(self.T)
        assert len(hmm.generate_observed_from_hidden(hidden)) == self.T

    def test_generate_observed_from_hidden_output(self, hmm):
        hidden = hmm.generate_hidden(self.T)
        for o in hmm.generate_observed_from_hidden(hidden):
            assert o in {"o0", "o1"}

    def test_generate_observed_length(self, hmm):
        assert len(hmm.generate_observed(self.T)) == self.T

    def test_generate_observed_output(self, hmm):
        for h in hmm.generate_observed(self.T):
            assert h in {"o0", "o1"}

    def test_hmm_is_valid_observed_state(self, hmm):
        assert hmm.is_valid_observed_state("o0")
        assert not hmm.is_valid_observed_state("h0")

    def test_hmm_is_valid_hidden_state(self, hmm):
        assert hmm.is_valid_hidden_state("h1")
        assert not hmm.is_valid_hidden_state("o1")

    def test_hmm_make_non_zero(self):
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
        hmm = HMM()
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

    def test_read_and_write(self, hmm):
        hmm.write_to_file("temp.txt")
        _hmm = HMM()
        _hmm.read_from_file("temp.txt")
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


class Test_Inference:
    def test_inference_default_values(self, hmm):
        inference = Inference(statistical_model=hmm)
        assert inference.num_solutions == 1
        assert inference.oracle_based is True

    def test_inference_with_positive_num_solutions(self, hmm):
        inference = Inference(
            statistical_model=hmm, num_solutions=5, oracle_based=False
        )
        assert inference.num_solutions == 5
        assert inference.oracle_based is False

    def test_inference_with_zero_num_solutions(self, hmm):
        with pytest.raises(InvalidInputError):
            Inference(statistical_model=hmm, num_solutions=0)

    def test_inference_with_negative_num_solutions(self, hmm):
        with pytest.raises(InvalidInputError):
            Inference(statistical_model=hmm, num_solutions=-1)

    def test_viterbi_1(self, hmm):
        inference = Inference(statistical_model=hmm)
        observed = ["o0", "o0", "o1", "o0", "o0"]
        assert inference(observed).solutions[0].hidden == ["h0", "h0", "h0", "h0", "h0"]

    def test_viterbi_2(self, hmm):
        inference = Inference(statistical_model=hmm)
        observed = ["o0", "o1", "o1", "o1", "o1"]
        assert inference(observed).solutions[0].hidden == ["h1", "h1", "h1", "h1", "h1"]

    def test_inference_invalid_observation(self, hmm):
        with pytest.raises(InvalidInputError):
            inference = Inference(statistical_model=hmm)
            observed = ["o2"]
            inference(observed)

    def test_a_star_equals_viterbi(self, hmm):
        inference = Inference(statistical_model=hmm)
        observed = hmm.generate_observed(25)
        assert (
            inference._viterbi(observed).solutions[0].hidden
            == inference._a_star(observed).solutions[0].hidden
        )
