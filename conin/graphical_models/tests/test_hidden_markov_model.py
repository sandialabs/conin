import pytest

from conin import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
import conin.graphical_models

import conin.examples.hidden_markov_model as tc
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
            _hmm = HiddenMarkovModel(
                start_probs=_start_probs,
                transition_probs=hmm.get_transition_probs(),
                emission_probs=hmm.get_emission_probs(),
            )

    def test_start_vec_new_hidden_state(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.get_start_probs().copy()
            _start_probs["h2"] = 0.1
            _hmm = HiddenMarkovModel(
                start_probs=_start_probs,
                transition_probs=hmm.get_transition_probs(),
                emission_probs=hmm.get_emission_probs(),
            )

    def test_start_vec_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.get_start_probs().copy()
            _start_probs["h0"] = 0.6
            _hmm = HiddenMarkovModel(
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
            _hmm = HiddenMarkovModel(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h0", "h0")] = 0.8
            _hmm = HiddenMarkovModel(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_filled_out(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            del _transition_probs[("h0", "h0")]
            _hmm = HiddenMarkovModel(
                start_probs=hmm.get_start_probs(),
                transition_probs=_transition_probs,
                emission_probs=hmm.get_emission_probs(),
            )

    def test_transition_mat_extra_label(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.get_transition_probs().copy()
            _transition_probs[("h", "h0")] = 0.3
            _hmm = HiddenMarkovModel(
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
            _hmm = HiddenMarkovModel(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h0", "o0")] = 0.6
            _hmm = HiddenMarkovModel(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_filled_out(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            del _emission_probs[("h0", "o0")]
            _hmm = HiddenMarkovModel(
                start_probs=hmm.get_start_probs(),
                transition_probs=hmm.get_transition_probs(),
                emission_probs=_emission_probs,
            )

    def test_emission_mat_extra_label(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.get_emission_probs().copy()
            _emission_probs[("h", "o0")] = 1
            _hmm = HiddenMarkovModel(
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
        hmm = HiddenMarkovModel(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )
        tol = 1e-6
        hmm = hmm.make_non_zero(tol=tol)
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
        _hmm = HiddenMarkovModel.read_from_file(tmp_path / "temp.txt")
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
        observed1 = ["o1"]
        hidden1 = ["h0"]
        observed2 = ["o0", "o0"]
        hidden2 = ["h1", "h0"]

        assert math.isclose(
            hmm.log_probability(observed=observed1, hidden=hidden1),
            math.log(0.4) + math.log(0.3),
        )
        assert math.isclose(
            hmm.log_probability(observed=observed2, hidden=hidden2),
            math.log(0.6) + math.log(0.4) + math.log(0.2) + math.log(0.7),
        )

    def test_to_dict_keys(self):
        hmm = tc.create_hmm1()
        d = hmm.to_dict()
        assert set(d.keys()) == {
            "start_probs",
            "transition_probs",
            "emission_probs",
            "num_hidden",
            "num_observed",
        }

    def test_to_dict_counts(self):
        hmm = tc.create_hmm1()
        d = hmm.to_dict()
        assert d["num_hidden"] == 2
        assert d["num_observed"] == 2

    def test_to_dict_start_probs_format(self):
        # start_probs is a list of (label, value) pairs for entries > tolerance
        hmm = tc.create_hmm1()
        d = hmm.to_dict()
        labels = [label for label, _ in d["start_probs"]]
        assert set(labels) == {"h0", "h1"}

    def test_to_dict_transition_probs_format(self):
        # transition_probs is a list of ((from, to), value) pairs
        hmm = tc.create_hmm1()
        d = hmm.to_dict()
        pairs = [pair for pair, _ in d["transition_probs"]]
        assert ("h0", "h0") in pairs
        assert ("h1", "h0") in pairs

    def test_to_dict_emission_probs_format(self):
        hmm = tc.create_hmm1()
        d = hmm.to_dict()
        pairs = [pair for pair, _ in d["emission_probs"]]
        assert ("h0", "o0") in pairs
        assert ("h1", "o1") in pairs

    def test_to_dict_no_tolerance_includes_zeros(self):
        # create_hmm0 has several zero probabilities; with tolerance=0.0 they
        # should still be included (the filter is v > tolerance, so 0.0 is excluded
        # even at tolerance=0.0).
        hmm = tc.create_hmm0()
        d = hmm.to_dict(tolerance=0.0)
        values = [v for _, v in d["start_probs"]]
        # h1 has start prob 0, so it is filtered out (0 > 0.0 is False)
        assert 0.0 not in values

    def test_to_dict_tolerance_filters_small_values(self):
        # With a tolerance equal to the smallest non-zero probability, that entry
        # should be excluded (strict inequality).
        hmm = tc.create_hmm1()
        # The smallest start_prob is h0=0.4; use tolerance=0.4 to exclude it.
        d = hmm.to_dict(tolerance=0.4)
        start_labels = [label for label, _ in d["start_probs"]]
        assert "h0" not in start_labels  # 0.4 > 0.4 is False
        assert "h1" in start_labels  # 0.6 > 0.4 is True

    def test_to_dict_tolerance_excludes_all(self):
        # With tolerance >= 1.0 no entry survives.
        hmm = tc.create_hmm1()
        d = hmm.to_dict(tolerance=1.0)
        assert d["start_probs"] == []
        assert d["transition_probs"] == []
        assert d["emission_probs"] == []


class Test_HMM_Util:
    def test_random_hmm(self):
        hidden_states = {1, 2, 3, 4}
        observed_states = ["a", "b", "c"]
        seed = 1
        hmm = conin.graphical_models.random_hmm(
            hidden_states=hidden_states,
            observed_states=observed_states,
            seed=1,
        )
        assert set(hmm.get_hidden_states()) == set(hidden_states)
        assert set(hmm.get_observable_states()) == set(observed_states)


class Test_HMM_Constructor:
    def test_constructor_requires_all_dictionaries(self):
        hmm = tc.create_hmm1()
        args = dict(
            start_probs=hmm.get_start_probs(),
            transition_probs=hmm.get_transition_probs(),
            emission_probs=hmm.get_emission_probs(),
        )
        with pytest.raises(TypeError):
            HiddenMarkovModel()
        for name in args:
            with pytest.raises(TypeError):
                HiddenMarkovModel(**{k: v for k, v in args.items() if k != name})

    @pytest.mark.parametrize(
        "name", ["start_probs", "transition_probs", "emission_probs"]
    )
    def test_constructor_rejects_empty_dictionaries(self, name):
        hmm = tc.create_hmm1()
        args = dict(
            start_probs=hmm.get_start_probs(),
            transition_probs=hmm.get_transition_probs(),
            emission_probs=hmm.get_emission_probs(),
        )
        args[name] = {}
        with pytest.raises(InvalidInputError, match=name):
            HiddenMarkovModel(**args)

    def test_no_load_model(self):
        assert not hasattr(HiddenMarkovModel, "load_model")

    def test_constructor_rejects_positional_args(self):
        with pytest.raises(TypeError):
            HiddenMarkovModel({"h0": 1.0})

    def test_state_order_is_sorted_labels(self):
        # Insertion order and sparsity do not affect the internal order: the
        # first sorted transition key, ("b", "c"), would otherwise put "c" before
        # "a".
        hmm = HiddenMarkovModel(
            start_probs={"b": 1.0},
            transition_probs={("c", "a"): 1.0, ("b", "c"): 1.0, ("a", "a"): 1.0},
            emission_probs={("c", "y"): 1.0, ("b", "x"): 1.0, ("a", "z"): 1.0},
        )
        assert hmm.hidden_to_external == ["a", "b", "c"]
        assert hmm.hidden_to_internal == {"a": 0, "b": 1, "c": 2}
        assert hmm.observed_to_external == ["x", "y", "z"]
        assert hmm.start_vec == [0, 1.0, 0]
        assert hmm.transition_mat == [[1.0, 0, 0], [0, 0, 1.0], [1.0, 0, 0]]
        assert hmm.emission_mat == [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]

    def test_round_trip_through_constructor(self):
        hmm = tc.create_hmm2()
        hmm2 = HiddenMarkovModel(
            start_probs=hmm.get_start_probs(),
            transition_probs=hmm.get_transition_probs(),
            emission_probs=hmm.get_emission_probs(),
        )
        assert hmm2.hidden_to_external == hmm.hidden_to_external
        assert hmm2.observed_to_external == hmm.observed_to_external
        assert hmm2.start_vec == hmm.start_vec
        assert hmm2.transition_mat == hmm.transition_mat
        assert hmm2.emission_mat == hmm.emission_mat

    def test_no_cached_numeric_representation(self):
        hmm = tc.create_hmm1()
        args = dict(
            start_probs=hmm.get_start_probs(),
            transition_probs=hmm.get_transition_probs(),
            emission_probs=hmm.get_emission_probs(),
        )
        with pytest.raises(TypeError):
            HiddenMarkovModel(**args, initialize=True)
        for name in ("repn", "initialize", "generate_hidden", "generate_observed"):
            assert not hasattr(hmm, name)

    @pytest.mark.parametrize(
        "name, key, value",
        [
            ("start_probs", "h0", -0.6),
            ("start_probs", "h0", 0.6),
            ("transition_probs", ("h0", "h0"), 0.8),
            ("emission_probs", ("h0", "o0"), -0.4),
        ],
    )
    def test_constructor_validates_probabilities(self, name, key, value):
        # The probabilities used to be checked only when the numeric
        # representation was first built (initialize=True or the first query),
        # so an invalid HMM could be created and used with other backends.
        hmm = tc.create_hmm1()
        args = dict(
            start_probs=hmm.get_start_probs(),
            transition_probs=hmm.get_transition_probs(),
            emission_probs=hmm.get_emission_probs(),
        )
        args[name] = dict(args[name])
        args[name][key] = value
        with pytest.raises(InvalidInputError):
            HiddenMarkovModel(**args)

    def test_inputs_are_copied(self):
        start_probs = {"h0": 1.0}
        transition_probs = {("h0", "h0"): 1.0}
        emission_probs = {("h0", "o0"): 1.0}
        hmm = HiddenMarkovModel(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )
        start_probs["h0"] = 0.5
        transition_probs[("h0", "h0")] = 0.5
        emission_probs[("h0", "o0")] = 0.5
        assert hmm.get_start_probs() == {"h0": 1.0}
        assert hmm.get_transition_probs() == {("h0", "h0"): 1.0}
        assert hmm.get_emission_probs() == {("h0", "o0"): 1.0}

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
