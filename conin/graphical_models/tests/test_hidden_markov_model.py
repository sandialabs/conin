import pytest

from conin import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
import conin.graphical_models

import conin.examples.hidden_markov_model as tc


class Test_HMM1:
    T = 25

    def test_hidden_map(self):
        hmm = tc.create_hmm1()
        assert hmm.representation("matvec").hidden_to_internal == {"h0": 0, "h1": 1}

    def test_hidden_map_inv(self):
        hmm = tc.create_hmm1()
        assert hmm.hidden_states == ["h0", "h1"]

    def test_num_hidden_states(self):
        hmm = tc.create_hmm1()
        assert hmm.num_hidden_states == 2

    def test_observed_map(self):
        hmm = tc.create_hmm1()
        assert hmm.representation("matvec").observable_to_internal == {"o0": 0, "o1": 1}

    def test_observed_map_inv(self):
        hmm = tc.create_hmm1()
        assert hmm.observable_states == ["o0", "o1"]

    def test_num_observed_states(self):
        hmm = tc.create_hmm1()
        assert hmm.num_observable_states == 2

    def test_start_vec(self):
        hmm = tc.create_hmm1()
        assert hmm.representation("matvec").start_vec.tolist() == [0.4, 0.6]

    def test_start_vec_negative(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.start_probs.copy()
            _start_probs["h0"] = -0.6
            HiddenMarkovModel(
                start_probs=_start_probs,
                transition_probs=hmm.transition_probs,
                emission_probs=hmm.emission_probs,
            )

    def test_start_vec_new_hidden_state(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.start_probs.copy()
            _start_probs["h2"] = 0.1
            HiddenMarkovModel(
                start_probs=_start_probs,
                transition_probs=hmm.transition_probs,
                emission_probs=hmm.emission_probs,
            )

    def test_start_vec_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _start_probs = hmm.start_probs.copy()
            _start_probs["h0"] = 0.6
            HiddenMarkovModel(
                start_probs=_start_probs,
                transition_probs=hmm.transition_probs,
                emission_probs=hmm.emission_probs,
            )

    def test_transition_matrix(self):
        hmm = tc.create_hmm1()
        assert hmm.representation("matvec").transition_mat.tolist() == [
            [0.9, 0.1],
            [0.2, 0.8],
        ]

    def test_transition_mat_negative(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.transition_probs.copy()
            _transition_probs[("h0", "h0")] = -0.1
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=_transition_probs,
                emission_probs=hmm.emission_probs,
            )

    def test_transition_mat_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.transition_probs.copy()
            _transition_probs[("h0", "h0")] = 0.8
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=_transition_probs,
                emission_probs=hmm.emission_probs,
            )

    def test_transition_mat_filled_out(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.transition_probs.copy()
            del _transition_probs[("h0", "h0")]
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=_transition_probs,
                emission_probs=hmm.emission_probs,
            )

    def test_transition_mat_extra_label(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _transition_probs = hmm.transition_probs.copy()
            _transition_probs[("h", "h0")] = 0.3
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=_transition_probs,
                emission_probs=hmm.emission_probs,
            )

    def test_emission_matrix(self):
        hmm = tc.create_hmm1()
        assert hmm.representation("matvec").emission_mat.tolist() == [
            [0.7, 0.3],
            [0.4, 0.6],
        ]

    def test_emission_mat_negative(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.emission_probs.copy()
            _emission_probs[("h0", "h0")] = -0.1
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=hmm.transition_probs,
                emission_probs=_emission_probs,
            )

    def test_emission_mat_sum_to_one(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.emission_probs.copy()
            _emission_probs[("h0", "o0")] = 0.6
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=hmm.transition_probs,
                emission_probs=_emission_probs,
            )

    def test_emission_mat_filled_out(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.emission_probs.copy()
            del _emission_probs[("h0", "o0")]
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=hmm.transition_probs,
                emission_probs=_emission_probs,
            )

    def test_emission_mat_extra_label(self):
        hmm = tc.create_hmm1()
        with pytest.raises(InvalidInputError):
            _emission_probs = hmm.emission_probs.copy()
            _emission_probs[("h", "o0")] = 1
            HiddenMarkovModel(
                start_probs=hmm.start_probs,
                transition_probs=hmm.transition_probs,
                emission_probs=_emission_probs,
            )

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

        assert hmm.start_probs == {"h0": high_val, "h1": low_val}
        assert hmm.transition_probs == {
            ("h0", "h0"): high_val,
            ("h0", "h1"): low_val,
            ("h1", "h0"): 0.6,
            ("h1", "h1"): 0.4,
        }
        assert hmm.emission_probs == {
            ("h0", "o0"): high_val,
            ("h0", "o1"): low_val,
            ("h1", "o0"): 0.4,
            ("h1", "o1"): 0.6,
        }

    def test_random_hmm(self):
        hidden_states = {1, 2, 3, 4}
        observable_states = ["a", "b", "c"]
        seed = 1
        hmm = conin.graphical_models.random_hmm(
            hidden_states=hidden_states,
            observable_states=observable_states,
            rng=seed,
        )
        assert set(hmm.hidden_states) == set(hidden_states)
        assert set(hmm.observable_states) == set(observable_states)

    def test_random_hmm_keeps_the_given_order(self):
        hmm = conin.graphical_models.random_hmm(
            hidden_states=["b", "a"], observable_states=["y", "x"], rng=0
        )
        assert hmm.hidden_states == ["b", "a"]
        assert hmm.observable_states == ["y", "x"]


class Test_HMM_Constructor:
    def test_constructor_requires_all_dictionaries(self):
        hmm = tc.create_hmm1()
        args = dict(
            start_probs=hmm.start_probs,
            transition_probs=hmm.transition_probs,
            emission_probs=hmm.emission_probs,
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
            start_probs=hmm.start_probs,
            transition_probs=hmm.transition_probs,
            emission_probs=hmm.emission_probs,
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
        assert hmm.hidden_states == ["a", "b", "c"]
        assert hmm.representation("matvec").hidden_to_internal == {
            "a": 0,
            "b": 1,
            "c": 2,
        }
        assert hmm.observable_states == ["x", "y", "z"]
        assert hmm.representation("matvec").start_vec.tolist() == [0, 1.0, 0]
        assert hmm.representation("matvec").transition_mat.tolist() == [
            [1.0, 0, 0],
            [0, 0, 1.0],
            [1.0, 0, 0],
        ]
        assert hmm.representation("matvec").emission_mat.tolist() == [
            [0, 0, 1.0],
            [1.0, 0, 0],
            [0, 1.0, 0],
        ]

    def test_round_trip_through_constructor(self):
        hmm = tc.create_hmm2()
        hmm2 = HiddenMarkovModel(
            start_probs=hmm.start_probs,
            transition_probs=hmm.transition_probs,
            emission_probs=hmm.emission_probs,
        )
        assert hmm2.hidden_states == hmm.hidden_states
        assert hmm2.observable_states == hmm.observable_states
        assert (
            hmm2.representation("matvec").start_vec.tolist()
            == hmm.representation("matvec").start_vec.tolist()
        )
        assert (
            hmm2.representation("matvec").transition_mat.tolist()
            == hmm.representation("matvec").transition_mat.tolist()
        )
        assert (
            hmm2.representation("matvec").emission_mat.tolist()
            == hmm.representation("matvec").emission_mat.tolist()
        )

    @pytest.mark.parametrize(
        "name",
        [
            "start_vec",
            "transition_mat",
            "emission_mat",
            "hidden_to_internal",
            "observable_to_internal",
            "hidden_to_external",
            "observed_to_external",
        ],
    )
    def test_index_view_is_in_matvec_only(self, name):
        hmm = tc.create_hmm1()
        assert not hasattr(hmm, name)
        if not name.endswith("_external"):
            assert hasattr(hmm.representation("matvec"), name)

    def test_state_labels_are_read_only(self):
        hmm = tc.create_hmm1()
        assert hmm.hidden_states is hmm.hidden_states
        with pytest.raises(TypeError, match="read-only"):
            hmm.hidden_states.append("h2")
        with pytest.raises(TypeError, match="read-only"):
            hmm.observable_states[0] = "o9"

    def test_no_cached_numeric_representation(self):
        hmm = tc.create_hmm1()
        args = dict(
            start_probs=hmm.start_probs,
            transition_probs=hmm.transition_probs,
            emission_probs=hmm.emission_probs,
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
            start_probs=hmm.start_probs,
            transition_probs=hmm.transition_probs,
            emission_probs=hmm.emission_probs,
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
        assert hmm.start_probs == {"h0": 1.0}
        assert hmm.transition_probs == {("h0", "h0"): 1.0}
        assert hmm.emission_probs == {("h0", "o0"): 1.0}

    def test_start_probs(self):
        hmm = tc.create_hmm1()
        assert hmm.start_probs == {"h0": 0.4, "h1": 0.6}

    def test_transition_probs(self):
        hmm = tc.create_hmm1()
        assert hmm.transition_probs == {
            ("h0", "h0"): 0.9,
            ("h0", "h1"): 0.1,
            ("h1", "h0"): 0.2,
            ("h1", "h1"): 0.8,
        }

    def test_emission_probs(self):
        hmm = tc.create_hmm1()
        assert hmm.emission_probs == {
            ("h0", "o0"): 0.7,
            ("h0", "o1"): 0.3,
            ("h1", "o0"): 0.4,
            ("h1", "o1"): 0.6,
        }

    def test_tables_are_dense_and_read_only(self):
        hmm = HiddenMarkovModel(
            start_probs={"a": 1.0},
            transition_probs={("a", "a"): 1.0, ("b", "a"): 1.0},
            emission_probs={("a", "x"): 1.0, ("b", "y"): 1.0},
        )
        assert hmm.start_probs == {"a": 1.0, "b": 0}
        assert hmm.transition_probs[("a", "b")] == 0
        assert hmm.emission_probs[("a", "y")] == 0
        for table in (hmm.start_probs, hmm.transition_probs, hmm.emission_probs):
            with pytest.raises(TypeError, match="read-only"):
                table.clear()

    @pytest.mark.parametrize(
        "name",
        [
            "get_start_probs",
            "get_transition_probs",
            "get_emission_probs",
            "get_hidden_states",
            "get_observable_states",
            "is_valid_hidden_state",
            "is_valid_observed_state",
            "to_dict",
            "log_probability",
            "write_to_file",
            "read_from_file",
            "observed_states",
            "num_observed_states",
        ],
    )
    def test_removed_methods(self, name):
        assert not hasattr(tc.create_hmm1(), name)


def _hmm1_args(**kwds):
    hmm = tc.create_hmm1()
    return dict(
        start_probs=hmm.start_probs,
        transition_probs=hmm.transition_probs,
        emission_probs=hmm.emission_probs,
        **kwds,
    )


class Test_HMM_State_Order:
    def test_given_order_is_the_matvec_order(self):
        hmm = HiddenMarkovModel(
            **_hmm1_args(hidden_states=["h1", "h0"], observable_states=["o1", "o0"])
        )
        assert hmm.hidden_states == ["h1", "h0"]
        assert hmm.observable_states == ["o1", "o0"]
        assert list(hmm.start_probs) == ["h1", "h0"]
        repn = hmm.representation("matvec")
        assert repn.hidden_states == ["h1", "h0"]
        assert repn.hidden_to_internal == {"h1": 0, "h0": 1}
        assert repn.start_vec.tolist() == [0.6, 0.4]
        assert repn.transition_mat.tolist() == [[0.8, 0.2], [0.1, 0.9]]
        assert repn.emission_mat.tolist() == [[0.6, 0.4], [0.3, 0.7]]

    def test_observable_state_that_is_never_emitted(self):
        hmm = HiddenMarkovModel(**_hmm1_args(observable_states=["o0", "o1", "o2"]))
        assert hmm.num_observable_states == 3
        assert hmm.emission_probs[("h0", "o2")] == 0
        assert hmm.representation("matvec").emission_mat[:, 2].tolist() == [0, 0]

    def test_labels_that_cannot_be_sorted(self):
        args = dict(
            start_probs={1: 1.0},
            transition_probs={(1, "a"): 1.0, ("a", 1): 1.0},
            emission_probs={(1, "x"): 1.0, ("a", "x"): 1.0},
        )
        with pytest.raises(TypeError):
            HiddenMarkovModel(**args)
        hmm = HiddenMarkovModel(**args, hidden_states=["a", 1])
        assert hmm.hidden_states == ["a", 1]

    @pytest.mark.parametrize(
        "kwds, match",
        [
            (dict(hidden_states=["h0"]), "start_probs has 'h1', which is not one"),
            (dict(observable_states=["o0"]), "emission_probs entry .* 'o1'"),
            (dict(hidden_states=[]), "hidden_states must not be empty"),
            (dict(hidden_states=["h0", "h1", "h0"]), "duplicate labels: 'h0'"),
            (dict(observable_states="o0o1"), "must be a list of labels"),
            (
                dict(hidden_states=["h0", "h1", "h2"]),
                "transition_probs from 'h2' must sum to 1, got 0",
            ),
        ],
    )
    def test_invalid_state_lists(self, kwds, match):
        with pytest.raises(InvalidInputError, match=match):
            HiddenMarkovModel(**_hmm1_args(**kwds))

    def test_start_probs_label_must_be_a_hidden_state(self):
        args = _hmm1_args()
        args["start_probs"] = {"h0": 0.4, "h1": 0.5, "x": 0.1}
        with pytest.raises(InvalidInputError, match="start_probs has 'x'"):
            HiddenMarkovModel(**args)

    def test_make_non_zero_keeps_the_order(self):
        hmm = HiddenMarkovModel(**_hmm1_args(hidden_states=["h1", "h0"]))
        assert hmm.make_non_zero().hidden_states == ["h1", "h0"]


class Test_HMM_Equality:
    def test_equal_models(self):
        a, b = tc.create_hmm1(), tc.create_hmm1()
        assert a == b and hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_order_and_probabilities_matter(self):
        hmm = tc.create_hmm1()
        assert hmm != HiddenMarkovModel(**_hmm1_args(hidden_states=["h1", "h0"]))
        assert hmm != hmm.make_non_zero(tol=0.4)
        assert hmm != tc.create_hmm2()
        assert hmm != "hmm"

    def test_repr(self):
        assert repr(tc.create_hmm1()) == (
            "HiddenMarkovModel(hidden_states=['h0', 'h1'], "
            "observable_states=['o0', 'o1'])"
        )

    def test_str_lists_the_nonzero_probabilities(self):
        text = str(tc.create_hmm1())
        assert "'num_observable': 2" in text and "('h0', 'o0')" in text
