from itertools import product

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.hmm import HiddenMarkovModel
from conin.hidden_markov_model.mvr import HomMVR, InhomMVR, MVR_MatVecRepn
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    return vec / vec.sum()


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / mat.sum(axis=1, keepdims=True)


def make_random_hmm(
    *,
    hidden_states: list[str],
    observed_states: list[str],
    seed: int = 123,
):
    rng = np.random.default_rng(seed)

    num_hidden = len(hidden_states)
    num_observed = len(observed_states)

    start_vec = normalize_vec(rng.random(num_hidden))
    transition_mat = normalize_rows(rng.random((num_hidden, num_hidden)))
    emission_mat = normalize_rows(rng.random((num_hidden, num_observed)))

    start_probs = {h: start_vec[i] for i, h in enumerate(hidden_states)}

    transition_probs = {
        (h1, h2): transition_mat[i, j]
        for i, h1 in enumerate(hidden_states)
        for j, h2 in enumerate(hidden_states)
    }

    emission_probs = {
        (h, o): emission_mat[i, j]
        for i, h in enumerate(hidden_states)
        for j, o in enumerate(observed_states)
    }

    hmm = HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        initialize=True,
    )

    return hmm


def make_forbid_state_mvr(
    *,
    hidden_states: list[str],
    forbidden_state: str,
    initialize: bool = False,
) -> HomMVR:
    """
    Construct a homogeneous MVR/DFA that rejects paths visiting a forbidden_state.
    """

    if forbidden_state not in hidden_states:
        raise ValueError("forbidden_state must be in hidden_states")

    mediation_states = ["ok", "violated"]

    ini = {
        h: "violated" if h == forbidden_state else "ok"
        for h in hidden_states
    }

    upd = {
        (m, h): "violated" if m == "violated" or h == forbidden_state else "ok"
        for m, h in product(mediation_states, hidden_states)
    }

    evl = {
        "ok": True,
        "violated": False,
    }

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
        initialize=initialize,
    )


def make_forbid_state_inhom_mvr(
    *,
    hidden_states: list[str],
    forbidden_state: str,
    time_horizon: int,
    initialize: bool = False,
) -> InhomMVR:
    """
    Construct an inhomogeneous MVR that rejects paths visiting forbidden_state. Essentially the same as above with dummy time dimension.
    """

    if forbidden_state not in hidden_states:
        raise ValueError("forbidden_state must be in hidden_states")

    if time_horizon <= 0:
        raise ValueError("time_horizon must be positive")

    mediation_states = [
        [f"ok_{t}", f"violated_{t}"]
        for t in range(time_horizon)
    ]

    ini = {
        h: f"violated_0" if h == forbidden_state else f"ok_0"
        for h in hidden_states
    }

    upd = []

    for t in range(time_horizon - 1):
        upd_t = {}

        for m_prev, h in product(mediation_states[t], hidden_states):
            if m_prev == f"violated_{t}" or h == forbidden_state:
                upd_t[(m_prev, h)] = f"violated_{t + 1}"
            else:
                upd_t[(m_prev, h)] = f"ok_{t + 1}"

        upd.append(upd_t)

    evl = [
        {
            f"ok_{t}": True,
            f"violated_{t}": False,
        }
        for t in range(time_horizon)
    ]

    return InhomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
        initialize=initialize,
    )


def make_valid_direct_mvr_repn_arrays():
    """
    Construct a simple valid homogeneous MVR_MatVecRepn.

    H = 2, M = 2.

    init_array:
        hidden 0 -> mediation 0
        hidden 1 -> mediation 1

    update_array:
        identity update on mediation state, independent of hidden state.
    """

    init_array = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    update_array = np.zeros((2, 2, 2), dtype=float)

    # update_array[h, m_curr, m_prev]
    # For both hidden states, m_curr = m_prev.
    update_array[:, 0, 0] = 1.0
    update_array[:, 1, 1] = 1.0

    eval_array = np.array([1.0, 0.0])

    return init_array, update_array, eval_array


def test_make_random_hmm_is_valid():
    hidden_states = ["A", "B", "C"]
    observed_states = ["o0", "o1"]

    hmm = make_random_hmm(
        hidden_states=hidden_states,
        observed_states=observed_states,
    )

    assert hmm.repn is not None
    assert set(hmm.hidden_states) == set(hidden_states)
    assert set(hmm.observed_states) == set(observed_states)

    assert np.isclose(sum(hmm.start_vec), 1.0)

    for row in hmm.transition_mat:
        assert np.isclose(sum(row), 1.0)

    for row in hmm.emission_mat:
        assert np.isclose(sum(row), 1.0)


def test_forbid_state_mvr_logic():
    hidden_states = ["A", "B", "C"]
    forbidden_state = "B"

    mvr = make_forbid_state_mvr(
        hidden_states=hidden_states,
        forbidden_state=forbidden_state,
    )

    assert mvr.ini["A"] == "ok"
    assert mvr.ini["B"] == "violated"
    assert mvr.ini["C"] == "ok"

    assert mvr.upd[("ok", "A")] == "ok"
    assert mvr.upd[("ok", "B")] == "violated"
    assert mvr.upd[("ok", "C")] == "ok"

    assert mvr.upd[("violated", "A")] == "violated"
    assert mvr.upd[("violated", "B")] == "violated"
    assert mvr.upd[("violated", "C")] == "violated"

    assert mvr.evl["ok"] is True
    assert mvr.evl["violated"] is False


def test_hom_mvr_matvec_repn_for_forbid_state_mvr():
    hidden_states = ["A", "B", "C"]
    forbidden_state = "B"

    mvr = make_forbid_state_mvr(
        hidden_states=hidden_states,
        forbidden_state=forbidden_state,
    )

    repn = mvr.repn

    assert isinstance(repn, MVR_MatVecRepn)

    assert repn.init_array.shape == (3, 2)
    assert repn.update_array.shape == (3, 2, 2)
    assert repn.eval_array.shape == (2,)

    # hidden order: A, B, C
    # mediation order: ok, violated
    expected_init_array = np.array(
        [
            [1.0, 0.0],  # A -> ok
            [0.0, 1.0],  # B -> violated
            [1.0, 0.0],  # C -> ok
        ]
    )

    expected_eval_array = np.array([1.0, 0.0])

    assert np.array_equal(repn.init_array, expected_init_array)
    assert np.array_equal(repn.eval_array, expected_eval_array)

    hidden_to_idx = {
        h: i for i, h in enumerate(hidden_states)
    }

    mediation_to_idx = {
        "ok": 0,
        "violated": 1,
    }

    for h in hidden_states:
        h_idx = hidden_to_idx[h]

        for m_prev in ["ok", "violated"]:
            m_prev_idx = mediation_to_idx[m_prev]

            expected_m_curr = (
                "violated"
                if m_prev == "violated" or h == forbidden_state
                else "ok"
            )
            expected_m_curr_idx = mediation_to_idx[expected_m_curr]

            assert repn.update_array[h_idx, expected_m_curr_idx, m_prev_idx] == 1.0
            assert np.isclose(repn.update_array[h_idx, :, m_prev_idx].sum(), 1.0)

    assert repn.num_hidden_states == 3
    assert repn.num_mediation_states == 2
    assert list(repn.hidden_states) == [0, 1, 2]
    assert list(repn.mediation_states) == [0, 1]


def test_hom_mvr_repn_is_cached():
    hidden_states = ["A", "B", "C"]

    mvr = make_forbid_state_mvr(
        hidden_states=hidden_states,
        forbidden_state="B",
    )

    repn_1 = mvr.repn
    repn_2 = mvr.repn
    repn_3 = mvr.initialize()

    assert repn_1 is repn_2
    assert repn_1 is repn_3


def test_hom_mvr_initialize_true_builds_repn_immediately():
    mvr = make_forbid_state_mvr(
        hidden_states=["A", "B", "C"],
        forbidden_state="B",
        initialize=True,
    )

    assert mvr._repn is not None
    assert isinstance(mvr._repn, MVR_MatVecRepn)


def test_inhom_mvr_matvec_repn_for_forbid_state_mvr():
    hidden_states = ["A", "B"]
    forbidden_state = "B"
    time_horizon = 3

    mvr = make_forbid_state_inhom_mvr(
        hidden_states=hidden_states,
        forbidden_state=forbidden_state,
        time_horizon=time_horizon,
    )

    repn = mvr.repn

    assert isinstance(repn, MVR_MatVecRepn)

    assert repn.init_array.shape == (2, 2)
    assert isinstance(repn.update_array, list)
    assert isinstance(repn.eval_array, list)

    assert len(repn.update_array) == time_horizon - 1
    assert len(repn.eval_array) == time_horizon

    expected_init_array = np.array(
        [
            [1.0, 0.0],  # A -> ok_0
            [0.0, 1.0],  # B -> violated_0
        ]
    )

    assert np.array_equal(repn.init_array, expected_init_array)

    for t in range(time_horizon):
        assert repn.eval_array[t].shape == (2,)
        assert np.array_equal(repn.eval_array[t], np.array([1.0, 0.0]))

    for t in range(time_horizon - 1):
        assert repn.update_array[t].shape == (2, 2, 2)

    hidden_to_idx = {
        h: i for i, h in enumerate(hidden_states)
    }

    # At each time, mediation order is [ok_t, violated_t].
    for t in range(time_horizon - 1):
        update_t = repn.update_array[t]

        for h in hidden_states:
            h_idx = hidden_to_idx[h]

            for m_prev_idx, m_prev_status in enumerate(["ok", "violated"]):
                expected_m_curr_idx = (
                    1
                    if m_prev_status == "violated" or h == forbidden_state
                    else 0
                )

                assert update_t[h_idx, expected_m_curr_idx, m_prev_idx] == 1.0
                assert np.isclose(update_t[h_idx, :, m_prev_idx].sum(), 1.0)

    assert repn.time_horizon == time_horizon
    assert repn.num_hidden_states == 2
    assert repn.num_mediation_states == [2, 2, 2]


def test_direct_mvr_matvec_repn_valid_homogeneous_arrays():
    init_array, update_array, eval_array = make_valid_direct_mvr_repn_arrays()

    repn = MVR_MatVecRepn(
        init_array=init_array,
        update_array=update_array,
        eval_array=eval_array,
    )

    assert np.array_equal(repn.init_array, init_array)
    assert np.array_equal(repn.update_array, update_array)
    assert np.array_equal(repn.eval_array, eval_array)

    assert repn.num_hidden_states == 2
    assert repn.num_mediation_states == 2
    assert list(repn.hidden_states) == [0, 1]
    assert list(repn.mediation_states) == [0, 1]


def test_direct_mvr_matvec_repn_rejects_invalid_init_array_rows():
    _, update_array, eval_array = make_valid_direct_mvr_repn_arrays()

    bad_init_array = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(
        InvalidInputError,
        match="init_array rows must sum to 1",
    ):
        MVR_MatVecRepn(
            init_array=bad_init_array,
            update_array=update_array,
            eval_array=eval_array,
        )


def test_direct_mvr_matvec_repn_rejects_invalid_update_array_sums():
    init_array, _, eval_array = make_valid_direct_mvr_repn_arrays()

    bad_update_array = np.zeros((2, 2, 2), dtype=float)

    with pytest.raises(
        InvalidInputError,
        match="must sum to 1 over the current mediation axis",
    ):
        MVR_MatVecRepn(
            init_array=init_array,
            update_array=bad_update_array,
            eval_array=eval_array,
        )


def test_direct_mvr_matvec_repn_rejects_invalid_eval_array_dimension():
    init_array, update_array, _ = make_valid_direct_mvr_repn_arrays()

    bad_eval_array = np.array(
        [
            [1.0, 0.0],
        ]
    )

    with pytest.raises(
        InvalidInputError,
        match="eval_array at index 0 must be a 1D array",
    ):
        MVR_MatVecRepn(
            init_array=init_array,
            update_array=update_array,
            eval_array=bad_eval_array,
        )


def test_direct_mvr_matvec_repn_rejects_dimension_mismatch():
    init_array, update_array, _ = make_valid_direct_mvr_repn_arrays()

    bad_eval_array = np.array([1.0, 0.0, 1.0])

    with pytest.raises(
        InvalidInputError,
        match="eval_array mediation dimension must match init_array mediation dimension",
    ):
        MVR_MatVecRepn(
            init_array=init_array,
            update_array=update_array,
            eval_array=bad_eval_array,
        )


def test_mvr_chmm_accepts_random_hmm_with_forbid_state_mvr():
    hidden_states = ["A", "B", "C"]
    observed_states = ["o0", "o1"]
    forbidden_state = "B"

    hmm = make_random_hmm(
        hidden_states=hidden_states,
        observed_states=observed_states,
    )

    mvr = make_forbid_state_mvr(
        hidden_states=hidden_states,
        forbidden_state=forbidden_state,
    )

    model = MVR_CHMM(
        hidden_markov_model=hmm,
        constraints=[mvr],
        data=None,
    )

    assert model is not None


def test_mvr_chmm_rejects_missing_hmm():
    with pytest.raises(
        InvalidInputError,
        match="hidden_markov_model is a required argument",
    ):
        MVR_CHMM(
            hidden_markov_model=None,
            constraints=None,
            data=None,
        )


def test_mvr_chmm_rejects_mismatched_hidden_states():
    hmm = make_random_hmm(
        hidden_states=["A", "B", "C"],
        observed_states=["o0", "o1"],
    )

    mvr = make_forbid_state_mvr(
        hidden_states=["A", "B", "D"],
        forbidden_state="D",
    )

    with pytest.raises(
        InvalidInputError,
        match="Hidden states of constraint 0 do not match",
    ):
        MVR_CHMM(
            hidden_markov_model=hmm,
            constraints=[mvr],
            data=None,
        )


def test_mvr_chmm_rejects_non_mvr_constraint():
    hmm = make_random_hmm(
        hidden_states=["A", "B", "C"],
        observed_states=["o0", "o1"],
    )

    bad_constraint = "not an MVR"

    with pytest.raises(
        InvalidInputError,
        match="Constraint 0 is not an MVR object",
    ):
        MVR_CHMM(
            hidden_markov_model=hmm,
            constraints=[bad_constraint],
            data=None,
        )