from itertools import product

import numpy as np
import pytest

from conin.constraint import mvr_constraint_fn
from conin.exceptions import InvalidInputError
from conin.hidden_markov_model import ConstrainedHiddenMarkovModel
from conin.hidden_markov_model.hmm import HiddenMarkovModel
from conin.hidden_markov_model.mvr import HomMVR, InhomMVR, MVR_MatVecRepn
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM

# ===========================
# Heplers
# ===========================


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
) -> HomMVR:
    """
    Construct a homogeneous MVR/DFA that rejects paths visiting a forbidden_state.
    """

    if forbidden_state not in hidden_states:
        raise ValueError("forbidden_state must be in hidden_states")

    mediation_states = ["ok", "violated"]

    ini = {h: "violated" if h == forbidden_state else "ok" for h in hidden_states}

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
    )


def make_forbid_state_inhom_mvr(
    *,
    hidden_states: list[str],
    forbidden_state: str,
    time_horizon: int,
) -> InhomMVR:
    """
    Construct an inhomogeneous MVR that rejects paths visiting forbidden_state.

    Essentially the same as make_forbid_state_mvr, but with a dummy time dimension.
    """

    if forbidden_state not in hidden_states:
        raise ValueError("forbidden_state must be in hidden_states")

    if time_horizon <= 0:
        raise ValueError("time_horizon must be positive")

    mediation_states = [[f"ok_{t}", f"violated_{t}"] for t in range(time_horizon)]

    ini = {h: f"violated_0" if h == forbidden_state else f"ok_0" for h in hidden_states}

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
    )


def make_valid_direct_mvr_repn_arrays():
    """
    Construct a simple valid homogeneous MVR_MatVecRepn. H = 2, M = 2.
    """

    ini_array = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    upd_array = np.zeros((2, 2, 2), dtype=float)

    # upd_array[h, m_curr, m_prev]
    # For both hidden states, m_curr = m_prev.
    upd_array[:, 0, 0] = 1.0
    upd_array[:, 1, 1] = 1.0

    evl_array = np.array([1.0, 0.0])

    return ini_array, upd_array, evl_array


# ===========================
# Tests
# ===========================


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


def test_hom_mvr_prefix_defaults_to_false():
    mvr = make_forbid_state_mvr(
        hidden_states=["A", "B", "C"],
        forbidden_state="B",
    )

    assert mvr.prefix is False
    assert mvr._prefix is False


def test_hom_mvr_prefix_property_is_read_only():
    mvr = make_forbid_state_mvr(
        hidden_states=["A", "B", "C"],
        forbidden_state="B",
    )

    with pytest.raises(AttributeError):
        mvr.prefix = True


def test_hom_mvr_matvec_repn_for_forbid_state_mvr():
    hidden_states = ["A", "B", "C"]
    forbidden_state = "B"

    mvr = make_forbid_state_mvr(
        hidden_states=hidden_states,
        forbidden_state=forbidden_state,
    )

    repn = mvr.repn

    assert isinstance(repn, MVR_MatVecRepn)

    assert repn.ini_array.shape == (3, 2)
    assert repn.upd_array.shape == (3, 2, 2)
    assert repn.evl_array.shape == (2,)

    # hidden order: A, B, C
    # mediation order: ok, violated
    expected_ini_array = np.array(
        [
            [1.0, 0.0],  # A -> ok
            [0.0, 1.0],  # B -> violated
            [1.0, 0.0],  # C -> ok
        ]
    )

    expected_evl_array = np.array([1.0, 0.0])

    assert np.array_equal(repn.ini_array, expected_ini_array)
    assert np.array_equal(repn.evl_array, expected_evl_array)

    hidden_to_idx = {h: i for i, h in enumerate(hidden_states)}

    mediation_to_idx = {
        "ok": 0,
        "violated": 1,
    }

    for h in hidden_states:
        h_idx = hidden_to_idx[h]

        for m_prev in ["ok", "violated"]:
            m_prev_idx = mediation_to_idx[m_prev]

            expected_m_curr = (
                "violated" if m_prev == "violated" or h == forbidden_state else "ok"
            )
            expected_m_curr_idx = mediation_to_idx[expected_m_curr]

            assert repn.upd_array[h_idx, expected_m_curr_idx, m_prev_idx] == 1.0
            assert np.isclose(repn.upd_array[h_idx, :, m_prev_idx].sum(), 1.0)

    assert repn.num_hidden_states == 3
    assert repn.num_mediation_states == 2
    assert list(repn.hidden_states) == [0, 1, 2]
    assert list(repn.mediation_states) == [0, 1]


def test_hom_mvr_repn_is_consistent():
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


def test_hom_mvr_constructor_builds_repn_immediately():
    mvr = make_forbid_state_mvr(
        hidden_states=["A", "B", "C"],
        forbidden_state="B",
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

    assert repn.ini_array.shape == (2, 2)
    assert isinstance(repn.upd_array, list)
    assert isinstance(repn.evl_array, list)

    assert len(repn.upd_array) == time_horizon - 1
    assert len(repn.evl_array) == time_horizon

    expected_ini_array = np.array(
        [
            [1.0, 0.0],  # A -> ok_0
            [0.0, 1.0],  # B -> violated_0
        ]
    )

    assert np.array_equal(repn.ini_array, expected_ini_array)

    for t in range(time_horizon):
        assert repn.evl_array[t].shape == (2,)
        assert np.array_equal(repn.evl_array[t], np.array([1.0, 0.0]))

    for t in range(time_horizon - 1):
        assert repn.upd_array[t].shape == (2, 2, 2)

    hidden_to_idx = {h: i for i, h in enumerate(hidden_states)}

    # At each time, mediation order is [ok_t, violated_t].
    for t in range(time_horizon - 1):
        update_t = repn.upd_array[t]

        for h in hidden_states:
            h_idx = hidden_to_idx[h]

            for m_prev_idx, m_prev_status in enumerate(["ok", "violated"]):
                expected_m_curr_idx = (
                    1 if m_prev_status == "violated" or h == forbidden_state else 0
                )

                assert update_t[h_idx, expected_m_curr_idx, m_prev_idx] == 1.0
                assert np.isclose(update_t[h_idx, :, m_prev_idx].sum(), 1.0)

    assert repn.time_horizon == time_horizon
    assert repn.num_hidden_states == 2
    assert repn.num_mediation_states == [2, 2, 2]


def test_direct_mvr_matvec_repn_valid_homogeneous_arrays():
    ini_array, upd_array, evl_array = make_valid_direct_mvr_repn_arrays()

    repn = MVR_MatVecRepn(
        ini_array=ini_array,
        upd_array=upd_array,
        evl_array=evl_array,
    )

    assert np.array_equal(repn.ini_array, ini_array)
    assert np.array_equal(repn.upd_array, upd_array)
    assert np.array_equal(repn.evl_array, evl_array)

    assert repn.num_hidden_states == 2
    assert repn.num_mediation_states == 2
    assert list(repn.hidden_states) == [0, 1]
    assert list(repn.mediation_states) == [0, 1]


def test_direct_mvr_matvec_repn_rejects_invalid_ini_array_rows():
    _, upd_array, evl_array = make_valid_direct_mvr_repn_arrays()

    bad_ini_array = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(
        InvalidInputError,
        match="ini_array rows must sum to 1",
    ):
        MVR_MatVecRepn(
            ini_array=bad_ini_array,
            upd_array=upd_array,
            evl_array=evl_array,
        )


def test_direct_mvr_matvec_repn_rejects_invalid_upd_array_sums():
    ini_array, _, evl_array = make_valid_direct_mvr_repn_arrays()

    bad_upd_array = np.zeros((2, 2, 2), dtype=float)

    with pytest.raises(
        InvalidInputError,
        match="must sum to 1 over the current mediation axis",
    ):
        MVR_MatVecRepn(
            ini_array=ini_array,
            upd_array=bad_upd_array,
            evl_array=evl_array,
        )


def test_direct_mvr_matvec_repn_rejects_invalid_evl_array_dimension():
    ini_array, upd_array, _ = make_valid_direct_mvr_repn_arrays()

    bad_evl_array = np.array(
        [
            [1.0, 0.0],
        ]
    )

    with pytest.raises(
        InvalidInputError,
        match="evl_array at index 0 must be a 1D array",
    ):
        MVR_MatVecRepn(
            ini_array=ini_array,
            upd_array=upd_array,
            evl_array=bad_evl_array,
        )


def test_direct_mvr_matvec_repn_rejects_dimension_mismatch():
    ini_array, upd_array, _ = make_valid_direct_mvr_repn_arrays()

    bad_evl_array = np.array([1.0, 0.0, 1.0])

    with pytest.raises(
        InvalidInputError,
        match="evl_array mediation dimension must match ini_array mediation dimension",
    ):
        MVR_MatVecRepn(
            ini_array=ini_array,
            upd_array=upd_array,
            evl_array=bad_evl_array,
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


def test_constrained_hmm_initializes_mvr_backend_from_functor():
    hidden_states = ["A", "B", "C"]
    observed_states = ["o0", "o1"]
    forbidden_state = "B"

    hmm = make_random_hmm(
        hidden_states=hidden_states,
        observed_states=observed_states,
    )

    @mvr_constraint_fn(name="forbid_state")
    def forbid_state_constraint(hidden_markov_model, data):
        assert hidden_markov_model is hmm
        assert data == {"forbidden_state": forbidden_state}
        return make_forbid_state_mvr(
            hidden_states=hidden_markov_model.hidden_states,
            forbidden_state=data["forbidden_state"],
        )

    chmm = ConstrainedHiddenMarkovModel(
        hmm=hmm,
        constraints=[forbid_state_constraint],
    )
    chmm.initialize_chmm(data={"forbidden_state": forbidden_state})

    assert chmm.constraint_type == "mvr"
    assert isinstance(chmm.chmm, MVR_CHMM)
    assert len(chmm.chmm.constraints) == 1
    assert isinstance(chmm.chmm.constraints[0], HomMVR)
    

# ===========================
# time_range tests
# ===========================


def _make_generated_mvr_with_time_range(mvr_type, time_range=None, time_horizon=3):
    if mvr_type == "hom":
        base_mvr = make_forbid_state_mvr(
            hidden_states=["A", "B", "C"],
            forbidden_state="B",
        )

        return HomMVR(
            hidden_states=base_mvr.hidden_states,
            mediation_states=base_mvr.mediation_states,
            ini=base_mvr.ini,
            upd=base_mvr.upd,
            evl=base_mvr.evl,
            time_range=time_range,
        )

    if mvr_type == "inhom":
        base_mvr = make_forbid_state_inhom_mvr(
            hidden_states=["A", "B", "C"],
            forbidden_state="B",
            time_horizon=time_horizon,
        )

        return InhomMVR(
            hidden_states=base_mvr.hidden_states,
            mediation_states=base_mvr.mediation_states,
            ini=base_mvr.ini,
            upd=base_mvr.upd,
            evl=base_mvr.evl,
            time_range=time_range,
        )

    raise ValueError("mvr_type must be either 'hom' or 'inhom'")


@pytest.mark.parametrize("mvr_type", ["hom", "inhom"])
def test_mvr_time_range_defaults_to_none(mvr_type):
    mvr = _make_generated_mvr_with_time_range(
        mvr_type=mvr_type,
        time_range=None,
        time_horizon=3,
    )

    assert mvr._time_range is None


@pytest.mark.parametrize("mvr_type", ["hom", "inhom"])
@pytest.mark.parametrize(
    "time_range",
    [
        [0, 0],  # inclusive width 1
        [0, 1],  # inclusive width 2
        [0, 2],  # inclusive width 3
        [1, 3],  # inclusive width 3
        [5, 7],  # inclusive width 3
    ],
)
def test_mvr_accepts_valid_time_range_list(mvr_type, time_range):
    mvr = _make_generated_mvr_with_time_range(
        mvr_type=mvr_type,
        time_range=time_range,
        time_horizon=3,
    )

    assert mvr._time_range == time_range
    assert mvr._time_range is time_range


@pytest.mark.parametrize("mvr_type", ["hom", "inhom"])
@pytest.mark.parametrize(
    "bad_time_range",
    [
        (0, 1),
        np.array([0, 1]),
        "01",
        {0: 0, 1: 1},
        range(2),
    ],
)
def test_mvr_rejects_non_list_time_range(mvr_type, bad_time_range):
    with pytest.raises(
        InvalidInputError,
        match="time range must be a list of two nonnegative integers",
    ):
        _make_generated_mvr_with_time_range(
            mvr_type=mvr_type,
            time_range=bad_time_range,
            time_horizon=3,
        )


@pytest.mark.parametrize("mvr_type", ["hom", "inhom"])
@pytest.mark.parametrize(
    "bad_time_range",
    [
        [],
        [0],
        [0, 1, 2],
        [0, 1, 2, 3],
    ],
)
def test_mvr_rejects_time_range_with_invalid_length(mvr_type, bad_time_range):
    with pytest.raises(
        InvalidInputError,
        match="time range must be a list of two nonnegative integers",
    ):
        _make_generated_mvr_with_time_range(
            mvr_type=mvr_type,
            time_range=bad_time_range,
            time_horizon=3,
        )


@pytest.mark.parametrize("mvr_type", ["hom", "inhom"])
@pytest.mark.parametrize(
    "bad_time_range",
    [
        [-1, 1],
        [0, -1],
        [0.0, 1],
        [True, 1],
        ["0", 1],
    ],
)
def test_mvr_rejects_time_range_with_invalid_entries(mvr_type, bad_time_range):
    with pytest.raises(
        InvalidInputError,
        match="time range must be a list of two nonnegative integers",
    ):
        _make_generated_mvr_with_time_range(
            mvr_type=mvr_type,
            time_range=bad_time_range,
            time_horizon=3,
        )


@pytest.mark.parametrize("mvr_type", ["hom", "inhom"])
@pytest.mark.parametrize(
    "bad_time_range",
    [
        [1, 0],
        [2, 1],
        [10, 9],
    ],
)
def test_mvr_rejects_time_range_with_start_greater_than_end(
    mvr_type,
    bad_time_range,
):
    with pytest.raises(
        InvalidInputError,
        match="time range must be a list of two nonnegative integers",
    ):
        _make_generated_mvr_with_time_range(
            mvr_type=mvr_type,
            time_range=bad_time_range,
            time_horizon=3,
        )


@pytest.mark.parametrize(
    "bad_time_range",
    [
        [0, 3],  # inclusive width 4 > time_horizon 3
        [1, 4],  # inclusive width 4 > time_horizon 3
        [2, 5],  # inclusive width 4 > time_horizon 3
        [0, 4],  # inclusive width 5 > time_horizon 3
        [5, 10],  # inclusive width 6 > time_horizon 3
    ],
)
def test_inhom_mvr_rejects_time_range_width_exceeding_time_horizon(
    bad_time_range,
):
    with pytest.raises(
        InvalidInputError,
        match="time range cannot be longer than the time horizon",
    ):
        _make_generated_mvr_with_time_range(
            mvr_type="inhom",
            time_range=bad_time_range,
            time_horizon=3,
        )