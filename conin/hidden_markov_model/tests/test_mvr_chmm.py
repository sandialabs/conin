from itertools import product

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.hmm import HiddenMarkovModel
from conin.hidden_markov_model.mvr import HomMVR
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

    start_probs = {
        h: start_vec[i]
        for i, h in enumerate(hidden_states)
    }

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
    )


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