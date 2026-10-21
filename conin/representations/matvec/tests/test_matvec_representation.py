import copy
import pickle

import numpy as np
import pytest

import conin.examples.hidden_markov_model as tc
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.constraints import OracleConstraint
from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
from conin.inference.oracle import a_star
from conin.inference.unconstrained import viterbi
from conin.inference.unconstrained.viterbi import viterbi_
from conin.representations import supported_representations
from conin.representations.matvec import MatVecHMMRepresentation

ARRAYS = (
    "start_vec",
    "transition_mat",
    "emission_mat",
    "log_start_vec",
    "log_transition_mat",
    "log_emission_mat",
)


def _sparse_hmm():
    # Zeros in every array, and labels whose sorted order differs from the
    # insertion order.
    return HiddenMarkovModel(
        start_probs={"b": 1.0},
        transition_probs={("c", "a"): 1.0, ("b", "c"): 1.0, ("a", "a"): 1.0},
        emission_probs={
            ("c", "y"): 1.0,
            ("b", "x"): 0.25,
            ("b", "z"): 0.75,
            ("a", "z"): 1.0,
        },
    )


def test_registered_for_the_bare_hmm_only():
    pairs = supported_representations(HiddenMarkovModel)
    assert [p for p in pairs if p[0] == "matvec"] == [("matvec", None)]
    cmodel = ConstrainedHiddenMarkovModel(
        tc.create_hmm1(), OracleConstraint(func=lambda s: True)
    )
    with pytest.raises(InvalidInputError, match="matvec"):
        cmodel.representation("matvec")


def test_no_structure_params():
    with pytest.raises(TypeError):
        ConstrainedHiddenMarkovModel(tc.create_hmm1()).representation("matvec", stop=3)


def test_class_and_attributes():
    hmm = _sparse_hmm()
    repn = ConstrainedHiddenMarkovModel(hmm).representation("matvec")
    assert isinstance(repn, MatVecHMMRepresentation)
    assert repn.kind == "matvec"
    assert repn.model is hmm
    assert (repn.num_hidden_states, repn.num_observed_states) == (3, 3)
    assert repn.start_vec.shape == (3,)
    assert repn.transition_mat.shape == (3, 3)
    assert repn.emission_mat.shape == (3, 3)
    for name in ARRAYS:
        assert getattr(repn, name).dtype == np.float64
    # Internal indices are the sorted labels.
    assert repn.start_vec.tolist() == hmm.start_vec == [0, 1.0, 0]
    assert repn.transition_mat.tolist() == hmm.transition_mat
    assert repn.emission_mat.tolist() == hmm.emission_mat
    for name in ("start_vec", "transition_mat", "emission_mat"):
        with np.errstate(divide="ignore"):
            expected = np.log(getattr(repn, name))
        assert np.array_equal(getattr(repn, "log_" + name), expected)
    assert repn.log_start_vec[0] == -np.inf
    assert repn.log_emission_mat[1, 0] == np.log(0.25)
    # No sampling methods and no state-label ranges.
    for name in ("generate_hidden", "generate_observed", "hidden_states"):
        assert not hasattr(repn, name)


def test_every_call_builds_a_new_representation():
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1())
    assert cmodel.representation("matvec") is not cmodel.representation("matvec")
    assert not hasattr(cmodel.model, "repn")


@pytest.mark.parametrize(
    "make",
    [
        lambda r: r,
        copy.deepcopy,
        lambda r: pickle.loads(pickle.dumps(r)),
    ],
    ids=["built", "deepcopy", "pickle"],
)
def test_arrays_are_read_only(make):
    repn = make(ConstrainedHiddenMarkovModel(tc.create_hmm1()).representation("matvec"))
    for name in ARRAYS:
        with pytest.raises(ValueError):
            getattr(repn, name)[0] = 0.5


def test_the_model_is_not_shared_with_the_arrays():
    hmm = tc.create_hmm1()
    repn = ConstrainedHiddenMarkovModel(hmm).representation("matvec")
    copy_ = np.array(repn.transition_mat)
    copy_[0, 0] = 0.0
    assert hmm.transition_mat[0][0] == 0.9
    assert repn.transition_mat[0, 0] == 0.9


def test_one_representation_serves_several_queries():
    hmm = tc.create_hmm2()
    repn = ConstrainedHiddenMarkovModel(hmm).representation("matvec")
    before = {name: getattr(repn, name).copy() for name in ARRAYS}
    observed = [hmm.observed_to_internal[o] for o in hmm.observed_states] * 3
    first = viterbi_(observed=observed, hmm=repn)
    second = viterbi_(observed=observed[:4], hmm=repn)
    again = viterbi_(observed=observed, hmm=repn)
    assert first == again and len(second.solution.hidden) == 4
    for name in ARRAYS:
        assert np.array_equal(getattr(repn, name), before[name])


def test_viterbi_and_a_star_accept_an_unconstrained_constrained_model():
    hmm = _sparse_hmm()
    observed = ["x", "y", "z", "z", "z"]
    expected = viterbi(observed=observed, hmm=hmm)
    assert viterbi(observed=observed, hmm=ConstrainedHiddenMarkovModel(hmm)) == expected
    assert expected.solution.hidden == ["b", "c", "a", "a", "a"]
    best = a_star(observed=observed, hmm=ConstrainedHiddenMarkovModel(hmm))
    assert best.solution.hidden == expected.solution.hidden
    assert best.solution.log_likelihood == expected.solution.log_likelihood


def test_viterbi_rejects_constraints():
    cmodel = tc.create_chmm1_oracle()
    with pytest.raises(InvalidInputError, match="matvec"):
        viterbi(observed=["o0"] * 3, hmm=cmodel)
