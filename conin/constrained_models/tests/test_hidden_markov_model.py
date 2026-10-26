import pytest

from conin import *
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models import HiddenMarkovModel

from conin.config import default_mip_solver
import conin.examples.hidden_markov_model as tc
import conin.constraints.oracle_library as oc
from conin.representations.matvec import MatVecHMMRepresentation
from conin.sampling.oracle import rejection_sample_hidden
from conin.sampling.unconstrained import (
    generate_hidden,
    generate_observed_from_hidden,
)


def _generate_observed(chmm, T):
    return generate_observed_from_hidden(chmm.model, rejection_sample_hidden(chmm, T))


class Test_Oracle_ConstrainedHiddenMarkovModel:
    T = 25

    def test_construct_model(self):
        cpgm = tc.create_chmm1_oracle()

        assert cpgm.representation("oracle").hmm.start_vec.tolist() == [
            cpgm.model.get_start_probs()[h] for h in cpgm.model.hidden_states
        ]

    def test_construct_model2(self):
        cpgm = tc.create_chmm1_oracle()

        _hmm = HiddenMarkovModel(
            start_probs=cpgm.model.get_start_probs(),
            emission_probs=cpgm.model.get_emission_probs(),
            transition_probs=cpgm.model.get_transition_probs(),
        )
        _repn = ConstrainedHiddenMarkovModel(_hmm).representation("oracle")

        assert _repn.hmm.start_vec.tolist() == [
            cpgm.model.get_start_probs()[h] for h in cpgm.model.hidden_states
        ]

    def test_construct_model3(self):
        pgm = tc.create_hmm0()
        constraints = [oc.all_diff_constraint]
        repn = ConstrainedHiddenMarkovModel(pgm, constraints).representation("oracle")
        assert isinstance(repn.hmm, MatVecHMMRepresentation)
        assert repn.hmm.transition_mat.tolist() == pgm.transition_mat
        assert len(repn.constraints) == 1

    def test_internal_is_feasible(self):
        cpgm = tc.create_chmm1_oracle()
        assert len(cpgm.constraints) == 2

        fail_seq1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        pass_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fail_seq2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert not cpgm.representation("oracle").is_feasible(fail_seq1)
        assert cpgm.representation("oracle").is_feasible(pass_seq)
        assert not cpgm.representation("oracle").is_feasible(fail_seq2)

    def test_is_valid_hidden_state(self):
        cpgm = tc.create_chmm1_oracle()
        assert cpgm.model.is_valid_hidden_state("h0")
        assert cpgm.model.is_valid_hidden_state("h1")
        assert not cpgm.model.is_valid_hidden_state("invalid")
        assert not cpgm.model.is_valid_hidden_state("")

    def test_set_seed_affects_generation(self):
        # Calling set_seed with the same value should produce identical sequences.
        chmm = tc.create_chmm1_oracle()
        chmm.model.set_seed(42)
        seq1 = generate_hidden(chmm.model, 10)
        chmm.model.set_seed(42)
        seq2 = generate_hidden(chmm.model, 10)
        assert seq1 == seq2

    def test_generate(self):
        cpgm = tc.create_chmm1_oracle()
        observed = generate_observed_from_hidden(
            cpgm.model,
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
            ],
        )
        assert len(observed) == 11
        T = 25
        observed2 = _generate_observed(cpgm, T)
        assert len(observed2) == T

        with pytest.raises(InvalidInputError):
            _generate_observed(cpgm, -1)

    def test_is_feasible(self):
        cpgm = tc.create_chmm1_oracle()
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
        chmm = tc.create_chmm1_oracle()
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
        chmm = tc.create_chmm1_oracle()
        assert len(rejection_sample_hidden(chmm, self.T)) == self.T
        with pytest.raises(InvalidInputError):
            rejection_sample_hidden(chmm, -1)

    def test_generate_hidden_output(self):
        chmm = tc.create_chmm1_oracle()
        for h in rejection_sample_hidden(chmm, self.T):
            assert h in {"h0", "h1"}

    def test_generate_hidden_constraint(self):
        chmm = tc.create_chmm1_oracle()
        assert chmm.is_feasible(rejection_sample_hidden(chmm, self.T))

    def test_generate_hidden_negative_time_steps(self):
        chmm = tc.create_chmm1_oracle()
        with pytest.raises(InvalidInputError):
            rejection_sample_hidden(chmm, -1)

    def test_generate_observed_from_hidden_length(self):
        chmm = tc.create_chmm1_oracle()
        hidden = rejection_sample_hidden(chmm, self.T)
        assert len(generate_observed_from_hidden(chmm.model, hidden)) == self.T

    def test_generate_observed_from_hidden_output(self):
        chmm = tc.create_chmm1_oracle()
        hidden = rejection_sample_hidden(chmm, self.T)
        for o in generate_observed_from_hidden(chmm.model, hidden):
            assert o in {"o0", "o1"}

    def test_generate_observed_length(self):
        chmm = tc.create_chmm1_oracle()
        assert len(_generate_observed(chmm, self.T)) == self.T
        with pytest.raises(InvalidInputError):
            _generate_observed(chmm, -1)

    def test_generate_observed_output(self):
        chmm = tc.create_chmm1_oracle()
        for h in _generate_observed(chmm, self.T):
            assert h in {"o0", "o1"}

    def test_generate_observed_negative_time_steps(self):
        chmm = tc.create_chmm1_oracle()
        with pytest.raises(InvalidInputError):
            _generate_observed(chmm, -1)


# ---------------------------------------------------------------------
# Ported from the deleted hidden_markov_model/tests/test_application_chmm.py
# (the Num_Zeros HMMApplication), on ConstrainedHiddenMarkovModel.
# ---------------------------------------------------------------------


def _num_zeros_hmm():
    p_stay, p_err = 0.6, 0.3
    return HiddenMarkovModel(
        start_probs={0: 0.5, 1: 0.5},
        transition_probs={
            (0, 0): p_stay,
            (0, 1): 1 - p_stay,
            (1, 0): 1 - p_stay,
            (1, 1): p_stay,
        },
        emission_probs={
            (0, 0): 1 - p_err,
            (0, 1): p_err,
            (1, 0): p_err,
            (1, 1): 1 - p_err,
        },
    )


def test_integer_states_oracle_sampling_and_feasibility():
    hmm = _num_zeros_hmm()
    chmm = ConstrainedHiddenMarkovModel(
        hmm, oc.has_exact_number_of_occurences_constraint(val=0, count=10)
    )
    assert not chmm.is_feasible([0] * 9)
    assert chmm.is_feasible([0] * 10)
    assert not chmm.is_feasible([0] * 11)

    hmm.set_seed(1)
    for _ in range(5):
        hidden = rejection_sample_hidden(chmm, 20)
        assert len(hidden) == 20 and hidden.count(0) == 10
        assert chmm.is_feasible(hidden)
        assert len(generate_observed_from_hidden(chmm.model, hidden)) == 20


@pytest.mark.skipif(not default_mip_solver, reason="No mip solver installed")
def test_network_flow_constraint_reads_the_application_data():
    # A Pyomo constraint that reads data.num_zeros gets the data passed to
    # ConstrainedHiddenMarkovModel(..., data=...).
    import munch
    import pyomo.environ as pyo

    from conin.inference.pyomo import ip_inference

    @pyomo_constraint_fn()
    def num_zeros(M, data):
        T = list(list(M.hmm.x.index_set().subsets())[0])
        M.num_zeros = pyo.Constraint(
            expr=sum(M.hmm.x[t, 0] for t in T) == data.num_zeros
        )

    chmm = ConstrainedHiddenMarkovModel(
        _num_zeros_hmm(), num_zeros, data=munch.Munch(num_zeros=3)
    )
    results = ip_inference(hmm=chmm, observed=[0] * 8, solver=default_mip_solver)
    assert results.solution.states.count(0) == 3
