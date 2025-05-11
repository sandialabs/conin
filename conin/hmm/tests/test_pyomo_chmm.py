import pytest

import conin.hmm
import conin.hmm.algebraic_chmm


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
    hmm = conin.hmm.HMM()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


class Test_IndexSets:

    def test_HMM(self, hmm):
        assert hmm is not None

    def test_data(self, hmm):
        assert hmm.num_hidden_states == 2
        assert hmm.start_vec == [0.4, 0.6]
        assert hmm.emission_mat == [[0.7, 0.3], [0.4, 0.6]]
        assert hmm.transition_mat == [[0.9, 0.1], [0.2, 0.8]]

    def test_index_sets(self, hmm):
        data = conin.hmm.algebraic_chmm._create_index_sets(
            hmm=hmm, observations=["o0", "o1", "o0", "o1", "o0"]
        )
        assert data.E == [
            (-1, -1, 0),
            (-1, -1, 1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
            (2, 0, 0),
            (2, 0, 1),
            (2, 1, 0),
            (2, 1, 1),
            (3, 0, 0),
            (3, 0, 1),
            (3, 1, 0),
            (3, 1, 1),
            (4, 0, -2),
            (4, 1, -2),
        ]
        assert data.F == {(0, 1), (1, 0), (1, 1), (0, 0)}
        assert data.FF == set()
        assert data.Gt == {
            (-1, -1, 0): -0.916290731874155,
            (-1, -1, 1): -0.5108256237659907,
            (0, 0, 0): -0.10536051565782628,
            (0, 0, 1): -2.3025850929940455,
            (0, 1, 0): -1.6094379124341003,
            (0, 1, 1): -0.2231435513142097,
            (1, 0, 0): -0.10536051565782628,
            (1, 0, 1): -2.3025850929940455,
            (1, 1, 0): -1.6094379124341003,
            (1, 1, 1): -0.2231435513142097,
            (2, 0, 0): -0.10536051565782628,
            (2, 0, 1): -2.3025850929940455,
            (2, 1, 0): -1.6094379124341003,
            (2, 1, 1): -0.2231435513142097,
            (3, 0, 0): -0.10536051565782628,
            (3, 0, 1): -2.3025850929940455,
            (3, 1, 0): -1.6094379124341003,
            (3, 1, 1): -0.2231435513142097,
        }
        assert data.Ge == {
            (0, 0): -0.35667494393873245,
            (0, 1): -0.916290731874155,
            (1, 0): -1.2039728043259361,
            (1, 1): -0.5108256237659907,
            (2, 0): -0.35667494393873245,
            (2, 1): -0.916290731874155,
            (3, 0): -1.2039728043259361,
            (3, 1): -0.5108256237659907,
            (4, 0): -0.35667494393873245,
            (4, 1): -0.916290731874155,
        }
