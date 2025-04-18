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
        assert data.G == {
            (-1, -1, 0): -1.2729656758128876,
            (-1, -1, 1): -1.4271163556401456,
            (0, 0, 0): -1.3093333199837625,
            (0, 0, 1): -2.8134107167600364,
            (0, 1, 0): -2.8134107167600364,
            (0, 1, 1): -0.7339691750802004,
            (1, 0, 0): -0.46203545959655873,
            (1, 0, 1): -3.2188758248682006,
            (1, 1, 0): -1.9661128563728327,
            (1, 1, 1): -1.1394342831883648,
            (2, 0, 0): -1.3093333199837625,
            (2, 0, 1): -2.8134107167600364,
            (2, 1, 0): -2.8134107167600364,
            (2, 1, 1): -0.7339691750802004,
            (3, 0, 0): -0.46203545959655873,
            (3, 0, 1): -3.2188758248682006,
            (3, 1, 0): -1.9661128563728327,
            (3, 1, 1): -1.1394342831883648,
        }
