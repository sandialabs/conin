import pytest

from conin.hmm import create_dbn_from_hmm
from conin.dynamic_bayesian_network import create_bn_from_dbn
import conin.hmm.tests.examples as tc


def test_hmm0_to_dbn():
    dbn = create_dbn_from_hmm(tc.create_hmm0())
    for cpd in dbn.cpds:
        if cpd.node[0] == "H" and cpd.node[1] == 0:
            assert cpd.values == {"h0": 1, "h1": 0}

        elif cpd.node[0] == "H":
            assert cpd.values == {"h0": {"h0": 0, "h1": 1}, "h1": {"h0": 0, "h1": 1}}

        elif cpd.node[0] == "E":
            assert cpd.values == {"h0": {"o0": 1, "o1": 0}, "h1": {"o0": 0, "o1": 1}}

        else:
            raise RuntimeError("Unexpected node")

    bn = create_bn_from_dbn(dbn=dbn, start=0, stop=3)
    nodes = {cpd.node for cpd in bn.cpds}
    assert nodes == {("H", t) for t in range(4)}.union({("E", t) for t in range(4)})


def test_hmm1_to_dbn():
    dbn = create_dbn_from_hmm(tc.create_hmm1())
    for cpd in dbn.cpds:
        if cpd.node[0] == "H" and cpd.node[1] == 0:
            assert cpd.values == {"h0": 0.4, "h1": 0.6}

        elif cpd.node[0] == "H":
            assert cpd.values == {
                "h0": {"h0": 0.9, "h1": 0.1},
                "h1": {"h0": 0.2, "h1": 0.8},
            }

        elif cpd.node[0] == "E":
            assert cpd.values == {
                "h0": {"o0": 0.7, "o1": 0.3},
                "h1": {"o0": 0.4, "o1": 0.6},
            }

        else:
            raise RuntimeError("Unexpected node")
