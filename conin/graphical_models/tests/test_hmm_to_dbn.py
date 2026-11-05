import pytest

from conin.graphical_models import create_dbn_from_hmm
from conin.graphical_models import create_bn_from_dbn
import conin.examples.hidden_markov_model as tc


@pytest.mark.parametrize(
    "create_hmm,start,transition,emission",
    [
        (
            tc.create_hmm0,
            {"h0": 1, "h1": 0},
            {"h0": {"h0": 0, "h1": 1}, "h1": {"h0": 0, "h1": 1}},
            {"h0": {"o0": 1, "o1": 0}, "h1": {"o0": 0, "o1": 1}},
        ),
        (
            tc.create_hmm1,
            {"h0": 0.4, "h1": 0.6},
            {"h0": {"h0": 0.9, "h1": 0.1}, "h1": {"h0": 0.2, "h1": 0.8}},
            {"h0": {"o0": 0.7, "o1": 0.3}, "h1": {"o0": 0.4, "o1": 0.6}},
        ),
    ],
)
def test_hmm_to_dbn(create_hmm, start, transition, emission):
    dbn = create_dbn_from_hmm(create_hmm())
    for cpd in dbn.cpds:
        if cpd.node[0] == "H" and cpd.node[1] == 0:
            assert cpd.values == start

        elif cpd.node[0] == "H":
            assert cpd.values == transition

        elif cpd.node[0] == "E":
            assert cpd.values == emission

        else:
            raise RuntimeError("Unexpected node")

    bn = create_bn_from_dbn(dbn=dbn, start=0, stop=3)
    nodes = {cpd.node for cpd in bn.cpds}
    assert nodes == {("H", t) for t in range(4)}.union({("E", t) for t in range(4)})
