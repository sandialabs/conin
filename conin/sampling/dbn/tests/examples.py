from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
from conin.bayesian_network.model import DiscreteCPD


def create_ddbn0():
    G = DynamicDiscreteBayesianNetwork()

    states = {"X": ["T", "F"]}
    G.states = states

    dynamic_states = {"A": ["a", "b", "c"], "B": [0, 1, 2, 3], "C": ["t", "f"]}
    G.dynamic_states = dynamic_states

    cpd_start_A = DiscreteCPD(node=("A", 0), values=[0.7, 0.1, 0.2])
    cpd_trans_A = DiscreteCPD(
        node=("A", G.t),
        parents=[("A", G.t - 1), ("B", G.t - 1), "X"],
        values={
            ("a", 0, "T"): [0.7, 0.2, 0.1],
            ("a", 0, "F"): [0.2, 0.1, 0.7],
            ("a", 1, "T"): [0.1, 0.1, 0.8],
            ("a", 1, "F"): [0.9, 0, 0.1],
            ("a", 2, "T"): [0.3, 0.3, 0.4],
            ("a", 2, "F"): [0.4, 0.4, 0.2],
            ("a", 3, "T"): [0.6, 0.1, 0.3],
            ("a", 3, "F"): [0.5, 0.4, 0.1],
            ("b", 0, "T"): [0.2, 0.2, 0.6],
            ("b", 0, "F"): [0.6, 0, 0.4],
            ("b", 1, "T"): [0.3, 0.4, 0.3],
            ("b", 1, "F"): [0, 0.9, 0.1],
            ("b", 2, "T"): [0.5, 0, 0.5],
            ("b", 2, "F"): [0.7, 0.3, 0],
            ("b", 3, "T"): [0.8, 0.1, 0.1],
            ("b", 3, "F"): [0, 0, 1.0],
            ("c", 0, "T"): [0.3, 0.5, 0.2],
            ("c", 0, "F"): [0.2, 0.3, 0.5],
            ("c", 1, "T"): [0.1, 0.7, 0.2],
            ("c", 1, "F"): [0.2, 0.6, 0.2],
            ("c", 2, "T"): [0, 0.5, 0.5],
            ("c", 2, "F"): [0.6, 0.2, 0.2],
            ("c", 3, "T"): [0.1, 0.8, 0.1],
            ("c", 3, "F"): [0.8, 0.1, 0.1],
        },
    )
    cpd_B = DiscreteCPD(
        node=("B", G.t),
        parents=[("A", G.t), ("C", G.t)],
        values={
            ("a", "t"): [0.2, 0.3, 0.3, 0.2],
            ("a", "f"): [0.2, 0.4, 0, 0.4],
            ("b", "t"): [0.6, 0.2, 0.1, 0.1],
            ("b", "f"): [0.8, 0, 0.1, 0.1],
            ("c", "t"): [0.3, 0.3, 0.1, 0.3],
            ("c", "f"): [0, 0.1, 0.2, 0.7],
        },
    )
    cpd_C = DiscreteCPD(node=("C", G.t), values=[0.5, 0.5])
    cpd_X = DiscreteCPD(node="X", values=[0.9, 0.1])
    G.cpds = [cpd_start_A, cpd_trans_A, cpd_B, cpd_C, cpd_X]

    # print(G.check_model())
    return G
