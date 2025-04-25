from math import log
import numpy as np
import pyomo.environ as pyo
from conin.markov_network import (
    create_MN_map_query_model,
    optimize_map_query_model,
    extract_factor_representation,
)

try:
    from pgmpy.models import MarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor

    pgmpy_available = True
except Exception as e:
    print(f"pgmpy not available: {e}")
    pgmpy_available = False


def test_example6():
    """
    See Obbens, p.18

    Non-uniform weights are applied to the A_B factors to remove a degeneracy w.r.t. the
    value of variable A.
    """
    S = {"A": [0, 1], "B": [0, 1]}

    J = {"A": [0, 1], "B": [0, 1], "A_B": [0, 1, 2, 3]}

    v = {
        ("A", 0, "A"): 0,
        ("A", 1, "A"): 1,
        ("B", 0, "B"): 0,
        ("B", 1, "B"): 1,
        ("A_B", 0, "A"): 0,
        ("A_B", 0, "B"): 0,
        ("A_B", 1, "A"): 0,
        ("A_B", 1, "B"): 1,
        ("A_B", 2, "A"): 1,
        ("A_B", 2, "B"): 0,
        ("A_B", 3, "A"): 1,
        ("A_B", 3, "B"): 1,
    }

    w = {
        ("A", 0): -log(2),
        ("A", 1): -log(2),
        ("B", 0): -log(3),
        ("B", 1): log(2 / 3),
        ("A_B", 0): -log(6),
        ("A_B", 1): -log(2),
        ("A_B", 2): -log(6),
        ("A_B", 3): -log(6),
    }

    model = create_MN_map_query_model(S=S, J=J, v=v, w=w)
    results = optimize_map_query_model(model)
    assert results.solutions[0].var_values == {"A": 0, "B": 1}

    if pgmpy_available:
        G = MarkovNetwork()
        G.add_nodes_from(["A", "B"])
        G.add_edge("A", "B")
        f1 = DiscreteFactor(["A"], [2], [1, 1])
        f2 = DiscreteFactor(["B"], [2], [1, 2])
        f3 = DiscreteFactor(["A", "B"], [2, 2], [1, 3, 1, 1])
        G.add_factors(f1, f2, f3)
        S_, J_, v_, w_ = extract_factor_representation(G)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_
        model = create_MN_map_query_model(pgm=G)
        results = optimize_map_query_model(model)
        assert results.solutions[0].var_values == {"A": 0, "B": 1}


def test_ABC():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights, so the MAP solution is defined by the weights for the
    factors that describe the individual variables.

    The MAP solution is A:2, B:2, C:1.
    """
    S = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]}

    J = {
        "A": [0, 1, 2],
        "B": [0, 1, 2],
        "C": [0, 1, 2],
        "A_B": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "B_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "A_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    v = {
        ("A", 0, "A"): 0,
        ("A", 1, "A"): 1,
        ("A", 2, "A"): 2,
        ("B", 0, "B"): 0,
        ("B", 1, "B"): 1,
        ("B", 2, "B"): 2,
        ("C", 0, "C"): 0,
        ("C", 1, "C"): 1,
        ("C", 2, "C"): 2,
        ("A_B", 0, "A"): 0,
        ("A_B", 0, "B"): 0,
        ("A_B", 1, "A"): 0,
        ("A_B", 1, "B"): 1,
        ("A_B", 2, "A"): 0,
        ("A_B", 2, "B"): 2,
        ("A_B", 3, "A"): 1,
        ("A_B", 3, "B"): 0,
        ("A_B", 4, "A"): 1,
        ("A_B", 4, "B"): 1,
        ("A_B", 5, "A"): 1,
        ("A_B", 5, "B"): 2,
        ("A_B", 6, "A"): 2,
        ("A_B", 6, "B"): 0,
        ("A_B", 7, "A"): 2,
        ("A_B", 7, "B"): 1,
        ("A_B", 8, "A"): 2,
        ("A_B", 8, "B"): 2,
        ("B_C", 0, "B"): 0,
        ("B_C", 0, "C"): 0,
        ("B_C", 1, "B"): 0,
        ("B_C", 1, "C"): 1,
        ("B_C", 2, "B"): 0,
        ("B_C", 2, "C"): 2,
        ("B_C", 3, "B"): 1,
        ("B_C", 3, "C"): 0,
        ("B_C", 4, "B"): 1,
        ("B_C", 4, "C"): 1,
        ("B_C", 5, "B"): 1,
        ("B_C", 5, "C"): 2,
        ("B_C", 6, "B"): 2,
        ("B_C", 6, "C"): 0,
        ("B_C", 7, "B"): 2,
        ("B_C", 7, "C"): 1,
        ("B_C", 8, "B"): 2,
        ("B_C", 8, "C"): 2,
        ("A_C", 0, "A"): 0,
        ("A_C", 0, "C"): 0,
        ("A_C", 1, "A"): 0,
        ("A_C", 1, "C"): 1,
        ("A_C", 2, "A"): 0,
        ("A_C", 2, "C"): 2,
        ("A_C", 3, "A"): 1,
        ("A_C", 3, "C"): 0,
        ("A_C", 4, "A"): 1,
        ("A_C", 4, "C"): 1,
        ("A_C", 5, "A"): 1,
        ("A_C", 5, "C"): 2,
        ("A_C", 6, "A"): 2,
        ("A_C", 6, "C"): 0,
        ("A_C", 7, "A"): 2,
        ("A_C", 7, "C"): 1,
        ("A_C", 8, "A"): 2,
        ("A_C", 8, "C"): 2,
    }

    w = {
        ("A", 0): -log(4),
        ("A", 1): -log(4),
        ("A", 2): -log(2),
        ("B", 0): -log(5),
        ("B", 1): -log(5),
        ("B", 2): log(3 / 5),
        ("C", 0): -log(4),
        ("C", 1): -log(2),
        ("C", 2): -log(4),
        ("A_B", 0): -log(9),
        ("A_B", 1): -log(9),
        ("A_B", 2): -log(9),
        ("A_B", 3): -log(9),
        ("A_B", 4): -log(9),
        ("A_B", 5): -log(9),
        ("A_B", 6): -log(9),
        ("A_B", 7): -log(9),
        ("A_B", 8): -log(9),
        ("B_C", 0): -log(9),
        ("B_C", 1): -log(9),
        ("B_C", 2): -log(9),
        ("B_C", 3): -log(9),
        ("B_C", 4): -log(9),
        ("B_C", 5): -log(9),
        ("B_C", 6): -log(9),
        ("B_C", 7): -log(9),
        ("B_C", 8): -log(9),
        ("A_C", 0): -log(9),
        ("A_C", 1): -log(9),
        ("A_C", 2): -log(9),
        ("A_C", 3): -log(9),
        ("A_C", 4): -log(9),
        ("A_C", 5): -log(9),
        ("A_C", 6): -log(9),
        ("A_C", 7): -log(9),
        ("A_C", 8): -log(9),
    }

    model = create_MN_map_query_model(S=S, J=J, v=v, w=w)
    results = optimize_map_query_model(model)
    assert results.solutions[0].var_values == {"A": 2, "B": 2, "C": 1}

    if pgmpy_available:
        G = MarkovNetwork()
        G.add_nodes_from(["A", "B", "C"])
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("A", "C")
        f1 = DiscreteFactor(["A"], [3], [1, 1, 2])
        f2 = DiscreteFactor(["B"], [3], [1, 1, 3])
        f3 = DiscreteFactor(["C"], [3], [1, 2, 1])
        f4 = DiscreteFactor(["A", "B"], [3, 3], np.ones(9))
        f5 = DiscreteFactor(["B", "C"], [3, 3], np.ones(9))
        f6 = DiscreteFactor(["A", "C"], [3, 3], np.ones(9))
        G.add_factors(f1, f2, f3, f4, f5, f6)
        S_, J_, v_, w_ = extract_factor_representation(G)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_
        model = create_MN_map_query_model(pgm=G)
        results = optimize_map_query_model(model)
        assert results.solutions[0].var_values == {"A": 2, "B": 2, "C": 1}


def test_ABC_constrained():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MAP solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MAP solution is A:0, B:2, C:1.
    """
    S = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]}

    J = {
        "A": [0, 1, 2],
        "B": [0, 1, 2],
        "C": [0, 1, 2],
        "A_B": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "B_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "A_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    v = {
        ("A", 0, "A"): 0,
        ("A", 1, "A"): 1,
        ("A", 2, "A"): 2,
        ("B", 0, "B"): 0,
        ("B", 1, "B"): 1,
        ("B", 2, "B"): 2,
        ("C", 0, "C"): 0,
        ("C", 1, "C"): 1,
        ("C", 2, "C"): 2,
        ("A_B", 0, "A"): 0,
        ("A_B", 0, "B"): 0,
        ("A_B", 1, "A"): 0,
        ("A_B", 1, "B"): 1,
        ("A_B", 2, "A"): 0,
        ("A_B", 2, "B"): 2,
        ("A_B", 3, "A"): 1,
        ("A_B", 3, "B"): 0,
        ("A_B", 4, "A"): 1,
        ("A_B", 4, "B"): 1,
        ("A_B", 5, "A"): 1,
        ("A_B", 5, "B"): 2,
        ("A_B", 6, "A"): 2,
        ("A_B", 6, "B"): 0,
        ("A_B", 7, "A"): 2,
        ("A_B", 7, "B"): 1,
        ("A_B", 8, "A"): 2,
        ("A_B", 8, "B"): 2,
        ("B_C", 0, "B"): 0,
        ("B_C", 0, "C"): 0,
        ("B_C", 1, "B"): 0,
        ("B_C", 1, "C"): 1,
        ("B_C", 2, "B"): 0,
        ("B_C", 2, "C"): 2,
        ("B_C", 3, "B"): 1,
        ("B_C", 3, "C"): 0,
        ("B_C", 4, "B"): 1,
        ("B_C", 4, "C"): 1,
        ("B_C", 5, "B"): 1,
        ("B_C", 5, "C"): 2,
        ("B_C", 6, "B"): 2,
        ("B_C", 6, "C"): 0,
        ("B_C", 7, "B"): 2,
        ("B_C", 7, "C"): 1,
        ("B_C", 8, "B"): 2,
        ("B_C", 8, "C"): 2,
        ("A_C", 0, "A"): 0,
        ("A_C", 0, "C"): 0,
        ("A_C", 1, "A"): 0,
        ("A_C", 1, "C"): 1,
        ("A_C", 2, "A"): 0,
        ("A_C", 2, "C"): 2,
        ("A_C", 3, "A"): 1,
        ("A_C", 3, "C"): 0,
        ("A_C", 4, "A"): 1,
        ("A_C", 4, "C"): 1,
        ("A_C", 5, "A"): 1,
        ("A_C", 5, "C"): 2,
        ("A_C", 6, "A"): 2,
        ("A_C", 6, "C"): 0,
        ("A_C", 7, "A"): 2,
        ("A_C", 7, "C"): 1,
        ("A_C", 8, "A"): 2,
        ("A_C", 8, "C"): 2,
    }

    w = {
        ("A", 0): -log(4),
        ("A", 1): -log(4),
        ("A", 2): -log(2),
        ("B", 0): -log(5),
        ("B", 1): -log(5),
        ("B", 2): log(3 / 5),
        ("C", 0): -log(4),
        ("C", 1): -log(2),
        ("C", 2): -log(4),
        ("A_B", 0): -log(9),
        ("A_B", 1): -log(9),
        ("A_B", 2): -log(9),
        ("A_B", 3): -log(9),
        ("A_B", 4): -log(9),
        ("A_B", 5): -log(9),
        ("A_B", 6): -log(9),
        ("A_B", 7): -log(9),
        ("A_B", 8): -log(9),
        ("B_C", 0): -log(9),
        ("B_C", 1): -log(9),
        ("B_C", 2): -log(9),
        ("B_C", 3): -log(9),
        ("B_C", 4): -log(9),
        ("B_C", 5): -log(9),
        ("B_C", 6): -log(9),
        ("B_C", 7): -log(9),
        ("B_C", 8): -log(9),
        ("A_C", 0): -log(9),
        ("A_C", 1): -log(9),
        ("A_C", 2): -log(9),
        ("A_C", 3): -log(9),
        ("A_C", 4): -log(9),
        ("A_C", 5): -log(9),
        ("A_C", 6): -log(9),
        ("A_C", 7): -log(9),
        ("A_C", 8): -log(9),
    }

    model = create_MN_map_query_model(S=S, J=J, v=v, w=w)

    # Constrain the inference to ensure that all variables have different values
    def diff_(M, s):
        return M.x["A", s] + M.x["B", s] + M.x["C", s] <= 1

    model.diff = pyo.Constraint([0, 1, 2], rule=diff_)

    results = optimize_map_query_model(model)
    assert results.solutions[0].var_values == {"A": 0, "B": 2, "C": 1}

    if pgmpy_available:
        G = MarkovNetwork()
        G.add_nodes_from(["A", "B", "C"])
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("A", "C")
        f1 = DiscreteFactor(["A"], [3], [1, 1, 2])
        f2 = DiscreteFactor(["B"], [3], [1, 1, 3])
        f3 = DiscreteFactor(["C"], [3], [1, 2, 1])
        f4 = DiscreteFactor(["A", "B"], [3, 3], np.ones(9))
        f5 = DiscreteFactor(["B", "C"], [3, 3], np.ones(9))
        f6 = DiscreteFactor(["A", "C"], [3, 3], np.ones(9))
        G.add_factors(f1, f2, f3, f4, f5, f6)
        S_, J_, v_, w_ = extract_factor_representation(G)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_
        model = create_MN_map_query_model(pgm=G)

        # Constrain the inference to ensure that all variables have different values
        def diff_(M, s):
            return M.X["A", s] + M.X["B", s] + M.X["C", s] <= 1

        model.diff = pyo.Constraint([0, 1, 2], rule=diff_)

        results = optimize_map_query_model(model)
        assert results.solutions[0].var_values == {"A": 0, "B": 2, "C": 1}
