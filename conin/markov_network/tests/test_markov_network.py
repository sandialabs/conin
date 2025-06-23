from math import log
import pyomo.environ as pyo
from conin.markov_network import (
    create_MN_map_query_model,
    optimize_map_query_model,
)
from conin.markov_network.factor_repn import extract_factor_representation, State
from conin.markov_network.inference import create_MN_map_query_model_from_factorial_repn
from . import test_cases

try:
    import pgmpy

    pgmpy_available = True
except Exception as e:
    pgmpy_available = False


def test_example6():
    """
    See Obbens, p.18

    Non-uniform weights are applied to the A_B factors to remove a degeneracy w.r.t. the
    value of variable A.
    """
    S = {"A": [State(0), State(1)], "B": [State(0), State(1)]}

    J = {"A": [0, 1], "B": [0, 1], "A_B": [0, 1, 2, 3]}

    v = {
        ("A", 0, "A"): State(0),
        ("A", 1, "A"): State(1),
        ("B", 0, "B"): State(0),
        ("B", 1, "B"): State(1),
        ("A_B", 0, "A"): State(0),
        ("A_B", 0, "B"): State(0),
        ("A_B", 1, "A"): State(0),
        ("A_B", 1, "B"): State(1),
        ("A_B", 2, "A"): State(1),
        ("A_B", 2, "B"): State(0),
        ("A_B", 3, "A"): State(1),
        ("A_B", 3, "B"): State(1),
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

    model = create_MN_map_query_model_from_factorial_repn(S=S, J=J, v=v, w=w)
    results = optimize_map_query_model(model, solver="glpk")
    assert results.solution.variable_value == {"A": 0, "B": 1}

    if pgmpy_available:
        pgm = test_cases.example6()
        S_, J_, v_, w_ = extract_factor_representation(pgm)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_
        model = create_MN_map_query_model(pgm=pgm)
        results = optimize_map_query_model(model, solver="glpk")
        assert results.solution.variable_value == {"A": 0, "B": 1}


def test_ABC():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights, so the MAP solution is defined by the weights for the
    factors that describe the individual variables.

    The MAP solution is A:2, B:2, C:1.
    """
    S = {
        "A": [State(0), State(1), State(2)],
        "B": [State(0), State(1), State(2)],
        "C": [State(0), State(1), State(2)],
    }

    J = {
        "A": [0, 1, 2],
        "B": [0, 1, 2],
        "C": [0, 1, 2],
        "A_B": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "B_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "A_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    v = {
        ("A", 0, "A"): State(0),
        ("A", 1, "A"): State(1),
        ("A", 2, "A"): State(2),
        ("B", 0, "B"): State(0),
        ("B", 1, "B"): State(1),
        ("B", 2, "B"): State(2),
        ("C", 0, "C"): State(0),
        ("C", 1, "C"): State(1),
        ("C", 2, "C"): State(2),
        ("A_B", 0, "A"): State(0),
        ("A_B", 0, "B"): State(0),
        ("A_B", 1, "A"): State(0),
        ("A_B", 1, "B"): State(1),
        ("A_B", 2, "A"): State(0),
        ("A_B", 2, "B"): State(2),
        ("A_B", 3, "A"): State(1),
        ("A_B", 3, "B"): State(0),
        ("A_B", 4, "A"): State(1),
        ("A_B", 4, "B"): State(1),
        ("A_B", 5, "A"): State(1),
        ("A_B", 5, "B"): State(2),
        ("A_B", 6, "A"): State(2),
        ("A_B", 6, "B"): State(0),
        ("A_B", 7, "A"): State(2),
        ("A_B", 7, "B"): State(1),
        ("A_B", 8, "A"): State(2),
        ("A_B", 8, "B"): State(2),
        ("B_C", 0, "B"): State(0),
        ("B_C", 0, "C"): State(0),
        ("B_C", 1, "B"): State(0),
        ("B_C", 1, "C"): State(1),
        ("B_C", 2, "B"): State(0),
        ("B_C", 2, "C"): State(2),
        ("B_C", 3, "B"): State(1),
        ("B_C", 3, "C"): State(0),
        ("B_C", 4, "B"): State(1),
        ("B_C", 4, "C"): State(1),
        ("B_C", 5, "B"): State(1),
        ("B_C", 5, "C"): State(2),
        ("B_C", 6, "B"): State(2),
        ("B_C", 6, "C"): State(0),
        ("B_C", 7, "B"): State(2),
        ("B_C", 7, "C"): State(1),
        ("B_C", 8, "B"): State(2),
        ("B_C", 8, "C"): State(2),
        ("A_C", 0, "A"): State(0),
        ("A_C", 0, "C"): State(0),
        ("A_C", 1, "A"): State(0),
        ("A_C", 1, "C"): State(1),
        ("A_C", 2, "A"): State(0),
        ("A_C", 2, "C"): State(2),
        ("A_C", 3, "A"): State(1),
        ("A_C", 3, "C"): State(0),
        ("A_C", 4, "A"): State(1),
        ("A_C", 4, "C"): State(1),
        ("A_C", 5, "A"): State(1),
        ("A_C", 5, "C"): State(2),
        ("A_C", 6, "A"): State(2),
        ("A_C", 6, "C"): State(0),
        ("A_C", 7, "A"): State(2),
        ("A_C", 7, "C"): State(1),
        ("A_C", 8, "A"): State(2),
        ("A_C", 8, "C"): State(2),
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

    model = create_MN_map_query_model_from_factorial_repn(S=S, J=J, v=v, w=w)
    results = optimize_map_query_model(model, solver="glpk")
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}

    if pgmpy_available:
        pgm = test_cases.ABC()
        S_, J_, v_, w_ = extract_factor_representation(pgm)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_
        model = create_MN_map_query_model(pgm=pgm)
        results = optimize_map_query_model(model, solver="glpk")
        assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


def test_ABC_constrained():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MAP solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MAP solution is A:0, B:2, C:1.
    """
    S = {
        "A": [State(0), State(1), State(2)],
        "B": [State(0), State(1), State(2)],
        "C": [State(0), State(1), State(2)],
    }

    J = {
        "A": [0, 1, 2],
        "B": [0, 1, 2],
        "C": [0, 1, 2],
        "A_B": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "B_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "A_C": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    v = {
        ("A", 0, "A"): State(0),
        ("A", 1, "A"): State(1),
        ("A", 2, "A"): State(2),
        ("B", 0, "B"): State(0),
        ("B", 1, "B"): State(1),
        ("B", 2, "B"): State(2),
        ("C", 0, "C"): State(0),
        ("C", 1, "C"): State(1),
        ("C", 2, "C"): State(2),
        ("A_B", 0, "A"): State(0),
        ("A_B", 0, "B"): State(0),
        ("A_B", 1, "A"): State(0),
        ("A_B", 1, "B"): State(1),
        ("A_B", 2, "A"): State(0),
        ("A_B", 2, "B"): State(2),
        ("A_B", 3, "A"): State(1),
        ("A_B", 3, "B"): State(0),
        ("A_B", 4, "A"): State(1),
        ("A_B", 4, "B"): State(1),
        ("A_B", 5, "A"): State(1),
        ("A_B", 5, "B"): State(2),
        ("A_B", 6, "A"): State(2),
        ("A_B", 6, "B"): State(0),
        ("A_B", 7, "A"): State(2),
        ("A_B", 7, "B"): State(1),
        ("A_B", 8, "A"): State(2),
        ("A_B", 8, "B"): State(2),
        ("B_C", 0, "B"): State(0),
        ("B_C", 0, "C"): State(0),
        ("B_C", 1, "B"): State(0),
        ("B_C", 1, "C"): State(1),
        ("B_C", 2, "B"): State(0),
        ("B_C", 2, "C"): State(2),
        ("B_C", 3, "B"): State(1),
        ("B_C", 3, "C"): State(0),
        ("B_C", 4, "B"): State(1),
        ("B_C", 4, "C"): State(1),
        ("B_C", 5, "B"): State(1),
        ("B_C", 5, "C"): State(2),
        ("B_C", 6, "B"): State(2),
        ("B_C", 6, "C"): State(0),
        ("B_C", 7, "B"): State(2),
        ("B_C", 7, "C"): State(1),
        ("B_C", 8, "B"): State(2),
        ("B_C", 8, "C"): State(2),
        ("A_C", 0, "A"): State(0),
        ("A_C", 0, "C"): State(0),
        ("A_C", 1, "A"): State(0),
        ("A_C", 1, "C"): State(1),
        ("A_C", 2, "A"): State(0),
        ("A_C", 2, "C"): State(2),
        ("A_C", 3, "A"): State(1),
        ("A_C", 3, "C"): State(0),
        ("A_C", 4, "A"): State(1),
        ("A_C", 4, "C"): State(1),
        ("A_C", 5, "A"): State(1),
        ("A_C", 5, "C"): State(2),
        ("A_C", 6, "A"): State(2),
        ("A_C", 6, "C"): State(0),
        ("A_C", 7, "A"): State(2),
        ("A_C", 7, "C"): State(1),
        ("A_C", 8, "A"): State(2),
        ("A_C", 8, "C"): State(2),
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

    model = create_MN_map_query_model_from_factorial_repn(S=S, J=J, v=v, w=w)

    # Constrain the inference to ensure that all variables have different
    # values
    def diff_(M, s):
        s = State(s)
        return M.x["A", s] + M.x["B", s] + M.x["C", s] <= 1

    model.diff = pyo.Constraint([0, 1, 2], rule=diff_)

    results = optimize_map_query_model(model, solver="glpk")
    assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}

    if pgmpy_available:
        # Double-check that we can extract the right factor representation
        cpgm = test_cases.ABC_constrained()
        S_, J_, v_, w_ = extract_factor_representation(cpgm.pgm)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_

        # Constrain the inference to ensure that all variables have different
        # values
        pgm = test_cases.ABC()
        model = create_MN_map_query_model(pgm=pgm)

        def diff_(M, s):
            s = State(s)
            return M.x["A", s] + M.x["B", s] + M.x["C", s] <= 1

        model.diff = pyo.Constraint([0, 1, 2], rule=diff_)

        results = optimize_map_query_model(model, solver="glpk")
        assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}

        # Constrain the inference to ensure that all variables have different
        # values
        cpgm = test_cases.ABC_constrained()
        results = optimize_map_query_model(
            cpgm.create_map_query_model(), solver="glpk")
        assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}
