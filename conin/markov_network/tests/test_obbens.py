from math import log
import pyomo.environ as pyo

from conin.util import try_import
from conin.markov_network import (
    create_MN_map_query_model,
    optimize_map_query_model,
)
from conin.markov_network.factor_repn import (
    extract_factor_representation,
    State,
)
from conin.markov_network.inference import (
    create_MN_map_query_model_from_factorial_repn,
)
from conin.markov_network.model import convert_to_DiscreteMarkovNetwork

from . import examples

with try_import() as pgmpy_available:
    import pgmpy


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

    if True:
        pgm = examples.example6_conin()
        pgm.check_model()
        S_, J_, v_, w_ = extract_factor_representation(pgm)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_
        model = create_MN_map_query_model(pgm=pgm)
        results = optimize_map_query_model(model, solver="glpk")
        assert results.solution.variable_value == {"A": 0, "B": 1}

    if pgmpy_available:
        pgm = examples.example6_pgmpy()
        pgm = convert_to_DiscreteMarkovNetwork(pgm)
        S_, J_, v_, w_ = extract_factor_representation(pgm)
        assert S == S_
        assert J == J_
        assert v == v_
        assert w == w_
        model = create_MN_map_query_model(pgm=pgm)
        results = optimize_map_query_model(model, solver="glpk")
        assert results.solution.variable_value == {"A": 0, "B": 1}
