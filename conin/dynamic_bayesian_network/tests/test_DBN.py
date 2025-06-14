import pytest
import pyomo.environ as pyo

try:
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, DBNInference

    pgmpy_available = True
except Exception as e:
    print(f"pgmpy not available: {e}")
    pgmpy_available = False

from conin.dynamic_bayesian_network import (
    create_DBN_map_query_model,
    optimize_map_query_model,
    MapCPD,
)


def simple0_DBN(debug=False):
    G = DBN()
    G.add_edges_from([(("Z", 0), ("Z", 1))])
    z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
    z_trans_cpd = TabularCPD(
        ("Z", 1), 2, [[0.7, 0.8], [0.3, 0.2]], evidence=[("Z", 0)], evidence_card=[2]
    )

    G.add_cpds(z_start_cpd, z_trans_cpd)
    G.initialize_initial_state()
    G.check_model()

    if debug:
        for cpd in G.get_cpds():
            print(cpd)
    return G


def simple2_DBN(debug=False):
    G = DBN()
    G.add_edges_from([(("Z", 0), ("Z", 1))])
    z_start_cpd = MapCPD(variable=("Z", 0), values=[0.5, 0.5])
    z_trans_cpd = MapCPD(
        variable=("Z", 1), evidence=[("Z", 0)], values={0: [0.7, 0.3], 1: [0.8, 0.2]}
    )

    G.add_cpds(z_start_cpd.cpd, z_trans_cpd.cpd)
    G.initialize_initial_state()
    G.check_model()

    if debug:
        for cpd in G.get_cpds():
            print(cpd)
    return G


def simple1_DBN(debug=False):
    G = DBN()
    G.add_nodes_from(["A", "B"])
    G.add_edge(("A", 0), ("B", 0))
    G.add_edge(("A", 0), ("A", 1))
    cpd_start_A = TabularCPD(variable=("A", 0), variable_card=2, values=[[0.9], [0.1]])
    cpd_start_B = TabularCPD(
        variable=("B", 0),
        variable_card=2,
        values=[[0.2, 0.9], [0.8, 0.1]],
        evidence=[("A", 0)],
        evidence_card=[2],
    )
    cpd_trans_A = TabularCPD(
        variable=("A", 1),
        variable_card=2,
        values=[[0.2, 0.9], [0.8, 0.1]],
        evidence=[("A", 0)],
        evidence_card=[2],
    )
    G.add_cpds(cpd_start_A, cpd_start_B, cpd_trans_A)
    G.initialize_initial_state()
    G.check_model()

    if debug:
        for cpd in G.get_cpds():
            print(cpd)
    return G


def simple3_DBN(debug=False):
    G = DBN()
    G.add_nodes_from(["A", "B"])
    G.add_edge(("A", 0), ("B", 0))
    G.add_edge(("A", 0), ("A", 1))
    cpd_start_A = MapCPD(variable=("A", 0), values=[0.9, 0.1])
    cpd_start_B = MapCPD(
        variable=("B", 0),
        evidence=[("A", 0)],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
        # values=[[0.2, 0.9], [0.8, 0.1]],
    )
    cpd_trans_A = MapCPD(
        variable=("A", 1),
        evidence=[("A", 0)],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
        # values=[[0.2, 0.9], [0.8, 0.1]],
    )
    G.add_cpds(cpd_start_A.cpd, cpd_start_B.cpd, cpd_trans_A.cpd)
    G.initialize_initial_state()
    G.check_model()

    if debug:
        for cpd in G.get_cpds():
            print(cpd)
    return G


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple0_ALL():
    """
    Z
    """
    G = simple0_DBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_ALL():
    """
    Z
    """
    G = simple3_DBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL():
    """
    A -> B

    No evidence
    """
    G = simple1_DBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_B():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = simple1_DBN()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    model = create_DBN_map_query_model(
        pgm=G, evidence={("A", 0): 1}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL_constrained():
    """
    A -> B

    No evidence
    """
    G = simple1_DBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    model.c = pyo.ConstraintList()
    model.c.add(model.X[("A", 0), 0] == model.X[("A", 1), 0])
    model.c.add(model.X[("B", 0), 0] == model.X[("B", 1), 0])

    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_ALL():
    """
    A -> B

    No evidence
    """
    G = simple3_DBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_B():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = simple3_DBN()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    model = create_DBN_map_query_model(
        pgm=G, evidence={("A", 0): 1}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_ALL_constrained():
    """
    A -> B

    No evidence
    """
    G = simple1_DBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    model.c = pyo.ConstraintList()
    model.c.add(model.X[("A", 0), 0] == model.X[("A", 1), 0])
    model.c.add(model.X[("B", 0), 0] == model.X[("B", 1), 0])

    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value
