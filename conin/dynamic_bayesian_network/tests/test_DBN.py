import pytest
from conin.dynamic_bayesian_network import (
    pyomo_DBN_map_query,
    optimize_pyomo_inference_model,
)

try:
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, DBNInference

    pgmpy_available = True
except Exception as e:
    print(f"pgmpy not available: {e}")
    pgmpy_available = False


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


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple0_ALL():
    """
    Z_0 -> Z_1
    """
    G = simple0_DBN(False)
    model = pyomo_DBN_map_query(pgm=G)  # variables=None, evidence=None
    results = optimize_pyomo_inference_model(model)  # num=1
    assert results.solutions[0].var_values == {("Z", 0): 1, ("Z", 1): 0}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL():
    """
    A -> B
    """
    G = simple1_DBN()
    # assert q == {'A':0, 'B':1}
    model = pyomo_DBN_map_query(pgm=G)  # variables=None, evidence=None
    results = optimize_pyomo_inference_model(model)  # num=1
    assert results.solutions[0].var_values == {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_B():
    """
    A -> B
    """
    G = simple1_DBN()
    # assert q == {'B':0}

    model = pyomo_DBN_map_query(
        pgm=G, evidence={("A", 0): 1}
    )  # variables=None, evidence=None
    results = optimize_pyomo_inference_model(model)  # num=1
    assert results.solutions[0].var_values == {("A", 1): 0, ("B", 0): 0, ("B", 1): 1}
