import pytest
from conin.bayesian_network import create_BN_map_query_model, optimize_map_query_model

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    pgmpy_available = True
except Exception as e:
    print(f"pgmpy not available: {e}")
    pgmpy_available = False


def cancer_BN(debug=False):
    # Step 1: Define the network structure.
    cancer_model = DiscreteBayesianNetwork(
        [
            ("Pollution", "Cancer"),
            ("Smoker", "Cancer"),
            ("Cancer", "Xray"),
            ("Cancer", "Dyspnoea"),
        ]
    )

    # Step 2: Define the CPDs.
    cpd_poll = TabularCPD(variable="Pollution", variable_card=2, values=[[0.9], [0.1]])
    cpd_smoke = TabularCPD(variable="Smoker", variable_card=2, values=[[0.3], [0.7]])
    cpd_cancer = TabularCPD(
        variable="Cancer",
        variable_card=2,
        values=[[0.03, 0.05, 0.001, 0.02], [0.97, 0.95, 0.999, 0.98]],
        evidence=["Smoker", "Pollution"],
        evidence_card=[2, 2],
    )
    cpd_xray = TabularCPD(
        variable="Xray",
        variable_card=2,
        values=[[0.9, 0.2], [0.1, 0.8]],
        evidence=["Cancer"],
        evidence_card=[2],
    )
    cpd_dysp = TabularCPD(
        variable="Dyspnoea",
        variable_card=2,
        values=[[0.65, 0.3], [0.35, 0.7]],
        evidence=["Cancer"],
        evidence_card=[2],
    )

    # Step 3: Add the CPDs to the model.
    if debug:
        print(cpd_poll)
        print(cpd_smoke)
        print(cpd_cancer)
        print(cpd_xray)
        print(cpd_dysp)
    cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)

    # Step 4: Check if the model is correctly defined.
    cancer_model.check_model()
    return cancer_model


def simple1_BN(debug=False):
    G = DiscreteBayesianNetwork()
    G.add_nodes_from(["A", "B"])
    G.add_edge("A", "B")
    cpd_A = TabularCPD(variable="A", variable_card=2, values=[[0.9], [0.1]])
    cpd_B = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.2, 0.9], [0.8, 0.1]],
        evidence=["A"],
        evidence_card=[2],
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
    G.add_cpds(cpd_A, cpd_B)
    G.check_model()
    return G


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL():
    """
    A -> B
    """
    G = simple1_BN()
    infer = VariableElimination(G)
    q = infer.map_query(variables=["A", "B"])
    assert q == {"A": 0, "B": 1}

    model = create_BN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model)  # num=1
    assert results.solutions[0].var_values == q


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_B():
    """
    A -> B
    """
    G = simple1_BN()
    infer = VariableElimination(G)
    q = infer.map_query(variables=["B"], evidence={"A": 1})
    assert q == {"B": 0}

    model = create_BN_map_query_model(
        pgm=G, evidence={"A": 1}
    )  # variables=None, evidence=None
    # model.pprint()
    results = optimize_map_query_model(model)  # num=1
    assert results.solutions[0].var_values == q


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer_ALL():
    """
    Cancer model from pgmpy examples
    """
    G = cancer_BN()
    infer = VariableElimination(G)
    q = infer.map_query(variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"])
    assert q == {"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1}

    model = create_BN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model)  # num=1
    assert results.solutions[0].var_values == q


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer_Cancer():
    """
    Cancer model from pgmpy examples
    """
    G = cancer_BN(True)
    infer = VariableElimination(G)
    q = infer.map_query(
        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"], evidence={"Cancer": 0}
    )
    assert q == {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}

    model = create_BN_map_query_model(
        pgm=G, evidence={"Cancer": 0}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model)  # num=1
    assert results.solutions[0].var_values == q
