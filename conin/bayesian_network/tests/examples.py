import pyomo.environ as pyo
from conin.bayesian_network import (
    ConstrainedDiscreteBayesianNetwork,
    MapCPD,
)

try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
except Exception as e:
    pass


def cancer1_BN(debug=False):
    """
    Cancer example using TabularCPD
    """
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


def cancer1_BN_constrained(debug=False):
    pgm = cancer1_BN(debug=debug)

    def constraints(model, data):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)
        return model

    return ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)


def cancer2_BN(debug=False):
    """
    Cancer example using MapCPD
    """
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
    cpd_poll = MapCPD(variable="Pollution", values=[0.9, 0.1])
    cpd_smoke = MapCPD(variable="Smoker", values=[0.3, 0.7])
    cpd_cancer = MapCPD(
        variable="Cancer",
        evidence=["Smoker", "Pollution"],
        values={
            (0, 0): [0.03, 0.97],
            (0, 1): [0.05, 0.95],
            (1, 0): [0.001, 0.999],
            (1, 1): [0.02, 0.98],
        },
        # values=[[0.03, 0.05, 0.001, 0.02], [0.97, 0.95, 0.999, 0.98]],
    )
    cpd_xray = MapCPD(
        variable="Xray",
        evidence=["Cancer"],
        values={0: [0.9, 0.1], 1: [0.2, 0.8]},
        # values=[[0.9, 0.2], [0.1, 0.8]],
    )
    cpd_dysp = MapCPD(
        variable="Dyspnoea",
        evidence=["Cancer"],
        values={0: [0.65, 0.35], 1: [0.3, 0.7]},
        # values=[[0.65, 0.3], [0.35, 0.7]],
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


def cancer2_BN_constrained(debug=False):
    pgm = cancer2_BN(debug=debug)

    def constraints(model, data):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)
        return model

    return ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)


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


def simple2_BN(debug=False):
    G = DiscreteBayesianNetwork()
    G.add_nodes_from(["A", "B"])
    G.add_edge("A", "B")
    cpd_A = MapCPD(variable="A", values=[0.9, 0.1])
    cpd_B = MapCPD(
        variable="B",
        evidence=["A"],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
    G.add_cpds(cpd_A, cpd_B)
    G.check_model()
    return G


def DBDA_5_1(debug=False):
    """
    Model used in exercise 5.1 from "Doing Bayesian Data Analysis" by John K. Kruschke:
    https://sites.google.com/site/doingbayesiandataanalysis/exercises
    """
    p_disease_present = 0.001
    p_test_positive_given_disease_present = 0.99
    p_test_positive_given_disease_absent = 0.05

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(["disease-state", "test-result1", "test-result2"])
    model.add_edge("disease-state", "test-result1")
    model.add_edge("disease-state", "test-result2")

    disease_state_CPD = TabularCPD(
        variable="disease-state",
        variable_card=2,
        values=[[p_disease_present], [1.0 - p_disease_present]],
    )

    test_result_CPD_1 = TabularCPD(
        variable="test-result1",
        variable_card=2,
        values=[
            [
                p_test_positive_given_disease_present,
                p_test_positive_given_disease_absent,
            ],
            [
                (1 - p_test_positive_given_disease_present),
                (1 - p_test_positive_given_disease_absent),
            ],
        ],
        evidence=["disease-state"],
        evidence_card=[2],
    )

    test_result_CPD_2 = TabularCPD(
        variable="test-result2",
        variable_card=2,
        values=[
            [
                p_test_positive_given_disease_present,
                p_test_positive_given_disease_absent,
            ],
            [
                (1 - p_test_positive_given_disease_present),
                (1 - p_test_positive_given_disease_absent),
            ],
        ],
        evidence=["disease-state"],
        evidence_card=[2],
    )

    model.add_cpds(disease_state_CPD, test_result_CPD_1, test_result_CPD_2)
    model.check_model()

    if debug:
        print(disease_state_CPD)
        print(test_result_CPD_1)
        print(test_result_CPD_2)

    return model
