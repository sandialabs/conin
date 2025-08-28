import pandas as pd
import numpy as np
import pyomo.environ as pyo

from conin.util import try_import
from conin.bayesian_network import (
    DiscreteBayesianNetwork,
    ConstrainedDiscreteBayesianNetwork,
    DiscreteCPD,
)

with try_import() as pgmpy_available:
    from pgmpy.models import DiscreteBayesianNetwork as pgmpy_DiscreteBayesianNetwork
    from pgmpy.estimators.MLE import (
        MaximumLikelihoodEstimator as pgmpy_MaximumLikelihoodEstimator,
    )
    from pgmpy.factors.discrete import TabularCPD as pgmpy_TabularCPD
    from conin.common.pgmpy import MapCPD


#
# cancer
#


def cancer1_BN_conin(debug=False):
    """
    Cancer example using conin
    """
    # Step 1: Define the network structure.
    cancer_model = DiscreteBayesianNetwork()

    cancer_model.states = {
        "Cancer": [0, 1],
        "Dyspnoea": [0, 1],
        "Pollution": [0, 1],
        "Smoker": [0, 1],
        "Xray": [0, 1],
    }

    # Step 2: Define the CPDs.
    cpd_poll = DiscreteCPD(variable="Pollution", values=[0.9, 0.1])
    cpd_smoke = DiscreteCPD(variable="Smoker", values=[0.3, 0.7])
    cpd_cancer = DiscreteCPD(
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
    cpd_xray = DiscreteCPD(
        variable="Xray",
        evidence=["Cancer"],
        values={0: [0.9, 0.1], 1: [0.2, 0.8]},
        # values=[[0.9, 0.2], [0.1, 0.8]],
    )
    cpd_dysp = DiscreteCPD(
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
    cancer_model.cpds = [cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp]

    # Step 4: Check if the model is correctly defined.
    cancer_model.check_model()
    return cancer_model


def cancer1_BN_pgmpy(debug=False):
    """
    Cancer example using pgmpy
    """
    # Step 1: Define the network structure.
    cancer_model = pgmpy_DiscreteBayesianNetwork(
        [
            ("Pollution", "Cancer"),
            ("Smoker", "Cancer"),
            ("Cancer", "Xray"),
            ("Cancer", "Dyspnoea"),
        ]
    )

    # Step 2: Define the CPDs.
    cpd_poll = pgmpy_TabularCPD(
        variable="Pollution", variable_card=2, values=[[0.9], [0.1]]
    )
    cpd_smoke = pgmpy_TabularCPD(
        variable="Smoker", variable_card=2, values=[[0.3], [0.7]]
    )
    cpd_cancer = pgmpy_TabularCPD(
        variable="Cancer",
        variable_card=2,
        values=[[0.03, 0.05, 0.001, 0.02], [0.97, 0.95, 0.999, 0.98]],
        evidence=["Smoker", "Pollution"],
        evidence_card=[2, 2],
    )
    cpd_xray = pgmpy_TabularCPD(
        variable="Xray",
        variable_card=2,
        values=[[0.9, 0.2], [0.1, 0.8]],
        evidence=["Cancer"],
        evidence_card=[2],
    )
    cpd_dysp = pgmpy_TabularCPD(
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
    cancer_model.cpds = [cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp]

    # Step 4: Check if the model is correctly defined.
    cancer_model.check_model()
    return cancer_model


def cancer2_BN_pgmpy(debug=False):
    """
    Cancer example using pgmpy with MapCPD
    """
    # Step 1: Define the network structure.
    cancer_model = pgmpy_DiscreteBayesianNetwork(
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
    cancer_model.cpds = [cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp]

    # Step 4: Check if the model is correctly defined.
    cancer_model.check_model()
    return cancer_model


#
# cancer constrained
#


def cancer1_BN_constrained_conin(debug=False):
    pgm = cancer1_BN_conin(debug=debug)

    def constraints(model, data):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)
        return model

    return ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)


def cancer1_BN_constrained_pgmpy(debug=False):
    pgm = cancer1_BN_pgmpy(debug=debug)

    def constraints(model, data):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)
        return model

    return ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)


def cancer2_BN_constrained_pgmpy(debug=False):
    pgm = cancer2_BN_pgmpy(debug=debug)

    def constraints(model, data):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)
        return model

    return ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)


#
# simple1
#


def simple1_BN_conin(debug=False):
    G = DiscreteBayesianNetwork()
    G.states = {"A": [0, 1], "B": [0, 1]}
    cpd_A = DiscreteCPD(variable="A", values=[0.9, 0.1])
    cpd_B = DiscreteCPD(
        variable="B",
        evidence=["A"],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
    G.cpds = [cpd_A, cpd_B]
    G.check_model()
    return G


def simple1_BN_pgmpy(debug=False):
    G = pgmpy_DiscreteBayesianNetwork()
    G.add_nodes_from(["A", "B"])
    G.add_edge("A", "B")
    cpd_A = pgmpy_TabularCPD(variable="A", variable_card=2, values=[[0.9], [0.1]])
    cpd_B = pgmpy_TabularCPD(
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


def simple2_BN_pgmpy(debug=False):
    G = pgmpy_DiscreteBayesianNetwork()
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


#
# DBDA_5_1
#


def DBDA_5_1_conin(debug=False):
    """
    Model used in exercise 5.1 from "Doing Bayesian Data Analysis" by John K. Kruschke:
    https://sites.google.com/site/doingbayesiandataanalysis/exercises
    """
    p_disease_present = 0.001
    p_test_positive_given_disease_present = 0.99
    p_test_positive_given_disease_absent = 0.05

    model = DiscreteBayesianNetwork()

    model.states = {
        "disease-state": [0, 1],
        "test-result1": [0, 1],
        "test-result2": [0, 1],
    }

    disease_state_CPD = DiscreteCPD(
        variable="disease-state",
        values=[p_disease_present, 1.0 - p_disease_present],
    )

    test_result_CPD_1 = DiscreteCPD(
        variable="test-result1",
        evidence=["disease-state"],
        values={
            0: [
                p_test_positive_given_disease_present,
                1 - p_test_positive_given_disease_present,
            ],
            1: [
                p_test_positive_given_disease_absent,
                1 - p_test_positive_given_disease_absent,
            ],
        },
    )

    test_result_CPD_2 = DiscreteCPD(
        variable="test-result2",
        evidence=["disease-state"],
        values={
            0: [
                p_test_positive_given_disease_present,
                1 - p_test_positive_given_disease_present,
            ],
            1: [
                p_test_positive_given_disease_absent,
                1 - p_test_positive_given_disease_absent,
            ],
        },
    )

    model.cpds = [disease_state_CPD, test_result_CPD_1, test_result_CPD_2]
    model.check_model()

    if debug:
        print(disease_state_CPD)
        print(test_result_CPD_1)
        print(test_result_CPD_2)

    return model


def DBDA_5_1_pgmpy(debug=False):
    """
    Model used in exercise 5.1 from "Doing Bayesian Data Analysis" by John K. Kruschke:
    https://sites.google.com/site/doingbayesiandataanalysis/exercises
    """
    p_disease_present = 0.001
    p_test_positive_given_disease_present = 0.99
    p_test_positive_given_disease_absent = 0.05

    model = pgmpy_DiscreteBayesianNetwork()
    model.add_nodes_from(["disease-state", "test-result1", "test-result2"])
    model.add_edge("disease-state", "test-result1")
    model.add_edge("disease-state", "test-result2")

    disease_state_CPD = pgmpy_TabularCPD(
        variable="disease-state",
        variable_card=2,
        values=[[p_disease_present], [1.0 - p_disease_present]],
    )

    test_result_CPD_1 = pgmpy_TabularCPD(
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

    test_result_CPD_2 = pgmpy_TabularCPD(
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


#
# holmes
#


def holmes_conin(debug=False):
    """
    Adapted from Lecture Notes by Alice Gao.

    W - Does Watson call Holmes?
    G - Does Gibbon call Holmes?
    A - Does Alarm go off?
    B - Does a burglary happen?
    E - Does an earthquake happen?
    R - Does he hear about an earthquake on the radio?

    A. Gao. "Lecture 13: Variable Elimination Algorithm", 2021.
    https://cs.uwaterloo.ca/~a23gao/cs486686_f18/schedule.shtml
    """
    G = DiscreteBayesianNetwork()
    G.states = {
        "W": ["w", "-w"],
        "G": ["g", "-g"],
        "A": ["a", "-a"],
        "B": ["b", "-b"],
        "E": ["e", "-e"],
        "R": ["r", "-r"],
    }

    cpd_E = DiscreteCPD(variable="E", values={"e": 0.0003, "-e": 0.9997})
    cpd_B = DiscreteCPD(variable="B", values={"b": 0.0001, "-b": 0.9999})
    cpd_R = DiscreteCPD(
        variable="R",
        evidence=["E"],
        values={"e": {"r": 0.0002, "-r": 0.9998}, "-e": {"r": 0.9, "-r": 0.1}},
    )
    cpd_A = DiscreteCPD(
        variable="A",
        evidence=["E", "B"],
        values={
            ("-e", "-b"): {"a": 0.01, "-a": 0.99},
            ("e", "-b"): {"a": 0.2, "-a": 0.8},
            ("-e", "b"): {"a": 0.95, "-a": 0.05},
            ("e", "b"): {"a": 0.96, "-a": 0.04},
        },
    )
    cpd_W = DiscreteCPD(
        variable="W",
        evidence=["A"],
        values={"-a": {"w": 0.4, "-w": 0.6}, "a": {"w": 0.8, "-w": 0.2}},
    )
    cpd_G = DiscreteCPD(
        variable="G",
        evidence=["A"],
        values={"-a": {"g": 0.04, "-g": 0.96}, "a": {"g": 0.4, "-g": 0.6}},
    )

    if debug:
        print(cpd_E)
        print(cpd_B)
        print(cpd_R)
        print(cpd_A)
        print(cpd_W)
        print(cpd_G)
    G.cpds = [cpd_E, cpd_B, cpd_R, cpd_A, cpd_W, cpd_G]
    G.check_model()
    return G


def holmes_pgmpy(debug=False):
    """
    Adapted from Lecture Notes by Alice Gao.

    W - Does Watson call Holmes?
    G - Does Gibbon call Holmes?
    A - Does Alarm go off?
    B - Does a burglary happen?
    E - Does an earthquake happen?
    R - Does he hear about an earthquake on the radio?

    A. Gao. "Lecture 13: Variable Elimination Algorithm", 2021.
    https://cs.uwaterloo.ca/~a23gao/cs486686_f18/schedule.shtml
    """
    G = pgmpy_DiscreteBayesianNetwork()
    G.add_nodes_from(["E", "B", "R", "A", "W", "G"])
    G.add_edge("E", "R")
    G.add_edge("E", "A")
    G.add_edge("B", "A")
    G.add_edge("A", "W")
    G.add_edge("A", "G")
    cpd_E = MapCPD(variable="E", values={"e": 0.0003, "-e": 0.9997})
    cpd_B = MapCPD(variable="B", values={"b": 0.0001, "-b": 0.9999})
    cpd_R = MapCPD(
        variable="R",
        evidence=["E"],
        values={"e": {"r": 0.0002, "-r": 0.9998}, "-e": {"r": 0.9, "-r": 0.1}},
    )
    cpd_A = MapCPD(
        variable="A",
        evidence=["E", "B"],
        values={
            ("-e", "-b"): {"a": 0.01, "-a": 0.99},
            ("e", "-b"): {"a": 0.2, "-a": 0.8},
            ("-e", "b"): {"a": 0.95, "-a": 0.05},
            ("e", "b"): {"a": 0.96, "-a": 0.04},
        },
    )
    cpd_W = MapCPD(
        variable="W",
        evidence=["A"],
        values={"-a": {"w": 0.4, "-w": 0.6}, "a": {"w": 0.8, "-w": 0.2}},
    )
    cpd_G = MapCPD(
        variable="G",
        evidence=["A"],
        values={"-a": {"g": 0.04, "-g": 0.96}, "a": {"g": 0.4, "-g": 0.6}},
    )
    if debug:
        print(cpd_E)
        print(cpd_B)
        print(cpd_R)
        print(cpd_A)
        print(cpd_W)
        print(cpd_G)
    G.add_cpds(cpd_E, cpd_B, cpd_R, cpd_A, cpd_W, cpd_G)
    G.check_model()
    return G


#
# pgmpy_issue_1177
#


def pgmpy_issue_1177_pgmpy(debug=False):
    model = pgmpy_DiscreteBayesianNetwork(
        [
            ("A", "SD"),
            ("DW", "SD"),
            ("A", "ES"),
            ("DW", "ES"),
            ("SD", "ES"),
            ("IW", "IE"),
            ("DNR", "IE"),
            ("G", "IE"),
            ("DRW", "LC"),
            ("DT", "LC"),
            ("DRD", "LC"),
            ("DW", "C"),
            ("LC", "C"),
            ("ES", "ELP"),
            ("IE", "ELP"),
            ("C", "ELP"),
            ("ELP", "STKJ"),
            ("ELP", "SCKJ"),
            ("ELP", "SHKJ"),
        ]
    )
    np.random.seed(837498373)
    data = pd.DataFrame(
        np.random.randint(0, 2, size=(1000, 17)),
        columns=[
            "A",
            "SD",
            "DW",
            "ES",
            "IW",
            "IE",
            "DNR",
            "G",
            "DRW",
            "LC",
            "DT",
            "DRD",
            "C",
            "ELP",
            "STKJ",
            "SCKJ",
            "SHKJ",
        ],
    )
    model.fit(data, estimator=pgmpy_MaximumLikelihoodEstimator)
    return model
