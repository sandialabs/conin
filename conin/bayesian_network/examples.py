from munch import Munch
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from conin.constraint import pyomo_constraint_fn
from conin.util import try_import, MPESolution
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
    cpd_poll = DiscreteCPD(node="Pollution", values=[0.9, 0.1])
    cpd_smoke = DiscreteCPD(node="Smoker", values=[0.3, 0.7])
    cpd_cancer = DiscreteCPD(
        node="Cancer",
        parents=["Smoker", "Pollution"],
        values={
            (0, 0): [0.03, 0.97],
            (0, 1): [0.05, 0.95],
            (1, 0): [0.001, 0.999],
            (1, 1): [0.02, 0.98],
        },
        # values=[[0.03, 0.05, 0.001, 0.02], [0.97, 0.95, 0.999, 0.98]],
    )
    cpd_xray = DiscreteCPD(
        node="Xray",
        parents=["Cancer"],
        values={0: [0.9, 0.1], 1: [0.2, 0.8]},
        # values=[[0.9, 0.2], [0.1, 0.8]],
    )
    cpd_dysp = DiscreteCPD(
        node="Dyspnoea",
        parents=["Cancer"],
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
    return Munch(
        pgm=cancer_model,
        solution={"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1},
    )


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
    return Munch(
        pgm=cancer_model,
        solution={"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1},
    )


def cancer2_BN_pgmpy(debug=False):
    """
    Cancer example using pgmpy with MapCPD
    """
    from conin.common.pgmpy import MapCPD

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
    return Munch(
        pgm=cancer_model,
        solution={"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1},
    )


#
# cancer constrained
#


def cancer1_BN_constrained_pyomo_conin(debug=False):
    pgm = cancer1_BN_conin(debug=debug).pgm

    @pyomo_constraint_fn()
    def constraints(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraints])
    return Munch(
        pgm=cpgm,
        solution={"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1},
    )


def cancer1_BN_constrained_pyomo_pgmpy(debug=False):
    pgm = cancer1_BN_pgmpy(debug=debug).pgm

    @pyomo_constraint_fn()
    def constraints(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)

    import conin.common.pgmpy

    pgm = conin.common.pgmpy.convert_pgmpy_to_conin(pgm)
    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraints])
    return Munch(
        pgm=cpgm,
        solution={"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1},
    )


def cancer2_BN_constrained_pyomo_pgmpy(debug=False):
    pgm = cancer2_BN_pgmpy(debug=debug).pgm

    @pyomo_constraint_fn()
    def constraints(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
        model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)

    import conin.common.pgmpy

    pgm = conin.common.pgmpy.convert_pgmpy_to_conin(pgm)
    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraints])
    return Munch(
        pgm=cpgm,
        solution={"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1},
    )


#
# simple1
#


def simple1_BN_conin(debug=False):
    G = DiscreteBayesianNetwork()
    G.states = {"A": [0, 1], "B": [0, 1]}
    cpd_A = DiscreteCPD(node="A", values=[0.9, 0.1])
    cpd_B = DiscreteCPD(
        node="B",
        parents=["A"],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
    G.cpds = [cpd_A, cpd_B]
    G.check_model()
    return Munch(pgm=G, solution={"A": 0, "B": 1})


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
    return Munch(pgm=G, solution={"A": 0, "B": 1})


def simple2_BN_pgmpy(debug=False):
    from conin.common.pgmpy import MapCPD

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
    return Munch(pgm=G, solution={"A": 0, "B": 1})


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
        node="disease-state",
        values=[p_disease_present, 1.0 - p_disease_present],
    )

    test_result_CPD_1 = DiscreteCPD(
        node="test-result1",
        parents=["disease-state"],
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
        node="test-result2",
        parents=["disease-state"],
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

    return Munch(
        pgm=model, solution={"disease-state": 1, "test-result1": 1, "test-result2": 1}
    )


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

    return Munch(
        pgm=model, solution={"disease-state": 1, "test-result1": 1, "test-result2": 1}
    )


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

    cpd_E = DiscreteCPD(node="E", values={"e": 0.0003, "-e": 0.9997})
    cpd_B = DiscreteCPD(node="B", values={"b": 0.0001, "-b": 0.9999})
    cpd_R = DiscreteCPD(
        node="R",
        parents=["E"],
        values={"e": {"r": 0.0002, "-r": 0.9998}, "-e": {"r": 0.9, "-r": 0.1}},
    )
    cpd_A = DiscreteCPD(
        node="A",
        parents=["E", "B"],
        values={
            ("-e", "-b"): {"a": 0.01, "-a": 0.99},
            ("e", "-b"): {"a": 0.2, "-a": 0.8},
            ("-e", "b"): {"a": 0.95, "-a": 0.05},
            ("e", "b"): {"a": 0.96, "-a": 0.04},
        },
    )
    cpd_W = DiscreteCPD(
        node="W",
        parents=["A"],
        values={"-a": {"w": 0.4, "-w": 0.6}, "a": {"w": 0.8, "-w": 0.2}},
    )
    cpd_G = DiscreteCPD(
        node="G",
        parents=["A"],
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
    return Munch(
        pgm=G,
        solution={"W": "-w", "G": "-g", "A": "-a", "B": "-b", "E": "-e", "R": "r"},
    )


def holmes_pgmpy(debug=False):
    """
    Adapted from Lecture Notes by Alice Gao.

    W - Does Watson call Holmes?
    G - Does Gibbon call Holmes?
    A - Does Alarm go off?
    B - Does a burglary happen?
    E - Does an earthquake happen?
    R - Does he hear about an earthquake on the radio?

    A. Gao. "Lecture 12: Variable Elimination Algorithm", 2021.
    https://cs.uwaterloo.ca/~a23gao/cs486686_f18/slides/lec12_inferences_bayes_nets_nosol.pdf
    """
    from conin.common.pgmpy import MapCPD

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
    return Munch(
        pgm=G,
        solution={"W": "-w", "G": "-g", "A": "-a", "B": "-b", "E": "-e", "R": "r"},
    )


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


#
# tb2
#


def tb2_BN_conin(debug=False):
    G = DiscreteBayesianNetwork()
    G.states = {"A": [0, 1], "B": [2, 3], "C": [4, 5, 6]}
    cpd_A = DiscreteCPD(node="A", values=[0.436, 0.564])
    cpd_B = DiscreteCPD(
        node="B",
        parents=["A"],
        values={0: [0.128, 0.872], 1: [0.92, 0.08]},
    )
    cpd_C = DiscreteCPD(
        node="C",
        parents=["B"],
        values={2: [0.21, 0.333, 0.457], 3: [0.811, 0.0, 0.189]},
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G.cpds = [cpd_A, cpd_B, cpd_C]
    G.check_model()
    return Munch(pgm=G, solution={"A": 0, "B": 3, "C": 4})


def tb2_BN_pgmpy(debug=False):
    G = pgmpy_DiscreteBayesianNetwork()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    cpd_A = pgmpy_TabularCPD(variable="A", variable_card=2, values=[[0.436], [0.564]])
    cpd_B = pgmpy_TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.128, 0.92], [0.872, 0.08]],
        evidence=["A"],
        evidence_card=[2],
        state_names={"A": [0, 1], "B": [2, 3]},
    )
    cpd_C = pgmpy_TabularCPD(
        variable="C",
        variable_card=3,
        values=[[0.21, 0.811], [0.333, 0.0], [0.457, 0.189]],
        evidence=["B"],
        evidence_card=[2],
        state_names={"C": [4, 5, 6], "B": [2, 3]},
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G.add_cpds(cpd_A, cpd_B, cpd_C)
    G.check_model()
    return Munch(pgm=G, solution={"A": 0, "B": 3, "C": 4})


def tb2_BN_pgmpy_mapcpd(debug=False):
    from conin.common.pgmpy import MapCPD

    G = pgmpy_DiscreteBayesianNetwork()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    cpd_A = MapCPD(variable="A", values={0: 0.436, 1: 0.564})
    cpd_B = MapCPD(
        variable="B",
        evidence=["A"],
        values={0: {2: 0.128, 3: 0.872}, 1: {2: 0.92, 3: 0.08}},
    )
    cpd_C = MapCPD(
        variable="C",
        evidence=["B"],
        values={2: {4: 0.21, 5: 0.333, 6: 0.457}, 3: {4: 0.811, 5: 0.0, 6: 0.189}},
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G.add_cpds(cpd_A, cpd_B, cpd_C)
    G.check_model()
    return Munch(pgm=G, solution={"A": 0, "B": 3, "C": 4})


#
# tb2*
#


def tb2_BN_conin(debug=False):
    G = DiscreteBayesianNetwork()
    G.states = {"A": [0, 1], "B": [2, 3], "C": [4, 5, 6]}
    cpd_A = DiscreteCPD(node="A", values=[0.436, 0.564])
    cpd_B = DiscreteCPD(
        node="B",
        parents=["A"],
        values={0: [0.128, 0.872], 1: [0.92, 0.08]},
    )
    cpd_C = DiscreteCPD(
        node="C",
        parents=["A", "B"],
        values={
            (0, 2): [0.21, 0.333, 0.457],
            (0, 3): [0.811, 0.0, 0.189],
            (1, 2): [0.2, 0.3, 0.5],
            (1, 3): [0.8, 0.0, 0.2],
        },
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G.cpds = [cpd_A, cpd_B, cpd_C]
    G.check_model()
    return Munch(pgm=G, solution={"A": 0, "B": 3, "C": 4})


# 0 2 0.21, 0.333, 0.457
# 0 3 0.811, 0.0, 0.189
# 1 2 0.2, 0.3, 0.4
# 1 3 0.8, 0.0, 0.2


def tb2_BN_pgmpy(debug=False):
    G = pgmpy_DiscreteBayesianNetwork()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edge("A", "B")
    G.add_edge("A", "C")
    G.add_edge("B", "C")
    cpd_A = pgmpy_TabularCPD(variable="A", variable_card=2, values=[[0.436], [0.564]])
    cpd_B = pgmpy_TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.128, 0.92], [0.872, 0.08]],
        evidence=["A"],
        evidence_card=[2],
        state_names={"A": [0, 1], "B": [2, 3]},
    )
    cpd_C = pgmpy_TabularCPD(
        variable="C",
        variable_card=3,
        values=[
            [0.21, 0.811, 0.2, 0.8],
            [0.333, 0.0, 0.3, 0.0],
            [0.457, 0.189, 0.5, 0.2],
        ],
        evidence=["A", "B"],
        evidence_card=[2, 2],
        state_names={"C": [4, 5, 6], "B": [2, 3], "A": [0, 1]},
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G.add_cpds(cpd_A, cpd_B, cpd_C)
    G.check_model()
    return Munch(pgm=G, solution={"A": 0, "B": 3, "C": 4})


def tb2_BN_pgmpy_mapcpd(debug=False):
    from conin.common.pgmpy import MapCPD

    G = pgmpy_DiscreteBayesianNetwork()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edge("A", "B")
    G.add_edge("A", "C")
    G.add_edge("B", "C")
    cpd_A = MapCPD(variable="A", values={0: 0.436, 1: 0.564})
    cpd_B = MapCPD(
        variable="B",
        evidence=["A"],
        values={0: {2: 0.128, 3: 0.872}, 1: {2: 0.92, 3: 0.08}},
    )
    cpd_C = MapCPD(
        variable="C",
        evidence=["A", "B"],
        values={
            (0, 2): {4: 0.21, 5: 0.333, 6: 0.457},
            (0, 3): {4: 0.811, 5: 0.0, 6: 0.189},
            (1, 2): {4: 0.2, 5: 0.3, 6: 0.5},
            (1, 3): {4: 0.8, 5: 0.0, 6: 0.2},
        },
    )
    if debug:
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G.add_cpds(cpd_A, cpd_B, cpd_C)
    G.check_model()
    return Munch(pgm=G, solution={"A": 0, "B": 3, "C": 4})
