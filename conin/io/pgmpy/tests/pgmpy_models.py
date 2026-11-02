"""pgmpy models for the converter tests (they need pgmpy).

Each ``*_pgmpy`` function returns a ``Munch`` with ``pgm`` (a pgmpy model)
and ``solutions`` (its MAP solutions), except ``pgmpy_issue_1177_pgmpy``,
which returns a fitted pgmpy model.  :data:`CONIN_TWINS` names the CONIN
example that describes the same model.
"""

from munch import Munch
import numpy as np
import pandas as pd

from conin.examples.dynamic_bayesian_network import q_weather_A
from conin.util import MPESolution

from pgmpy.models import DiscreteBayesianNetwork as pgmpy_DiscreteBayesianNetwork
from pgmpy.models import DiscreteMarkovNetwork as pgmpy_MarkovNetwork
from pgmpy.models import DynamicBayesianNetwork as pgmpy_DynamicBayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor as pgmpy_DiscreteFactor
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import TabularCPD as pgmpy_TabularCPD

try:
    # Newer pgmpy versions fit with an estimator instance.
    from pgmpy.parameter_estimator import DiscreteMLE as _Estimator

    _ESTIMATOR = _Estimator()
except ImportError:  # pragma: no cover
    from pgmpy.estimators.MLE import MaximumLikelihoodEstimator as _ESTIMATOR


def example6_pgmpy():
    """
    See Obbens, p.18

    Non-uniform weights are applied to the A_B factors to remove a degeneracy w.r.t. the
    value of variable A.
    """
    pgm = pgmpy_MarkovNetwork()
    pgm.add_nodes_from(["A", "B"])
    pgm.add_edge("A", "B")
    f1 = pgmpy_DiscreteFactor(["A"], [2], [1, 1])
    f2 = pgmpy_DiscreteFactor(["B"], [2], [1, 2])
    f3 = pgmpy_DiscreteFactor(["A", "B"], [2, 2], [1, 3, 1, 1])
    pgm.add_factors(f1, f2, f3)

    return Munch(pgm=pgm, solutions=[MPESolution()])


def ABC_pgmpy():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights, so the MPE solution is defined by the weights for the
    factors that describe the individual variables.

    The MPE solution is A:2, B:2, C:1.
    """
    pgm = pgmpy_MarkovNetwork()
    pgm.add_nodes_from(["A", "B", "C"])
    pgm.add_edge("A", "B")
    pgm.add_edge("B", "C")
    pgm.add_edge("A", "C")
    f1 = pgmpy_DiscreteFactor(["A"], [3], [1, 1, 2])
    f2 = pgmpy_DiscreteFactor(["B"], [3], [1, 1, 3])
    f3 = pgmpy_DiscreteFactor(["C"], [3], [1, 2, 1])
    f4 = pgmpy_DiscreteFactor(["A", "B"], [3, 3], np.ones(9))
    f5 = pgmpy_DiscreteFactor(["B", "C"], [3, 3], np.ones(9))
    f6 = pgmpy_DiscreteFactor(["A", "C"], [3, 3], np.ones(9))
    pgm.factors = [f1, f2, f3, f4, f5, f6]

    return Munch(pgm=pgm, solutions=[MPESolution(states={"A": 2, "B": 2, "C": 1})])


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
    if debug:  # pragma:nocover
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
        solutions=[
            MPESolution(
                states={
                    "Cancer": 1,
                    "Dyspnoea": 1,
                    "Pollution": 0,
                    "Smoker": 1,
                    "Xray": 1,
                }
            )
        ],
    )


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
    if debug:  # pragma:nocover
        print(cpd_A)
        print(cpd_B)
    G.add_cpds(cpd_A, cpd_B)
    G.check_model()
    return Munch(pgm=G, solutions=[MPESolution(states={"A": 0, "B": 1})])


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

    if debug:  # pragma:nocover
        print(disease_state_CPD)
        print(test_result_CPD_1)
        print(test_result_CPD_2)

    return Munch(
        pgm=model,
        solutions=[
            MPESolution(
                states={"disease-state": 1, "test-result1": 1, "test-result2": 1}
            )
        ],
    )


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
    model.fit(data, estimator=_ESTIMATOR)
    return model


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
    if debug:  # pragma:nocover
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G.add_cpds(cpd_A, cpd_B, cpd_C)
    G.check_model()
    return Munch(pgm=G, solutions=[MPESolution(states={"A": 0, "B": 3, "C": 4})])


def simple0_DDBN1_pgmpy(debug=False):
    G = pgmpy_DynamicBayesianNetwork()
    G.add_edges_from([(("Z", 0), ("Z", 1))])
    z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
    z_trans_cpd = TabularCPD(
        ("Z", 1),
        2,
        [[0.7, 0.8], [0.3, 0.2]],
        evidence=[("Z", 0)],
        evidence_card=[2],
    )

    G.add_cpds(z_start_cpd, z_trans_cpd)
    G.initialize_initial_state()
    G.check_model()

    if debug:  # pragma:nocover
        for cpd in G.get_cpds():
            print(cpd)
    return Munch(pgm=G, solutions=[MPESolution(states={("Z", 0): 1, ("Z", 1): 0})])


def simple1_DDBN_pgmpy(debug=False):
    G = pgmpy_DynamicBayesianNetwork()
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

    if debug:  # pragma:nocover
        for cpd in G.get_cpds():
            print(cpd)
    return Munch(
        pgm=G,
        solutions=[
            MPESolution(
                states={
                    ("A", 0): 0,
                    ("A", 1): 1,
                    ("B", 0): 1,
                    ("B", 1): 0,
                }
            )
        ],
    )


def weather1_pgmpy(debug=False):
    """
    A DDBN example adapted from pgmpy documentation.
    """
    # Initialize a simple DDBN model modeling the Weather (W), Rain (O),
    # Temperature (T), and Humidity (H).

    dbn = pgmpy_DynamicBayesianNetwork()

    # pgmpy requires the user to define the structure of the first time slice and the edges
    # connecting the first time slice to the second time slice.
    # pgmpy assumes that this structure remains constant for further time
    # slices, i.e., it is a 2-TBN.

    W_states = ["Sunny", "Cloudy", "Rainy"]
    T_states = ["Hot", "Mild", "Cold"]
    O_states = ["Dry", "Wet"]
    H_states = ["Low", "Medium", "High"]

    # Add intra-slice edges for both time slices
    dbn.add_edges_from(
        [
            (("W", 0), ("O", 0)),  # Weather influences ground observation
            (("T", 0), ("H", 0)),  # Temperature influences humidity
            (("W", 0), ("H", 0)),  # Weather influences humidity
        ]
    )

    # Add inter-slice edges
    dbn.add_edges_from(
        [
            (("W", 0), ("W", 1)),  # Weather transition
            (("T", 0), ("T", 1)),  # Temperature transition
            (("W", 0), ("T", 1)),  # Weather influences future temperature
        ]
    )

    # Define the parameters of the model. Again pgmpy assumes that these CPDs
    # remain the same for future time slices.

    # Define CPDs
    # CPD for W (Weather transition)
    cpd_w_0 = TabularCPD(
        variable=("W", 0),
        variable_card=3,  # Sunny, Cloudy, Rainy
        values=[[0.6], [0.3], [0.1]],  # Initial probabilities
        state_names={("W", 0): W_states},
    )

    cpd_w_1 = TabularCPD(
        variable=("W", 1),
        variable_card=3,
        evidence=[("W", 0)],
        evidence_card=[3],
        values=[
            [0.7, 0.3, 0.2],  # P(Sunny | W_0)
            [0.2, 0.4, 0.3],  # P(Cloudy | W_0)
            [0.1, 0.3, 0.5],  # P(Rainy | W_0)
        ],
        state_names={
            ("W", 0): W_states,
            ("W", 1): W_states,
        },
    )

    # CPD for T (Temperature transition)
    cpd_t_0 = TabularCPD(
        variable=("T", 0),
        variable_card=3,  # Hot, Mild, Cold
        values=[[0.5], [0.4], [0.1]],  # Initial probabilities
        state_names={("T", 0): T_states},
    )

    cpd_t_1 = TabularCPD(
        variable=("T", 1),
        variable_card=3,
        evidence=[("T", 0), ("W", 0)],
        evidence_card=[3, 3],
        values=[
            [0.8, 0.6, 0.1, 0.7, 0.4, 0.2, 0.6, 0.3, 0.1],  # P(Hot | T_0, W_0)
            [
                0.2,
                0.3,
                0.7,
                0.2,
                0.5,
                0.3,
                0.3,
                0.4,
                0.3,
            ],  # P(Mild | T_0, W_0)
            [
                0.0,
                0.1,
                0.2,
                0.1,
                0.1,
                0.5,
                0.1,
                0.3,
                0.6,
            ],  # P(Cold | T_0, W_0)
        ],
        state_names={
            ("T", 1): T_states,
            ("T", 0): T_states,
            ("W", 0): W_states,
        },
    )

    # CPD for O (Ground observation)
    cpd_o = TabularCPD(
        variable=("O", 0),
        variable_card=2,  # Dry, Wet
        evidence=[("W", 0)],
        evidence_card=[3],
        values=[
            [0.9, 0.6, 0.2],  # P(Dry | Sunny, Cloudy, Rainy)
            [0.1, 0.4, 0.8],  # P(Wet | Sunny, Cloudy, Rainy)
        ],
        state_names={("O", 0): O_states, ("W", 0): W_states},
    )

    # CPD for H (Humidity observation)
    cpd_h = TabularCPD(
        variable=("H", 0),
        variable_card=3,  # Low, Medium, High
        evidence=[("T", 0), ("W", 0)],
        evidence_card=[3, 3],
        values=[
            [0.7, 0.4, 0.1, 0.5, 0.3, 0.2, 0.3, 0.2, 0.1],  # P(Low | T_0, W_0)
            [
                0.2,
                0.5,
                0.3,
                0.4,
                0.5,
                0.3,
                0.4,
                0.3,
                0.2,
            ],  # P(Medium | T_0, W_0)
            [
                0.1,
                0.1,
                0.6,
                0.1,
                0.2,
                0.5,
                0.3,
                0.5,
                0.7,
            ],  # P(High | T_0, W_0)
        ],
        state_names={
            ("H", 0): H_states,
            ("T", 0): T_states,
            ("W", 0): W_states,
        },
    )

    # Add CPDs to the DDBN
    dbn.add_cpds(cpd_w_0, cpd_w_1, cpd_t_0, cpd_t_1, cpd_o, cpd_h)

    if debug:  # pragma:nocover
        for cpd in dbn.get_cpds():
            print(cpd)
    return Munch(pgm=dbn, solutions=[MPESolution(states=q_weather_A)])


CONIN_TWINS = {
    "example6_pgmpy": "markov_network.example6_conin",
    "ABC_pgmpy": "markov_network.ABC_conin",
    "cancer1_BN_pgmpy": "bayesian_network.cancer1_BN_conin",
    "simple1_BN_pgmpy": "bayesian_network.simple1_BN_conin",
    "DBDA_5_1_pgmpy": "bayesian_network.DBDA_5_1_conin",
    "tb2_BN_pgmpy": "bayesian_network.tb2_BN_conin",
    "simple0_DDBN1_pgmpy": "dynamic_bayesian_network.simple0_DDBN_conin",
    "simple1_DDBN_pgmpy": "dynamic_bayesian_network.simple1_DDBN_conin",
    "weather1_pgmpy": "dynamic_bayesian_network.weather_conin",
}
"""``*_pgmpy`` builder -> the CONIN example (module.function) of the same model."""
