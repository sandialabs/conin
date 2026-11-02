import itertools
from munch import Munch
import pyomo.environ as pyo

from conin.constraints import (
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
    oracle_constraint_fn,
    smoek_constraint_fn,
)
from conin.util import try_import, MPESolution
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD
from conin.constrained_models import ConstrainedDiscreteBayesianNetwork

with try_import() as smoek_available:
    import smoek as smk


#
# cancer
#


def cancer1_BN_conin(debug=False):
    """
    Cancer example using conin
    """
    # Step 1: Define the network structure.
    states = {
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
    if debug:  # pragma:nocover
        print(cpd_poll)
        print(cpd_smoke)
        print(cpd_cancer)
        print(cpd_xray)
        print(cpd_dysp)
    cancer_model = DiscreteBayesianNetwork(
        states=states, cpds=[cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp]
    )

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


#
# cancer constrained
#


def cancer1_BN_constrained_pyomo_conin(debug=False):
    pgm = cancer1_BN_conin(debug=debug).pgm

    @pyomo_constraint_fn()
    def constraints(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
        model.c.add(model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    return Munch(
        pgm=cpgm,
        solutions=[
            MPESolution(
                states={
                    "Cancer": 1,
                    "Dyspnoea": 0,
                    "Pollution": 0,
                    "Smoker": 1,
                    "Xray": 1,
                }
            )
        ],
    )


def cancer1_BN_constrained_smoek_conin(debug=False):
    pgm = cancer1_BN_conin(debug=debug).pgm

    @smoek_constraint_fn()
    def constraints(model):
        return [
            model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
            model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
        ]

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    return Munch(
        pgm=cpgm,
        solutions=[
            MPESolution(
                states={
                    "Cancer": 1,
                    "Dyspnoea": 0,
                    "Pollution": 0,
                    "Smoker": 1,
                    "Xray": 1,
                }
            )
        ],
    )


def cancer1_BN_constrained_oracle_conin(debug=False):
    pgm = cancer1_BN_conin(debug=debug).pgm

    @oracle_constraint_fn(nodes=["Dyspnoea", "Xray"])
    def constraints(states):
        return states["Dyspnoea"] != states["Xray"]

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    return Munch(
        pgm=cpgm,
        solutions=[
            MPESolution(
                states={
                    "Cancer": 1,
                    "Dyspnoea": 0,
                    "Pollution": 0,
                    "Smoker": 1,
                    "Xray": 1,
                }
            )
        ],
    )


def cancer1_BN_constrained_toulbar2_conin(debug=False):
    pgm = cancer1_BN_conin(debug=debug).pgm

    @toulbar2_constraint_fn()
    def constraints(M):
        M.AddGeneralizedLinearConstraint([M.V("Dyspnoea", 1), M.V("Xray", 1)], "<=", 1)
        M.AddGeneralizedLinearConstraint([M.V("Dyspnoea", 0), M.V("Xray", 0)], "<=", 1)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    return Munch(
        pgm=cpgm,
        solutions=[
            MPESolution(
                states={
                    "Cancer": 1,
                    "Dyspnoea": 0,
                    "Pollution": 0,
                    "Smoker": 1,
                    "Xray": 1,
                }
            )
        ],
    )


#
# simple1
#


def simple1_BN_conin(debug=False):
    cpd_A = DiscreteCPD(node="A", values=[0.9, 0.1])
    cpd_B = DiscreteCPD(
        node="B",
        parents=["A"],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    if debug:  # pragma:nocover
        print(cpd_A)
        print(cpd_B)
    G = DiscreteBayesianNetwork(states={"A": [0, 1], "B": [0, 1]}, cpds=[cpd_A, cpd_B])
    G.check_model()
    return Munch(pgm=G, solutions=[MPESolution(states={"A": 0, "B": 1})])


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

    model_states = {
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

    model = DiscreteBayesianNetwork(
        states=model_states,
        cpds=[disease_state_CPD, test_result_CPD_1, test_result_CPD_2],
    )
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
    g_states = {
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

    if debug:  # pragma:nocover
        print(cpd_E)
        print(cpd_B)
        print(cpd_R)
        print(cpd_A)
        print(cpd_W)
        print(cpd_G)
    G = DiscreteBayesianNetwork(
        states=g_states, cpds=[cpd_E, cpd_B, cpd_R, cpd_A, cpd_W, cpd_G]
    )
    G.check_model()
    return Munch(
        pgm=G,
        solutions=[
            MPESolution(
                {"W": "-w", "G": "-g", "A": "-a", "B": "-b", "E": "-e", "R": "r"}
            )
        ],
    )


#
# tb2
#


def tb2_BN_conin(debug=False):
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
    if debug:  # pragma:nocover
        print(cpd_A)
        print(cpd_B)
        print(cpd_C)
    G = DiscreteBayesianNetwork(
        states={"A": [0, 1], "B": [2, 3], "C": [4, 5, 6]},
        cpds=[cpd_A, cpd_B, cpd_C],
    )
    G.check_model()
    return Munch(pgm=G, solutions=[MPESolution(states={"A": 0, "B": 3, "C": 4})])


# 0 2 0.21, 0.333, 0.457
# 0 3 0.811, 0.0, 0.189
# 1 2 0.2, 0.3, 0.4
# 1 3 0.8, 0.0, 0.2


#
# wide
#


def wide_BN_conin(debug=False):
    N = 10

    states = {("A", i): [0, 1] for i in range(N)}
    states["B"] = [0, 1]

    cpds = []
    for i in range(N):
        cpds.append(
            DiscreteCPD(node=("A", i), values=[1.0 / (i + 1), 1.0 - 1.0 / (i + 1)])
        )

    B_values = {}
    tmp = [[0, 1] for i in range(N)]
    for index in itertools.product(*tmp):
        count = sum(i for i in index)
        B_values[index] = [1.0 - 1.0 * count / N, 1.0 * count / N]
    cpds.append(
        DiscreteCPD(
            node="B",
            parents=[("A", i) for i in range(N)],
            values=B_values,
        )
    )

    if debug:
        print(states)
        for cpd in cpds:
            print(cpd)

    G = DiscreteBayesianNetwork(states=states, cpds=cpds)
    G.check_model()

    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["B"] = 1
    return Munch(pgm=G, solutions=[MPESolution(states=solution_states)])


#
# wide constrained1
#


def wide_BN_constrained1_conin_pyomo(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @pyomo_constraint_fn()
    def constraints(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.V(("A", 8), 1) + model.V(("A", 9), 1) <= 1)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 1
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])


def wide_BN_constrained1_conin_smoek(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @smoek_constraint_fn()
    def constraints(model):
        return model.V(("A", 8), 1) + model.V(("A", 9), 1) <= 1

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 1
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])


def wide_BN_constrained1_conin_toulbar2(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @toulbar2_constraint_fn()
    def constraints(M):
        M.AddGeneralizedLinearConstraint([M.V(("A", 8), 1), M.V(("A", 9), 1)], "<=", 1)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 1
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])


def wide_BN_constrained1_conin_oracle(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @oracle_constraint_fn(nodes=[("A", 8), ("A", 9)])
    def constraints(states):
        return states["A", 8] + states["A", 9] <= 1

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 1
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])


#
# wide constrained2
#


def wide_BN_constrained2_conin_pyomo(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @pyomo_constraint_fn()
    def constraints(model):
        model.I = pyo.RangeSet(0, N - 1)

        def rule(model, i):
            if i == 0:
                return pyo.Constraint.Skip
            return model.V(("A", i - 1), 1) + model.V(("A", i), 1) <= 1

        model.c = pyo.Constraint(model.I, rule=rule)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 1] = 0
    solution_states["A", 2] = 0
    solution_states["A", 4] = 0
    solution_states["A", 6] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 0
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])


def wide_BN_constrained2_conin_smoek(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @smoek_constraint_fn()
    def constraints(M):
        I = smk.sequence(start=1, stop=N - 1)
        i = smk.index()
        c = (
            smk.constraint()
            .expr(M.V(("A", i - 1), 1) + M.V(("A", i), 1) <= 1)
            .forall(i in I)
        )
        M.add_components(I=I, i=i, c=c)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 1] = 0
    solution_states["A", 2] = 0
    solution_states["A", 4] = 0
    solution_states["A", 6] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 0
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])


def wide_BN_constrained2_conin_toulbar2(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @toulbar2_constraint_fn()
    def constraints(M):
        for i in range(1, N):
            if i == 0:
                continue
            M.AddGeneralizedLinearConstraint(
                [M.V(("A", i - 1), 1), M.V(("A", i), 1)], "<=", 1
            )

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 1] = 0
    solution_states["A", 2] = 0
    solution_states["A", 4] = 0
    solution_states["A", 6] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 0
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])


def wide_BN_constrained2_conin_oracle(debug=False):
    N = 10
    pgm = wide_BN_conin(debug=debug).pgm

    @oracle_constraint_fn(nodes=[("A", t) for t in range(N)])
    def constraints(states):
        for i in range(1, N):
            if states["A", i - 1] + states["A", i] > 1:
                return False
        return True

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=constraints)
    solution_states = {("A", i): 1 for i in range(N)}
    solution_states["A", 0] = 0
    solution_states["A", 1] = 0
    solution_states["A", 2] = 0
    solution_states["A", 4] = 0
    solution_states["A", 6] = 0
    solution_states["A", 8] = 0
    solution_states["B"] = 0
    return Munch(pgm=cpgm, solutions=[MPESolution(states=solution_states)])
