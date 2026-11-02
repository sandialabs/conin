from munch import Munch
import pyomo.environ as pyo

from conin.constraints import (
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
    oracle_constraint_fn,
    smoek_constraint_fn,
)
from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
from conin.graphical_models import (
    DiscreteCPD,
    DynamicDiscreteBayesianNetwork,
    TimeIndex,
)
from conin.util import MPESolution

#
# simple0 example
#


def simple0_DDBN_conin(debug=False):
    t = TimeIndex()
    z_start_cpd = DiscreteCPD(node=("Z", 0), values=[0.5, 0.5])
    z_trans_cpd = DiscreteCPD(
        node=("Z", t),
        parents=[("Z", t - 1)],
        values={0: [0.7, 0.3], 1: [0.8, 0.2]},
    )

    G = DynamicDiscreteBayesianNetwork(
        dynamic_states={"Z": [0, 1]}, cpds=[z_start_cpd, z_trans_cpd], t=t
    )
    G.check_model()

    if debug:  # pragma:nocover
        for cpd in G.cpds:
            print(cpd)
    return Munch(pgm=G, solutions=[MPESolution(states={("Z", 0): 1, ("Z", 1): 0})])


#
# simple1
#


def simple1_DDBN_conin(debug=False):
    t = TimeIndex()

    cpd_start_A = DiscreteCPD(node=("A", 0), values=[0.9, 0.1])
    cpd_start_B = DiscreteCPD(
        node=("B", t),
        parents=[("A", t)],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    cpd_trans_A = DiscreteCPD(
        node=("A", t),
        parents=[("A", t - 1)],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    G = DynamicDiscreteBayesianNetwork(
        dynamic_states={"A": [0, 1], "B": [0, 1]},
        cpds=[cpd_start_A, cpd_start_B, cpd_trans_A],
        t=t,
    )
    G.check_model()

    if debug:  # pragma:nocover
        for cpd in G.cpds:
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


def simple1_DDBN_constrained_pyomo_conin(debug=False):
    pgm = simple1_DDBN_conin(debug=debug).pgm

    @pyomo_constraint_fn()
    def constraints(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.V("A", 0, 0) == model.V("A", 1, 0))
        model.c.add(model.V("B", 0, 0) == model.V("B", 1, 0))

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[
            MPESolution(
                states={
                    ("A", 0): 0,
                    ("A", 1): 0,
                    ("B", 0): 1,
                    ("B", 1): 1,
                }
            )
        ],
    )


def simple1_DDBN_constrained_smoek_conin(debug=False):
    pgm = simple1_DDBN_conin(debug=debug).pgm

    @smoek_constraint_fn()
    def constraints(model):
        return [
            model.V("A", 0, 0) == model.V("A", 1, 0),
            model.V("B", 0, 0) == model.V("B", 1, 0),
        ]

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[
            MPESolution(
                states={
                    ("A", 0): 0,
                    ("A", 1): 0,
                    ("B", 0): 1,
                    ("B", 1): 1,
                }
            )
        ],
    )


def simple1_DDBN_constrained_oracle_conin(debug=False):
    pgm = simple1_DDBN_conin(debug=debug).pgm

    def nodes(data):
        for t in data.T:
            yield ("A", t)
            yield ("B", t)

    @oracle_constraint_fn(nodes=nodes)
    def constraints(states):
        return states["A", 0] == states["A", 1] and states["B", 0] == states["B", 1]

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[
            MPESolution(
                states={
                    ("A", 0): 0,
                    ("A", 1): 0,
                    ("B", 0): 1,
                    ("B", 1): 1,
                }
            )
        ],
    )


def simple1_DDBN_constrained_toulbar2_conin(debug=False):
    pgm = simple1_DDBN_conin(debug=debug).pgm

    @toulbar2_constraint_fn()
    def constraints(M):
        M.AddGeneralizedLinearConstraint(
            [M.V("A", 0, 0), M.V("A", 1, 0, coef=-1)], "==", 0
        )
        M.AddGeneralizedLinearConstraint(
            [M.V("B", 0, 0), M.V("B", 1, 0, coef=-1)], "==", 0
        )

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[
            MPESolution(
                states={
                    ("A", 0): 0,
                    ("A", 1): 0,
                    ("B", 0): 1,
                    ("B", 1): 1,
                }
            )
        ],
    )


#
# weather
#

q_weather_A = {
    ("H", 0): "Low",
    ("H", 1): "Low",
    ("H", 2): "Low",
    ("H", 3): "Low",
    ("H", 4): "Low",
    ("O", 0): "Dry",
    ("O", 1): "Dry",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Hot",
    ("T", 2): "Hot",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Sunny",
    ("W", 1): "Sunny",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}


def weather_conin(debug=False):
    """
    A DDBN example adapted from pgmpy documentation.

    Using explicit state names.
    """
    # Initialize a simple DDBN model modeling the Weather (W), Rain (O),
    # Temperature (T), and Humidity (H).

    # pgmpy requires the user to define the structure of the first time slice and the edges
    # connecting the first time slice to the second time slice.
    # pgmpy assumes that this structure remains constant for further time
    # slices, i.e., it is a 2-TBN.

    W_states = ["Sunny", "Cloudy", "Rainy"]
    T_states = ["Hot", "Mild", "Cold"]
    O_states = ["Dry", "Wet"]
    H_states = ["Low", "Medium", "High"]

    t = TimeIndex()

    # Add intra-slice edges for both time slices
    # dbn.add_edges_from(
    #    [
    #        (("W", 0), ("O", 0)),  # Weather influences ground observation
    #        (("T", 0), ("H", 0)),  # Temperature influences humidity
    #        (("W", 0), ("H", 0)),  # Weather influences humidity
    #    ]
    # )

    # Add inter-slice edges
    # dbn.add_edges_from(
    #    [
    #        (("W", 0), ("W", 1)),  # Weather transition
    #        (("T", 0), ("T", 1)),  # Temperature transition
    #        (("W", 0), ("T", 1)),  # Weather influences future temperature
    #    ]
    # )

    # Define the parameters of the model. Again pgmpy assumes that these CPDs
    # remain the same for future time slices.

    # Define CPDs
    # CPD for W (Weather transition)
    cpd_w_0 = DiscreteCPD(
        node=("W", 0),
        values={"Sunny": 0.6, "Cloudy": 0.3, "Rainy": 0.1},
    )

    cpd_w_1 = DiscreteCPD(
        node=("W", t),
        parents=[("W", t - 1)],
        values={
            "Sunny": {"Sunny": 0.7, "Cloudy": 0.2, "Rainy": 0.1},
            "Cloudy": {"Sunny": 0.3, "Cloudy": 0.4, "Rainy": 0.3},
            "Rainy": {"Sunny": 0.2, "Cloudy": 0.3, "Rainy": 0.5},
        },
    )

    # CPD for T (Temperature transition)
    cpd_t_0 = DiscreteCPD(
        node=("T", 0),
        values={"Hot": 0.5, "Mild": 0.4, "Cold": 0.1},
    )

    cpd_t_1 = DiscreteCPD(
        node=("T", t),
        parents=[("T", t - 1), ("W", t - 1)],
        values={
            ("Hot", "Sunny"): {"Hot": 0.8, "Mild": 0.2, "Cold": 0.0},
            ("Hot", "Cloudy"): {"Hot": 0.6, "Mild": 0.3, "Cold": 0.1},
            ("Hot", "Rainy"): {"Hot": 0.1, "Mild": 0.7, "Cold": 0.2},
            ("Mild", "Sunny"): {"Hot": 0.7, "Mild": 0.2, "Cold": 0.1},
            ("Mild", "Cloudy"): {"Hot": 0.4, "Mild": 0.5, "Cold": 0.1},
            ("Mild", "Rainy"): {"Hot": 0.2, "Mild": 0.3, "Cold": 0.5},
            ("Cold", "Sunny"): {"Hot": 0.6, "Mild": 0.3, "Cold": 0.1},
            ("Cold", "Cloudy"): {"Hot": 0.3, "Mild": 0.4, "Cold": 0.3},
            ("Cold", "Rainy"): {"Hot": 0.1, "Mild": 0.3, "Cold": 0.6},
        },
    )

    # CPD for O (Ground observation)
    cpd_o = DiscreteCPD(
        node=("O", t),
        parents=[("W", t)],
        values={
            "Sunny": {"Dry": 0.9, "Wet": 0.1},
            "Cloudy": {"Dry": 0.6, "Wet": 0.4},
            "Rainy": {"Dry": 0.2, "Wet": 0.8},
        },
    )

    # CPD for H (Humidity observation)
    cpd_h = DiscreteCPD(
        node=("H", t),
        parents=[("T", t), ("W", t)],
        values={
            ("Hot", "Sunny"): {"Low": 0.7, "Medium": 0.2, "High": 0.1},
            ("Hot", "Cloudy"): {"Low": 0.4, "Medium": 0.5, "High": 0.1},
            ("Hot", "Rainy"): {"Low": 0.1, "Medium": 0.3, "High": 0.6},
            ("Mild", "Sunny"): {"Low": 0.5, "Medium": 0.4, "High": 0.1},
            ("Mild", "Cloudy"): {"Low": 0.3, "Medium": 0.5, "High": 0.2},
            ("Mild", "Rainy"): {"Low": 0.2, "Medium": 0.3, "High": 0.5},
            ("Cold", "Sunny"): {"Low": 0.3, "Medium": 0.4, "High": 0.3},
            ("Cold", "Cloudy"): {"Low": 0.2, "Medium": 0.3, "High": 0.5},
            ("Cold", "Rainy"): {"Low": 0.1, "Medium": 0.2, "High": 0.7},
        },
    )

    # Add CPDs to the DDBN
    dbn = DynamicDiscreteBayesianNetwork(
        dynamic_states={"W": W_states, "T": T_states, "O": O_states, "H": H_states},
        cpds=[cpd_w_0, cpd_w_1, cpd_t_0, cpd_t_1, cpd_o, cpd_h],
        t=t,
    )

    if debug:  # pragma:nocover
        for cpd in dbn.cpds:
            print(cpd)
    return Munch(pgm=dbn, solutions=[MPESolution(states=q_weather_A)])


q_weather_A_constrained = {
    ("W", 0): "Sunny",
    ("T", 0): "Hot",
    ("O", 0): "Dry",
    ("H", 0): "Low",
    ("W", 1): "Sunny",
    ("T", 1): "Hot",
    ("O", 1): "Dry",
    ("H", 1): "Low",
    ("W", 2): "Sunny",
    ("T", 2): "Hot",
    ("O", 2): "Dry",
    ("H", 2): "Low",
    ("W", 3): "Rainy",
    ("T", 3): "Hot",
    ("O", 3): "Wet",
    ("H", 3): "High",
    ("W", 4): "Rainy",
    ("T", 4): "Mild",
    ("O", 4): "Wet",
    ("H", 4): "High",
}


def weather_constrained_pyomo_conin(debug=False):
    pgm = weather_conin(debug).pgm

    @pyomo_constraint_fn()
    def constraints(model, data):
        """2 rainy days"""
        model.c = pyo.Constraint(
            expr=sum(model.V("W", t, "Rainy") for t in data.T) == 2
        )

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[MPESolution(states=q_weather_A_constrained)],
    )


def weather_constrained_smoek_conin(debug=False):
    pgm = weather_conin(debug).pgm

    @smoek_constraint_fn()
    def constraints(model, data):
        """2 rainy days"""
        return sum(model.V("W", t, "Rainy") for t in data.T) == 2

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[MPESolution(states=q_weather_A_constrained)],
    )


def weather_constrained_oracle_conin(debug=False):
    pgm = weather_conin(debug).pgm

    def nodes(data):
        for t in data.T:
            yield ("W", t)

    @oracle_constraint_fn(nodes=nodes)
    def constraints(states, data):
        """2 rainy days"""
        num = sum(states["W", t] == "Rainy" for t in data.T)
        if num == 2:
            return True
        return False

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[MPESolution(states=q_weather_A_constrained)],
    )


def weather_constrained_toulbar2_conin(debug=False):
    pgm = weather_conin(debug).pgm

    @toulbar2_constraint_fn()
    def constraints(M, data):
        """2 rainy days"""
        M.AddGeneralizedLinearConstraint(
            [M.V("W", t, "Rainy") for t in data.T], "==", 2
        )

    return Munch(
        pgm=ConstrainedDynamicDiscreteBayesianNetwork(pgm, constraints=constraints),
        solutions=[MPESolution(states=q_weather_A_constrained)],
    )
