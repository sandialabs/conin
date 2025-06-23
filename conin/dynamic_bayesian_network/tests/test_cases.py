import pyomo.environ as pyo

try:
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.factors.discrete import TabularCPD
except Exception as e:
    pass

from conin.dynamic_bayesian_network import (
    MapCPD,
    ConstrainedDynamicBayesianNetwork,
)


def simple0_DBN(debug=False):
    G = DBN()
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

    if debug:
        for cpd in G.get_cpds():
            print(cpd)
    return G


def simple2_DBN(debug=False):
    G = DBN()
    G.add_edges_from([(("Z", 0), ("Z", 1))])
    z_start_cpd = MapCPD(variable=("Z", 0), values=[0.5, 0.5])
    z_trans_cpd = MapCPD(
        variable=("Z", 1),
        evidence=[("Z", 0)],
        values={0: [0.7, 0.3], 1: [0.8, 0.2]},
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


def simple1_DBN_constrained(debug=False):
    pgm = simple1_DBN(debug=debug)

    def constraints(model, data):
        model.c = pyo.ConstraintList()
        model.c.add(model.X[("A", 0), 0] == model.X[("A", 1), 0])
        model.c.add(model.X[("B", 0), 0] == model.X[("B", 1), 0])
        return model

    return ConstrainedDynamicBayesianNetwork(pgm, constraints=constraints)


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
    )
    cpd_trans_A = MapCPD(
        variable=("A", 1),
        evidence=[("A", 0)],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    G.add_cpds(cpd_start_A, cpd_start_B, cpd_trans_A)
    G.initialize_initial_state()
    G.check_model()

    if debug:
        for cpd in G.get_cpds():
            print(cpd)
    return G


def simple3_DBN_constrained(debug=False):
    pgm = simple3_DBN(debug=debug)

    def constraints(model, data):
        model.c = pyo.ConstraintList()
        model.c.add(model.X[("A", 0), 0] == model.X[("A", 1), 0])
        model.c.add(model.X[("B", 0), 0] == model.X[("B", 1), 0])
        return model

    return ConstrainedDynamicBayesianNetwork(pgm, constraints=constraints)


def pgmpy_weather1(debug=False):
    """
    A DBN example adapted from pgmpy documentation.
    """
    # Initialize a simple DBN model modeling the Weather (W), Rain (O),
    # Temperature (T), and Humidity (H).

    dbn = DBN()

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

    # Add CPDs to the DBN
    dbn.add_cpds(cpd_w_0, cpd_w_1, cpd_t_0, cpd_t_1, cpd_o, cpd_h)

    if debug:
        for cpd in dbn.get_cpds():
            print(cpd)
    return dbn


def pgmpy_weather2(debug=False):
    """
    A DBN example adapted from pgmpy documentation.

    Using explicit state names, and MapCPD declarations.
    """
    # Initialize a simple DBN model modeling the Weather (W), Rain (O),
    # Temperature (T), and Humidity (H).

    dbn = DBN()

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
    cpd_w_0 = MapCPD(
        variable=("W", 0),
        values={"Sunny": 0.6, "Cloudy": 0.3, "Rainy": 0.1},
        state_names={("W", 0): W_states},
    )

    cpd_w_1 = MapCPD(
        variable=("W", 1),
        evidence=[("W", 0)],
        values={
            "Sunny": {"Sunny": 0.7, "Cloudy": 0.2, "Rainy": 0.1},
            "Cloudy": {"Sunny": 0.3, "Cloudy": 0.4, "Rainy": 0.3},
            "Rainy": {"Sunny": 0.2, "Cloudy": 0.3, "Rainy": 0.5},
        },
        state_names={
            ("W", 0): W_states,
            ("W", 1): W_states,
        },
    )

    # CPD for T (Temperature transition)
    cpd_t_0 = MapCPD(
        variable=("T", 0),
        values={"Hot": 0.5, "Mild": 0.4, "Cold": 0.1},
        state_names={("T", 0): T_states},
    )

    cpd_t_1 = MapCPD(
        variable=("T", 1),
        evidence=[("T", 0), ("W", 0)],
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
        state_names={
            ("T", 1): T_states,
            ("T", 0): T_states,
            ("W", 0): W_states,
        },
    )

    # CPD for O (Ground observation)
    cpd_o = MapCPD(
        variable=("O", 0),
        evidence=[("W", 0)],
        values={
            "Sunny": {"Dry": 0.9, "Wet": 0.1},
            "Cloudy": {"Dry": 0.6, "Wet": 0.4},
            "Rainy": {"Dry": 0.2, "Wet": 0.8},
        },
        state_names={("O", 0): O_states, ("W", 0): W_states},
    )

    # CPD for H (Humidity observation)
    cpd_h = MapCPD(
        variable=("H", 0),
        evidence=[("T", 0), ("W", 0)],
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
        state_names={
            ("H", 0): H_states,
            ("T", 0): T_states,
            ("W", 0): W_states,
        },
    )

    # Add CPDs to the DBN
    dbn.add_cpds(cpd_w_0, cpd_w_1, cpd_t_0, cpd_t_1, cpd_o, cpd_h)

    if debug:
        for cpd in dbn.get_cpds():
            print(cpd)
    return dbn


def pgmpy_weather_constrained1(debug=False):
    pgm = pgmpy_weather2(debug)

    def constraints(model, data):
        """2 rainy days"""
        model.c = pyo.Constraint(
            expr=sum(model.X[("W", t), "Rainy"] for t in data.T) == 2
        )
        return model

    return ConstrainedDynamicBayesianNetwork(pgm, constraints=constraints)
