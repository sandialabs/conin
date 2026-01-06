from collections import defaultdict
import os
import tempfile
import pprint
import munch
import pytoulbar2
from pyomo.common.timing import TicTocTimer

from .factor_repn import extract_factor_representation_, State
from conin.markov_network import (
    ConstrainedDiscreteMarkovNetwork,
    DiscreteMarkovNetwork,
    DiscreteFactor,
)
import conin.common


def create_reduced_MN(
    *,
    pgm,
    variables=None,
    evidence=None,
    var_index_map=None,
    timing=False,
    **options,
):
    """Create a reduced MN

    Parameters
    ----------
    pgm : DiscreteMarkovNetwork or ConstrainedDiscreteMarkovNetwork
        The graphical model that is used to construct the Pyomo model.
    variables : Iterable, optional
        Nodes for which the MAP configuration is requested.
    evidence : dict, optional
        Observed states keyed by node.
    timing : bool, optional
        If ``True``, return inference statistics along with the MAP result.
    var_index_map : dict, optional
        A dictionary of that is used construct mapped varibles
    **options
        Additional keyword arguments forwarded to the inference backend.

    Returns
    -------
    ConcreteModel
        The pyomo optimization model that supports inference with MAP queries.
    """
    return pgm

    """
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("create_MN_map_query_pyomo_model - START")
    pgm_ = pgm.pgm if isinstance(pgm, ConstrainedDiscreteMarkovNetwork) else pgm

    if variables or evidence:
        variables_ = [] if variables is None else variables
        evidence_ = {} if evidence is None else evidence
        if not evidence_ and len(variables_) == len(pgm_.nodes):
            factors = pgm_.factors
        else:
            raise RuntimeError("VariableElimination is not supported for CONIN models")
        if variables_:
            states = {var: pgm_.states[var] for var in variables_}
        else:
            states = {
                var: pgm_.states[var] for var in pgm_.nodes() if var not in evidence_
            }
    else:
        states = pgm_.states
        factors = pgm_.factors
    if timing:  # pragma:nocover
        timer.toc("Setup states and factors")

    S, J, v, w = extract_factor_representation_(states, factors, var_index_map)
    if timing:  # pragma:nocover
        timer.toc("Created factor repn")

    model = create_MN_map_query_model_from_factorial_repn(
        S=S,
        J=J,
        v=v,
        w=w,
        var_index_map=var_index_map,
        variables=variables,
        timing=timing,
    )

    # if evidence:
    #    for k, v in evidence.items():
    #        model.X[k, State(v)].fix(1)


    if timing:  # pragma:nocover
        timer.toc("create_MN_map_query_model - STOP")
    return model
    """


def create_toulbar2_map_query_model_MN(
    pgm,
    *,
    variables=None,
    evidence=None,
    timing=False,
):
    # Ignoring variables and evidence for now
    if timing:  # pragma:nocover
        timer.toc("create_toulbar2_model - START")
    pgm_ = pgm.pgm if isinstance(pgm, ConstrainedDiscreteMarkovNetwork) else pgm

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "model.uai")
        conin.common.save_model(pgm_, filename)

        model = pytoulbar2.CFN()
        model.Read(filename)

    model.X = {name: i for i, name in enumerate(pgm.nodes)}
    model.states = {i: pgm.states_of(name) for i, name in enumerate(pgm.nodes)}

    if isinstance(pgm, ConstrainedDiscreteMarkovNetwork) and pgm.constraints:
        data = munch.Munch(variables=variables, evidence=evidence)
        for func in pgm.constraints:
            model = func(model, data)

    if timing:  # pragma:nocover
        timer.toc("create_toulbar2_model - STOP")
    return model


def solve_toulbar2_map_query_model(
    model,
    *,
    # tee=False,
    # with_fixed=False,
    timing=False,
    # solver_options=None,
):
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("CFN_map_query - START")

    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    res = model.Solve()
    solvetime = solver_timer.toc(None)

    solution, primal_bound, num_solutions = res
    var = {name: model.states[i][solution[i]] for name, i in model.X.items()}
    soln = munch.Munch(
        variable_value=var, log_factor_sum=None, primal_bound=primal_bound
    )

    if timing:  # pragma:nocover
        timer.toc("Completed optimization")

    return munch.Munch(
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
        solvetime=solvetime,
    )


def inference_toulbar2_map_query_MN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    **options,
):
    model = create_toulbar2_map_query_model_MN(
        pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
    )
    return solve_toulbar2_map_query_model(model, timing=timing, **options)
