from collections import defaultdict
import pprint
import munch

from .factor_repn import extract_factor_representation_, State
from .variable_elimination import _variable_elimination
from .model import ConstrainedDiscreteMarkovNetwork


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

    if isinstance(pgm, ConstrainedDiscreteMarkovNetwork) and pgm.constraints:
        data = munch.Munch(variables=variables, evidence=evidence)
        for func in pgm.constraints:
            model = func(model, data)

    if timing:  # pragma:nocover
        timer.toc("create_MN_map_query_model - STOP")
    return model
    """


def CFN_map_query(
    model,
    *,
    solver="gurobi",
    tee=False,
    with_fixed=False,
    timing=False,
    solver_options=None,
):
    pass
    """
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("optimize_map_query_model - START")
    opt = pe.SolverFactory(solver)
    if solver_options:
        opt.options = solver_options
    if timing:  # pragma:nocover
        timer.toc("Initialize solver")
    timer = TicTocTimer()
    timer.tic(None)
    res = opt.solve(model, tee=tee)
    solvetime = timer.toc(None)
    pe.assert_optimal_termination(res)
    if timing:  # pragma:nocover
        timer.toc("Completed optimization")

    var = {}
    variables = set()
    fixed_variables = set()
    for r, s in model.X:
        variables.add(r)
        if model.X[r, s].is_fixed():
            fixed_variables.add(r)
            if with_fixed and pe.value(model.X[r, s]) > 0.5:
                var[r] = s.value
        elif pe.value(model.X[r, s]) > 0.5:
            var[r] = s.value
    assert variables == set(var.keys()).union(
        fixed_variables
    ), "Some variables do not have values."

    soln = munch.Munch(variable_value=var, log_factor_sum=pe.value(model.o))
    if timing:  # pragma:nocover
        timer.toc("optimize_map_query_model - STOP")
    return munch.Munch(
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
        solvetime=solvetime,
    )
    """
