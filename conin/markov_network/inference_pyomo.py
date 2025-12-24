from collections import defaultdict
import pprint
import munch
import pyomo.environ as pe
from pyomo.common.timing import TicTocTimer

from .factor_repn import extract_factor_representation_, State
from .variable_elimination import _variable_elimination
from .model import ConstrainedDiscreteMarkovNetwork


"""
Notes on this integer program:

The equations for Example 6 in Obbens appear to have an error.
Specifically, the constants associate with c4 are missing.

The likelihood model needs to reference calX_c not X_c

Need to define calX_r and calX_i

Why use k_i?  Use r instead.
"""

"""
Notes on the size of this integer program:

Suppose we have N variables with at most k values, and M factors with at
most l variables in each configuration.  I agree that the maximum number
of variables is Nk + Mk^l.  But it might be more intuitive to characterize
the number of variables w.r.t. the maximum number of configurations, c.
This would give a maximum number of variables as Nk + Mc.  Clearly, c <=
k^l, but in practice models may be more sparse and thus the original
complexity analysis would be too pessimistic.

Similarly, the number of constraints in c4 is O(Mc), so the total number
of constraints is O(N + M + Mcl + Mc) = O(N + Mcl).
"""


class VarWrapper(dict):
    def __init__(self, *arg, **kw):
        super(VarWrapper, self).__init__(*arg, **kw)

    def pprint(self):
        pprint.pprint(self)

    def __getitem__(self, index):
        r, s = index
        if type(s) is not State:
            s = State(s)
        return dict.__getitem__(self, (r, s))


def create_MN_pyomo_map_query_model(
    *,
    pgm,
    variables=None,
    evidence=None,
    var_index_map=None,
    timing=False,
    **options,
):
    """Build a inference model for MAP queries.

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

    model = create_MN_pyomo_map_query_model_from_factorial_repn(
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


def create_MN_pyomo_map_query_model_from_factorial_repn(
    *,
    S=None,
    J=None,
    v=None,
    w=None,
    var_index_map=None,
    variables=None,
    timing=False,
    tuple_repn=True,
):
    #
    # S[r]: the (finite) set of possible values of variable X_r
    #           Variable values s can be integers or strings
    #
    # J[i]: the (finite) set of possible configurations (rows) of factor i
    #           J[i] contains the configuration ids for factor i
    #
    # v[i,j,r]: the value of variable r in row j of factor i
    #           Note that v[i,j,r] \in S[r]
    #           Note that j \in J[i]
    #
    # w[i,j]: the log-probability of factor i in configuration j
    #           Note that j \in J[i]
    #
    # var_index_map[hr]: a dictionary that maps hashable value hr to variable name X_r
    #
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("create_MN_map_query_model_from_factorial - START")
    R = list(S.keys())
    RS = [(r, s) for r, values in S.items() for s in values]

    I = list(J.keys())
    # IJ = [(i, j) for i, values in J.items() for j in values]
    IJ = sorted(w.keys())
    IJset = set(w.keys())

    if tuple_repn:
        IRS = defaultdict(set)
        for i, j, r in v:
            if not (i, j) in IJset:
                continue
            IRS[i, r, v[i, j, r]].add(j)
    else:
        V = {(i, j): [] for i, j in IJ}
        for i, j, r in v:
            V[i, j].append(r)
        IJR = list(v.keys())

    # TODO: Figure out how to marginalize everything except the specified
    #           variables

    # TODO: consistency checks on inputs
    #   v[i,j,r] in S[r]
    #   {(i,j) in w} == IJ
    if timing:  # pragma:nocover
        timer.toc("DATA")

    #
    # Integer programming formulation
    #
    model = pe.ConcreteModel()

    # x[r,s] is 1 iff variable X_r = s
    model.x = pe.Var(RS, within=pe.Binary)
    # y[i,j] is 1 iff factor i is in configuration (row) j
    model.y = pe.Var(IJ, within=pe.Binary)

    if var_index_map is None:
        model.X = VarWrapper({rs: model.x[rs] for rs in RS})
    else:
        model.X = VarWrapper(
            {
                (r, s): model.x[index, s]
                for r, index in var_index_map.items()
                for s in S.get(index, [])
            }
        )

    if timing:  # pragma:nocover
        timer.toc("VARIABLES")

    # Each variable X_r only assumes one value
    def c1_(M, r):
        return sum(M.x[r, s] for s in S[r]) == 1

    model.c1 = pe.Constraint(R, rule=c1_)

    if timing:  # pragma:nocover
        timer.toc("c1")

    # Each factor i can only be in one configration for the joint distribution
    def c2_(M, i):
        return sum(M.y[i, j] for j in J[i] if (i, j) in IJset) == 1

    model.c2 = pe.Constraint(I, rule=c2_)

    if timing:  # pragma:nocover
        timer.toc("c2")

    if tuple_repn:

        def c5_(M, i, r, s):
            return sum(M.y[i, j] for j in IRS[i, r, s]) == M.x[r, s]

        model.c5 = pe.Constraint(sorted(IRS.keys()), rule=c5_)

        if timing:  # pragma:nocover
            timer.toc("c5")

    else:
        # Factor i cannot assume configuration j unless its corresponding
        # variables are set to the correct values
        def c3_(M, i, j, r):
            return M.y[i, j] <= M.x[r, v[i, j, r]]

        model.c3 = pe.Constraint(IJR, rule=c3_)

        if timing:  # pragma:nocover
            timer.toc("c3")

        # If factor i is not in configuration j, then at least one of its
        # corresponding variables is not set to the values for configuration j
        def c4_(M, i, j):
            return sum(M.x[r, v[i, j, r]] for r in V.get((i, j), [])) <= M.y[i, j] + (
                len(V.get((i, j), [])) - 1
            )

        model.c4 = pe.Constraint(IJ, rule=c4_)

        if timing:  # pragma:nocover
            timer.toc("c4")

    # Maximize the sum of log-values of all posible factors and configurations
    model.o = pe.Objective(
        expr=sum(w[i, j] * model.y[i, j] for i, j in IJ), sense=pe.maximize
    )

    if timing:  # pragma:nocover
        timer.toc("create_MN_map_query_model_from_factorial - STOP")

    return model


def solve_pyomo_map_query_model(
    model,
    *,
    solver="gurobi",
    tee=False,
    with_fixed=False,
    timing=False,
    solver_options=None,
):
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("optimize_map_query_model - START")
    opt = pe.SolverFactory(solver)
    if solver_options:
        opt.options = solver_options
    if timing:  # pragma:nocover
        timer.toc("Initialize solver")
    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    res = opt.solve(model, tee=tee)
    solvetime = solver_timer.toc(None)
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
