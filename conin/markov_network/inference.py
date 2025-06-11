import munch
import pyomo.environ as pe
from pyomo.common.timing import tic, toc

from .factor_repn import extract_factor_representation, State


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

    def __getitem__(self, index):
        r,s = index
        if type(s) is not State:
            s = State(s)
        return dict.__getitem__(self, (r, s))


def create_MN_map_query_model(pgm, X=None, evidence=None):
    S, J, v, w = extract_factor_representation(pgm)
    model = create_MN_map_query_model_from_factorial_repn(S=S, J=J, v=v, w=w, X=X)

    if evidence is not None:
        for k, v in evidence.items():
            model.X[k, State(v)].fix(1)

    return model



def create_MN_map_query_model_from_factorial_repn(
    *, S=None, J=None, v=None, w=None, X=None
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
    # X[hr]: a dictionary that maps hashable value hr to variable name X_r
    #
    tic()
    R = list(S.keys())
    RS = [(r, s) for r, values in S.items() for s in values]

    I = list(J.keys())
    # IJ = [(i, j) for i, values in J.items() for j in values]
    IJ = list(sorted(w.keys()))
    IJset = set(w.keys())

    V = {(i, j): [] for i, j in IJ}
    for i, j, r in v:
        V[i, j].append(r)
    IJR = list(v.keys())

    # TODO: consistency checks on inputs
    #   v[i,j,r] in S[r]
    #   {(i,j) in w} == IJ
    toc("DATA")

    #
    # Integer programming formulation
    #
    model = pe.ConcreteModel()

    # x[r,s] is 1 iff variable X_r = s
    model.x = pe.Var(RS, within=pe.Binary)
    # y[i,j] is 1 iff factor i is in configuration (row) j
    model.y = pe.Var(IJ, within=pe.Binary)

    if X is None:
        model.X = VarWrapper({rs: model.x[rs] for rs in RS})
    else:
        model.X = VarWrapper({(r, s): model.x[X[r], s] for r in X for s in S[X[r]]})

    toc("VARIABLES")

    # Each variable X_r only assumes one value
    def c1_(M, r):
        return sum(M.x[r, s] for s in S[r]) == 1

    model.c1 = pe.Constraint(R, rule=c1_)

    toc("c1")

    # Each factor i can only be in one configration for the joint distribution
    def c2_(M, i):
        return sum(M.y[i, j] for j in J[i] if (i,j) in IJset) == 1

    model.c2 = pe.Constraint(I, rule=c2_)

    toc("c2")

    # Factor i cannot assume configuration j unless its corresponding variables are set to the correct values
    def c3_(M, i, j, r):
        return M.y[i, j] <= M.x[r, v[i, j, r]]

    model.c3 = pe.Constraint(IJR, rule=c3_)

    toc("c3")

    # If factor i is not in configuration j, then at least one of its corresponding variables is not set to the values for configuration j
    def c4_(M, i, j):
        return sum(M.x[r, v[i, j, r]] for r in V.get((i, j), [])) <= M.y[i, j] + (
            len(V.get((i, j), [])) - 1
        )

    model.c4 = pe.Constraint(IJ, rule=c4_)

    toc("c4")

    # Maximize the sum of log-values of all posible factors and configurations
    model.o = pe.Objective(
        expr=sum(w[i, j] * model.y[i, j] for i, j in IJ), sense=pe.maximize
    )

    toc("DONE")

    return model


def optimize_map_query_model(model, *, solver="gurobi", tee=False, with_fixed=False):
    opt = pe.SolverFactory(solver)
    res = opt.solve(model, tee=tee)
    # TODO: check optimality conditions

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
    return munch.Munch(
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
    )
