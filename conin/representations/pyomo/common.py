"""Shared Pyomo helpers for the ``"pyomo"`` and ``"pyomo_network_flow"`` representations.

* :class:`PyomoVarWrapper`: the ``model.V(...)`` accessor that user constraint
  functions see (defined in :mod:`conin.util.var_wrapper`, re-exported here);
* :func:`apply_pyomo_constraints`: the single constraint applier for the
  Pyomo-backend families (``"pyomo"`` and ``"algebraic"``), including the
  smoek->Pyomo translation (:func:`add_algebraic_constraints_to_pyomo_model`);
* :func:`create_MN_pyomo_map_query_model_from_factorial_repn`: the IP built
  from a factor representation ``(S, J, v, w)``, plus the c5 helpers
  (:func:`add_c5_constraints`, :func:`has_omitted_c5`,
  :class:`C5LazyCutGenerator`);
* :class:`PyomoFactorRepresentation`: the base class of the MN/BN/DBN/HMM
  ``"pyomo"`` representations.
"""

from collections import defaultdict

import munch
import pyomo.environ as pe
from pyomo.common.timing import TicTocTimer

from conin.constraints import PyomoConstraint
from conin.constraints.algebraic import AlgebraicConstraint
from conin.representations.base import Representation
from conin.util import try_import, State
from conin.util.var_wrapper import PyomoVarWrapper

with try_import() as smoek_available:
    import smoek

with try_import() as gurobipy_available:
    import gurobipy


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


def add_algebraic_constraints_to_pyomo_model(*, pgm, constraints, model, data):
    """Translate smoek (algebraic) constraints into Pyomo components on ``model``.

    ``model.V`` must be the :class:`PyomoVarWrapper` of the model.  ``pgm`` is
    unused (kept for symmetry with the other backends).
    """
    if not smoek_available:
        raise TypeError(
            f"The smoek package must be installed to use algebraic constraints."
        )

    smoek_model = smoek.model()
    for func in constraints:
        func(smoek_model, data)
    smoek_model._update_smoek_components()
    smoek.pymodel.pyomo.generate(
        model=smoek_model,
        pyomo_model=model,
        data=data,
        component_map=dict(V=model.V),
    )
    return model


def apply_pyomo_constraints(*, pgm, constraints, model, data):
    """Apply constraints of a Pyomo-backend family to a Pyomo ``model``.

    This is the single constraint applier for the ``"pyomo"`` and
    ``"pyomo_network_flow"`` representations.

    Parameters
    ----------
    pgm : GraphicalModel
        The (possibly unrolled) model the formulation was built from.
    constraints : sequence of PyomoConstraint or AlgebraicConstraint
        Constraints of one family.
    model : pyomo.environ.ConcreteModel
        The model; constraint functions see ``model.V(...)``.
    data : object
        Passed to each constraint function.

    Returns
    -------
    pyomo.environ.ConcreteModel
    """
    if len(constraints) == 0:
        return model

    if isinstance(constraints[0], PyomoConstraint):
        for func in constraints:
            assert isinstance(
                func, PyomoConstraint
            ), f"Unexpected constraint type ({type(func)}) when performing inference with an integer program. If the first constraint is a pyomo constraint, then all subsequent contraints must be the same."
            model = func(model, data)
        return model

    if isinstance(constraints[0], AlgebraicConstraint):
        return add_algebraic_constraints_to_pyomo_model(
            pgm=pgm, constraints=constraints, model=model, data=data
        )

    raise TypeError(
        f"Unexpected constraint type ({type(constraints[0])}) when performing inference with an integer program."
    )


def has_omitted_c5(model):
    """Returns ``True`` if the model was created with ``use_cut_generation=True`` and the
    c5 constraints have not been added."""
    return getattr(model, "_c5_data", None) is not None


def add_c5_constraints(model, IRS):
    """Add the full set of c5 constraints to a model that was created with
    ``use_cut_generation=True``.  This is a no-op for models that already include c5.
    """

    def c5_(M, i, r, s):
        return sum(M.y[i, j] for j in IRS[i, r, s]) == M.x[r, s]

    model.c5 = pe.Constraint(sorted(IRS.keys()), rule=c5_)
    return model


class C5LazyCutGenerator:
    """
    Gurobi callback that generates the c5 constraints on demand for models
    created with ``use_cut_generation=True``.

    Without c5, the factor configuration variables (y) are decoupled from
    the variable-value indicators (x).  This callback separates violated c5
    rows:

    * MIPSOL: every violated c5 row for the candidate integer solution is
      added as a lazy constraint.  This is required for correctness.  Only
      the violated side of each c5 equality is added.
    * MIPNODE (optional, ``user_cuts=True``): the most violated c5 rows for
      the node LP relaxation are added as user cuts to tighten the bound.

    Each cut is an exact c5 row, so the optimal solution is identical to the
    solution of the full model.
    """

    def __init__(self, model, *, user_cuts=False, tol=1e-6, max_cuts_per_node=100):
        self.model = model
        self.IRS = model._c5_data.IRS
        self.user_cuts = user_cuts
        self.tol = tol
        self.max_cuts_per_node = max_cuts_per_node
        self.variables = list(model.x.values()) + list(model.y.values())
        # Map from (i,r,s,sense) to the Pyomo constraint created for it
        self.cuts = {}
        self.num_lazy_cuts = 0
        self.num_user_cuts = 0

    def _get_rows(self):
        # WEH - Should we cache this?  This could be big.
        model = self.model
        for key in sorted(self.IRS.keys()):
            yield (
                key,
                [model.y[key[0], j] for j in self.IRS[key]],
                model.x[key[1], key[2]],
            )

    def _violated_rows(self, tol, skip_added=False):
        """
        Returns a list of tuples (violation, cut_key), where cut_key is
        (i, r, s, sense).  If sense is 1, then sum y[i,j] > x[r,s] and the
        cut is sum y[i,j] <= x[r,s].  Otherwise, sense is -1 and the cut is
        sum y[i,j] >= x[r,s].
        """
        for key, yvars, xvar in self._get_rows():
            diff = sum(y.value for y in yvars) - xvar.value
            if diff > tol:
                cut_key = key + (1,)
            elif diff < -tol:
                cut_key = key + (-1,)
            else:
                continue
            if skip_added and cut_key in self.cuts:
                continue
            yield (abs(diff), cut_key)

    def _get_cut(self, cut_key):
        # N.B. The gurobi_persistent cbLazy() and cbCut() methods do not
        # support equality constraints, so each side of c5 is added separately.
        con = self.cuts.get(cut_key, None)
        if con is None:
            i, r, s, sense = cut_key
            lhs = sum(self.model.y[i, j] for j in self.IRS[i, r, s])
            rhs = self.model.x[r, s]
            if sense > 0:
                con = self.model.c5_cuts.add(lhs <= rhs)
            else:
                con = self.model.c5_cuts.add(lhs >= rhs)
            self.cuts[cut_key] = con
        return con

    def __call__(self, cb_m, cb_opt, cb_where):
        GRB = gurobipy.GRB
        if cb_where == GRB.Callback.MIPSOL:
            cb_opt.cbGetSolution(vars=self.variables)
            for _, cut_key in self._violated_rows(0.5):
                cb_opt.cbLazy(self._get_cut(cut_key))
                self.num_lazy_cuts += 1

        elif self.user_cuts and cb_where == GRB.Callback.MIPNODE:
            if cb_opt.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            cb_opt.cbGetNodeRel(vars=self.variables)
            violated = list(self._violated_rows(self.tol, skip_added=True))
            violated.sort(key=lambda t: -t[0])
            for _, cut_key in violated[: self.max_cuts_per_node]:
                cb_opt.cbCut(self._get_cut(cut_key))
                self.num_user_cuts += 1


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
    omit_c5=False,
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
    # omit_c5: If True, the c5 constraints are not created.  Instead, the data
    #           needed to generate them is stored in model._c5_data, and the
    #           model.c5_cuts constraint list is used to store generated cuts.
    #
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("create_MN_map_query_model_from_factorial - START")
    R = list(S.keys())
    RS = [(r, s) for r, values in S.items() for s in values]

    J_keys = list(J.keys())
    # IJ = [(i, j) for i, values in J.items() for j in values]
    IJ = sorted(w.keys())
    IJset = set(w.keys())

    if tuple_repn:
        IRS = defaultdict(set)
        for i, j, r in v:
            if not (i, j) in IJset:
                continue
            IRS[i, r, v[i, j, r]].add(j)
    else:  # pragma:nocover
        v = {(i, j): [] for i, j in IJ}
        for i, j, r in v:
            v[i, j].append(r)
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
        model.V = PyomoVarWrapper({rs: model.x[rs] for rs in RS})
    else:
        model.V = PyomoVarWrapper(
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

    model.c2 = pe.Constraint(J_keys, rule=c2_)

    if timing:  # pragma:nocover
        timer.toc("c2")

    if tuple_repn:
        if omit_c5:
            # The c5 constraints are generated with cut generation when solving
            model._c5_data = munch.Munch(IRS=IRS)
            model.c5_cuts = pe.ConstraintList()
        else:
            add_c5_constraints(model, IRS)

        if timing:  # pragma:nocover
            timer.toc("c5")

    else:  # pragma:nocover
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
            return sum(M.x[r, v[i, j, r]] for r in model.V.get((i, j), [])) <= M.y[
                i, j
            ] + (len(model.V.get((i, j), [])) - 1)

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


class PyomoFactorRepresentation(Representation):
    """The factor-repn integer program of a constrained MN/BN/DBN/HMM.

    The representation holds only evidence-independent data: the factor
    representation ``(S, J, v, w)`` of the (unrolled) Markov network, the
    variable-name map and the constraints.
    :meth:`create_model` builds a fresh Pyomo model on every call, fixes the
    evidence and then applies the constraints.

    Attributes
    ----------
    S, J, v, w : dict
        The factor representation (see
        :func:`create_MN_pyomo_map_query_model_from_factorial_repn`).
    var_index_map : dict or None
        Node -> internal variable name, or ``None`` (nodes used as-is).
    pgm : GraphicalModel
        The model passed to the constraint applier (the MN, BN or unrolled BN/DBN).
    constraints : tuple
        The user constraints (family ``"pyomo"`` or ``"algebraic"``).
    nodes : tuple
        The nodes of the (unrolled) model the IP ranges over.

    Notes
    -----
    Nothing is mutated after construction, so one representation can serve
    several queries.  The factor dicts are shared with every model built from
    them, but the IP builder only reads them.
    """

    kind = "pyomo"

    def __init__(self, *, S, J, v, w, var_index_map, pgm, constraints, nodes):
        self.S = S
        self.J = J
        self.v = v
        self.w = w
        self.var_index_map = var_index_map
        self.pgm = pgm
        self.constraints = tuple(constraints)
        self.nodes = tuple(nodes)

    def constraint_data(self, *, evidence=None, variables=None):
        """The ``data`` object passed to the constraint functions for a query.

        Subclasses add their structure fields (``T``, ``start``, ``hmm``, ...).
        The constrained model's own ``data`` is not passed (as before PR 4a).
        """
        return munch.Munch(variables=variables, evidence=evidence)

    def create_model(
        self, *, evidence=None, variables=None, omit_c5=False, timing=False
    ):
        """Build a fresh Pyomo model for one query.

        Parameters
        ----------
        evidence : dict, optional
            ``{node: state}``; each ``model.V(node, state)`` is fixed to 1.
        variables : list, optional
            Passed to the constraint functions as ``data.variables``.
        omit_c5 : bool, optional
            Omit the c5 constraints (they are generated as cuts with gurobi).
        timing : bool, optional
            Print timing information.

        Returns
        -------
        pyomo.environ.ConcreteModel
        """
        model = create_MN_pyomo_map_query_model_from_factorial_repn(
            S=self.S,
            J=self.J,
            v=self.v,
            w=self.w,
            var_index_map=self.var_index_map,
            variables=variables,
            timing=timing,
            omit_c5=omit_c5,
        )

        if evidence:
            for k, val in evidence.items():
                model.V(k, State(val)).fix(1)

        if self.constraints:
            apply_pyomo_constraints(
                pgm=self.pgm,
                constraints=self.constraints,
                model=model,
                data=self.constraint_data(evidence=evidence, variables=variables),
            )
        return model
