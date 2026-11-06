"""
Tests for MAP query models that omit the c5 constraints, which are then
generated with cut generation when using the gurobi solver.
"""

import itertools
import random
from types import SimpleNamespace

import pytest
import pyomo.environ as pyo
import pyomo.opt

from conin.util import State
from conin.examples import markov_network as examples
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.algorithms.inference.pyomo import solve_pyomo_map_query_model
from conin.representations.pyomo import common
from conin.representations.pyomo.common import (
    _create_pyomo_map_query_model,
    has_omitted_c5,
)
from conin.algorithms.inference.pyomo.solve import solve_model
from conin.algorithms.inference import map_query

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None
other_mip_solver = pyomo.opt.check_available_solvers("highs", "glpk")
other_mip_solver = other_mip_solver[0] if other_mip_solver else None
try:
    gurobi_available = pyo.SolverFactory("gurobi_persistent").available(
        exception_flag=False
    )
except Exception:
    gurobi_available = False

skipif_no_mip_solvers = pytest.mark.skipif(
    not mip_solver, reason="No mip solver installed"
)
skipif_no_other_mip_solvers = pytest.mark.skipif(
    not other_mip_solver, reason="No non-gurobi mip solver installed"
)
skipif_gurobi_not_available = pytest.mark.skipif(
    not gurobi_available, reason="gurobi not installed"
)

GUROBI_QUIET = {"OutputFlag": 0}


def random_factor_repn(seed, N=10, M=15, k=3, max_vars=3, density=1.0):
    rng = random.Random(seed)
    S = {r: [State(s) for s in range(k)] for r in range(N)}
    J, v, w = {}, {}, {}
    for i in range(M):
        vars_ = rng.sample(range(N), rng.randint(2, max_vars))
        rows = list(itertools.product(*[S[r] for r in vars_]))
        rng.shuffle(rows)
        rows = rows[: max(2, int(len(rows) * density))]
        J[i] = list(range(len(rows)))
        for j, row in enumerate(rows):
            w[i, j] = rng.uniform(-5, 0)
            for r, s in zip(vars_, row):
                v[i, j, r] = s
    return S, J, v, w


def test_omit_c5_model():
    pgm = examples.ABC_conin().pgm
    repn = ConstrainedDiscreteMarkovNetwork(pgm).representation("pyomo")
    full = repn.create_model()
    small = repn.create_model(omit_c5=True)

    assert not has_omitted_c5(full)
    assert hasattr(full, "c5")
    assert has_omitted_c5(small)
    assert not hasattr(small, "c5")
    assert len(small.c5_cuts) == 0
    assert small.nconstraints() < full.nconstraints()


@skipif_gurobi_not_available
@pytest.mark.parametrize("user_cuts", [False, True])
@pytest.mark.parametrize(
    "name", ["example6_conin", "ABC_conin", "ABC2_conin", "ABC_constrained_pyomo_conin"]
)
def test_examples_gurobi_cut_generation(name, user_cuts):
    example = getattr(examples, name)()

    full = map_query(
        example.pgm,
        method="integer_program",
        solver="gurobi",
        solver_options=dict(GUROBI_QUIET),
    )
    assert "cut_stats" not in full

    cuts = map_query(
        example.pgm,
        method="integer_program",
        solver="gurobi",
        use_cut_generation=True,
        c5_user_cuts=user_cuts,
        solver_options=dict(GUROBI_QUIET),
    )
    assert cuts.solution.states == full.solution.states
    assert cuts.solution.solver_objective == pytest.approx(
        full.solution.solver_objective
    )
    assert cuts.cut_stats.num_lazy_cuts > 0
    if not user_cuts:
        assert cuts.cut_stats.num_user_cuts == 0


@skipif_gurobi_not_available
def test_ABC_evidence_gurobi_cut_generation():
    example = examples.ABC_conin()
    results = map_query(
        example.pgm,
        method="integer_program",
        solver="gurobi",
        evidence={"B": 0},
        use_cut_generation=True,
        solution_with_evidence=True,
        solver_options=dict(GUROBI_QUIET),
    )
    assert results.solution.states == {"A": 2, "B": 0, "C": 1}


@skipif_gurobi_not_available
@skipif_no_other_mip_solvers
@pytest.mark.parametrize("density", [1.0, 0.7])
@pytest.mark.parametrize("user_cuts", [False, True])
def test_random_gurobi_cut_generation(density, user_cuts):
    for seed in range(10):
        S, J, v, w = random_factor_repn(seed, density=density)
        model = _create_pyomo_map_query_model(S=S, J=J, v=v, w=w)
        ref = solve_pyomo_map_query_model(model, solver=other_mip_solver)

        model = _create_pyomo_map_query_model(S=S, J=J, v=v, w=w, omit_c5=True)
        results = solve_pyomo_map_query_model(
            model,
            solver="gurobi",
            c5_user_cuts=user_cuts,
            solver_options=dict(GUROBI_QUIET),
        )
        assert results.solution.solver_objective == pytest.approx(
            ref.solution.solver_objective
        )


@skipif_no_other_mip_solvers
def test_omit_c5_fallback():
    """
    The c5 constraints are added to the model for solvers that do not
    support cut generation.
    """
    example = examples.ABC_conin()
    with pytest.raises(
        NotImplementedError,
        match=r"The c5 constraints were omitted, but cut generation is not supported for solver=",
    ):
        map_query(
            example.pgm,
            method="integer_program",
            solver=other_mip_solver,
            use_cut_generation=True,
        )


# ---------------------------------------------------------------------
# C5LazyCutGenerator without Gurobi: the callback is called with a fake
# gurobipy, a fake callback interface and the x/y values set by hand or
# by another MIP solver.
# ---------------------------------------------------------------------

FAKE_GRB = SimpleNamespace(
    Callback=SimpleNamespace(MIPSOL="MIPSOL", MIPNODE="MIPNODE", MIPNODE_STATUS="S"),
    OPTIMAL="optimal",
)


class FakeCallback:
    """Records the lazy constraints and cuts the generator adds."""

    def __init__(self, node_status=FAKE_GRB.OPTIMAL):
        self.node_status = node_status
        self.lazy = []
        self.cuts = []

    def cbGetSolution(self, vars):
        pass

    def cbGetNodeRel(self, vars):
        pass

    def cbGet(self, what):
        assert what == FAKE_GRB.Callback.MIPNODE_STATUS
        return self.node_status

    def cbLazy(self, con):
        self.lazy.append(con)

    def cbCut(self, con):
        self.cuts.append(con)


@pytest.fixture
def fake_gurobipy(monkeypatch):
    monkeypatch.setattr(
        common, "gurobipy", SimpleNamespace(GRB=FAKE_GRB), raising=False
    )


def _abc_model_without_c5():
    pgm = examples.ABC_conin().pgm
    return (
        ConstrainedDiscreteMarkovNetwork(pgm)
        .representation("pyomo")
        .create_model(omit_c5=True)
    )


def _set_values(model, x_ones, y_values):
    """Set x[r, s] to 1 for (r, s) in x_ones and y[i, j] to y_values.get((i, j), 0)."""
    for key, var in model.x.items():
        var.set_value(1 if key in x_ones else 0)
    for key, var in model.y.items():
        var.set_value(y_values.get(key, 0))


def _c5_row_values(model, cut_key):
    i, r, s, _ = cut_key
    IRS = model._c5_data.IRS
    return sum(model.y[i, j].value for j in IRS[i, r, s]), model.x[r, s].value


def test_c5_generator_finds_the_violated_rows():
    model = _abc_model_without_c5()
    IRS = model._c5_data.IRS
    # One factor row chosen (y = 1) for x = 0 everywhere: each of its
    # variables' c5 rows has sum y = 1 > x = 0.
    i, r, s = next(key for key in sorted(IRS) if len(IRS[key]) > 0)
    j = sorted(IRS[i, r, s])[0]
    _set_values(model, x_ones=set(), y_values={(i, j): 1})

    gen = common.C5LazyCutGenerator(model)
    violated = list(gen._violated_rows(0.5))
    assert violated
    for violation, cut_key in violated:
        assert cut_key[0] == i and cut_key[3] == 1
        total, x = _c5_row_values(model, cut_key)
        assert violation == pytest.approx(total - x) and total > x
    # Every c5 row that is violated is found.
    expected = {
        key + (1,)
        for key in IRS
        if sum(model.y[key[0], jj].value for jj in IRS[key]) > model.x[key[1:]].value
    }
    assert {cut_key for _, cut_key in violated} == expected

    # x = 1 with no row chosen violates the other side (sum y < x).
    _set_values(model, x_ones={(r, s)}, y_values={})
    violated = list(gen._violated_rows(0.5))
    assert {cut_key[3] for _, cut_key in violated} == {-1}
    assert all(cut_key[1:3] == (r, s) for _, cut_key in violated)


def _satisfied(con):
    body = pyo.value(con.body)
    lower, upper = pyo.value(con.lower), pyo.value(con.upper)
    return (lower is None or body >= lower - 1e-9) and (
        upper is None or body <= upper + 1e-9
    )


def test_c5_generator_adds_each_cut_once():
    model = _abc_model_without_c5()
    IRS = model._c5_data.IRS
    gen = common.C5LazyCutGenerator(model)
    i, r, s = key = sorted(IRS)[0]
    upper = gen._get_cut(key + (1,))
    assert gen._get_cut(key + (1,)) is upper
    lower = gen._get_cut(key + (-1,))
    assert lower is not upper
    assert len(model.c5_cuts) == 2

    # The two cuts are the two sides of sum_j y[i, j] == x[r, s].
    j = sorted(IRS[key])[0]
    _set_values(model, x_ones=set(), y_values={(i, j): 1})
    assert not _satisfied(upper) and _satisfied(lower)
    _set_values(model, x_ones={(r, s)}, y_values={})
    assert _satisfied(upper) and not _satisfied(lower)
    _set_values(model, x_ones={(r, s)}, y_values={(i, j): 1})
    assert _satisfied(upper) and _satisfied(lower)


@pytest.mark.usefixtures("fake_gurobipy")
def test_c5_generator_mipnode_adds_the_most_violated_new_rows():
    model = _abc_model_without_c5()
    IRS = model._c5_data.IRS
    # A fractional relaxation: every y is 0.3 and every x is 0, so all the
    # rows with more y terms are more violated.
    for var in model.x.values():
        var.set_value(0)
    for var in model.y.values():
        var.set_value(0.3)

    gen = common.C5LazyCutGenerator(model, user_cuts=True, max_cuts_per_node=3)
    cb = FakeCallback(node_status="not optimal")
    gen(None, cb, FAKE_GRB.Callback.MIPNODE)
    assert cb.cuts == [] and gen.num_user_cuts == 0

    cb = FakeCallback()
    gen(None, cb, FAKE_GRB.Callback.MIPNODE)
    assert len(cb.cuts) == 3 and gen.num_user_cuts == 3
    violations = sorted((0.3 * len(rows) for rows in IRS.values()), reverse=True)
    added = [key for key, con in gen.cuts.items() if con in cb.cuts]
    assert sorted(
        (0.3 * len(IRS[key[:3]]) for key in added), reverse=True
    ) == pytest.approx(violations[:3])

    # The next node skips the rows already added.
    cb2 = FakeCallback()
    gen(None, cb2, FAKE_GRB.Callback.MIPNODE)
    assert len(cb2.cuts) == 3
    assert not set(map(id, cb2.cuts)) & set(map(id, cb.cuts))

    # Without user_cuts, MIPNODE does nothing.
    gen = common.C5LazyCutGenerator(model)
    cb = FakeCallback()
    gen(None, cb, FAKE_GRB.Callback.MIPNODE)
    assert cb.cuts == [] and cb.lazy == []


@skipif_no_other_mip_solvers
@pytest.mark.usefixtures("fake_gurobipy")
@pytest.mark.parametrize("density", [1.0, 0.7])
@pytest.mark.parametrize("seed", range(5))
def test_c5_lazy_cut_loop_reaches_the_full_model_optimum(seed, density):
    # Emulate Gurobi's lazy constraints: solve the model without c5, let the
    # MIPSOL callback add the violated rows, and solve again until the
    # solution violates none.
    S, J, v, w = random_factor_repn(seed, N=6, M=8, density=density)
    full = _create_pyomo_map_query_model(S=S, J=J, v=v, w=w)
    ref = solve_pyomo_map_query_model(full, solver=other_mip_solver)

    model = _create_pyomo_map_query_model(S=S, J=J, v=v, w=w, omit_c5=True)
    gen = common.C5LazyCutGenerator(model)
    opt = pyo.SolverFactory(other_mip_solver)
    for _ in range(100):
        _, infeasible = solve_model(opt, model)
        assert not infeasible
        cb = FakeCallback()
        gen(None, cb, FAKE_GRB.Callback.MIPSOL)
        if not cb.lazy:
            break
        assert all(con.parent_component() is model.c5_cuts for con in cb.lazy)
    else:
        pytest.fail("the lazy cut loop did not converge")

    assert gen.num_lazy_cuts > 0
    assert pyo.value(model.o) == pytest.approx(ref.solution.solver_objective)
