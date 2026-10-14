"""
Tests for MAP query models that omit the c5 constraints, which are then
generated with cut generation when using the gurobi solver.
"""

import itertools
import random

import pytest
import pyomo.environ as pyo
import pyomo.opt

from conin.util import State
from conin.examples import markov_network as examples
from conin.inference.mn import inference_pyomo_map_query_MN
from conin.inference.mn.inference_pyomo import (
    create_pyomo_map_query_model_MN,
    create_MN_pyomo_map_query_model_from_factorial_repn,
    solve_pyomo_map_query_model,
    has_omitted_c5,
)

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None
other_mip_solver = pyomo.opt.check_available_solvers("highs", "glpk")
other_mip_solver = other_mip_solver[0] if other_mip_solver else None
try:
    gurobi_available = pyo.SolverFactory("gurobi_persistent").available(
        exception_flag=False
    )
except:
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
    full = create_pyomo_map_query_model_MN(pgm=pgm)
    small = create_pyomo_map_query_model_MN(pgm=pgm, omit_c5=True)

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

    full = inference_pyomo_map_query_MN(
        pgm=example.pgm, solver="gurobi", solver_options=dict(GUROBI_QUIET)
    )
    assert "cut_stats" not in full

    cuts = inference_pyomo_map_query_MN(
        pgm=example.pgm,
        solver="gurobi",
        use_cut_generation=True,
        c5_user_cuts=user_cuts,
        solver_options=dict(GUROBI_QUIET),
    )
    assert cuts.solution.states == full.solution.states
    assert cuts.solution.log_factor_sum == pytest.approx(full.solution.log_factor_sum)
    assert cuts.cut_stats.num_lazy_cuts > 0
    if not user_cuts:
        assert cuts.cut_stats.num_user_cuts == 0


@skipif_gurobi_not_available
def test_ABC_evidence_gurobi_cut_generation():
    example = examples.ABC_conin()
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm,
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
        model = create_MN_pyomo_map_query_model_from_factorial_repn(S=S, J=J, v=v, w=w)
        ref = solve_pyomo_map_query_model(model, solver=other_mip_solver)

        model = create_MN_pyomo_map_query_model_from_factorial_repn(
            S=S, J=J, v=v, w=w, omit_c5=True
        )
        results = solve_pyomo_map_query_model(
            model,
            solver="gurobi",
            c5_user_cuts=user_cuts,
            solver_options=dict(GUROBI_QUIET),
        )
        assert results.solution.log_factor_sum == pytest.approx(
            ref.solution.log_factor_sum
        )


@skipif_no_other_mip_solvers
def test_omit_c5_fallback():
    """
    The c5 constraints are added to the model for solvers that do not
    support cut generation.
    """
    example = examples.ABC_conin()
    with pytest.raises(
        RuntimeError,
        match=r"The c5 constraints were omitted, but cut generation is not supported for solver=",
    ):
        results = inference_pyomo_map_query_MN(
            pgm=example.pgm, solver=other_mip_solver, use_cut_generation=True
        )
