"""
Tests for MAP query models that omit the c5 constraints, which are then
generated with cut generation when using the gurobi solver.
"""

import pytest
import pyomo.environ as pyo
import pyomo.opt

from conin.examples import markov_network as examples
from conin.examples.markov_network import random_markov_network
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.algorithms.inference.pyomo import solve_pyomo_map_query_model
from conin.representations.pyomo.common import has_omitted_c5
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
        pgm = random_markov_network(seed, density=density)
        repn = ConstrainedDiscreteMarkovNetwork(pgm).representation("pyomo")
        ref = solve_pyomo_map_query_model(
            repn.create_model(), solver=other_mip_solver
        )

        model = repn.create_model(omit_c5=True)
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
