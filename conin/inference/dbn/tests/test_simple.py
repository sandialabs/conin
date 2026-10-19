import pyomo.opt

from conin.util import try_import
from conin.inference.pyomo import inference_pyomo_map_query_DDBN
from conin.inference.toulbar2 import inference_toulbar2_map_query_DDBN

from conin.examples import dynamic_bayesian_network as examples

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None

import pytest

skipif_no_mip_solver = pytest.mark.skipif(
    not mip_solver, reason="No mip solver installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)
skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)


# ===============================================================================
#
# Pyomo tests
#
# ===============================================================================

#
# simple0
#


@skipif_no_mip_solver
def test_simple0_DDBN_conin_mip():
    example = examples.simple0_DDBN_conin()
    results = inference_pyomo_map_query_DDBN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_no_mip_solver
def test_simple0_DDBN1_pgmpy_mip():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple0_DDBN1_pgmpy()
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_DDBN(pgm=pgm, solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_no_mip_solver
def test_simple0_DDBN2_pgmpy_mip():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple0_DDBN2_pgmpy()
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_DDBN(pgm=pgm, solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


#
# simple1
#


@skipif_no_mip_solver
def test_simple1_DDBN_conin_mip():
    example = examples.simple1_DDBN_conin()
    results = inference_pyomo_map_query_DDBN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_no_mip_solver
def test_simple1_DDBN_pgmpy_mip():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple1_DDBN_pgmpy()
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_DDBN(pgm=pgm, solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


@skipif_no_mip_solver
def test2_simple1_DDBN_conin_mip():
    example = examples.simple1_DDBN_conin()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    results = inference_pyomo_map_query_DDBN(
        pgm=example.pgm, solver=mip_solver, evidence={("A", 0): 1}
    )
    assert q == results.solution.states


@skipif_pgmpy_not_available
@skipif_no_mip_solver
def test2_simple1_DDBN_pgmpy_mip():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple1_DDBN_pgmpy()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_DDBN(
        pgm=pgm, solver=mip_solver, evidence={("A", 0): 1}
    )
    assert q == results.solution.states


@skipif_no_mip_solver
def test_simple1_DDBN_constrained_pyomo_conin():
    example = examples.simple1_DDBN_constrained_pyomo_conin()
    results = inference_pyomo_map_query_DDBN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_no_mip_solver
def test_simple1_DDBN_constrained_pyomo_pgmpy():
    example = examples.simple1_DDBN_constrained_pyomo_pgmpy()
    results = inference_pyomo_map_query_DDBN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# Toulbar2 tests
#
# ===============================================================================

#
# simple0
#


@skipif_toulbar2_not_available
def test_simple0_DDBN_conin_toulbar2():
    example = examples.simple0_DDBN_conin()
    results = inference_toulbar2_map_query_DDBN(pgm=example.pgm)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_simple0_DDBN1_pgmpy_toulbar2():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple0_DDBN1_pgmpy()
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_toulbar2_map_query_DDBN(pgm=pgm)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_simple0_DDBN2_pgmpy_toulbar2():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple0_DDBN2_pgmpy()
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_toulbar2_map_query_DDBN(pgm=pgm)
    assert results.solution.states == example.solutions[0].states


#
# simple1
#


@skipif_toulbar2_not_available
def test_simple1_conin_toulbar2():
    example = examples.simple1_DDBN_conin()
    results = inference_toulbar2_map_query_DDBN(pgm=example.pgm)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_simple1_pgmpy_toulbar2():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple1_DDBN_pgmpy()
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_toulbar2_map_query_DDBN(pgm=pgm)
    assert results.solution.states == example.solutions[0].states


@skipif_toulbar2_not_available
def test2_simple1_DDBN_conin_toulbar2():
    example = examples.simple1_DDBN_conin()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    results = inference_toulbar2_map_query_DDBN(pgm=example.pgm, evidence={("A", 0): 1})
    assert q == results.solution.states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test2_simple1_DDBN_pgmpy_toulbar2():
    from conin.common.pgmpy import convert_pgmpy_to_conin

    example = examples.simple1_DDBN_pgmpy()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_toulbar2_map_query_DDBN(pgm=pgm, evidence={("A", 0): 1})
    assert q == results.solution.states


@skipif_toulbar2_not_available
def test_simple1_DDBN_constrained_toulbar2_conin():
    example = examples.simple1_DDBN_constrained_toulbar2_conin()
    results = inference_toulbar2_map_query_DDBN(pgm=example.pgm)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_simple1_DDBN_constrained_toulbar2_pgmpy():
    example = examples.simple1_DDBN_constrained_toulbar2_pgmpy()
    results = inference_toulbar2_map_query_DDBN(pgm=example.pgm)
    assert results.solution.states == example.solutions[0].states


#
# Mixed static and dynamic nodes
#


def _static_and_dynamic_DDBN():
    """A DBN with a static node ``S`` that is a parent of the dynamic node ``Z``.

    Unrolling it gives a Bayesian network whose nodes mix plain names
    (``"S"``) and tuples (``("Z", t)``).
    """
    from conin.graphical_models import (
        DiscreteCPD,
        DynamicDiscreteBayesianNetwork,
        ExpressionVariable,
    )

    t = ExpressionVariable()
    s_cpd = DiscreteCPD(node="S", values=[0.4, 0.6])
    z_start_cpd = DiscreteCPD(
        node=("Z", 0), parents=["S"], values={0: [0.9, 0.1], 1: [0.2, 0.8]}
    )
    z_trans_cpd = DiscreteCPD(
        node=("Z", t),
        parents=[("Z", t - 1)],
        values={0: [0.7, 0.3], 1: [0.1, 0.9]},
    )
    pgm = DynamicDiscreteBayesianNetwork(
        states={"S": [0, 1]},
        dynamic_states={"Z": [0, 1]},
        cpds=[s_cpd, z_start_cpd, z_trans_cpd],
        t=t,
    )
    pgm.check_model()
    # MAP by enumeration: S=1, Z0=1, Z1=1 has probability 0.6*0.8*0.9 = 0.432
    return pgm, {"S": 1, ("Z", 0): 1, ("Z", 1): 1}


@skipif_no_mip_solver
def test_static_and_dynamic_DDBN_conin_mip():
    pgm, expected = _static_and_dynamic_DDBN()
    results = inference_pyomo_map_query_DDBN(pgm=pgm, solver=mip_solver)
    assert results.solution.states == expected


@skipif_toulbar2_not_available
def test_static_and_dynamic_DDBN_conin_toulbar2():
    pgm, expected = _static_and_dynamic_DDBN()
    results = inference_toulbar2_map_query_DDBN(pgm=pgm)
    assert results.solution.states == expected
