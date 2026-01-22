import pytest
import numpy as np
import pyomo.environ as pyo

from conin.util import try_import

# from conin.bayesian_network.inference_pyomo import (
#   create_pyomo_map_query_model_BN,
# )
# from conin.markov_network.inference_pyomo import (
#    solve_pyomo_map_query_model,
# )

from conin.bayesian_network import examples

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination

    # from conin.common.pgmpy import convert_pgmpy_to_conin


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_pgmpy_issue_1177_pgmpy():
    pgm = examples.pgmpy_issue_1177_pgmpy()
    q = {
        "SCKJ": np.int64(1),
        "SHKJ": np.int64(0),
        "STKJ": np.int64(1),
    }
    variables = ["STKJ", "SCKJ", "SHKJ"]
    evidence = {"ELP": np.int64(1)}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    # pgm = convert_pgmpy_to_conin(pgm)
    # with pytest.raises(RuntimeError):
    #    model = create_pyomo_map_query_model_BN(
    #        pgm=pgm, variables=variables, evidence=evidence
    #    )
    #    results = solve_pyomo_map_query_model(model, solver="glpk")
    #    assert q == results.solution.states
