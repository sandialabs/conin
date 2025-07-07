import pytest
import numpy as np
import pyomo.environ as pyo
from conin.bayesian_network import (
    create_BN_map_query_model,
    optimize_map_query_model,
)
from . import examples

try:
    from pgmpy.inference import VariableElimination

    pgmpy_available = True
except Exception as e:
    pgmpy_available = False


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_pgmpy_issue_1177():
    pgm = examples.pgmpy_issue_1177()
    q = {
        "SCKJ": np.int64(1),
        "SHKJ": np.int64(0),
        "STKJ": np.int64(1),
    }
    variables = ["STKJ", "SCKJ", "SHKJ"]
    evidence = {"ELP": np.int64(1)}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    model = create_BN_map_query_model(pgm=pgm, variables=variables, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value
