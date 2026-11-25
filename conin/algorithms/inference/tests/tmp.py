import pyomo.opt

# from conin.util import try_import
from conin.algorithms.inference.query import map_query
import conin.examples.markov_network

if True:
    example = conin.examples.markov_network.ABC_conin()
    kwargs = dict(method="integer_program", solver="gurobi", variables=["A"])

    results = map_query(example.pgm, **kwargs)
    print(results.solution.states)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float
    print("DONE")
