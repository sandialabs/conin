"""MAP inference with variable elimination.

The models are built by the ``"variable_elimination"`` representations
(:mod:`conin.representations.variable_elimination`); the elimination itself
is :mod:`conin.algorithms.util.variable_elimination`.
"""

from .solve import solve_variable_elimination_map_query_model
