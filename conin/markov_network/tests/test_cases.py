import numpy as np
import pyomo.environ as pyo
from conin.markov_network import ConstrainedMarkovNetwork

try:
    from pgmpy.models import MarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor
except Exception as e:
    pass


def example6():
    """
    See Obbens, p.18

    Non-uniform weights are applied to the A_B factors to remove a degeneracy w.r.t. the
    value of variable A.
    """
    pgm = MarkovNetwork()
    pgm.add_nodes_from(["A", "B"])
    pgm.add_edge("A", "B")
    f1 = DiscreteFactor(["A"], [2], [1, 1])
    f2 = DiscreteFactor(["B"], [2], [1, 2])
    f3 = DiscreteFactor(["A", "B"], [2, 2], [1, 3, 1, 1])
    pgm.add_factors(f1, f2, f3)

    return pgm


def ABC():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights, so the MAP solution is defined by the weights for the
    factors that describe the individual variables.

    The MAP solution is A:2, B:2, C:1.
    """
    pgm = MarkovNetwork()
    pgm.add_nodes_from(["A", "B", "C"])
    pgm.add_edge("A", "B")
    pgm.add_edge("B", "C")
    pgm.add_edge("A", "C")
    f1 = DiscreteFactor(["A"], [3], [1, 1, 2])
    f2 = DiscreteFactor(["B"], [3], [1, 1, 3])
    f3 = DiscreteFactor(["C"], [3], [1, 2, 1])
    f4 = DiscreteFactor(["A", "B"], [3, 3], np.ones(9))
    f5 = DiscreteFactor(["B", "C"], [3, 3], np.ones(9))
    f6 = DiscreteFactor(["A", "C"], [3, 3], np.ones(9))
    pgm.add_factors(f1, f2, f3, f4, f5, f6)

    return pgm


def ABC_constrained():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MAP solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MAP solution is A:0, B:2, C:1.
    """
    pgm = ABC()

    def constraint_fn(model):
        # @model.Constraint
        def diff_(M, s):
            return M.X["A", s] + M.X["B", s] + M.X["C", s] <= 1

        model.diff = pyo.Constraint([0, 1, 2], rule=diff_)

        return model

    cpgm = ConstrainedMarkovNetwork(pgm)
    cpgm.add_constraints(constraint_fn)

    return cpgm
