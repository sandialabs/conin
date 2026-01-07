from munch import Munch
import munch
import numpy as np
import pyomo.environ as pyo

from conin.constraint import pyomo_constraint_fn, toulbar2_constraint_fn
from conin.util import try_import
from conin.markov_network import (
    DiscreteFactor,
    DiscreteMarkovNetwork,
    ConstrainedDiscreteMarkovNetwork,
)

with try_import() as pgmpy_available:
    from pgmpy.models import MarkovNetwork as pgmpy_MarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor as pgmpy_DiscreteFactor


#
# example6
#


def example6_conin():
    """
    See Obbens, p.18

    Non-uniform weights are applied to the A_B factors to remove a degeneracy w.r.t. the
    value of variable A.
    """
    pgm = DiscreteMarkovNetwork()
    pgm.states = {"A": [0, 1], "B": [0, 1]}
    pgm.edges = [("A", "B")]
    f1 = DiscreteFactor(["A"], {0: 1, 1: 1})
    f2 = DiscreteFactor(["B"], {0: 1, 1: 2})
    f3 = DiscreteFactor(["A", "B"], {(0, 0): 1, (0, 1): 3, (1, 0): 1, (1, 1): 1})
    pgm.factors = [f1, f2, f3]

    return Munch(pgm=pgm, solution=[])


def example6_pgmpy():
    """
    See Obbens, p.18

    Non-uniform weights are applied to the A_B factors to remove a degeneracy w.r.t. the
    value of variable A.
    """
    pgm = pgmpy_MarkovNetwork()
    pgm.add_nodes_from(["A", "B"])
    pgm.add_edge("A", "B")
    f1 = pgmpy_DiscreteFactor(["A"], [2], [1, 1])
    f2 = pgmpy_DiscreteFactor(["B"], [2], [1, 2])
    f3 = pgmpy_DiscreteFactor(["A", "B"], [2, 2], [1, 3, 1, 1])
    pgm.add_factors(f1, f2, f3)

    return Munch(pgm=pgm, solution=[])


#
# ABC
#


def ABC_conin():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights, so the MPE solution is defined by the weights for the
    factors that describe the individual variables.

    The MPE solution is A:2, B:2, C:1.
    """
    pgm = DiscreteMarkovNetwork()
    pgm.states = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]}
    pgm.edges = [("A", "B"), ("B", "C"), ("A", "C")]
    f1 = DiscreteFactor(nodes=["A"], values=[1, 1, 2])
    f2 = DiscreteFactor(nodes=["B"], values=[1, 1, 3])
    f3 = DiscreteFactor(nodes=["C"], values=[1, 2, 1])
    f4 = DiscreteFactor(nodes=["A", "B"], values=np.ones(9))
    f5 = DiscreteFactor(nodes=["B", "C"], values=np.ones(9))
    f6 = DiscreteFactor(nodes=["A", "C"], values=np.ones(9))
    pgm.factors = [f1, f2, f3, f4, f5, f6]

    return Munch(pgm=pgm, solution={"A": 2, "B": 2, "C": 1})


def ABC_pgmpy():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights, so the MPE solution is defined by the weights for the
    factors that describe the individual variables.

    The MPE solution is A:2, B:2, C:1.
    """
    pgm = pgmpy_MarkovNetwork()
    pgm.add_nodes_from(["A", "B", "C"])
    pgm.add_edge("A", "B")
    pgm.add_edge("B", "C")
    pgm.add_edge("A", "C")
    f1 = pgmpy_DiscreteFactor(["A"], [3], [1, 1, 2])
    f2 = pgmpy_DiscreteFactor(["B"], [3], [1, 1, 3])
    f3 = pgmpy_DiscreteFactor(["C"], [3], [1, 2, 1])
    f4 = pgmpy_DiscreteFactor(["A", "B"], [3, 3], np.ones(9))
    f5 = pgmpy_DiscreteFactor(["B", "C"], [3, 3], np.ones(9))
    f6 = pgmpy_DiscreteFactor(["A", "C"], [3, 3], np.ones(9))
    pgm.factors = [f1, f2, f3, f4, f5, f6]

    return Munch(pgm=pgm, solution={"A": 2, "B": 2, "C": 1})


def ABC_constrained_pyomo_pgmpy():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MPE solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MPE solution is A:0, B:2, C:1.
    """
    pgm = ABC_pgmpy()

    @pyomo_constraint_fn()
    def constraint_fn(model):
        @model.Constraint([0, 1, 2])
        def diff(M, s):
            return M.X["A", s] + M.X["B", s] + M.X["C", s] <= 1

    import conin.common.pgmpy

    pgm = conin.common.pgmpy.convert_pgmpy_to_conin(pgm.pgm)
    cpgm = ConstrainedDiscreteMarkovNetwork(pgm, constraints=[constraint_fn])
    return Munch(pgm=cpgm, solution={"A": 0, "B": 2, "C": 1})


def ABC_constrained_pyomo_conin():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MPE solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MPE solution is A:0, B:2, C:1.
    """
    pgm = ABC_conin()

    @pyomo_constraint_fn()
    def constraint_fn(model):
        @model.Constraint([0, 1, 2])
        def diff(M, s):
            return M.X["A", s] + M.X["B", s] + M.X["C", s] <= 1

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=[constraint_fn])
    return Munch(pgm=cpgm, solution={"A": 0, "B": 2, "C": 1})


def ABC_constrained_toulbar2_pgmpy():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MPE solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MPE solution is A:0, B:2, C:1.
    """
    pgm = ABC_pgmpy()

    @toulbar2_constraint_fn()
    def constraint_fn(M):
        for i in [0, 1, 2]:
            M.AddGeneralizedLinearConstraint(
                [(M.X["A"], i, 1), (M.X["B"], i, 1), (M.X["C"], i, 1)], "<=", 1
            )

    import conin.common.pgmpy

    pgm = conin.common.pgmpy.convert_pgmpy_to_conin(pgm.pgm)
    cpgm = ConstrainedDiscreteMarkovNetwork(pgm, constraints=[constraint_fn])
    return Munch(pgm=cpgm, solution={"A": 0, "B": 2, "C": 1})


def ABC_constrained_toulbar2_conin():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MPE solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MPE solution is A:0, B:2, C:1.
    """
    pgm = ABC_conin()

    @toulbar2_constraint_fn()
    def constraints(M):
        for i in [0, 1, 2]:
            M.AddGeneralizedLinearConstraint(
                [(M.X["A"], i, 1), (M.X["B"], i, 1), (M.X["C"], i, 1)], "<=", 1
            )

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=[constraints])
    return Munch(pgm=cpgm, solution={"A": 0, "B": 2, "C": 1})
