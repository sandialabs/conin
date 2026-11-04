from munch import Munch
import numpy as np

from conin.constraints import (
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
    oracle_constraint_fn,
    smoek_constraint_fn,
)
from conin.util import MPESolution
from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork

#
# example6
#


def example6_conin():
    """
    See Obbens, p.18

    Non-uniform weights are applied to the A_B factors to remove a degeneracy w.r.t. the
    value of variable A.
    """
    f1 = DiscreteFactor(["A"], {0: 1, 1: 1})
    f2 = DiscreteFactor(["B"], {0: 1, 1: 2})
    f3 = DiscreteFactor(["A", "B"], {(0, 0): 1, (0, 1): 3, (1, 0): 1, (1, 1): 1})
    pgm = DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        edges=[("A", "B")],
        factors=[f1, f2, f3],
    )

    return Munch(pgm=pgm, solutions=[MPESolution()])


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

    f1 = DiscreteFactor(nodes=["A"], values=[1, 1, 2])
    f2 = DiscreteFactor(nodes=["B"], values=[1, 1, 3])
    f3 = DiscreteFactor(nodes=["C"], values=[1, 2, 1])
    f4 = DiscreteFactor(nodes=["A", "B"], values=np.ones(9))
    f5 = DiscreteFactor(nodes=["B", "C"], values=np.ones(9))
    f6 = DiscreteFactor(nodes=["A", "C"], values=np.ones(9))
    pgm = DiscreteMarkovNetwork(
        states={"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]},
        edges=[("A", "B"), ("B", "C"), ("A", "C")],
        factors=[f1, f2, f3, f4, f5, f6],
    )

    return Munch(pgm=pgm, solutions=[MPESolution(states={"A": 2, "B": 2, "C": 1})])


def ABC2_conin():
    """
    Unconstrained AOS example for three variables with pair-wise interactions.

    The edge interactions have equal weights, so the MPE solution is defined by the weights for the
    factors that describe the individual state variables.

    The best solution is A:2, B:2, C:1.
    Second best is A:1, B:2, C:1
    """

    f1 = DiscreteFactor(nodes=["A"], values=[10, 19, 20])
    f2 = DiscreteFactor(nodes=["B"], values=[10, 10, 30])
    f3 = DiscreteFactor(nodes=["C"], values=[10, 20, 10])
    f4 = DiscreteFactor(nodes=["A", "B"], values=np.ones(9))
    f5 = DiscreteFactor(nodes=["B", "C"], values=np.ones(9))
    f6 = DiscreteFactor(nodes=["A", "C"], values=np.ones(9))
    pgm = DiscreteMarkovNetwork(
        states={"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]},
        edges=[("A", "B"), ("B", "C"), ("A", "C")],
        factors=[f1, f2, f3, f4, f5, f6],
    )

    return Munch(
        pgm=pgm,
        solutions=[
            MPESolution(states={"A": 2, "B": 2, "C": 1}),
            MPESolution(states={"A": 1, "B": 2, "C": 1}),
        ],
    )


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
            return M.V("A", s) + M.V("B", s) + M.V("C", s) <= 1

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=constraint_fn)
    return Munch(pgm=cpgm, solutions=[MPESolution(states={"A": 0, "B": 2, "C": 1})])


def ABC2_constrained_pyomo_conin():
    """
    Constrained AOS example for three variables with pair-wise interactions.
    Based off ABC2_conin.
    We add a constraint that excludes variable assignments to values that are equal.

    The edge interactions have equal weights, so the MPE solution is defined by the weights for the
    factors that describe the individual state variables.

    The best solution is A:0, B:2, C:1.
    Second best is A:1, B:2, C:0
    """

    pgm = ABC2_conin()

    @pyomo_constraint_fn()
    def constraint_fn(model):
        @model.Constraint([0, 1, 2])
        def diff(M, s):
            return M.V("A", s) + M.V("B", s) + M.V("C", s) <= 1

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=constraint_fn)
    return Munch(
        pgm=cpgm,
        solutions=[
            MPESolution(states={"A": 0, "B": 2, "C": 1}),
            MPESolution(states={"A": 1, "B": 2, "C": 0}),
        ],
    )


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
                [M.V("A", i), M.V("B", i), M.V("C", i)], "<=", 1
            )

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=constraints)
    return Munch(pgm=cpgm, solutions=[MPESolution(states={"A": 0, "B": 2, "C": 1})])


def ABC_constrained_smoek_conin():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MPE solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MPE solution is A:0, B:2, C:1.
    """
    pgm = ABC_conin()

    @smoek_constraint_fn()
    def constraint_fn(model):
        return [
            model.V("A", s) + model.V("B", s) + model.V("C", s) <= 1 for s in [0, 1, 2]
        ]

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=constraint_fn)
    return Munch(pgm=cpgm, solutions=[MPESolution(states={"A": 0, "B": 2, "C": 1})])


def ABC2_constrained_smoek_conin():
    """
    Constrained AOS example for three variables with pair-wise interactions.
    Based off ABC2_conin.
    We add a constraint that excludes variable assignments to values that are equal.

    The edge interactions have equal weights, so the MPE solution is defined by the weights for the
    factors that describe the individual state variables.

    The best solution is A:0, B:2, C:1.
    Second best is A:1, B:2, C:0
    """

    pgm = ABC2_conin()

    @smoek_constraint_fn()
    def constraint_fn(model):
        return [
            model.V("A", s) + model.V("B", s) + model.V("C", s) <= 1 for s in [0, 1, 2]
        ]

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=constraint_fn)
    return Munch(
        pgm=cpgm,
        solutions=[
            MPESolution(states={"A": 0, "B": 2, "C": 1}),
            MPESolution(states={"A": 1, "B": 2, "C": 0}),
        ],
    )


def ABC_constrained_oracle_conin():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MPE solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MPE solution is A:0, B:2, C:1.
    """
    pgm = ABC_conin()

    @oracle_constraint_fn(nodes=["A", "B", "C"])
    def constraint_fn(states):
        values = set(x for x in states.values())
        return len(values) == 3  # All r.v. values are different

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=constraint_fn)
    return Munch(pgm=cpgm, solutions=[MPESolution(states={"A": 0, "B": 2, "C": 1})])


def ABC2_constrained_oracle_conin():
    """
    Constrained AOS example for three variables with pair-wise interactions.
    Based off ABC2_conin.
    We add a constraint that excludes variable assignments to values that are equal.

    The edge interactions have equal weights, so the MPE solution is defined by the weights for the
    factors that describe the individual state variables.

    The best solution is A:0, B:2, C:1.
    Second best is A:1, B:2, C:0
    """

    pgm = ABC2_conin()

    @oracle_constraint_fn(nodes=["A", "B", "C"])
    def constraint_fn(states):
        values = set(x for x in states.values())
        return len(values) == 3  # All r.v. values are different

    cpgm = ConstrainedDiscreteMarkovNetwork(pgm.pgm, constraints=constraint_fn)
    return Munch(
        pgm=cpgm,
        solutions=[
            MPESolution(states={"A": 0, "B": 2, "C": 1}),
            MPESolution(states={"A": 1, "B": 2, "C": 0}),
        ],
    )
