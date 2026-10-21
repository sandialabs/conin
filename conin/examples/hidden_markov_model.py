import pyomo.environ as pe
from conin.constraints import (
    OracleConstraint,
    FactorOracleConstraint,
    RecursiveConstraint,
)
from conin.constraints import (
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
    oracle_constraint_fn,
    algebraic_constraint_fn,
)
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models import HiddenMarkovModel

"""
This script has a collection of HMM test cases that can be used to
test conin capabilities.
"""


def create_hmm0():
    start_probs = {"h0": 1, "h1": 0}
    transition_probs = {
        ("h0", "h0"): 0,
        ("h0", "h1"): 1,
        ("h1", "h0"): 0,
        ("h1", "h1"): 1,
    }
    emission_probs = {
        ("h0", "o0"): 1,
        ("h0", "o1"): 0,
        ("h1", "o0"): 0,
        ("h1", "o1"): 1,
    }
    hmm = HiddenMarkovModel(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm1():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.9,
        ("h0", "h1"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.8,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
    }
    hmm = HiddenMarkovModel(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm1_aos():
    """
    Unconstrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.

    For all observed state Lengths > 0, the follow behavior occurs:
    best sol should be h0*Length
    second best sol should be h1*Length
    all other sols should have prob 0
    """

    start_probs = {"h0": 0.51, "h1": 0.49}
    transition_probs = {
        ("h0", "h0"): 1.0,
        ("h1", "h1"): 1.0,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.7,
        ("h1", "o1"): 0.3,
    }
    hmm = HiddenMarkovModel(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm2():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.8,
        ("h0", "h1"): 0.1,
        ("h0", "h2"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.7,
        ("h1", "h2"): 0.1,
        ("h2", "h2"): 1.0,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
        ("h2", "o2"): 1.0,
    }
    hmm = HiddenMarkovModel(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm2_aos():
    """
    Unconstrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.

    For all observed state Lengths > 0, the follow behavior occurs:
    best sol is h0*length
    second best is h1*length
    third best is h2, h0*(length-1)
    fourth best is h1,h2, h0*(length-2)
    """
    start_probs = {"h0": 0.6, "h1": 0.3, "h2": 0.1}
    transition_probs = {
        ("h0", "h0"): 1.0,
        ("h1", "h1"): 0.99,
        ("h1", "h2"): 0.01,
        ("h2", "h0"): 1.0,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.2,
        ("h0", "o2"): 0.1,
        ("h1", "o0"): 0.7,
        ("h1", "o1"): 0.2,
        ("h1", "o2"): 0.1,
        ("h2", "o0"): 0.7,
        ("h2", "o1"): 0.2,
        ("h2", "o2"): 0.1,
    }
    hmm = HiddenMarkovModel(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_chmm1_oracle():
    hmm = create_hmm1()

    num_zeros_greater_than_nine = OracleConstraint(
        func=lambda states: sum(1 for v in states.values() if v == "h0") > 9,
        partial_func=lambda T, states: T
        - len(states)
        + sum(1 for v in states.values() if v == "h0")
        >= 10,
    )
    num_zeros_less_than_thirteen = OracleConstraint(
        func=lambda states: sum(1 for v in states.values() if v == "h0") < 13,
        partial_func=lambda T, states: sum(1 for v in states.values() if v == "h0")
        < 13,
    )
    constraints = [num_zeros_greater_than_nine, num_zeros_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm1_oracle_ve():
    """Oracle CHMM with node-scoped constraints for variable elimination."""
    hmm = create_hmm1()

    def nodes(data):
        for t in data.hmm.T:
            yield ("H", t)

    num_zeros_greater_than_nine = FactorOracleConstraint(
        func=lambda states: sum(1 for v in states.values() if v == "h0") > 9,
        nodes=nodes,
        name="num_zeros_greater_than_nine",
    )
    num_zeros_less_than_thirteen = FactorOracleConstraint(
        func=lambda states: sum(1 for v in states.values() if v == "h0") < 13,
        nodes=nodes,
        name="num_zeros_less_than_thirteen",
    )
    constraints = [num_zeros_greater_than_nine, num_zeros_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm1_pyomo():
    hmm = create_hmm1()

    @pyomo_constraint_fn()
    def num_zeros_greater_than_nine(M, D):
        M.h0_lower = pe.Constraint(expr=sum(M.V("H", t, "h0") for t in D.hmm.T) >= 10)

    @pyomo_constraint_fn()
    def num_zeros_less_than_thirteen(M, D):
        M.h0_upper = pe.Constraint(expr=sum(M.V("H", t, "h0") for t in D.hmm.T) <= 12)

    constraints = [num_zeros_greater_than_nine, num_zeros_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm1_algebraic():
    hmm = create_hmm1()

    @algebraic_constraint_fn()
    def num_zeros_greater_than_nine(M, D):
        return sum(M.V("H", t, "h0") for t in D.hmm.T) >= 10

    @algebraic_constraint_fn()
    def num_zeros_less_than_thirteen(M, D):
        return sum(M.V("H", t, "h0") for t in D.hmm.T) <= 12

    constraints = [num_zeros_greater_than_nine, num_zeros_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm1_pyomo_aos():
    """
    Constrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.
    Modification of create_hmm1_aos to add constraints

    Based on observed state Lengths > 0, the follow behavior occurs:
    if length > 12, only sol with prob > 0 is h1*length
    if length <= 12, best sol h0*length, second best h1*length
    """
    hmm = create_hmm1_aos()

    @pyomo_constraint_fn()
    def num_zeros_less_than_thirteen(M, D):
        M.h0_upper = pe.Constraint(expr=sum(M.V("H", t, "h0") for t in D.hmm.T) <= 12)

    constraints = [num_zeros_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm1_algebraic_aos():
    """
    Constrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.
    Modification of create_hmm1_aos to add constraints

    Based on observed state Lengths > 0, the follow behavior occurs:
    if length > 12, only sol with prob > 0 is h1*length
    if length <= 12, best sol h0*length, second best h1*length
    """
    hmm = create_hmm1_aos()

    @algebraic_constraint_fn()
    def num_zeros_less_than_thirteen(M, D):
        return sum(M.V("H", t, "h0") for t in D.hmm.T) <= 12

    constraints = [num_zeros_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm2_pyomo_aos():
    """
    Constrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.
    Modification of create_hmm2_aos to add constraints

    Based on observed state Lengths, the follow behavior occurs:
    if length > 12, second best can't happen, move sols up one
    best sol is h0*length
    second best is h1*length,
    third best is h2, h0*(length-1)
    fourth best is h1,h2, h0*(length-2)
    """
    hmm = create_hmm2_aos()

    @pyomo_constraint_fn()
    def num_ones_less_than_thirteen(M, D):
        M.h0_upper = pe.Constraint(expr=sum(M.V("H", t, "h1") for t in D.hmm.T) <= 12)

    constraints = [num_ones_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm2_algebraic_aos():
    """
    Constrained HMM example for testing AOS capabilities.
    Designed so the known solutions should be well-ordered.
    Modification of create_hmm2_aos to add constraints

    Based on observed state Lengths, the follow behavior occurs:
    if length > 12, second best can't happen, move sols up one
    best sol is h0*length
    second best is h1*length,
    third best is h2, h0*(length-1)
    fourth best is h1,h2, h0*(length-2)
    """
    hmm = create_hmm2_aos()

    @algebraic_constraint_fn()
    def num_ones_less_than_thirteen(M, D):
        return sum(M.V("H", t, "h1") for t in D.hmm.T) <= 12

    constraints = [num_ones_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm1_toulbar2():
    hmm = create_hmm1()

    @toulbar2_constraint_fn()
    def num_zeros_greater_than_nine(M, D):
        M.AddGeneralizedLinearConstraint([M.V("H", t, "h0") for t in D.hmm.T], ">=", 10)

    @toulbar2_constraint_fn()
    def num_zeros_less_than_thirteen(M, D):
        M.AddGeneralizedLinearConstraint([M.V("H", t, "h0") for t in D.hmm.T], "<=", 12)

    constraints = [num_zeros_greater_than_nine, num_zeros_less_than_thirteen]

    chmm = ConstrainedHiddenMarkovModel(hmm, constraints=constraints)
    return chmm


def create_chmm1_recursive(lb=10, ub=12):
    """Recursive CHMM: the number of ``"h0"`` states is between ``lb`` and ``ub``.

    The same condition as :func:`create_chmm1_oracle` (with the defaults),
    written as a :class:`~conin.constraints.RecursiveConstraint` for
    :func:`conin.inference.recursive.recursive_a_star`.  The state is the
    number of ``"h0"`` states so far.
    """
    hmm = create_hmm1()

    num_zeros = RecursiveConstraint(
        initial=lambda h: 1 if h == "h0" else 0,
        update=lambda n, h: n + 1 if h == "h0" else n,
        feasible=lambda n: lb <= n <= ub,
        feasible_partial=lambda n, t, T: n + (T - t) >= lb and n <= ub,
        name="num_zeros",
    )

    chmm = ConstrainedHiddenMarkovModel(hmm, num_zeros)
    return chmm
