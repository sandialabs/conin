"""
Oracle constraints: predicates on (partial) assignments of states.

This package contains ``OracleConstraint`` and its decorator
``oracle_constraint_fn`` (a predicate over the whole assignment),
``FactorOracleConstraint`` (a predicate with an explicit node scope), and the
library of reusable oracle constraints (``conin.constraints.oracle.library``).
"""

from .constraint import OracleConstraint, oracle_accepts, oracle_constraint_fn
from .factor import FactorOracleConstraint
