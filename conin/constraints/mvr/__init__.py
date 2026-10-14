"""
Mediation variable representations (MVRs) for constrained hidden Markov models.

An MVR is an auxiliary automaton that tracks a constraint on hidden-state
sequences.  This package contains the MVR containers (``HomMVR``,
``InhomMVR``), the ``MVRConstraint`` factory, the MVR primitives and
operators, and the MVR formula language (``build_mvr``,
``build_mvr_functor``).  Everything here depends only on NumPy; the optional
``greenery`` dependency is needed only by ``mvr_regex``.
"""

from .base import BaseMVR, HomMVR, InhomMVR, MVR_MatVecRepn
from .constraint import MVRConstraint, mvr_constraint_fn
from .operators import (
    MVROperator,
    mvr_operator_fn,
    mvr_timerange,
    mvr_and,
    mvr_or,
    mvr_not,
    mvr_not_yet,
    mvr_already_satisfied,
    mvr_sattime,
    mvr_setdiff,
    mvr_concatenate,
    mvr_concatenate_prefix,
    mvr_kfold_product,
    mvr_kfold_product_prefix,
    mvr_kleene_closure,
    mvr_kleene_closure_prefix,
    mvr_reverse,
    mvr_precedence,
    mvr_count,
)
from .primitives import (
    mvr_constant,
    mvr_current_state,
    mvr_current_transition,
    mvr_current_sequencelist,
    mvr_jump,
    mvr_visit_state,
    mvr_forbid_state,
    mvr_visit_transition,
    mvr_forbid_transition,
    mvr_visit_sequencelist,
    mvr_forbid_sequencelist,
    mvr_holdingtime,
    mvr_jumpcounts,
    mvr_withinbounds,
    mvr_regex,
)
from .formula import build_mvr, build_mvr_functor
