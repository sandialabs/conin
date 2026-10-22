Oracle Constraint Helpers
=========================

``conin.constraints.oracle_library`` provides reusable ``OracleConstraint`` objects and
factory functions for common sequence constraints. These helpers are most useful
with hidden Markov models, where an oracle constraint evaluates a candidate
hidden-state sequence directly.

An oracle constraint is called with a dict that maps keys (for HMMs, the time
steps) to states, ``{0: "h0", 1: "h1", ...}``; sequence constraints read the
states in key order.

Ready-made Constraints
----------------------

``all_diff_constraint`` and ``citation_constraint`` are already constructed
``OracleConstraint`` instances:

.. testcode::

   from conin.constraints.oracle_library import all_diff_constraint, citation_constraint

   assert all_diff_constraint({0: "a", 1: "b", 2: "c"})
   assert not all_diff_constraint({0: "a", 1: "b", 2: "a"})

   assert citation_constraint({0: "a", 1: "a", 2: "b", 3: "b", 4: "c"})
   assert not citation_constraint({0: "a", 1: "a", 2: "b", 3: "a"})

Occurrence Constraints
----------------------

Use these helpers when a state must appear a minimum, maximum, or exact number
of times in a sequence.

.. testcode::

   from conin.constraints.oracle_library import (
       appears_at_least_once_constraint,
       does_not_occur_constraint,
       has_exact_number_of_occurences_constraint,
       has_maximum_number_of_occurences_constraint,
       has_minimum_number_of_occurences_constraint,
   )

   at_least_two_h0 = has_minimum_number_of_occurences_constraint(val="h0", count=2)
   at_most_three_h1 = has_maximum_number_of_occurences_constraint(val="h1", count=3)
   exactly_one_h2 = has_exact_number_of_occurences_constraint(val="h2", count=1)
   includes_h0 = appears_at_least_once_constraint("h0")
   excludes_h3 = does_not_occur_constraint("h3")

   assert at_least_two_h0({0: "h0", 1: "h1", 2: "h0"})
   assert at_most_three_h1({0: "h1", 1: "h1"})
   assert exactly_one_h2({0: "h0", 1: "h2", 2: "h1"})
   assert includes_h0({0: "h1", 1: "h0"})
   assert excludes_h3({0: "h0", 1: "h1"})

Ordering Constraints
--------------------

Ordering helpers constrain where states may appear relative to one another or to
a time window.

.. testcode::

   from conin.constraints.oracle_library import (
       always_appears_after_constraint,
       always_appears_before_constraint,
       appears_at_least_once_after_constraint,
       appears_at_least_once_before_constraint,
       fix_final_state_constraint,
       occurs_at_least_once_in_time_frame_constraint,
       occurs_only_in_time_frame_constraint,
   )

   before = always_appears_before_constraint("start", "stop")
   after = always_appears_after_constraint("stop", "start")
   once_before = appears_at_least_once_before_constraint("start", "stop")
   once_after = appears_at_least_once_after_constraint("stop", "start")
   final_stop = fix_final_state_constraint("stop")
   only_middle = occurs_only_in_time_frame_constraint("middle", lower_t=1, upper_t=4)
   at_least_middle = occurs_at_least_once_in_time_frame_constraint(
       "middle",
       lower_t=1,
       upper_t=3,
   )

   assert before({0: "start", 1: "start", 2: "stop"})
   assert after({0: "start", 1: "stop"})
   assert once_before({0: "start", 1: "stop"})
   assert once_after({0: "start", 1: "stop"})
   assert final_stop({0: "start", 1: "stop"})
   assert only_middle({0: "start", 1: "middle", 2: "middle", 3: "stop"})
   assert at_least_middle({0: "start", 1: "middle", 2: "stop"})

Combining Constraints
---------------------

Oracle constraints can be combined with Boolean helper functions.

.. testcode::

   from conin.constraints.oracle_library import (
       and_constraints,
       appears_at_least_once_constraint,
       does_not_occur_constraint,
       not_constraint,
       or_constraints,
       xor_constraints,
   )

   has_a = appears_at_least_once_constraint("a")
   has_b = appears_at_least_once_constraint("b")
   no_c = does_not_occur_constraint("c")

   assert and_constraints([has_a, no_c])({0: "a", 1: "b"})
   assert or_constraints([has_a, has_b])({0: "b"})
   assert xor_constraints([has_a, has_b])({0: "a"})
   assert not_constraint(has_a)({0: "b", 1: "c"})

Each helper returns an ``OracleConstraint`` with a ``partial_func`` when the code
can safely evaluate partial sequences during search.

Reference
---------

.. automodule:: conin.constraints.oracle_library
   :members:
   :undoc-members:
