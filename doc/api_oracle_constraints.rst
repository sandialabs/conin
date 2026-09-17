Oracle Constraint Helpers
=========================

``conin.oracle_constraints`` provides reusable ``OracleConstraint`` objects and
factory functions for common sequence constraints. These helpers are most useful
with hidden Markov models, where an oracle constraint evaluates a candidate
hidden-state sequence directly.

Ready-made Constraints
----------------------

``all_diff_constraint`` and ``citation_constraint`` are already constructed
``OracleConstraint`` instances:

.. code-block:: python

   from conin.oracle_constraints import all_diff_constraint, citation_constraint

   assert all_diff_constraint(["a", "b", "c"])
   assert not all_diff_constraint(["a", "b", "a"])

   assert citation_constraint(["a", "a", "b", "b", "c"])
   assert not citation_constraint(["a", "a", "b", "a"])

Occurrence Constraints
----------------------

Use these helpers when a state must appear a minimum, maximum, or exact number
of times in a sequence.

.. code-block:: python

   from conin.oracle_constraints import (
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

   assert at_least_two_h0(["h0", "h1", "h0"])
   assert at_most_three_h1(["h1", "h1"])
   assert exactly_one_h2(["h0", "h2", "h1"])
   assert includes_h0(["h1", "h0"])
   assert excludes_h3(["h0", "h1"])

Ordering Constraints
--------------------

Ordering helpers constrain where states may appear relative to one another or to
a time window.

.. code-block:: python

   from conin.oracle_constraints import (
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

   assert before(["start", "start", "stop"])
   assert after(["start", "stop"])
   assert once_before(["start", "stop"])
   assert once_after(["start", "stop"])
   assert final_stop(["start", "stop"])
   assert only_middle(["start", "middle", "middle", "stop"])
   assert at_least_middle(["start", "middle", "stop"])

Combining Constraints
---------------------

Oracle constraints can be combined with Boolean helper functions.

.. code-block:: python

   from conin.oracle_constraints import (
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

   assert and_constraints([has_a, no_c])(["a", "b"])
   assert or_constraints([has_a, has_b])(["b"])
   assert xor_constraints([has_a, has_b])(["a"])
   assert not_constraint(has_a)(["b", "c"])

Each helper returns an ``OracleConstraint`` with a ``partial_func`` when the code
can safely evaluate partial sequences during search.

Reference
---------

.. automodule:: conin.oracle_constraints
   :members:
   :undoc-members:
