MVR Constraints
===============

``conin.constraints.mvr`` holds the mediation variable representations
(MVRs): automata over the hidden states of an HMM that express constraints on
hidden-state sequences, for the torch MVR algorithms (see
:doc:`api_inference`, :doc:`api_sampling_learning` and ``map_query(...,
method="mvr_viterbi")``). Declaring MVRs needs only NumPy; ``mvr_regex``
needs ``greenery``. Built MVRs (``HomMVR``, ``InhomMVR``) and
``MVRConstraint`` factories (called with the HMM when the ``"mvr"``
representation is built) are both in the ``"mvr"`` constraint family, so they
can be mixed in one constrained model. The notebooks in ``doc/notebooks/MVR``
and the cheat sheets there show the formula and regex languages.

Containers and factories
------------------------

.. autoclass:: conin.constraints.mvr.BaseMVR
   :members:
   :show-inheritance:

.. autoclass:: conin.constraints.mvr.HomMVR
   :show-inheritance:

.. autoclass:: conin.constraints.mvr.InhomMVR
   :show-inheritance:

.. autoclass:: conin.constraints.mvr.MVR_MatVecRepn
   :show-inheritance:

.. autoclass:: conin.constraints.mvr.MVRConstraint
   :show-inheritance:

.. autofunction:: conin.constraints.mvr.mvr_constraint_fn

Formulas
--------

.. autofunction:: conin.constraints.mvr.build_mvr

.. autofunction:: conin.constraints.mvr.build_mvr_functor

Primitives
----------

.. autofunction:: conin.constraints.mvr.mvr_constant
.. autofunction:: conin.constraints.mvr.mvr_current_state
.. autofunction:: conin.constraints.mvr.mvr_current_transition
.. autofunction:: conin.constraints.mvr.mvr_current_sequencelist
.. autofunction:: conin.constraints.mvr.mvr_jump
.. autofunction:: conin.constraints.mvr.mvr_visit_state
.. autofunction:: conin.constraints.mvr.mvr_forbid_state
.. autofunction:: conin.constraints.mvr.mvr_visit_transition
.. autofunction:: conin.constraints.mvr.mvr_forbid_transition
.. autofunction:: conin.constraints.mvr.mvr_visit_sequencelist
.. autofunction:: conin.constraints.mvr.mvr_forbid_sequencelist
.. autofunction:: conin.constraints.mvr.mvr_holdingtime
.. autofunction:: conin.constraints.mvr.mvr_jumpcounts
.. autofunction:: conin.constraints.mvr.mvr_withinbounds
.. autofunction:: conin.constraints.mvr.mvr_regex

Operators
---------

.. autoclass:: conin.constraints.mvr.MVROperator
   :show-inheritance:

.. autofunction:: conin.constraints.mvr.mvr_operator_fn
.. autofunction:: conin.constraints.mvr.mvr_timerange
.. autofunction:: conin.constraints.mvr.mvr_and
.. autofunction:: conin.constraints.mvr.mvr_or
.. autofunction:: conin.constraints.mvr.mvr_not
.. autofunction:: conin.constraints.mvr.mvr_not_yet
.. autofunction:: conin.constraints.mvr.mvr_already_satisfied
.. autofunction:: conin.constraints.mvr.mvr_sattime
.. autofunction:: conin.constraints.mvr.mvr_setdiff
.. autofunction:: conin.constraints.mvr.mvr_concatenate
.. autofunction:: conin.constraints.mvr.mvr_concatenate_prefix
.. autofunction:: conin.constraints.mvr.mvr_kfold_product
.. autofunction:: conin.constraints.mvr.mvr_kfold_product_prefix
.. autofunction:: conin.constraints.mvr.mvr_kleene_closure
.. autofunction:: conin.constraints.mvr.mvr_kleene_closure_prefix
.. autofunction:: conin.constraints.mvr.mvr_reverse
.. autofunction:: conin.constraints.mvr.mvr_precedence
.. autofunction:: conin.constraints.mvr.mvr_count
