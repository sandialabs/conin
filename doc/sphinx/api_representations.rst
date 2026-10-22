Representations
===============

A representation is what a backend needs to answer queries on a
``(model, constraints)`` pair: an integer program, a cost function network,
factor tables, aligned MVRs, the HMM parameters as arrays, and so on.
``cmodel.representation(kind, **structure_params)`` builds one; every call
builds a new object. Structure parameters (such as ``start`` and ``stop`` for
DBNs and HMMs) are build arguments, while per-query inputs (evidence, solver
options) are passed to the representation's ``create_model()`` or to the
algorithm, so one representation can serve several queries.

Builders are registered per ``(kind, model_type, constraint_family)``. The
registry also decides which constraint families a constrained model accepts:
a family is accepted for a model type if some builder is registered for it.

.. list-table:: Registered kinds and constraint families
   :header-rows: 1

   * - Kind
     - Models
     - Constraint families (besides none)
     - Used by
   * - ``"pyomo"``
     - MN, BN, DBN, HMM
     - ``pyomo``, ``algebraic``
     - ``integer_program``
   * - ``"pyomo_network_flow"``
     - HMM
     - ``pyomo``, ``algebraic``
     - ``integer_program`` with ``ip_formulation="network_flow"``
   * - ``"toulbar2"``
     - MN, BN, DBN, HMM
     - ``toulbar2``, ``algebraic``
     - ``toulbar2``
   * - ``"pgmpy"``
     - MN, BN, DBN, HMM
     - ``factor_oracle``, ``algebraic``
     - ``variable_elimination``, ``log_probability_query``
   * - ``"matvec"``
     - HMM
     - (none only)
     - ``viterbi``, A*, sampling, the MVR kernels
   * - ``"oracle"``
     - HMM
     - ``oracle``
     - ``a_star``, rejection sampling
   * - ``"recursive"``
     - HMM
     - ``recursive``
     - ``a_star`` with recursive constraints
   * - ``"mvr"``
     - HMM
     - ``mvr``
     - ``mvr_viterbi`` and the torch MVR algorithms

.. doctest:: representations

   >>> from conin.constrained_models import ConstrainedHiddenMarkovModel
   >>> from conin.examples.hidden_markov_model import create_hmm1
   >>> repn = ConstrainedHiddenMarkovModel(create_hmm1()).representation("matvec")
   >>> repn.num_hidden_states, repn.transition_mat.shape
   (2, (2, 2))
   >>> repn.transition_mat.flags.writeable
   False

Registry
--------

.. autoclass:: conin.representations.Representation
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: conin.representations.register_representation

.. autofunction:: conin.representations.unregister_representation

.. autofunction:: conin.representations.build_representation

.. autofunction:: conin.representations.supported_representations

.. autofunction:: conin.representations.get_representation_builder

Pyomo
-----

.. autoclass:: conin.representations.pyomo.PyomoFactorRepresentation
   :members:
   :show-inheritance:

.. autoclass:: conin.representations.pyomo.PyomoMNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.pyomo.PyomoBNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.pyomo.PyomoDBNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.pyomo.PyomoHMMRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.pyomo.PyomoNetworkFlowHMMRepresentation
   :members:
   :show-inheritance:

.. autoclass:: conin.representations.pyomo.PyomoVarWrapper
   :members:
   :show-inheritance:

.. autofunction:: conin.representations.pyomo.common.apply_pyomo_constraints

Toulbar2
--------

.. autoclass:: conin.representations.toulbar2.Toulbar2FactorRepresentation
   :members:
   :show-inheritance:

.. autoclass:: conin.representations.toulbar2.Toulbar2MNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.toulbar2.Toulbar2BNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.toulbar2.Toulbar2DBNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.toulbar2.Toulbar2HMMRepresentation
   :show-inheritance:

The ``Toulbar2VarWrapper`` and the constraint helpers of
``conin.representations.toulbar2.common`` are documented in
:doc:`api_constraints`.

pgmpy (variable elimination)
----------------------------

.. autoclass:: conin.representations.pgmpy.PgmpyRepresentation
   :members:
   :show-inheritance:

.. autoclass:: conin.representations.pgmpy.PgmpyMNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.pgmpy.PgmpyBNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.pgmpy.PgmpyDBNRepresentation
   :show-inheritance:

.. autoclass:: conin.representations.pgmpy.PgmpyHMMRepresentation
   :show-inheritance:

.. autofunction:: conin.representations.pgmpy.apply_pgmpy_constraints

.. autofunction:: conin.representations.pgmpy.materialise_factor_oracle

HMM representations
-------------------

.. autoclass:: conin.representations.matvec.MatVecHMMRepresentation
   :members:
   :show-inheritance:

.. autoclass:: conin.representations.oracle.OracleHMMRepresentation
   :members:
   :show-inheritance:

.. autoclass:: conin.representations.recursive.RecursiveHMMRepresentation
   :members:
   :show-inheritance:

.. autoclass:: conin.representations.mvr.MVRHMMRepresentation
   :members:
   :show-inheritance:

.. autofunction:: conin.representations.mvr.align_hidden_states
