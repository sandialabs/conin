Constrained Graphical Models
============================

The following constrained probabilistic graphical models are defined in conin.

Base class
----------

.. autoclass:: conin.constrained_models.ConstrainedGraphicalModel
   :members:
   :undoc-members:
   :show-inheritance:

Markov networks
---------------

.. autoclass:: conin.constrained_models.ConstrainedDiscreteMarkovNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Discrete Bayesian networks
--------------------------

.. autoclass:: conin.constrained_models.ConstrainedDiscreteBayesianNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Hidden Markov models
--------------------

.. autoclass:: conin.constrained_models.ConstrainedHiddenMarkovModel
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic Discrete Bayesian networks
----------------------------------

.. autoclass:: conin.constrained_models.ConstrainedDynamicDiscreteBayesianNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Representations
---------------

Constrained models build backend representations lazily with
``representation(kind)``. Builders are registered per
``(kind, model_type, constraint_family)``. The registry also decides which
constraint families a constrained model accepts: a family is accepted for a
model type if some builder is registered for it (for example, oracle
constraints only for hidden Markov models; use ``FactorOracleConstraint`` with
the other models).

.. autoclass:: conin.representations.Representation
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: conin.representations.register_representation

.. autofunction:: conin.representations.build_representation

.. autofunction:: conin.representations.supported_representations
