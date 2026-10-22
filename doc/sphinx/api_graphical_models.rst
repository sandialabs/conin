Graphical Models
================

The following probabilistic graphical models are defined in conin.

Base class
----------

.. autoclass:: conin.graphical_models.GraphicalModel
   :show-inheritance:

Markov networks
---------------

.. autoclass:: conin.graphical_models.DiscreteFactor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.graphical_models.DiscreteMarkovNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Bayesian networks
-----------------

.. autoclass:: conin.graphical_models.DiscreteCPD
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.graphical_models.DiscreteBayesianNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Hidden Markov models
--------------------

.. autoclass:: conin.graphical_models.HiddenMarkovModel
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic Bayesian networks
-------------------------

.. autoclass:: conin.graphical_models.DynamicDiscreteBayesianNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Expressions and conversions
---------------------------

.. autoclass:: conin.graphical_models.ExpressionVariable
   :show-inheritance:

.. autofunction:: conin.graphical_models.random_hmm

.. autofunction:: conin.graphical_models.create_mn_from_bn

.. autofunction:: conin.graphical_models.create_bn_from_dbn

.. autofunction:: conin.graphical_models.create_dbn_from_hmm

.. autofunction:: conin.graphical_models.is_polytree
