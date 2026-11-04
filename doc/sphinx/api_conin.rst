The ``conin`` Package
=====================

The names below are the top-level public API (``conin.__all__``): import them
from ``conin`` directly.

.. doctest::

   >>> from conin import HiddenMarkovModel, map_query, load_model

Each name is the same object as the one in its subpackage (for example,
``conin.map_query is conin.algorithms.inference.map_query``), and is
documented there:

Models
   :class:`~conin.graphical_models.DiscreteFactor`,
   :class:`~conin.graphical_models.DiscreteMarkovNetwork`,
   :class:`~conin.graphical_models.DiscreteCPD`,
   :class:`~conin.graphical_models.DiscreteBayesianNetwork`,
   :class:`~conin.graphical_models.DynamicDiscreteBayesianNetwork`,
   :class:`~conin.graphical_models.HiddenMarkovModel`

Constrained models
   :class:`~conin.constrained_models.ConstrainedDiscreteMarkovNetwork`,
   :class:`~conin.constrained_models.ConstrainedDiscreteBayesianNetwork`,
   :class:`~conin.constrained_models.ConstrainedDynamicDiscreteBayesianNetwork`,
   :class:`~conin.constrained_models.ConstrainedHiddenMarkovModel`

Constraints
   :class:`~conin.constraints.Constraint`,
   :func:`~conin.constraints.mvr.mvr_constraint_fn`,
   :func:`~conin.constraints.oracle_constraint_fn`,
   :func:`~conin.constraints.pyomo_constraint_fn`,
   :func:`~conin.constraints.smoek_constraint_fn`,
   :func:`~conin.constraints.toulbar2_constraint_fn`

Inference, scoring and feasibility
   :func:`~conin.algorithms.inference.map_query`,
   :func:`~conin.algorithms.scoring.log_probability_query`,
   :func:`~conin.algorithms.scoring.log_probability`,
   :func:`~conin.algorithms.scoring.log_potential`,
   :func:`~conin.algorithms.feasibility.is_feasible`

Model files
   :func:`~conin.io.load_model`, :func:`~conin.io.save_model`

Exceptions
   :exc:`~conin.exceptions.InvalidInputError`,
   :exc:`~conin.exceptions.InsufficientSolutionsError`,
   :exc:`~conin.exceptions.ModelFileError`

Everything else is in the subpackages, which ``import conin`` also loads:
:mod:`conin.graphical_models` (e.g. ``random_hmm``, the model conversions),
:mod:`conin.constrained_models`, :mod:`conin.constraints` (the constraint
classes and constraint libraries), :mod:`conin.algorithms` (the HMM-specific
inference functions, ``partial_is_feasible``, sampling and learning),
:mod:`conin.io` (the format-specific readers and writers),
:mod:`conin.representations` and :mod:`conin.config`.  The example models in
:mod:`conin.examples` are imported by module, e.g.
``from conin.examples.hidden_markov_model import create_hmm1``.

Solver Configuration
--------------------

.. automodule:: conin.config

.. autofunction:: conin.config.mip_solver
