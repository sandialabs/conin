Architecture and Design Notes
=============================

This page is for contributors. It describes how ``conin`` is organised and
records design decisions that are not obvious from the code.

Package layout
--------------

``conin.graphical_models``
   Unconstrained, immutable models (``DiscreteMarkovNetwork``,
   ``DiscreteBayesianNetwork``, ``DynamicDiscreteBayesianNetwork``,
   ``HiddenMarkovModel``) and the conversions between them.
``conin.constraints``
   Constraint containers. Each class declares a ``family`` (``"pyomo"``,
   ``"toulbar2"``, ``"algebraic"``, ``"oracle"``, ``"factor_oracle"``,
   ``"recursive"``, ``"mvr"``) and holds no backend logic.
``conin.constrained_models``
   ``Constrained*(model, constraints=(), *, data=None)``: an immutable pair of
   a model and a tuple of constraints of one family, plus application data.
``conin.representations``
   The registry and the representations built from a constrained model, one
   subpackage per backend (``pyomo``, ``toulbar2``, ``factor``, ``matvec``,
   ``oracle``, ``recursive``, ``mvr``).
``conin.inference``, ``conin.learning``, ``conin.sampling``, ``conin.scoring``
   Algorithms, organised by representation (``unconstrained``, ``pyomo``,
   ``toulbar2``, ``variable_elimination``, ``oracle``, ``recursive``,
   ``mvr``). ``map_query`` and ``log_probability_query`` are the generic entry
   points.
``conin.io``, ``conin.common``
   File I/O, and non-I/O helpers (pgmpy conversion, ``is_polytree``).
``conin.examples``
   The example factories used by the docs and the tests.
``conin.util``, ``conin.exceptions``
   Small shared helpers.

Layering
--------

- ``graphical_models``, ``common``, ``util`` and ``exceptions`` are the bottom
  layer: they import nothing from the packages above them (``common`` imports
  only ``graphical_models`` and ``util``).
- ``constraints`` does not import the models.
- ``constrained_models`` and ``representations`` sit above them; the
  algorithm packages sit above the representations.
- There are no import cycles between modules. Function-level imports are
  used only for optional dependencies (pgmpy, torch).
- ``import conin`` must not import torch. The MVR packages export their
  algorithms through a lazy module ``__getattr__``.
- Optional dependencies are imported with ``conin.util.try_import``, which
  catches only ``ImportError``; code that probes a dependency which can fail
  with ``OSError`` (a broken torch install) catches both.

Representations and dispatch
----------------------------

- Builders are registered with ``@register_representation(kind, model_type,
  families)``. The registry is the single source of truth for which
  ``(kind, model type, constraint family)`` combinations work, and a
  constrained model accepts a constraint family only if some builder is
  registered for it. Supporting a new family or model means registering a
  builder.
- Each backend has one function that applies the constraints of a family
  (``apply_pyomo_constraints``, ``apply_toulbar2_constraints``,
  ``apply_factor_constraints``).
- ``map_query`` (and ``log_probability_query``) normalise the model (pgmpy
  models are converted, bare models are wrapped as ``Constrained*(model)``),
  map the method to a representation kind, build the representation with its
  structure parameters, and dispatch with ``ovld`` on the representation
  class: one overload per class, each calling a private core that the public
  per-backend function also calls.
- Structure parameters (``start``/``stop`` for DBNs and HMMs,
  ``x_binary``/``y_binary`` for network flow) are build arguments. Per-query
  inputs (evidence, ``variables``, solver options, ``verbose``,
  ``write_uai_file``) are ``create_model()`` or algorithm options, and
  ``create_model()`` does not change the representation, so one
  representation can serve several queries.
- Constraint functions see the constrained model's ``data``: a shallow copy
  with the backend's fields (``T``, ``hmm``, ``start``, ``stop``, ...) set.

Decisions
---------

No representation cache
   ``representation()`` builds a new object on every call. A cache existed
   briefly and was removed because nothing reused it (``map_query`` wraps a
   bare model anew each call, and learning changes the model every
   iteration), while it made it necessary to reason about mutation after the
   build and about which arguments are cache keys. If one is needed, it can be
   added inside ``representation()`` without an API change.
Variable elimination uses pgmpy
   The ``"factor"`` representation materialises the constraints once and
   converts the result to a pgmpy model per query; VE does not run on conin's
   own factor tables. This kept the results identical to the earlier
   implementation. pgmpy's memory use limits VE on larger unrolled models.
Composition, not inheritance
   A constrained model holds its model (``cmodel.model``) rather than
   subclassing it, so code written for bare models cannot silently accept a
   constrained model and drop its constraints. ``ensure(model)`` wraps a bare
   model.
Immutability
   Models, constrained models and the read-only arrays of the ``"matvec"``
   and MVR representations are not changed after construction. Use
   ``with_model``/``with_constraints`` (or ``with_cpd``/``with_factor``) to
   derive new objects.
MVR static context
   The torch kernels build their per-query context (tensors, shapes, active
   sets) from ``repn.mvrs``, because it depends on the horizon, the dtype, the
   device and the semiring. The ``"mvr"`` representation deep-copies prebuilt
   MVRs at build time, so the user's objects are never changed.
DBN unrolling
   DBNs and HMMs are unrolled from ``start=0``; unrolling from a later time
   would need the marginal distribution at that time.

Open questions
--------------

- The unconstrained samplers, ``DBNSampler`` and rejection sampling use
  NumPy's global random state, while the torch MVR samplers take a
  ``generator``. A common random-state argument has not been designed.
- The MVR algorithms build ``representation("mvr")`` per call, so stopped
  sampling (one ``sat_time`` call plus one ``ffbs`` call per distinct stopping
  time) deep-copies the prebuilt MVRs each time (about 1 ms per MVR). Measure
  before optimising (a cache inside ``representation()``, or passing the
  built MVRs on internally).
