Architecture and Design Notes
=============================

This page is for contributors. It describes how ``conin`` is organised and
records design decisions that are not obvious from the code.

Package layout
--------------

``conin``
   The top-level public API, ``conin.__all__`` (see :doc:`api_conin`): the
   names most code needs, re-exported from the subpackages below. Functions
   specific to one model type (the HMM samplers, learning and inference
   functions, ``partial_is_feasible``) stay in their subpackages.
   ``conin/tests/test_public_api.py`` pins the list.
``conin.graphical_models``
   Unconstrained, immutable models (``DiscreteMarkovNetwork``,
   ``DiscreteBayesianNetwork``, ``DynamicDiscreteBayesianNetwork``,
   ``HiddenMarkovModel``) and the conversions between them.
``conin.constraints``
   Constraint containers. Each class declares a ``family`` (``"pyomo"``,
   ``"toulbar2"``, ``"smoek"``, ``"oracle"``, ``"factor_oracle"``,
   ``"recursive"``, ``"mvr"``) and holds no backend logic.
``conin.constrained_models``
   ``Constrained*(model, constraints=(), *, data=None)``: an immutable pair of
   a model and a tuple of constraints of one family, plus application data.
``conin.representations``
   The registry and the representations built from a constrained model, one
   subpackage per backend (``pyomo``, ``toulbar2``, ``variable_elimination``,
   ``matvec``, ``oracle``, ``recursive``, ``mvr``).
``conin.algorithms``
   The algorithms. ``conin.algorithms.inference``,
   ``conin.algorithms.learning`` and ``conin.algorithms.sampling`` are each
   organised by representation (``unconstrained``, ``pyomo``, ``toulbar2``,
   ``variable_elimination``, ``oracle``, ``recursive``, ``mvr``);
   ``conin.algorithms.scoring`` scores assignments (``log_potential``,
   ``log_probability``), and ``conin.algorithms.feasibility`` checks them
   against the constraints (``is_feasible``, dispatched on the constrained
   model and then on each constraint's class, and ``partial_is_feasible``
   for constrained HMMs). ``conin.algorithms.util`` holds the code they
   share (the torch MVR kernels and variable elimination). ``map_query`` and ``log_probability_query``
   are the generic entry points.
``conin.io``
   File I/O (UAI, CFN, JSON, BIF). ``conin.io.pgmpy`` is the only place that uses
   pgmpy: BIF files and converting models between pgmpy and CONIN
   (``convert_pgmpy_to_conin``, ``convert_conin_to_pgmpy_bn``/``_mn``).
   Needs pgmpy only when called.
``conin.examples``
   The example factories used by the docs and the tests.
``conin.util``, ``conin.exceptions``
   Small shared helpers.

Layering
--------

- ``util``, ``exceptions`` and ``config`` are the bottom layer: they import
  nothing from the packages above them.
- ``graphical_models`` and ``constraints`` sit above them and do not import
  each other (``constraints`` does not import the models).
- ``constrained_models``, ``representations`` and ``io`` form the model
  layer. They import each other (``constrained_models`` imports the
  representation registry, the toulbar2 representations import ``io``, and
  ``io`` builds constrained models), which is why the registry imports its
  builders on first use.
- The ``algorithms`` packages sit above the model layer, in the order
  ``util``, ``feasibility``, ``scoring``, ``inference``, ``sampling``,
  ``learning``: each imports only the ones before it. ``examples`` and the
  ``algorithms`` package itself (which imports all of them) come next, and
  the top-level ``conin`` package, which re-exports the public API, is
  above everything.
- There are no module-level import cycles between modules. A conin module is
  imported inside a function only if it needs torch (the MVR algorithms and
  ``algorithms.util.mvr_kernels``); other function-level imports are of
  optional dependencies (torch).
- ``import conin`` must not import torch or pgmpy (pgmpy imports torch, and
  takes several seconds to import). The MVR packages export their
  algorithms through a lazy module ``__getattr__``, and ``conin.io.pgmpy``
  imports pgmpy with ``conin.util.lazy_import``.
- ``conin/tests/test_layering.py`` checks these rules; its ``LAYERS`` table
  is the authoritative layer order, and a new package must be added to it.
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
- A constrained model class registers itself as the constrained class of its
  ``model_type`` when it is defined (``ConstrainedGraphicalModel.__init_subclass__``,
  into ``conin.graphical_models.base``, the layer below). Queries wrap bare
  models in it (``as_constrained_model``), and ``model.representation(kind)``
  is ``ConstrainedX(model).representation(kind)``. A new model type needs a
  ``GraphicalModel`` subclass and a ``ConstrainedGraphicalModel`` subclass
  with ``model_type`` set, and no list to edit.
- Each backend has one function that applies the constraints of a family
  (``apply_pyomo_constraints``, ``apply_toulbar2_constraints``,
  ``apply_variable_elimination_constraints``). Each passes ``apply_constraints``
  (``conin.representations.base``) a table from family to applier;
  ``apply_constraints`` rejects mixed or unsupported families with one
  message. The smoek translators all start from ``build_smoek_model``.
- The DBN and HMM representations of every backend get ``start``, ``stop``,
  ``time_steps`` and their constraint-``data`` time fields from the mixins in
  ``conin.representations.unrolled``, so the backends cannot drift apart.
- ``map_query`` (and ``log_probability_query``) normalise the model (bare
  models are wrapped as ``Constrained*(model)``; pgmpy models are rejected),
  map the method to a representation kind (one table for both queries,
  ``conin.representations.methods``), build the representation with its
  structure parameters, and dispatch with ``ovld`` on the representation
  class. The factor backends have one solver each (``_solve_pyomo``,
  ``_solve_toulbar2``, ``_solve_variable_elimination``), registered on the
  backend's base representation class for MN, BN and DBN; their HMM
  representations are solved as the unrolled DBN by ``solve_unrolled_hmm``
  (``conin.algorithms.inference.hmm_results``). Network flow, Viterbi, A*,
  recursive A* and the MVR Viterbi have their own overloads. A* and
  recursive A* are thin wrappers around one search
  (``conin.algorithms.inference.hmm_a_star``), which sees the constraints
  through a search state (``initial``, ``update``, ``feasible_partial``,
  ``feasible``); for oracle constraints the state is the prefix itself, so no
  prefixes are merged.
- Every MAP result has the shape of ``conin.algorithms.inference.results``:
  the same top-level keys for every method, and solutions with ``states``,
  ``log_potential`` and ``solver_objective``. ``log_potential`` is computed
  from the assignment by ``conin.algorithms.scoring`` (the factor backends'
  solutions are scored in ``map_query``, the HMM algorithms score their
  own), never read from a solver objective, so it is the same for every
  method. This is why the method table is in ``conin.representations``
  rather than ``conin.algorithms.inference``: ``conin.algorithms.scoring``
  imports it, and ``conin.algorithms.inference`` imports
  ``conin.algorithms.scoring``.
- Argument names: the generic queries (``map_query``,
  ``log_probability_query``) take ``evidence=`` for every model type,
  including the observations of an HMM. The HMM-specific functions
  (``viterbi``, ``a_star``, ``recursive_a_star``, ``lp_inference``,
  ``ip_inference``, the MVR algorithms, ``is_feasible``, ``log_probability``
  and ``log_potential``) take ``observed``, and the MVR learners, which take
  several sequences, take ``observations``.
- Structure parameters (``start``/``stop`` for DBNs and HMMs,
  ``x_binary``/``y_binary`` for network flow) are build arguments. Per-query
  inputs (evidence, ``variables``, solver options, ``verbose``,
  ``write_uai_file``) are ``create_model()`` or algorithm options, and
  ``create_model()`` does not change the representation, so one
  representation can serve several queries.
- Constraint functions see the constrained model's ``data``: a shallow copy
  with the backend's fields set (``conin.util.constraint_data``). The time
  fields are the same everywhere: ``start``, ``stop`` and ``T`` for a DBN;
  ``start``, ``stop`` and ``hmm = Munch(T, hidden_to_internal)`` for an HMM.
  Backends add their own fields on top (``variables`` and ``evidence`` for
  Pyomo and toulbar2, the index sets in ``hmm`` for network flow).

Decisions
---------

No representation cache
   ``representation()`` builds a new object on every call. A cache existed
   briefly and was removed because nothing reused it (``map_query`` wraps a
   bare model anew each call, and learning changes the model every
   iteration), while it made it necessary to reason about mutation after the
   build and about which arguments are cache keys. If one is needed, it can be
   added inside ``representation()`` without an API change.
Variable elimination is CONIN's own
   The ``"variable_elimination"`` representation materialises each constraint
   once, as an indicator factor, into a Markov network of the model's factors
   (a Bayesian network's CPDs). ``conin.algorithms.util.variable_elimination``
   eliminates on numpy tables of log values, in a greedy (smallest new table)
   order, for MAP, partial MAP and ``log_probability_query``. It replaced
   pgmpy's variable elimination, which needed a pgmpy model per query, a
   constraint node with evidence for each constraint of a Bayesian network,
   a second elimination to detect infeasibility, and summed in linear space
   (long sequences underflowed).
Composition, not inheritance
   A constrained model holds its model (``cmodel.model``) rather than
   subclassing it, so code written for bare models cannot silently accept a
   constrained model and drop its constraints. ``ensure(model)`` wraps a bare
   model.
Immutability
   Models, constrained models and the read-only arrays of the ``"matvec"``
   and MVR representations are not changed after construction. Use
   ``with_model``/``with_constraints`` (or ``with_cpd``/``with_factor``) to
   derive new objects. The lists and dicts of the models, factors and CPDs
   are ``FrozenList``/``FrozenDict`` (``conin.util.frozen``): read-only
   subclasses of ``list`` and ``dict``, so they compare equal to and print
   like the built-in types, and code that reads them needs no change. Test
   for them with ``isinstance(x, list)``/``isinstance(x, dict)``, not
   ``type(x) is list``. The constructors copy their arguments into them
   (``freeze``). A DBN's ``t`` is the exception: it
   is the time index that unrolling and sampling set while they evaluate the
   CPD templates.
MVR static context
   The torch kernels (``conin.algorithms.util.mvr_kernels``) build their
   per-query context (tensors, shapes, active sets) from ``repn.mvrs``,
   because it depends on the horizon, the dtype, the device and the semiring.
   The ``"mvr"`` representation deep-copies prebuilt MVRs at build time, so
   the user's objects are never changed.
DBN unrolling
   DBNs and HMMs are unrolled from ``start=0``; unrolling from a later time
   would need the marginal distribution at that time.

Open questions
--------------

- The MVR algorithms build ``representation("mvr")`` per call, so stopped
  sampling (one ``sat_time`` call plus one ``ffbs`` call per distinct stopping
  time) deep-copies the prebuilt MVRs each time (about 1 ms per MVR). Measure
  before optimising (a cache inside ``representation()``, or passing the
  built MVRs on internally).
