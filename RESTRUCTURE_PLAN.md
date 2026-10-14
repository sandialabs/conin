# Conin Restructure Plan

Status: **In progress.** PR 0 is merged into `main` (#241). PR 1 is merged into `restructure` (#242). PR 2 is merged into `restructure` (#243). PR 3 is done (branch `restructure-pr3-constrained-models`, targets `restructure`). The later PRs target the `restructure` branch.
This file lives at the repo root while the work is in progress. Delete it in the final PR.

All decisions are recorded in §11; §12 lists what is still open. Critiques of the
original sketch are in §10.

---

## 1. Goals

1. Separate *unconstrained* graphical models from *constrained* ones.
2. Add an explicit **representation** layer. A representation is built from a
   `(graphical model, constraints)` pair and holds whatever a backend needs:
   a Pyomo model, a toulbar2 CFN, aligned MVRs, wrapped oracles, factor tables, and so on.
3. Make `constraints/` a set of simple *containers* with no backend logic.
4. Make a constrained model simple: an immutable graphical model, an immutable
   tuple of same-family constraints, and representations that are built lazily and cached.
5. Organize algorithms (inference, learning, sampling, queries) **by
   representation/backend**, so generic code (e.g. Pyomo solve logic) is shared.
6. Keep the public API concepts (`map_query` with ovld dispatch, constraint
   decorators, model constructors, examples). Import paths **will** change: this is a
   hard break with no compatibility shims, released as conin 2.0.

Non-goals: new algorithms (beyond wiring MVR Viterbi into `map_query`),
performance work, and changing the `model.V(...)` / `data` API that user
constraint functions see.

---

## 2. Current state (summary of findings)

### 2.1 Layout today
Before PR 1, each model package held both its unconstrained and constrained classes
(`markov_network/model.py`, `bayesian_network/model.py` + `bn_to_mn.py`,
`dynamic_bayesian_network/model.py` + `dbn_to_bn.py` + `expr.py`,
`hidden_markov_model/hmm.py` + `hmm_util.py` + `hmm_to_dbn.py`) plus an `examples.py`.
After PR 1:
```
conin/
  graphical_models/      base.py (GraphicalModel), markov_network.py, bayesian_network.py,
                         dynamic_bayesian_network.py, expr.py, hidden_markov_model.py, conversions.py
  examples/              markov_network.py, bayesian_network.py, dynamic_bayesian_network.py,
                         hidden_markov_model.py (same names as graphical_models/)
  markov_network/        constrained_model.py (ConstrainedDiscreteMarkovNetwork)
  bayesian_network/      constrained_model.py (ConstrainedDiscreteBayesianNetwork)
  dynamic_bayesian_network/ constrained_model.py (ConstrainedDynamicDiscreteBayesianNetwork)
  hidden_markov_model/   constrained_hmm.py, chmm*.py, mvr*.py, hmm_application.py,
                         inference/ learning/ sampling/ other_queries/
  constraints/           constraint.py (all functor classes), oracle.py (oracle library), algebraic/ (smoek)
  inference/             map_query.py, {Optimization,CFN,AStar,Viterbi}Inference.py, VariableElimination.py,
                         mn/ bn/ dbn/ hmm/  (pyomo + toulbar2 per model)
  operators.py           MVR operator framework (only used by mvr_operators.py; import cycle)
  sampling/dbn/          DBNSampler (namespace package, not imported by `import conin`)
  common/                IO + pgmpy/pgmax/pomegranate conversion + misc helpers
  mediation_variables/   research prototype (KEEP, untouched)
  util/                  State, try_import, MPESolution, ...
```
After PR 2:
```
conin/
  constraints/           base.py (Constraint ABC, `family`), pyomo.py, toulbar2.py, oracle.py (OracleConstraint),
                         factor_oracle.py (FactorOracleConstraint), oracle_library.py (the oracle library),
                         algebraic/ (AlgebraicConstraint + the add_constraints_* backend modules, which move in PR 4a/4b),
                         mvr/ (constraint.py, base.py, primitives.py, operators.py, formula.py; __init__ exports the public MVR names)
  representations/factor/ materialise.py (materialise_factor_oracle; temporary home, used by inference/VariableElimination.py)
  hidden_markov_model/   constrained_hmm.py, chmm*.py, mvr_common.py, hmm_application.py,
                         inference/ learning/ sampling/ other_queries/  (torch MVR code moves in PR 4d)
  operators.py           removed (merged into constraints/mvr/operators.py)
```
After PR 3:
```
conin/
  constrained_models/    base.py (ConstrainedGraphicalModel), markov_network.py, bayesian_network.py,
                         dynamic_bayesian_network.py, hidden_markov_model.py, tests/
  representations/       base.py (Representation, registry, register_representation, build_representation),
                         oracle/hmm.py (OracleHMMRepresentation, wraps Oracle_CHMM),
                         pyomo/hmm_network_flow.py (PyomoNetworkFlowHMMRepresentation, wraps PyomoAlgebraic_CHMM),
                         mvr/hmm.py (MVRHMMRepresentation, wraps MVR_CHMM, exposes .mvrs), factor/ (unchanged)
  hidden_markov_model/   chmm*.py, mvr_common.py, hmm_application.py, inference/ learning/ sampling/ other_queries/
  markov_network/, bayesian_network/, dynamic_bayesian_network/   removed (emptied)
```

### 2.2 Problems the restructure should fix
- **The constrained wrappers are inconsistent.**
  - The BN/MN/DBN wrappers validate nothing.
  - `ConstrainedHiddenMarkovModel` validates with `assert` and keeps a mutable
    `constraint_type`.
  - It also needs a manual `initialize_chmm()` call, which attaches `self.chmm`
    *after* construction. That object goes stale if the HMM or constraints change, and the
    call is a silent no-op for algebraic, toulbar2 and oracle-with-nodes constraints.
- **Representation building is scattered across the code:**
  - inside constraint classes: `OracleConstraint._materialise` builds factors and CPDs
    (PR 2: moved to `representations/factor/materialise.py`; becomes the factor repn in 4e);
  - inside containers: `initialize_chmm` (PR 3: removed; replaced by the registered "oracle",
    "pyomo_network_flow" and "mvr" HMM representations, which still wrap `Oracle_CHMM`,
    `PyomoAlgebraic_CHMM` and `MVR_CHMM` until 4a/4c/4d);
  - inside "model" classes: `PyomoAlgebraic_CHMM.generate_model` (PR 3: reached only through the
    "pyomo_network_flow" repn; the class itself is replaced in 4a);
  - inside inference functions: `create_*_map_query_model_*` (remain; 4a/4b), and `lp_inference`
    builds a CHMM (PR 3: uses `representation("pyomo_network_flow")`);
  - inside the dispatch layer: `VariableElimination._add_*_constraints_as_evidence` (remains; 4e).
- **Almost nothing is cached.** Only `HiddenMarkovModel.repn` is. The
  HMM→DBN→BN→MN→factor-repn→Pyomo chain is rebuilt on every query.
- **There is a lot of duplication:**
  - 3 copies of `PyomoVarWrapper`;
  - the MN and BN toulbar2 builders are near-identical;
  - every ovld dispatcher has identical Constrained/unconstrained overload pairs;
  - there are 3 separate `add_constraints` dispatchers.
- **The imports are tangled.**
  - `constraints` lazily imports MN, BN and HMM.
  - `operators.py` and `mvr_operators` import each other (a cycle).
  - Some code uses implicit attribute-path imports (`conin.inference.mn.inference_pyomo.solve_…`)
    that only work because of import order.
- `mvr_common.py` mixes representation building with torch kernels.
- `recursive_a_star` relies on an undeclared duck-typed protocol on an
  `HMMApplication` subclass instead of a constraint object.

### 2.3 Baseline
- Before PR 0, `pytest conin` in `.venv-opencode` gave **765 passed, 178 skipped, 8
  collection errors**. All 8 errors were the torch MVR test files: torch is broken in that
  env (`OSError`, which `importorskip` doesn't catch).
- **After PR 0 (the baseline for later PRs):** `.venv-opencode/bin/python -m pytest conin -q`
  gives **959 passed, 195 skipped, 0 errors**. This breaks down as:
  - the old suite: 765 passed, 186 skipped (178 + the 8 torch MVR modules, which now skip);
  - the new `conin/tests/test_public_api.py`: 194 passed, 9 skipped (the torch rows).
- **After PR 1:** **962 passed, 195 skipped, 0 errors**. The +3 are the new DBN `.t`
  regression tests (2 in `graphical_models/tests/test_conversions.py`, 1 in
  `sampling/dbn/tests/test_sampling.py`). `test_public_api.py` is still 194 passed, 9 skipped
  (rows moved to the new paths, duplicates merged, a `GraphicalModel` row added).
- **After PR 2:** **1009 passed, 195 skipped, 0 errors**. The +47 are new tests:
  - 5 VE regression tests for the two §9 bugs fixed in PR 2 (`inference/tests/test_variable_elimination_helpers.py`);
  - 29 unit tests: `constraints/tests/test_families.py` (16), `representations/factor/tests/test_materialise.py` (6),
    `constraints/mvr/tests/test_mvr_constraint_family.py` (7);
  - a net +13 `test_public_api.py` rows: `Constraint` and `FactorOracleConstraint` (api_constraints.rst) replace the
    deleted `ConstraintFunctor` row, and a new `RESTRUCTURE_PLAN_API` group holds 12 rows for the §5 names. The MVR
    notebook rows now point at `conin.constraints.mvr`. `test_public_api.py` is 207 passed, 9 skipped.
  The moved MVR spec tests (`constraints/mvr/tests/`) give 281 passed, 0 skipped in `/projects/.conin-scratch/venv`
  (it has greenery).
- **After PR 3:** **1074 passed, 195 skipped, 0 errors** (+65). The new and rewritten tests:
  - `constrained_models/tests/test_base.py` (33: immutability, validation, lazy build, cache, get_copy isolation,
    the unsupported-kind error, copy/deepcopy/pickle, with_*);
  - `constrained_models/tests/test_{markov_network,bayesian_network,dynamic_bayesian_network}.py` (the MN file is the
    moved `markov_network/tests/test_constrained_model.py`, rewritten for read-only constraints; BN/DBN are new,
    including the §9 `nodes`-is-a-property regression test);
  - `representations/tests/test_hmm_representations.py` (11: oracle, pyomo_network_flow and mvr parity with the
    wrapped classes, data isolation, get_copy);
  - +12 `test_public_api.py` rows (`conin.constrained_models.*`, `conin.representations.*`): 219 passed, 9 skipped.
  The MVR spec tests are still 281 passed in `/projects/.conin-scratch/venv`.
- pgmpy, or_topas, gurobi and glpk are not installed in `.venv-opencode`; torch is broken
  there; greenery is missing.
- PR 0 adds `doc/notebooks/MVR/*.ipynb` to the CI notebook job and fixes the
  `regex_tutorial.ipynb` import (`conin.constraint` → `conin.constraints`).
  The job forces `--nbmake-kernel=python3`, because the MVR notebooks declare a `conin` kernel.
- Torch tests and the MVR notebooks are verified in GitHub Actions, not locally
  (the dev machine is an emulated CPU; see /projects/AGENTS.md).

---

## 3. Target layout

```text
conin/
├── __init__.py                 # top-level re-exports: map_query, *_constraint_fn, exceptions
├── graphical_models/           # UNCONSTRAINED, immutable models only
│   ├── __init__.py             # exports every model class + conversions
│   ├── base.py                 # GraphicalModel marker ABC (for ovld + typing)
│   ├── markov_network.py       # DiscreteFactor, DiscreteMarkovNetwork
│   ├── bayesian_network.py     # DiscreteCPD, DiscreteBayesianNetwork
│   ├── dynamic_bayesian_network.py
│   ├── expr.py                 # ExpressionVariable etc.
│   ├── hidden_markov_model.py  # HiddenMarkovModel, HMM_MatVecRepn, random_hmm
│   └── conversions.py          # create_dbn_from_hmm, create_bn_from_dbn, create_mn_from_bn
├── constraints/                # simple containers: no backend logic, no model imports
│   ├── __init__.py
│   ├── base.py                 # Constraint ABC (renamed from ConstraintFunctor)
│   ├── pyomo.py                # PyomoConstraint, pyomo_constraint_fn
│   ├── toulbar2.py             # Toulbar2Constraint, toulbar2_constraint_fn
│   ├── algebraic/              # AlgebraicConstraint, algebraic_constraint_fn, bridge.py (smoek nodes)
│   ├── oracle.py               # OracleConstraint (sequence/assignment predicate), oracle_constraint_fn
│   ├── factor_oracle.py        # FactorOracleConstraint (predicate + node scope)
│   ├── recursive.py            # NEW RecursiveConstraint (formalizes the Num_Zeros protocol)
│   ├── oracle_library.py       # current constraints/oracle.py (all_diff, citation, and/or/not, ...)
│   └── mvr/
│       ├── constraint.py       # MVRConstraint, mvr_constraint_fn (factories: hmm -> MVR)
│       ├── base.py             # BaseMVR(Constraint), HomMVR, InhomMVR, MVR_MatVecRepn
│       ├── primitives.py       # current mvr_constraints.py (mvr_forbid_state, mvr_regex, ...)
│       ├── operators.py        # current mvr_operators.py + conin/operators.py (cycle removed)
│       └── formula.py          # current mvr_formula.py (build_mvr, build_mvr_functor)
├── constrained_models/
│   ├── __init__.py
│   ├── base.py                 # ConstrainedGraphicalModel
│   ├── hidden_markov_model.py  # ConstrainedHiddenMarkovModel (+ generate_hidden, is_feasible helpers)
│   ├── bayesian_network.py     # ConstrainedDiscreteBayesianNetwork
│   ├── markov_network.py       # ConstrainedDiscreteMarkovNetwork
│   └── dynamic_bayesian_network.py
├── representations/
│   ├── __init__.py
│   ├── base.py                 # Representation ABC (get_copy), registry, build_representation()
│   ├── factor/                 # factor repn (S,J,v,w), BN->MN skeleton, FactorOracle -> factor/CPD (materialise.py: materialise_factor_oracle, since PR 2)
│   ├── pyomo/
│   │   ├── common.py           # ONE PyomoVarWrapper, IP-from-factor-repn, c5 constraints, smoek->pyomo
│   │   ├── mn.py  bn.py  dbn.py  hmm.py
│   │   └── hmm_network_flow.py # from PyomoAlgebraic_CHMM / lp.py (ip_formulation="network_flow")
│   ├── toulbar2/
│   │   ├── common.py           # UAI->CFN, Toulbar2VarWrapper, smoek->toulbar2 linear translator
│   │   └── mn.py  bn.py  dbn.py  hmm.py
│   ├── mvr/
│   │   ├── hmm.py              # MVRRepresentation (replaces MVR_CHMM): aligned MVRs + static context
│   │   └── kernels.py          # shared torch kernels from mvr_common.py (used by inference/learning/sampling/queries)
│   ├── oracle/hmm.py           # OracleRepresentation (replaces Oracle_CHMM); smoek->oracle
│   └── recursive/hmm.py
├── inference/
│   ├── __init__.py             # exports map_query
│   ├── map_query.py            # public entry point (method -> repn kind -> ovld on repn type)
│   ├── unconstrained/          # viterbi.py, a_star core search (a_star_)
│   ├── pyomo/                  # solve.py (generic solve/parse), per-model result post-processing
│   ├── toulbar2/               # solve.py, per-model post-processing
│   ├── variable_elimination/   # VE on factor repn; custom_VE.py moves here
│   ├── mvr/                    # viterbi.py, marginal_map.py
│   ├── oracle/                 # a_star_hmm.py
│   └── recursive/              # a_star_hmm.py
├── learning/
│   ├── unconstrained/          # supervised.py, helpers.py
│   ├── oracle/                 # mcem.py
│   └── mvr/                    # baum_welch.py (+ forward_backward), generalized_em.py
├── sampling/
│   ├── unconstrained/          # dbn_forward.py (DBNSampler)
│   ├── oracle/                 # rejection_hmm.py (from Oracle_CHMM.generate_hidden)
│   └── mvr/                    # ffbs.py, stopped_sampling.py
├── queries/mvr/                # sat_prob.py, sat_time.py (from other_queries/)
├── applications/               # hmm_application.py (HMMApplication)
├── examples/                   # markov_network.py, bayesian_network.py, dynamic_bayesian_network.py, hidden_markov_model.py (from */examples.py)
├── io/                         # file IO: UAI/CFN load+save, unified load_model/save_model, convert_pgm CLI, pyagrum loader
├── common/                     # non-IO helpers: pgmpy/ (pgmpy<->conin conversion, MapCPD), is_polytree, log_potential
├── mediation_variables/        # UNTOUCHED (research prototype)
└── util/
```

Removed outright:
- the pgmax and pomegranate support: `to_pgmax`, `to_pomegranate`, the
  `pgmax/` and `pomegranate/` loaders (these just load with pgmpy and convert),
  the `model_type` branches in unified `load_model`, and the pyproject extras;
- `initialize_chmm`, `CHMM`, `Oracle_CHMM`, `Algebraic_CHMM`,
  `PyomoAlgebraic_CHMM` and `MVR_CHMM` (they become representations);
- `create_algebraic_chmm`, `Xgenerate_hidden`, and the duplicated `PyomoVarWrapper`s.

Kept even though untested (they have plausible uses): `factor_repn.get_factor_value`, and
similar code. Rule for subagents: **delete only code that becomes redundant because of the
restructure. When in doubt, keep it and flag it in the PR description.**

Tests stay next to the code they test, as today (e.g. `representations/pyomo/tests/`).

MVR imports: code outside `conin/constraints/mvr/` imports public names from the package
(`from conin.constraints.mvr import HomMVR`), not from its modules. Modules inside the package use relative
imports (`from .base import HomMVR`). The package's own tests may import from submodules.

---

## 4. Core design

### 4.1 Constraints
- `Constraint` is an ABC. Each subclass is a plain container holding `func`, `name`,
  `num_args` and any class-specific metadata.
- `constraints/` must **not** import graphical models.
  `OracleConstraint._materialise` moves to `representations/factor/` (PR 2: `materialise_factor_oracle(constraint, pgm, data=None)`
  in `representations/factor/materialise.py`). The lazy
  `BaseMVR` import goes away because MVRs now live under `constraints/mvr/`.
- **Constraint family.** Each class declares `family` (a class attribute). A
  constrained model needs all of its constraints to be in one family. Most classes are
  their own family. The exception is MVRs: `MVRConstraint` (a factory) and `BaseMVR` (a
  built automaton) are both in the `"mvr"` family. That lets users pass
  `mvr_forbid_state(hmm, {"A"})` and `build_mvr_functor("never A")` side by side. The
  MVR representation turns factories into MVRs by calling them with the HMM.
- `OracleConstraint` is split:
  - `OracleConstraint`: a pure predicate on `{key: state}`. Optional
    `partial_func` for A* pruning.
  - `FactorOracleConstraint`: a predicate plus a `nodes` scope (a list or
    callable). It gets materialised into a factor/CPD for VE.
  - `oracle_constraint_fn(nodes=...)` returns whichever class matches the arguments, so
    decorator users see no change.
  - (PR 2) `FactorOracleConstraint` has no `partial_func`; `oracle_constraint_fn(nodes=..., same_partial_as_func=True)`
    raises `InvalidInputError`. Calling a `FactorOracleConstraint` evaluates the predicate; it no longer materialises
    when called with a pgm. The "all-nodes mode" (an oracle called with a pgm and `nodes=None`) is removed.
  - (PR 2) `Constraint` declares no abstract `__call__` (`BaseMVR` has none).
  - `family` values: `"pyomo"`, `"toulbar2"`, `"algebraic"`, `"oracle"`, `"factor_oracle"`, `"mvr"`.

Compatibility matrix (representation kind × constraint family):

| Constraint family           | pyomo | pyomo_network_flow | toulbar2 | factor (VE) | oracle (A*) | mvr | recursive |
|-----------------------------|:-----:|:------------------:|:--------:|:-----------:|:-----------:|:---:|:---------:|
| *(none)*                    |  ✓    |  ✓ (HMM)           |   ✓      |   ✓         |  ✓ (HMM)    | ✓ (HMM) | ✓ (HMM) |
| PyomoConstraint             |  ✓    |  ✓ (HMM)           |          |             |             |     |           |
| Toulbar2Constraint          |       |                    |   ✓      |             |             |     |           |
| AlgebraicConstraint         |  ✓    |  ✓ (HMM)           |   ✓      |   ✓ (via factor oracle) | ✓ (HMM) | |      |
| OracleConstraint            |       |                    |          |             |  ✓ (HMM)    |     |           |
| FactorOracleConstraint      |       |                    |          |   ✓         |             |     |           |
| MVR (MVRConstraint/BaseMVR) |       |                    |          |             |             | ✓ (HMM) |       |
| RecursiveConstraint         |       |                    |          |             |             |     | ✓ (HMM)   |

The builder registry (§4.3) is the single source of truth for this table.
Unsupported combinations raise a clear `InvalidInputError` that lists the supported options.

### 4.2 ConstrainedGraphicalModel
```python
class ConstrainedGraphicalModel:
    model_type: type[GraphicalModel]          # set by subclass

    def __init__(self, model, constraints=(), *, data=None):
        # type-check model against model_type
        # check every constraint is a Constraint and they share one family -> InvalidInputError
        self._model = model                   # already immutable
        self._constraints = tuple(constraints)
        self._family = ...                    # derived
        self._data = data                     # passed to builders, not part of the cache key
        self._cache = {}                      # key -> Representation

    @property model / constraints / constraint_family / data   # read-only
    def representation(self, kind, **structure_params):    # build lazily, cache, return repn.get_copy()
    def with_constraints(self, constraints):               # new object, empty cache
    def with_model(self, model):                           # new object, empty cache (used by learning)
```
- The object is fully immutable: no `add_constraint`, no setters, no `initialize_chmm`.
  (PR 3) It uses `__slots__`, and `__setattr__`/`__delattr__` raise `AttributeError`. `copy`, `deepcopy` and
  pickle work (`__getstate__`/`__setstate__`); copies start with an empty cache.
- (PR 3) `constraints` is a single `Constraint` (wrapped into a 1-tuple) or an iterable (list, tuple, generator)
  of `Constraint`; a string or a non-`Constraint` element raises `InvalidInputError`, as do mixed families and
  a wrong model type.
  Subclasses may set `supported_families`; `ConstrainedHiddenMarkovModel` accepts None, "oracle",
  "factor_oracle", "pyomo", "algebraic", "toulbar2" and "mvr".
  **(Decided after PR 3, done at the end of PR 4.)** `supported_families` is removed. The constructor derives
  the accepted families from the registry: a family is accepted for a model type iff
  `supported_representations(model_type)` has at least one `(kind, family)` pair for it. Then adding a constraint
  family or a model type means only registering builders. Families must not list model types, because
  support is a property of the (model type, family) pair, which the registry already records. This waits until
  4a/4b/4e have registered the HMM "pyomo", "algebraic", "toulbar2" and "factor_oracle" builders. Before that, the
  registry would reject HMMs that work today through the legacy `map_query` path.
- (PR 3) `representation()` raises `InvalidInputError` naming the supported `(kind, family)` pairs when no
  builder is registered.
- Every class has the same constructor shape:
  `ConstrainedHiddenMarkovModel(hmm, constraints=[...])`,
  `ConstrainedDiscreteBayesianNetwork(bn, constraints=[...])`, and so on.
- The thin per-model subclasses exist for three reasons:
  - the model type check;
  - as ovld/registry keys;
  - HMM-specific helpers (`generate_hidden`, `generate_observed`,
    `is_feasible`, `partial_is_feasible`), which delegate to a representation.
- `cmodel.model` replaces `.pgm` and `.hidden_markov_model`, with no aliases.

### 4.3 Building representations
- There is a dict registry keyed by `(kind, model_type, constraint_family)`. Each entry
  is a builder `(cmodel, **structure_params) -> Representation`. A plain dict makes it
  easy to say which combinations are supported, and builders register themselves with a
  decorator.
- Builders can build on each other. For example, the HMM Pyomo builder takes the factor
  representation of the unrolled DBN and then applies HMM-level constraints using
  today's `data` layout (`data.hmm.T`, `data.hmm.hidden_to_internal`). User constraint
  functions see the same `model.V(...)` and `data` as today.
- Each backend has exactly one function that applies constraints of a given family.
  This replaces the 3 duplicated `add_constraints` dispatchers.
- (PR 3) API, in `conin/representations/base.py`:
  - `Representation` (ABC): class attribute `kind`, and `get_copy()` (default `copy.deepcopy(self)`);
  - `@register_representation(kind, model_type, families=(None,))` registers `builder(cmodel, **structure_params)`
    under each `(kind, model_type, family)`; `None` means "no constraints"; re-registering a key with another
    builder raises `ValueError`; `unregister_representation(...)` is for tests;
  - `build_representation(cmodel, kind, **structure_params)` (uncached; `cmodel.representation()` caches it),
    `supported_representations(model_type)` (sorted `(kind, family)` pairs) and `get_representation_builder(...)`;
  - the built-in builder modules are imported lazily on first registry use, because they import
    `conin.hidden_markov_model`, which imports the constrained models.
- (PR 3) Registered HMM kinds: `"oracle"` (None, "oracle"), `"pyomo_network_flow"` (None, "pyomo") and
  `"mvr"` (None, "mvr"). `"algebraic"` is not registered for `"pyomo_network_flow"`: it never worked before
  PR 3 (initialize_chmm built no `.chmm` for it).

### 4.4 Caching and copy-on-get
- Representations are cached on the constrained model.
  `cmodel.representation(kind, ...)` always returns `repn.get_copy()`.
- `Representation.get_copy()` defaults to `copy.deepcopy(self)`. Each subclass
  overrides it with whatever is safest and cheapest:

  | Representation             | `get_copy()` strategy |
  |----------------------------|------------------------|
  | Factor (S,J,v,w), oracle, recursive, MVR | immutable (frozen data, read-only arrays) → return `self` |
  | (PR 3) oracle HMM (wraps `Oracle_CHMM`) | `self` (no per-query state; sampling uses the global NumPy RNG) |
  | (PR 3) MVR HMM (wraps `MVR_CHMM`) | shallow copy with deep-copied MVRs (MVRs are mutable until 4d); the HMM is shared |
  | (PR 3) pyomo_network_flow HMM | a fresh wrapper with a shallow copy of `data` (the CHMM writes `data.hmm` per query) |
  | Pyomo                      | `pyomo_model.clone()` |
  | toulbar2                   | cache the intermediate (UAI text + constraint/evidence spec); rebuild a fresh `pytoulbar2.CFN` on each get |

- Evidence and per-query options are applied to the *copy* by the inference
  code. They are never applied to the cached object.
- **Cache key:** `(kind, structure_params)`. Structure params are whatever changes the
  model's shape: `T` for HMMs, `(start, stop)` for DBNs, `ip_formulation`, and so on.
  Evidence is never part of the key for Pyomo or toulbar2, because it is applied after
  `get_copy()`. For VE, evidence changes the factors, so VE caches only the
  evidence-independent factor representation.
- The cache has no size limit (one entry per distinct structure). This can be revisited if memory becomes a problem.
- Open point, to be checked in PR 3: `clone()` may not be cheaper than a rebuild for
  big Pyomo models. If it isn't, the Pyomo repn can also use a rebuild strategy.
  `get_copy()` keeps this an internal decision.

### 4.5 Inference dispatch (keeps ovld + `map_query`)
```
map_query(pgm, method=..., evidence=..., ...)
  1. normalize: pgmpy -> conin; bare GraphicalModel -> Constrained*(gm, ())
  2. method -> representation kind:
       integer_program      -> "pyomo" | "pyomo_network_flow" (ip_formulation)
       toulbar2             -> "toulbar2"
       a_star               -> "oracle" | "recursive" (picked by constraint family)
       viterbi              -> unconstrained only
       variable_elimination -> "factor"
       mvr_viterbi  (NEW)   -> "mvr"
  3. repn = cmodel.representation(kind, **structure_params)
  4. _map_query_<method>(repn, ...)    # ovld dispatch on the Representation subclass
```
- There is one ovld overload per representation class. The duplicated
  Constrained/unconstrained pairs go away.
- Bare models are wrapped as `Constrained*(gm, ())` so that unconstrained queries take
  the same code path. The empty wrapper is transparent to every method, e.g. `viterbi` works on
  `ConstrainedHiddenMarkovModel(hmm, ())`.
- The return shape is unchanged: a Munch with `solution`, `solutions`,
  `termination_condition` and `solvetime`.

### 4.6 MVR specifics
- `MVR_CHMM` is removed. `MVRRepresentation` takes its role:
  - it calls any `MVRConstraint` factories with the HMM;
  - it validates hidden states;
  - it aligns hidden-state order (from `chmm_mvr._align_hidden_states`);
  - it holds the static context (from `mvr_common._build_static_context`).
- The MVR algorithms keep their names (`viterbi_torch_mvr_chmm`,
  `baum_welch_mvr_chmm`, `ffbs_torch_mvr_chmm`, `sat_time_torch_mvr_chmm`, …).
  They take a `ConstrainedHiddenMarkovModel` and call
  `cmodel.representation("mvr")` internally.
- Learning (Baum-Welch/GEM) currently deep-copies the model and swaps in
  `hidden_markov_model` on each iteration. It will use
  `cmodel.with_model(new_hmm)` instead.
- Stopped sampling currently mutates `_time_range` and swaps constraints on copies.
  It will use `with_constraints(...)` plus non-mutating MVR operators.
  `mvr_timerange` becomes non-mutating by default (breaking change).
- Notebook impact: `MVR_CHMM(hidden_markov_model=hmm, constraints=[m])` becomes
  `ConstrainedHiddenMarkovModel(hmm, constraints=[m])`, and the import paths change.
  `chmm.chmm.constraints[i]` becomes `chmm.representation("mvr").mvrs[i]`.
  That is the full extent of the API change.

### 4.7 Recursive A*
- `RecursiveConstraint` packages the four callbacks `Num_Zeros` uses today:
  - `initial(h)`,
  - `update(state, h)`,
  - `feasible_partial(state, t, T)`,
  - `feasible(state)`.
- `recursive_a_star` takes a `ConstrainedHiddenMarkovModel` with these constraints.
  `HMMApplication` no longer has to implement this protocol.
- Multiple `RecursiveConstraint`s combine automatically; the combined state is a tuple of the per-constraint states.

---

## 5. Public API after the restructure (sketch)

```python
from conin import map_query, pyomo_constraint_fn, oracle_constraint_fn, ...
from conin.graphical_models import HiddenMarkovModel, DiscreteBayesianNetwork, DiscreteCPD, ...
from conin.constrained_models import ConstrainedHiddenMarkovModel, ConstrainedDiscreteBayesianNetwork, ...
from conin.constraints import OracleConstraint, FactorOracleConstraint, ...
from conin.constraints.oracle_library import has_minimum_number_of_occurences_constraint, ...
from conin.constraints.mvr import HomMVR, mvr_forbid_state, mvr_not_yet, build_mvr, build_mvr_functor
from conin.inference.mvr import viterbi_torch_mvr_chmm, marginal_map_torch_mvr_chmm
from conin.learning.mvr import baum_welch_mvr_chmm, generalized_em_mvr_chmm, forward_backward_mvr_chmm
from conin.learning import supervised_learning, mcem, convert_to_simulations, add_unknowns
from conin.sampling.mvr import ffbs_torch_mvr_chmm, stopped_sampling_torch_mvr_chmm
from conin.queries.mvr import sat_prob_torch_mvr_chmm, sat_time_torch_mvr_chmm
from conin.examples.hidden_markov_model import create_hmm1, create_chmm1_pyomo
from conin.io import load_model, save_model
from conin.common import log_potential, is_polytree, convert_pgmpy_to_conin
```

Breaking changes (these go in the CHANGELOG for 2.0):
- All model, constraint and example import paths change (see above).
- `initialize_chmm()`, `chmm.chmm` and `constraint_type` are removed. Use
  `representation(kind)` and `constraint_family` instead (`chmm.chmm.constraints` for MVRs →
  `chmm.representation("mvr").mvrs`; `initialize_chmm(data=d)` → `Constrained*(model, constraints, data=d)`).
  `constraint_family` values are the constraint classes' `family`; `"oracle_with_nodes"` becomes `"factor_oracle"`.
- Constrained models are immutable, and their constructors are unified as `(model, constraints=(), *, data=None)`:
  there are no `hmm=`/`pgm=` keywords, no `constraints` setters and no `add_constraint`, and `.constraints` is a tuple.
- `.model` replaces `.pgm` (BN/MN/DBN) and `.hidden_markov_model` (HMM).
- The constrained classes move to `conin.constrained_models`. `conin.markov_network`, `conin.bayesian_network`
  and `conin.dynamic_bayesian_network` are removed, and `conin.hidden_markov_model` no longer exports
  `ConstrainedHiddenMarkovModel`.
- A single `Constraint` is accepted and wrapped into a 1-tuple. Invalid constraints (mixed families,
  non-`Constraint` objects, a string) and a wrong
  model type raise `InvalidInputError` at construction (some mixes used to fail later with `AssertionError`).
- `ConstrainedDynamicDiscreteBayesianNetwork.nodes` is a property (it was a method).
- `map_query` no longer accepts `CHMM` objects (those overloads were broken).
- `mcem` no longer writes the re-estimated HMM back to `app`.
- `MVR_CHMM` is removed; use `ConstrainedHiddenMarkovModel`.
- `ConstraintFunctor` is renamed `Constraint` (no alias). `Constraint` no longer requires `__call__`.
- The oracle library moves from `conin.constraints.oracle` to `conin.constraints.oracle_library` and is no longer
  star-exported from `conin.constraints`. `conin.constraints.oracle` is now the `OracleConstraint` module.
- The MVR spec modules move to `conin.constraints.mvr` (`hidden_markov_model.mvr` → `constraints.mvr.base`,
  `mvr_constraints` → `primitives`, `mvr_operators` + `conin.operators` → `operators`, `mvr_formula` → `formula`).
- Calling a `FactorOracleConstraint` evaluates the predicate; it no longer materialises when called with a pgm.
  `OracleConstraint` has no `nodes`, and the "all-nodes" materialisation mode is removed.
- `mvr_timerange` no longer mutates its input by default.
- `OracleConstraint(nodes=...)` becomes `FactorOracleConstraint`. The decorator still works as before.
- pgmax and pomegranate support is removed.
- New: `map_query(..., method="mvr_viterbi")`.

---

## 6. PR breakdown

**Branching:** all restructure PRs target a long-lived `restructure` integration
branch. That branch merges to `main` once, as conin 2.0. Since there are no shims,
**each PR must update every importer it affects**: tests, examples, docs and notebooks.
Each PR must leave `pytest conin` and the notebooks green on the integration branch.

Principle: put pure moves in their own PRs, separate from logic changes, so git rename
detection keeps the diffs reviewable.

| PR | Title | Kind | Contents |
|----|-------|------|----------|
| 0 | Guardrails | infra | **DONE (#241, merged to `main`).** **Target `main`.** Make torch tests skip robustly (`try_import` instead of `importorskip`). Add `doc/notebooks/MVR/*.ipynb` to CI and fix the `regex_tutorial` import. Add `tests/test_public_api.py`, which imports every public symbol used by the docs and notebooks; later PRs update it instead of breaking it silently. Record the baseline test counts. |
| 1 | Unconstrained models → `graphical_models/`; examples → `examples/` | move | **DONE (#242, merged to `restructure`).** Pure moves plus import updates. Conversions move too. One behaviour fix: stop `create_bn_from_dbn` and `DBNSampler` from mutating `dbn.t` (the immutability leak). |
| 2 | Constraints → containers | move + small refactor | **DONE (branch `restructure-pr2-constraints`).** Split `constraint.py`. Move the oracle library and the MVR spec code (mvr, primitives, operators, formula) into `constraints/`. Merge `operators.py` (removes the cycle). Add `family`. Split `FactorOracleConstraint`. Move `_materialise` into a temporary home in `representations/factor/`. |
| 3 | `ConstrainedGraphicalModel` + representation framework | **design** | **DONE (branch `restructure-pr3-constrained-models`).** Base class, registry, cache, `get_copy`, immutability. Port the four constrained classes. Builders initially wrap the *existing* create functions, so behaviour stays identical. Update every `initialize_chmm` and `.chmm` call site (examples, tests, docs). |
| 4a | Pyomo repns + `inference/pyomo` | refactor | One `PyomoVarWrapper` and one constraint applier. Split create from solve. Network-flow repn. Fix Pyomo-path bugs from §9. |
| 4b | toulbar2 repns + `inference/toulbar2` | refactor | Dedupe the MN/BN builders. Implement the rebuild-on-get strategy. Fix toulbar2 bugs. |
| 4c | Oracle + Recursive (A*), oracle sampling, mcem | refactor | `Oracle_CHMM` becomes `OracleRepresentation`. Add `RecursiveConstraint`, decouple `HMMApplication` and move it to `applications/`. Fix the A* pruning bugs. |
| 4d | MVR | refactor | `chmm_mvr` and `mvr_common` become repn plus kernels. Move inference, learning, sampling and queries. Update all MVR notebooks. |
| 4e | Factor / VE | refactor | VE constraint handling moves into the factor repn. Move `custom_VE`. Fix the VE evidence bugs. |
| 4f | `common/` → `io/` + `common/`; remove pgmax/pomegranate | move + delete | Independent of 3 and 4a–e. It can happen right after PR 1. |
| 5 | Dispatch rewrite | **design** | `map_query` → repn → ovld on the repn type. Remove the duplicated overloads. Add `method="mvr_viterbi"`. |
| 6 | Docs + cleanup | docs | Sphinx API pages, prose docs, AGENTS.md (examples location), CHANGELOG, version 2.0. Delete this plan file. |

Dependencies: 0 → 1 → 2 → 3 → {4a, 4b, 4c, 4d, 4e} → 5 → 6. 4f can go anywhere after 1.
4a–4e can be done in parallel.

Bugs are fixed inside the backend PR that touches the code (§9). Each fix gets its own
test and is called out in the PR description.

---

## 7. Subagent execution plan

Who does what:
- A strong model (or you) owns the **design PRs (3, 5)**, every review, and any
  test failure that needs judgment.
- Cheaper models do the mechanical PRs (1, 2, 4f, 6) and the bulk of 4a–4e, working
  from a precise spec.

Each subagent prompt includes:
1. **Scope**: the exact files and symbols to move or refactor, their target paths, and
   the files it may touch. Everything else is off-limits.
2. **Contract**: the relevant interface text from §4, copied in verbatim. Plus: "Do not
   change behaviour except for the listed bug fixes. Delete only code made redundant by
   this task. When unsure, keep the code and report it."
3. **Import update rule**: after moving, run
   `rg "<old.module.path>"` over `conin/ doc/ AGENTS.md README.md`, including
   `.ipynb` files, and update every hit.
4. **Verification**:
   - targeted `pytest` paths;
   - `pytest conin/tests/test_public_api.py`;
   - `black --check`;
   - for docs and notebook changes, the Sphinx doctest and html builds from AGENTS.md, plus
     `pytest --nbmake` on the notebooks it touched.
5. **Report back**: the files changed, the test commands and their results, any
   skips, and any deletions or kept-dead-code decisions.

Parallel fan-out:
- PR 1: one agent per model type (MN, BN, DBN, HMM). They run in sequence on shared
  files such as `__init__.py`, or a coordinator merges them.
- PR 2: one agent for the core constraint classes, one for the oracle library, and one
  for the MVR spec code.
- PR 4: one agent per backend (4a–4e). Each runs in its own worktree or branch off the
  merged PR 3.
- PR 6: one agent per group of docs pages or notebooks.

A shared "conventions" snippet goes into every prompt. It covers file naming, test
placement, docstring style (numpydoc, as used now), `try_import` for optional
dependencies, and no new top-level imports of optional backends.

---

## 8. Testing strategy
- PR 0 adds a public-API smoke test and records the baseline counts (959 passed / 195 skipped / 0 errors; see §2.3). After PR 1: 962 passed / 195 skipped. After PR 2: 1009 passed / 195 skipped (+47 new tests). After PR 3: 1074 passed / 195 skipped (+65).
- Every PR must keep the pass count ≥ baseline, after adjusting for deliberately deleted tests.
- PR 3 adds tests for:
  - immutability (setattr raises; `with_*` returns a new object);
  - mixed-family rejection;
  - lazy building (nothing built at `__init__`);
  - cache hits;
  - `get_copy` isolation (mutating a returned Pyomo model doesn't affect the next get).
  (PR 3: all done, using a test representation registered in the test; the Pyomo `get_copy` test moves to 4a.)
- Each bug fix gets a regression test.
- Notebooks run in CI, including `MVR/`.

---

## 9. Bugs found during exploration (fix in the relevant backend PR)

| Bug | PR |
|-----|----|
| `a_star_`: the partial-feasibility pruning condition is always true, and it checks `seq` instead of `newSeq` | 4c |
| `recursive_a_star`: partial feasibility uses the old `constraint_data`; `n_infeasible` is in the wrong branch | 4c |
| `mcem` replaces the HMM but the oracle keeps sampling from the old one; it also scores one sample and stores another | 4c |
| The `CHMM` overloads in the AStar/Optimization/CFN dispatchers are broken | 3 (removed) |
| VE: the ConstrainedMN overload replaces user evidence with `{}`; the MN factor oracle forces an existing scope node to 1 | 4e |
| VE: `_add_{bn,mn}_constraints_as_evidence` return `None` for an empty constraint list, and their `TypeError` strings were missing the `f` prefix | 2 (fixed, regression tests in `inference/tests/test_variable_elimination_helpers.py`) |
| `solve_toulbar2_map_query_model` rejects extra `**options` (e.g. `verbose`) | 4b |
| The DBN/HMM toulbar2 paths don't check that pytoulbar2 is available | 4b |
| The HMM `network_flow` branch can return `None` | 4a |
| `CallableDict.__getitem__` returns `KeyError` instead of raising it | 4a |
| `ConstrainedDynamicDiscreteBayesianNetwork.nodes` is not a property | 3 (fixed, regression test in `constrained_models/tests/test_dynamic_bayesian_network.py`) |
| `a_star` on an unconstrained `ConstrainedHiddenMarkovModel` raised `AttributeError` (a bare `CHMM` has no `is_feasible`) | 3 (fixed as a side effect of the "oracle" repn) |
| `mcem` calls `app.generate_hidden` twice per sample (it scores one sequence and trains on another) | 4c |
| `HMMApplication.create_chmm("pyomo")` attaches no `data`, so constraints that read application data (e.g. `data.num_zeros` in test_application_chmm) would fail if solved | 4c |
| `MVR/formula.ipynb` prose says `MVR_HMM`/`MVRFunctor` (typos for `MVR_CHMM`/`MVRConstraint`) | 4d |
| `DynamicDiscreteBayesianNetwork.t` is public and gets mutated by `create_bn_from_dbn` and `DBNSampler` (an immutability leak) | 1 (fixed: `create_bn_from_dbn` restores `t`; `DBNSampler` works on a deep copy) |
| `backends.rst` calls smoek optional although pyproject requires it; torch and pytoulbar2 aren't declared as extras | 6 |

---

## 10. Critiques of the original sketch

- **[C1] Copy-on-get can't be one strategy.** Pyomo `clone()` is about as costly as a
  build, and pytoulbar2 CFNs can't be copied at all. Resolution:
  `Representation.get_copy()` lets each class choose, with `deepcopy` as the default (§4.4).
- **[C2] A representation is not purely a function of (model, constraints).**
  HMM and DBN representations depend on the horizon `T` / `start, stop`, so the cache is
  keyed by structure parameters as well (§4.4).
- **[C3] The "MVR repn = expanded HMM, MVR inference just calls viterbi.py" idea doesn't
  hold.** The code never builds an expanded HMM. It runs torch kernels over an implicit
  augmented chain. Building it out would blow up the state space to |H|·Π|M_i|, and
  `InhomMVR` makes the chain time-inhomogeneous, which `HiddenMarkovModel` can't
  represent. Resolution: the MVR repn holds the aligned MVRs plus a static context, and
  MVR inference keeps its own kernels (§4.6).
- **[C4] "Generic/" was ambiguous** because Viterbi is HMM-specific. Renamed to `unconstrained/`.
- **[C5] Splitting by backend in every top-level directory can over-fragment.**
  `queries/` would only ever hold `mvr/`, and `learning/oracle/` holds one file. The
  plan accepts this for the sake of uniformity.
- **[C6] Pyomo constraint ↔ Pyomo representation is not 1:1.** There are two Pyomo HMM
  formulations, and a user's Pyomo constraint is written against one formulation's
  `model.V` shape. So there are two repn kinds.
- **[C7] A single PR would be unreviewable** at about 39k lines. The plan splits it
  into moves and logic changes on an integration branch.
- **[C8] "Hard break + multi-PR" is costlier per PR**, because each PR must fix every
  importer, including docs and notebooks. The upside is a cleaner end state.

---

## 11. Decisions made

| Topic | Decision |
|-------|----------|
| Caching | Cache representations; a getter returns `repn.get_copy()`, which defaults to deepcopy and is overridden per class |
| Constrained-model mutability | Fully immutable; use `with_constraints` / `with_model` |
| Oracle split | `OracleConstraint` (predicate) and `FactorOracleConstraint` (with nodes) |
| MVR_CHMM | Becomes `MVRRepresentation`; prebuilt MVRs are valid constraints ("mvr" family) |
| MVR function inputs | Take a `ConstrainedHiddenMarkovModel`; names kept |
| MVR in map_query | `method="mvr_viterbi"`, added in PR 5 |
| Old import paths | Hard break, no shims |
| PR strategy | Multiple PRs into a `restructure` integration branch; one 2.0 merge |
| Bugs | Fixed in the relevant backend PR, each with a test |
| Dead code | Keep `mediation_variables/` untouched; delete code made redundant by the refactor; keep untested-but-plausibly-useful code |
| Naming | `graphical_models`, `constraints`, `constrained_models`, `representations`, `inference`, `learning`, `sampling`, `queries`; lowercase backend subdirs |
| No-constraint algorithms | `unconstrained/` |
| Constructors | `Constrained*(model, constraints=...)` |
| Examples | `conin/examples/{markov_network,bayesian_network,dynamic_bayesian_network,hidden_markov_model}.py` (same module names as `graphical_models/`); update AGENTS.md |
| HMMApplication | Keep it, move it to `applications/`, decouple it via `RecursiveConstraint` |
| `common/` | Split into `io/` (file IO) and `common/` (non-IO helpers) |
| pgmax/pomegranate | Remove the converters and the loaders (the loaders just load with pgmpy and convert); keep pyagrum (it reads files natively) |
| network_flow IP | Keep it as the `pyomo_network_flow` repn |
| Plan location | Root `RESTRUCTURE_PLAN.md`, deleted at the end |
| Constraint base class | Renamed `ConstraintFunctor` → `Constraint` |
| MVR kernels | `representations/mvr/kernels.py` |
| Wrapped-model property | `.model` only, no aliases |
| Tests | Stay next to the code |
| Unconstrained models in `map_query` | Wrapped as `Constrained*(gm, ())`; the wrapper is transparent to all methods |
| Cache size | Unbounded per object |
| `mvr_timerange` | Non-mutating by default |
| Multiple `RecursiveConstraint`s | Combined automatically (tuple of states) |
| DBN `.t` leak | Fixed in PR 1 |
| pgmpy conversion | Stays in `common/pgmpy/` |
| Version | 2.0.0 |
| Application `data` (PR 3) | Constructor kwarg: `Constrained*(model, constraints=(), data=None)`, read-only `.data`, passed to builders; not part of the cache key |
| PR 3 representation scope | Framework + the HMM kinds that replace `initialize_chmm`: `"oracle"` (Oracle_CHMM), `"pyomo_network_flow"` (PyomoAlgebraic_CHMM), `"mvr"` (MVR_CHMM). pyomo/toulbar2/factor builders come in 4a/4b/4e; `map_query` dispatch is unchanged until PR 5 |
| MVR in PR 3 | `MVR_CHMM` and the torch algorithms are untouched until 4d; the `"mvr"` repn wraps `MVR_CHMM` and already exposes `.mvrs` |
| Constrained-class location | PR 3 `git mv`s them to `constrained_models/` (rename-only first commit) and deletes the empty `markov_network/`, `bayesian_network/`, `dynamic_bayesian_network/` packages |
| Representation API (PR 3) | `Representation` ABC with `kind` and `get_copy()`; `@register_representation(kind, model_type, families=(None,))`; `build_representation(cmodel, kind, **params)`; `supported_representations(model_type)` (§4.3) |
| `with_model` / `with_constraints` cache (PR 3) | Always start with an empty cache |
| Single constraint (after PR 3) | `Constrained*(model, c)` is accepted (wrapped as `(c,)`); single-element lists dropped in examples/docs/tests |
| Accepted constraint families (after PR 3) | Derived from the representation registry at the end of PR 4; `supported_families` is removed then. Not stored on constraint classes (§4.2) |
| `ConstrainedHiddenMarkovModel` model type (PR 3) | Requires `HiddenMarkovModel`; `HMM_MatVecRepn` is rejected (no caller passed one) |
| HMM repn modules (PR 3) | `representations/oracle/hmm.py`, `representations/pyomo/hmm_network_flow.py`, `representations/mvr/hmm.py` |
| mcem (PR 3) | Keeps the model in a local variable and does not write it back to `app`; same results |
| Constructor names / CHMM overloads | First parameter is `model` (no `hmm=`/`pgm=` alias); the broken `CHMM` overloads in the AStar/Optimization/CFN dispatchers are removed in PR 3 |

---

## 12. Open questions

None blocking. These can be settled during PR 3 (the design PR):
- Whether Pyomo `get_copy()` should use `clone()` or rebuild, depending on what benchmarks show (§4.4).
- (`with_model` cache behaviour, the `Representation` signatures and the registration decorator were settled in PR 3; see §11.)
