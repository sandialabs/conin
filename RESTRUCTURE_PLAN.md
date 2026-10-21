# Conin Restructure Plan

Status: **In progress.** PR 0 is merged into `main` (#241). PR 1 is merged into `restructure` (#242). PR 2 is merged into `restructure` (#243). PR 3 is merged into `restructure` (#244). PR 4 is split into sub-PRs 4f, 4a, 4b, 4e, 4c, 4d (in that order, stacked branches); 4f is done (branch `restructure-pr4f-io`, off `restructure`, merged as #245); 4a is done (branch `restructure-pr4a-pyomo`, off 4f, merged as #246); 4b is done (branch `restructure-pr4b-toulbar2`, off 4a, merged as #247); the representation cache was removed after 4b (#248); 4e is done (branch `restructure-pr4e-factor`, off `restructure` 3cf8d04; its follow-up removes the dead pgmpy log-probability code and adds `log_probability` and `is_feasible`; a second follow-up removes the last import cycle and moves the unneeded function-level imports to the top of their modules; merged as #249/#250); 4c is done (branch `restructure-pr4c-oracle`, off `restructure` 657c29f); a comment-only cleanup PR removes the PR-number and plan-section references from the code (branch `restructure-comment-cleanup`, off 4c; rule in §6); 4d is done, including the final step of PR 4 (branch `restructure-pr4d-mvr`, off `restructure` e5b7c04), merged as #253; 4g is done (branch `restructure-pr4g-hmm-repn`, off `restructure` b008502, the 4d merge; a follow-up moves `DBNSampler` to `conin.sampling.unconstrained`), merged as #254, so PR 4 is complete; PR 5 is done (branch `restructure-pr5-dispatch`, off `restructure` 50f9eb1, the 4g merge; a follow-up on the same branch gives HMM observations one reading in `graphical_models/hmm_observations.py`, makes VE honour `stop`, and renames the underscore modules). Next: PR 5b (scoring) and PR 6. The later PRs target the `restructure` branch.
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
   tuple of same-family constraints, and representations built on demand (not cached; see §4.4).
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
After PR 4f:
```
conin/
  io/                    __init__ (load_model, save_model, load_conin_model_from_uai, save_model_uai, load_conin_model_from_cfn),
                         load_model.py (unified load_model + load_conin_model), save_model.py (unified save_model + save_conin_model),
                         uai.py (UAI load + save_model_uai), cfn.py (CFN load), pgmpy_files.py (pgmpy BIF/UAI load + save),
                         convert_pgm.py (the `convert_pgm` console script), tests/ (IO tests + all model data files)
  common/                __init__ (log_potential, is_polytree, convert_pgmpy_to_conin, convert_conin_to_pgmpy_bn/mn),
                         log_potential.py (unified dispatcher + log_potential_conin), is_polytree.py,
                         pgmpy/ (to_conin, to_pgmpy_bn, to_pgmpy_mn, mapcpd, log_potential, polytree, custom_VE), tests/
  common/{conin,unified,bin,pgmax,pomegranate,pyagrum}/   removed
```
After PR 4a:
```
conin/
  representations/pyomo/ common.py (PyomoVarWrapper, apply_pyomo_constraints + add_algebraic_constraints_to_pyomo_model
                         (smoek->pyomo), create_MN_pyomo_map_query_model_from_factorial_repn, c5 helpers + C5LazyCutGenerator,
                         factor_representation_of_mn, skeleton_mn_of_bn, PyomoFactorRepresentation),
                         mn.py bn.py dbn.py hmm.py (Pyomo{MN,BN,DBN,HMM}Representation, kind "pyomo"),
                         hmm_network_flow.py (PyomoNetworkFlowHMMRepresentation + _create_index_sets), tests/
  inference/pyomo/       solve.py (solve_pyomo_map_query_model + parsers), mn.py bn.py dbn.py hmm.py
                         (inference_pyomo_map_query_{MN,BN,DDBN,HMM}), hmm_network_flow.py (lp_inference, ip_inference), tests/
  inference/{mn,bn,dbn,hmm}/   only toulbar2 left (+ mn/factor_repn.py, until 4e)
  hidden_markov_model/   chmm_algebraic.py and inference/lp.py removed
  constraints/algebraic/ add_constraints_pyomo.py removed (now representations/pyomo/common.py)
```
After PR 4b:
```
conin/
  representations/toulbar2/ common.py (Toulbar2VarWrapper, apply_toulbar2_constraints, Toulbar2FactorRepresentation,
                         add_algebraic_constraints_to_toulbar2_model + add_toulbar2_constraints (smoek->toulbar2)),
                         mn.py bn.py dbn.py hmm.py (Toulbar2{MN,BN,DBN,HMM}Representation, kind "toulbar2"), tests/
  inference/toulbar2/    solve.py (solve_toulbar2_map_query_model + inference_toulbar2_map_query_MN), bn.py dbn.py hmm.py
                         (inference_toulbar2_map_query_{BN,DDBN,HMM})
  inference/{mn,bn,dbn}/ no toulbar2 code left: empty __init__ + tests/ (+ mn/factor_repn.py, until 4e)
  inference/hmm/         removed
  constraints/algebraic/ add_constraints_toulbar2.py removed (now representations/toulbar2/common.py)
  io/uai.py              save_model_uai writes the text of the private _uai_text(pgm) (shared with the toulbar2 repns)
```
After PR 4e:
```
conin/
  representations/factor/ factor_repn.py (moved from inference/mn/: extract_factor_representation[_], get_factor_value, plus
                         factor_representation_of_mn, skeleton_mn_of_bn, tuple_var_index_map moved from pyomo/common.py),
                         materialise.py, common.py (apply_factor_constraints, FactorRepresentation),
                         mn.py bn.py dbn.py hmm.py (Factor{MN,BN,DBN,HMM}Representation, kind "factor"), tests/
  inference/variable_elimination/ solve.py (solve_variable_elimination_map_query_model), mn.py bn.py dbn.py hmm.py
                         (inference_variable_elimination_map_query_{MN,BN,DDBN,HMM}), custom_VE.py (from common/pgmpy/)
  inference/VariableElimination.py  the ovld dispatcher only (calls the functions above; native pgmpy MN/BN overloads unchanged)
  inference/{mn,bn,dbn}/ no code left: empty __init__ + tests/
```
After the PR 4e follow-up:
```
  common/                __init__ (log_potential, log_probability, is_polytree, convert_*), log_potential.py
                         (log_probability + log_potential for conin and pgmpy models), is_polytree.py,
                         pgmpy/ (to_conin, to_pgmpy_bn, to_pgmpy_mn, mapcpd, polytree), tests/
  common/pgmpy/log_potential.py, inference/variable_elimination/custom_VE.py   deleted (dead code)
  constrained_models/    feasibility.py (is_feasible checks: oracle, factor_oracle, algebraic, mvr), tests/test_feasibility.py
  util/                  var_wrapper.py (PyomoVarWrapper, moved from representations/pyomo/common.py, which re-exports it;
                         this removes the constraints.algebraic <-> representations.pyomo import cycle)
```
After PR 4c:
```
conin/
  constraints/           recursive.py (RecursiveConstraint, family "recursive")
  representations/oracle/hmm.py      OracleHMMRepresentation (a real repn: is_feasible / partial_is_feasible on internal indices)
  representations/recursive/hmm.py   RecursiveHMMRepresentation (the combined tuple-state constraint)
  inference/unconstrained/ viterbi.py (from hidden_markov_model/inference/)
  inference/oracle/      a_star_hmm.py (a_star, a_star_; from hidden_markov_model/inference/a_star.py), tests/
  inference/recursive/   a_star_hmm.py (recursive_a_star, Recursive_Heap_Item, Unique_Heapq), tests/
  inference/tests/       test_hmm_inference.py (from hidden_markov_model/tests/test_inference.py)
  learning/              __init__ (supervised_learning, add_unknowns, convert_to_simulations, mcem),
                         unconstrained/ (supervised.py, helpers.py), oracle/ (mcem.py), tests/test_learning.py
  sampling/oracle/       rejection_hmm.py (rejection_sample_hidden, from Oracle_CHMM.generate_hidden)
  constrained_models/tests/test_hidden_markov_model.py   (from hidden_markov_model/tests/test_oracle_chmm.py)
  hidden_markov_model/   only the MVR code is left: chmm_mvr.py (MVR_CHMM, no base class), mvr_common.py,
                         inference/ learning/ sampling/ other_queries/ (the *_mvr modules), tests/ (MVR tests + test_util.py);
                         __init__ exports nothing. chmm.py (CHMM), chmm_oracle.py (Oracle_CHMM), hmm_application.py
                         (HMMApplication) and tests/test_application_chmm.py are deleted. No conin/applications/ package.
```
After PR 4d:
```
conin/
  representations/mvr/   hmm.py (MVRHMMRepresentation: model, mvrs; align_hidden_states, from chmm_mvr._align_hidden_states),
                         kernels.py (the torch kernels, from hidden_markov_model/mvr_common.py), tests/test_mvr_representation.py
                         (from hidden_markov_model/tests/test_mvr_chmm.py); __init__ does not import kernels (no torch)
  inference/mvr/         viterbi.py, marginal_map.py (from hidden_markov_model/inference/*_mvr.py), tests/
  learning/mvr/          baum_welch.py (+ forward_backward_mvr_chmm), generalized_em.py, tests/
  sampling/mvr/          ffbs.py, stopped_sampling.py, tests/
  queries/               __init__ (new package); mvr/ sat_prob.py, sat_time.py (from hidden_markov_model/other_queries/), tests/
                         The four mvr/__init__ export the algorithms lazily (module __getattr__), so importing them needs no torch
  util/tests/test_util.py (from hidden_markov_model/tests/; it tests conin.util.Util)
  hidden_markov_model/   removed (chmm_mvr.py / MVR_CHMM deleted; tests/probabilities.json was empty and unused);
                         `conin/__init__.py` no longer imports it
  constrained_models/    base.py: the accepted families come from the registry (`supported_families` removed)
```
After PR 4g:
```
conin/
  representations/matvec/ hmm.py (MatVecHMMRepresentation, kind "matvec": the HMM parameters as read-only float64 arrays + their
                         logs on internal indices), tests/test_matvec_representation.py
  sampling/unconstrained/ hmm.py (generate_hidden, generate_hidden_until_state, generate_observed_from_hidden, generate_observed:
                         the former HiddenMarkovModel methods, as functions of a bare or unconstrained HMM), tests/test_hmm.py
  graphical_models/      hidden_markov_model.py: HMM_MatVecRepn, repn, initialize() and the generate_* methods removed;
                         the constructor validates the probabilities
  constrained_models/    hidden_markov_model.py: the generate_* methods removed
```
After the PR 4g follow-up:
```
conin/
  sampling/unconstrained/ dbn.py (DBNSampler, moved from sampling/dbn/sampling.py), tests/test_dbn.py + tests/dbn_examples.py
                         (from sampling/dbn/tests/{test_sampling,examples}.py); __init__ also exports DBNSampler
  sampling/dbn/          removed
```
After PR 5:
```
conin/
  inference/             map_query.py (map_query: normalise -> method -> representation kind -> _solve_map_query, one ovld overload
                         per representation class; METHODS), hmm_results.py (key_solutions_by_time, hidden_states_from_unrolled:
                         the shape of HMM MAP results), query_checks.py (reject_pruning_options, full_query_variables: shared by
                         integer_program and toulbar2; they replace pyomo/bn._validate_bn_query); each backend module has a private repn-level core
                         (_map_query_pyomo_{mn,bn,dbn,hmm}, _map_query_network_flow, _lp_inference, _ip_inference,
                         _map_query_toulbar2_{mn,bn,dbn,hmm}, _map_query_variable_elimination_{mn,bn,dbn,hmm}, _viterbi, _a_star,
                         _recursive_a_star, _viterbi_torch_mvr) that the public function and the overload call
  inference/{Optimization,CFN,AStar,Viterbi}Inference.py, VariableElimination.py   deleted
  inference/{mn,bn,dbn}/ removed; their tests are inference/tests/{mn,bn,dbn}/ (rename-only commit)
  util/                  constraint_data.py (merge_constraint_data: the application data + a representation's fields)
  graphical_models/      hmm_observations.py (PR 5 follow-up: as_sequence, require_sequence_type, time_range, unrolled_evidence:
                         the one reading of HMM observations, used by inference, feasibility.hmm_sequences and later scoring)
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
    `PyomoAlgebraic_CHMM` and `MVR_CHMM` until 4a/4c/4d; **4c: "oracle" is a real representation and `Oracle_CHMM`
    is gone; its `generate_hidden` is `conin.sampling.oracle.rejection_sample_hidden`**; **4d: "mvr" is a real
    representation (`MVRHMMRepresentation`: builds, validates and aligns the MVRs) and `MVR_CHMM` is gone**);
  - inside "model" classes: `PyomoAlgebraic_CHMM.generate_model` (PR 3: reached only through the
    "pyomo_network_flow" repn; **4a: replaced by `PyomoNetworkFlowHMMRepresentation.create_model`**);
  - inside inference functions: `create_*_map_query_model_*` (**4a: the Pyomo ones are removed; the
    "pyomo" representations build the models**; **4b: the toulbar2 ones too, the "toulbar2" representations build the CFNs**), and `lp_inference`
    builds a CHMM (PR 3: uses `representation("pyomo_network_flow")`; 4a: `create_model(observed=)`);
  - inside the dispatch layer: `VariableElimination._add_*_constraints_as_evidence` (**4e: replaced by
    `representations/factor/common.apply_factor_constraints`, run once when the "factor" repn is built**);
  - inside the model: `HiddenMarkovModel.initialize()` built and cached `HMM_MatVecRepn` (**4g: replaced by the registered
    `"matvec"` representation; the samplers it held are `conin.sampling.unconstrained`**).
- **Almost nothing is cached.** Only `HiddenMarkovModel.repn` is (**4g: removed; the `"matvec"` repn is built per call like the others**). The
  HMM→DBN→BN→MN→factor-repn→Pyomo chain is rebuilt on every query (4a: cached up to the factor repn,
  per `(start, stop)`, when the same constrained model is reused; 4b: the toulbar2 chain is cached up to the UAI text; 4e: the VE path builds the "factor" repn (constraints materialised once) and converts to pgmpy per query; `map_query` on a bare model still wraps it anew each call). **The representation cache was removed after 4b (§4.4, §11): every `representation()` call builds a new repn again; a repn's `create_model()` can still be called for several queries.**
- **There is a lot of duplication:**
  - 3 copies of `PyomoVarWrapper` (**4a: one, in `representations/pyomo/common.py`**);
  - the MN and BN toulbar2 builders are near-identical (**4b: one `Toulbar2FactorRepresentation.create_model`**);
  - every ovld dispatcher has identical Constrained/unconstrained overload pairs;
  - there are 3 separate `add_constraints` dispatchers (**4a: the Pyomo one is
    `representations/pyomo/common.apply_pyomo_constraints`**; **4b: the toulbar2 one is
    `representations/toulbar2/common.apply_toulbar2_constraints`**; **4e: the VE one is
    `representations/factor/common.apply_factor_constraints`**).
- **The imports are tangled.**
  - `constraints` lazily imports MN, BN and HMM.
  - `operators.py` and `mvr_operators` import each other (a cycle).
  - Some code uses implicit attribute-path imports (`conin.inference.mn.inference_pyomo.solve_…`)
    that only work because of import order (**4a: gone from the Pyomo code; 4b: gone from the toulbar2 code**).
- `mvr_common.py` mixes representation building with torch kernels (**4d: the MVR building and alignment are in
  `representations/mvr/hmm.py`; the torch kernels are `representations/mvr/kernels.py`**).
- `recursive_a_star` relies on an undeclared duck-typed protocol on an
  `HMMApplication` subclass instead of a constraint object (**4c: it takes a `ConstrainedHiddenMarkovModel` with
  `RecursiveConstraint`s; `HMMApplication` is deleted**).

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
- **After the PR 3 follow-ups** (single constraint accepted, brackets dropped): **1075 passed, 195 skipped,
  0 errors** in `.venv-opencode` (+1, `test_single_constraint_accepted`). The same suite in the
  `conin_scaling_experiments` venv (pgmpy + torch) gives 1434 passed, 119 skipped.
- **PR 4 branch point** (`restructure` = last-fetched `origin/restructure`, ea7a5b0, the PR 3 merge): 1075 passed,
  195 skipped, 0 errors in `.venv-opencode` (re-run; matches the PR 3 follow-ups).
- **After PR 4f:** **1076 passed, 193 skipped, 0 errors** in `.venv-opencode`. Removed: the 4 pgmax/pomegranate/pyagrum
  loader tests in `io/tests/test_load_model.py` (passed) and the 2 pomegranate load tests (skipped). `test_public_api.py`:
  224 passed, 9 skipped (+5 `RESTRUCTURE_PLAN_API` rows for the §5 `conin.io` / `conin.common` names; the other
  `conin.common*` rows moved to `conin.io*` / `conin.common.pgmpy`). In the `conin_scaling_experiments` venv (pgmpy):
  `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged); `conin/io conin/common conin/inference/{bn,dbn,mn}` +
  `test_MAP_Inference.py` + `test_public_api.py`: 496 passed, 65 skipped, 0 failed.
- **PR 4a branch point** (`restructure-pr4a-pyomo` = `restructure-pr4f-io` = b59605a, the 4f head; `restructure` =
  last-fetched `origin/restructure` 1eb3fc0 is its merge, same tree): 1076 passed, 193 skipped, 0 errors (re-run).
- **After PR 4a:** **1115 passed, 193 skipped, 0 errors** in `.venv-opencode` (+39). New:
  `representations/pyomo/tests/test_pyomo_representations.py` (30: registry pairs, repn classes, get_copy isolation,
  evidence isolation, cache keys, the constraint `data` each model passes, the two §9 regression groups, network flow
  with algebraic constraints, BN build options) and +9 `test_public_api.py` rows (`conin.inference.pyomo.*`,
  `conin.representations.pyomo.{PyomoNetworkFlowHMMRepresentation,PyomoVarWrapper}`): 233 passed, 9 skipped.
  Moved: `test_cut_generation.py` (→ `inference/pyomo/tests/`), `test_pyomo_chmm.py` (→
  `representations/pyomo/tests/test_hmm_network_flow.py`). Rewritten: the PR 3 `pyomo_network_flow` tests in
  `representations/tests/test_hmm_representations.py` (`create_model` returns `(model, data)`; get_copy returns the
  cached object), `test_application_chmm.test_algebraic_type`, `test_base.test_supported_representations_hmm` (more
  pairs). No test removed. In the `conin_scaling_experiments` venv (pgmpy): `test_MAP_Inference.py` 69 passed,
  41 skipped (unchanged). LP files written for 15 MN/BN/DBN/HMM/network-flow cases (constrained, algebraic,
  evidence, omit_c5) are byte-identical to the old path.
- **After the PR 4a follow-ups** (`_validate_bn_query` merges the two BN query helpers; `ConstrainedGraphicalModel.ensure`):
  **1116 passed, 193 skipped, 0 errors** in `.venv-opencode` (+1, `test_base.test_ensure`).
- **PR 4b branch point** (`restructure-pr4b-toulbar2` off `restructure-pr4a-pyomo` = 994ce9a; `restructure` = last-fetched
  `origin/restructure` 08f9d95 is its merge, same tree): 1116 passed, 193 skipped, 0 errors (re-run).
- **After PR 4b:** **1151 passed, 193 skipped, 0 errors** in `.venv-opencode` (+35). New:
  `representations/toulbar2/tests/test_toulbar2_representations.py` (29: registry pairs, repn classes and get_copy, the
  cached UAI text = `save_model` output, the var-wrapper state order, two gets give independent CFNs, evidence isolation,
  cache keys, `verbose`/`write_uai_file` not cache keys, `write_uai_file` with a dotted directory, the constraint `data`
  each model passes, the two §9 regression groups) and +6 `test_public_api.py` rows (`conin.inference.toulbar2.*`,
  `conin.representations.toulbar2.Toulbar2VarWrapper`; the `conin.constraints.algebraic.add_constraints_toulbar2` module
  row moved to `conin.representations.toulbar2.common`): 239 passed, 9 skipped. Rewritten:
  `test_base.test_supported_representations_hmm` (+3 toulbar2 pairs). No test removed. Commit 1 (rename-only) gave
  1116/193. In the `conin_scaling_experiments` venv (pgmpy): `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged).
  The UAI text, solutions, primal bounds, constraint `data` and `M.V` are identical to the old path on 26 cases
  (MN/BN/DBN/HMM; unconstrained, toulbar2, algebraic, evidence, `solution_with_evidence`, list/dict HMM evidence,
  `write_uai_file`).
- **PR 4e branch point** (`restructure-pr4e-factor` off `restructure` = last-fetched `origin/restructure` 3cf8d04, the
  cache-removal merge; `git fetch` is not possible here): 1150 passed, 193 skipped, 0 errors in `.venv-opencode` (re-run).
- **After PR 4e:** **1181 passed, 202 skipped, 0 errors** in `.venv-opencode` (+31 passed, +9 skipped). New:
  `representations/factor/tests/test_factor_representations.py` (25: 16 pass and 9 need pgmpy in `.venv-opencode`; registry
  pairs, repn classes, structure params, the constraint `data` each model passes, build-time materialisation, evidence not
  mutating the repn, two `create_model()` calls give independent pgmpy models, one repn serves several queries,
  `write_uai_file`, the §9 regression tests, brute-force parity for constrained MNs, the pyomo-family rejection) and +12
  `test_public_api.py` rows (`conin.inference.variable_elimination.*`, `conin.representations.factor.*`): 251 passed,
  9 skipped. Moved and rewritten: `inference/tests/test_variable_elimination_helpers.py` →
  `representations/factor/tests/test_apply_factor_constraints.py` (5 tests for the deleted `_add_{bn,mn}_constraints_as_evidence`
  → 8 for `apply_factor_constraints`, incl. the MN no-forced-evidence regression). Moved: `inference/mn/tests/test_factorial_repn.py`
  → `representations/factor/tests/`. Rewritten: `test_base.test_supported_representations_hmm` (+3 factor pairs). No test
  removed. Commit 1 (rename-only) was checked with the targeted tests. In the `conin_scaling_experiments` venv (pgmpy):
  `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged); with smoek on `PYTHONPATH` (unzipped from the wheelhouse),
  `representations/factor`, `common`, `inference/{mn,bn,dbn}`, `constrained_models` and `test_public_api.py`: 456 passed,
  9 skipped. VE parity with the old path on 47 cases (MN/BN/DBN/HMM, conin and pgmpy inputs; unconstrained, factor-oracle,
  algebraic; evidence, `variables`, `solution_with_evidence`, list/dict HMM evidence, `start`/`stop`): identical solutions and
  identical `write_uai_file` output (41 files), except the two §9 MN fixes (constrained MN with evidence; constrained MN with
  `variables`, which no longer returns the forced scope node) and the unsupported families (`TypeError` → `InvalidInputError`).
- pgmpy, or_topas, gurobi and glpk are not installed in `.venv-opencode`; torch is broken
  there; greenery is missing.
- PR 0 adds `doc/notebooks/MVR/*.ipynb` to the CI notebook job and fixes the
  `regex_tutorial.ipynb` import (`conin.constraint` → `conin.constraints`).
  The job forces `--nbmake-kernel=python3`, because the MVR notebooks declare a `conin` kernel.
- Torch tests and the MVR notebooks are verified in GitHub Actions, not locally
  (the dev machine is an emulated CPU; see /projects/AGENTS.md).

---

- **After the PR 4e follow-up** (dead pgmpy log-probability code removed; `log_probability`/`log_potential`/`is_feasible`):
  **1269 passed, 204 skipped, 0 errors** in `.venv-opencode` (+88 passed, +2 skipped). Rewritten:
  `common/tests/test_log_potential.py` (2 tests → 62: MN/BN/DBN/HMM values against hand computations and
  normalisation, sparse factors, input errors, constrained models of every checkable family, pgmpy conversion).
  New: `constrained_models/tests/test_feasibility.py` (26: every family, the HMM `is_feasible` §9 regression tests,
  MVR windows). +4 `test_public_api.py` rows (`log_probability` ×3, `cancer1_BN_constrained_oracle_conin`): 255 passed,
  9 skipped. Changed: `test_to_pgmpy_bn.test_model_equivalence` (uses `log_probability`, also checks pgmpy's
  `get_state_probability`), `test_factor_representations._brute_force` (`log_potential`). Sphinx doctest: 62 tests,
  0 failures; HTML builds. In the `conin_scaling_experiments` venv with smoek on `PYTHONPATH`: `conin/common`,
  `conin/constrained_models`, `conin/representations`, `test_load_cfn.py`, `test_public_api.py`, the oracle/application
  CHMM tests: 540 passed, 0 skipped; `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged).
- **After the import-cycle follow-up** (`PyomoVarWrapper` → `conin/util/var_wrapper.py`; function-level imports moved to
  the top): **1270 passed, 204 skipped** in `.venv-opencode` (+1: `test_var_wrapper_is_one_class`). Sphinx doctest: 62
  tests, 0 failures. With pgmpy + smoek (`conin/common`, `conin/constrained_models`, `conin/representations`,
  `test_load_cfn.py`, `test_public_api.py`, the oracle/application CHMM tests): 606 passed. No conin module imports
  another inside a function to dodge a cycle (checked by moving each function-level import to the top).
- **PR 4c branch point** (`restructure-pr4c-oracle` off `restructure` 657c29f = last-fetched `origin/restructure`, which
  contains 4e and both follow-ups; `git fetch` is not possible here): 1270 passed, 204 skipped, 0 errors (re-run).
- **After PR 4c:** **1330 passed, 204 skipped, 0 errors** in `.venv-opencode` (+60). Commit 1 (rename-only) was checked
  with the targeted tests (346 passed, 14 skipped). Per file (before → after):
  - new: `inference/oracle/tests/test_a_star_hmm.py` (17: the §9 pruning regressions, brute-force parity, the time-frame
    partial-function fixes, family rejection), `inference/recursive/tests/test_a_star_hmm.py` (16: the two §9
    regressions, brute force, tuple-state combination, bare HMM, `is_feasible`);
  - rewritten: `representations/tests/test_hmm_representations.py` 11 → 27 (the `Oracle_CHMM` parity test became
    "matches the old checks" against an inline copy of the old logic; new: repn attributes, `data` for 2-argument
    predicates, no mutation by checks, rejection-sampling parity with the old `generate_hidden` for 3 seeds, the
    "recursive" repn and `RecursiveConstraint` validation); `learning/tests/test_learning.py` 14 → 17 (the two mcem §9
    regressions, bare HMM); `constrained_models/tests/test_hidden_markov_model.py` (moved `test_oracle_chmm.py`) 19 → 21
    (2 tests ported from `test_application_chmm.py`); `test_feasibility.py` 26 → 27 (`test_hmm_recursive`);
    `test_constraints.py` (`test_occurs_at_least_once_in_time_frame`: 4 partial-function expectations changed, see §9);
    `test_base.test_supported_representations_hmm` (+2 recursive pairs, asserts no ("oracle", "algebraic"));
    `inference/tests/test_hmm_inference.py` (moved `test_inference.py`, 23 → 23: `recursive_a_star(hmm=create_chmm1_recursive())`
    instead of the `Num_Zeros` app; same expectations);
  - removed: `hidden_markov_model/tests/test_application_chmm.py` (8 tests; `HMMApplication` is deleted; the
    oracle-sampling/feasibility and the network-flow-with-`data` tests were ported, the others tested only
    `HMMApplication` itself);
  - `test_public_api.py` 264 → 277 (+14 rows, −1 `HMMApplication` row; see §5): 268 passed, 9 skipped.
  In the `conin_scaling_experiments` venv (pgmpy): `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged; with smoek on
  `PYTHONPATH` 96 passed, 14 skipped). Sphinx doctest: 67 tests, 0 failures; HTML: 0 warnings. nbmake
  (`/projects/.conin-scratch/venv`): `constraints.ipynb`, `HMM_basics.ipynb`, `inference.ipynb` pass. Parity with the old
  path: A* (`a_star` and `map_query(method="a_star")`) on 5 models × 4 observation sequences × `num_solutions` 1/3 and
  `generate_hidden`/`generate_observed` for 3 seeds: 90 results identical; A* with `max_iterations` caps: identical
  outcomes on 15 cases (chmm1, T=12/15/20). Recursive A* on 162 cases (3 bounds × 6 sequences × `num_solutions` 1/2/4,
  with and without `max_iterations`): 158 identical, 4 changed by the §9 fix (see §9).
- **PR 4d branch point** (`restructure-pr4d-mvr` off `restructure` e5b7c04 = last-fetched `origin/restructure`, which contains
  4c (#251) and the comment cleanup (#252); `git fetch` is not possible here): 1330 passed, 204 skipped, 0 errors (re-run).
- **After PR 4d:** **1349 passed, 213 skipped, 0 errors** in `.venv-opencode` (+19 passed, +9 skipped; 1356 → 1562 collected
  is +26 tests). Commit 1 (rename-only) gave 1330/204; commit 2 (MVR repn) 1340/213; commit 3 (registry-derived families)
  1349/213. Per file (before → after):
  - `representations/tests/test_hmm_representations.py` 27 → 37 (new: repn attributes, no constraints / bare HMM, hidden-state
    alignment, factories called per build and with the new HMM after `with_model`, read-only arrays (hom, inhom, after
    deepcopy/pickle), `MVR_MatVecRepn` copies its inputs, `mvr_timerange` does not change the repn, other families raise);
  - `hidden_markov_model/tests/test_mvr_chmm.py` → `representations/mvr/tests/test_mvr_representation.py` 24 → 23 (removed the 3
    `MVR_CHMM` constructor tests: missing HMM (moot), mismatched hidden states and non-MVR constraint, rewritten as the 2
    repn tests `..._rejects_mismatched_hidden_states`, `..._rejects_a_factory_that_returns_no_mvr`);
  - `constrained_models/tests/test_base.py` 31 → 38 (accepted families from the registry ×4 model types, a family without a
    builder is rejected and named, registering a builder makes a family accepted, no `supported_families`;
    `test_supported_representations_hmm` asserts `("mvr", None)` too);
  - `constrained_models/tests/test_feasibility.py` 27 → 29 (`test_mn_oracle_and_factor_oracle` → `test_mn_factor_oracle`,
    `test_all_constraints_must_hold` uses factor oracles; new `test_oracle_constraints_are_rejected_...` ×2);
  - `common/tests/test_log_potential.py` 62 → 62 (`test_constrained_mn_oracle` → `..._factor_oracle`,
    `test_constrained_dbn_oracle_sees_data_T` → `..._factor_oracle_sees_data_T`, `test_constrained_checks_assignment_...`
    uses a factor oracle; same expectations);
  - `constraints/mvr/tests/test_mvr_operators.py` 47 → 48 (`test_mvr_timerange` and `..._inplace_...` pass `inplace=True`;
    new `test_mvr_timerange_does_not_mutate_by_default`);
  - the 8 torch MVR test files moved (`hidden_markov_model/tests/test_*_mvr.py` → `{inference,learning,sampling,queries}/mvr/tests/`,
    same names): `MVR_CHMM(hidden_markov_model=h, constraints=c)` → `ConstrainedHiddenMarkovModel(h, c)`, `model.constraints` →
    `model.representation("mvr").mvrs` where the built MVRs were meant, the GEM test builds `_build_sumprod_ctx(hmm, mvrs, ...)`;
    no test added or removed, same expectations;
  - `test_public_api.py` 277 → 286 (−1 `MVR_CHMM` row, +10 `PUBLIC_API` rows; the notebook rows use the package paths): 268
    passed, 18 skipped (9 torch rows skip in `.venv-opencode`).
  In `/projects/.conin-scratch/venv` (torch): all 8 torch MVR test files pass (637 tests in 3 detached runs, with the repn and
  public-API tests), the MVR spec tests 282 passed (was 281 + the new `mvr_timerange` test). In the `conin_scaling_experiments`
  venv: `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged); with smoek on `PYTHONPATH`, `conin/common`,
  `conin/constrained_models`, `conin/representations/factor`, `test_public_api.py`: 525 passed. Sphinx doctest: 67 tests, 0
  failures; HTML 0 warnings. nbmake: `MVR/regex_tutorial.ipynb` passes. The other MVR notebooks need torch for their main
  cells (left to CI): a script ran each notebook's cells in order with the torch algorithms stubbed, and every notebook built
  its models and representations and reached its first torch call (regex_tutorial ran completely). Parity with the old path
  (old tree via `git archive`): 132 results (Viterbi dense/sparse, marginal MAP, forward-backward with augmented posteriors, FFBS,
  Baum-Welch, GEM, sat_prob, sat_time, stopped sampling with and without `min_length`, and the user MVRs' time ranges afterwards;
  3 seeds × 4 constraint sets incl. hidden states out of sorted order, windows, a formula; same torch generator seeds) are
  bit-identical, including 6 identical `InvalidInputError`s.
- **PR 4g branch point** (`restructure-pr4g-hmm-repn` off `restructure` b008502 = last-fetched `origin/restructure`, the 4d merge
  (#253), which contains 533878f; `git fetch` is not possible here): 1349 passed, 213 skipped, 0 errors (re-run).
- **After PR 4g:** **1380 passed, 213 skipped, 0 errors** in `.venv-opencode` (+31; 1554 → 1585 collected). One commit (no file
  was moved, so there is no rename-only commit; the sampling tests that moved were rewritten as functions). Per file (before → after):
  - new: `representations/matvec/tests/test_matvec_representation.py` (11: registry pair `("matvec", None)` only and constraints
    rejected, no structure params, class/attributes/dtypes/logs/internal order on a sparse HMM, a new repn per call and no
    `hmm.repn`, read-only arrays (built, deepcopy, pickle), arrays not shared with the model, one repn serves several Viterbi
    queries unchanged, `viterbi`/`a_star` on an unconstrained `ConstrainedHiddenMarkovModel`, `viterbi` rejects constraints);
    `sampling/unconstrained/tests/test_hmm.py` (16: the 8 `generate_*` tests moved from `graphical_models/tests` and rewritten as
    function calls, same expectations; new: draws identical to the reference sampler ×3 seeds, unconstrained constrained model gives
    the same draws, constraints are refused ×4, negative `time_steps` raises before drawing);
  - `graphical_models/tests/test_hidden_markov_model.py` 58 → 55 (−8 moved `generate_*` tests; `test_initialize_flag` →
    `test_no_cached_numeric_representation`; +4 `test_constructor_validates_probabilities` (the §9 regression); the
    `initialize=True`/`initialize()` calls dropped, same expectations);
  - `constrained_models/tests/test_hidden_markov_model.py` 21 → 20 (removed `test_generate_observed_from_hidden_failure`: it tested
    the feasibility check of the removed `ConstrainedHiddenMarkovModel.generate_observed_from_hidden`; the other `generate_*`
    tests use `rejection_sample_hidden` + `conin.sampling.unconstrained`, same expectations; `repn.hmm == pgm.repn` →
    `isinstance(repn.hmm, MatVecHMMRepresentation)` + equal arrays; `.start_vec ==` lists → `.start_vec.tolist() ==`);
  - rewritten, same counts: `constrained_models/tests/test_base.py` (`test_hmm_matvec_repn_rejected` passes a matvec repn;
    `test_supported_representations_hmm` asserts `("matvec", None)` is the only matvec pair;
    `test_hmm_helpers_require_oracle_family` → `test_hmm_rejection_sampling_requires_oracle_family`, also asserts the methods are
    gone), `representations/tests/test_hmm_representations.py` (oracle `.hmm` is a matvec repn; the rejection-sampling reference
    uses `generate_hidden`; the `cmodel.generate_hidden` asserts became `rejection_sample_hidden`),
    `learning/tests/test_learning.py` (the mcem spy patches `rejection_sample_hidden` in the mcem module instead of the removed
    method), the torch MVR tests (`initialize=True` dropped; `score_path` reads `hmm.start_vec` etc.; `test_baum_welch_mvr`'s
    stale-arrays check uses the matvec repn), `constraints/mvr/tests/test_mvr_constraints.py`,
    `representations/mvr/tests/test_mvr_representation.py` (`initialize=True` dropped);
  - `test_public_api.py` 286 → 294 (+8 rows, none removed: `conin.sampling.unconstrained.{generate_hidden, generate_observed,
    generate_observed_from_hidden}` in SPHINX_CODE; `conin.representations.matvec.MatVecHMMRepresentation` and the four
    `conin.sampling.unconstrained.generate_*` in PUBLIC_API). There was no `HMM_MatVecRepn` row.
  In `/projects/.conin-scratch/venv` (torch): all 8 torch MVR test files pass (85 + 52 + 154 in three detached runs), and the MVR
  spec tests + `representations/mvr` + `representations/matvec` + `test_public_api.py` 610 passed. In the `conin_scaling_experiments`
  venv: `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged). Sphinx doctest: 67 tests, 0 failures; HTML 0 warnings. nbmake:
  `HMM_basics.ipynb`, `inference.ipynb`, `MVR/regex_tutorial.ipynb` pass; `nbstripout --verify` passes. The other 8 MVR notebooks
  (left to CI) were run cell by cell with the torch algorithms stubbed: every edited cell ran up to its first torch call, and the
  helpers defined in the edited cells (path weights, the notebooks' own rejection samplers, the brute-force posterior) were called
  on the new repn. Parity with the old tree (`git archive b008502`): 174 non-torch results (sampling draws of all four samplers ×4
  HMMs ×3 seeds, `viterbi`, `map_query` viterbi/a_star with list and dict evidence, `a_star` with 1/3 solutions, unconstrained and
  2 oracle-constrained models with and without `max_iterations` (incl. 4 identical `IndexError`s), recursive A* bare and
  constrained, rejection sampling, the old `ConstrainedHiddenMarkovModel.generate_observed` vs rejection sampling +
  `generate_observed_from_hidden`, mcem) are identical, including the result types (`np.float64` log-likelihoods); 35 torch
  results (`_hmm_to_torch` plain/log × float32/float64, Viterbi, marginal MAP, forward-backward, Baum-Welch, GEM, FFBS ×2 seeds,
  sat_prob, sat_time, stopped sampling with and without `min_length`, on an HMM whose labels are not in sorted order; same torch
  generator seeds) are bit-identical.
- **After the PR 4g follow-up** (`DBNSampler` → `conin.sampling.unconstrained`): **1381 passed, 213 skipped, 0 errors** in
  `.venv-opencode` (+1: the `conin.sampling.unconstrained.DBNSampler` `PUBLIC_API` row; `test_public_api.py` 294 → 295). The 12 DBN
  sampler tests moved unchanged (`sampling/dbn/tests/test_sampling.py` → `sampling/unconstrained/tests/test_dbn.py`; only imports).
- **PR 5 branch point** (`restructure-pr5-dispatch` off `restructure` 50f9eb1 = last-fetched `origin/restructure`, the 4g merge (#254),
  which contains 37eac77; `git fetch` is not possible here): 1381 passed, 213 skipped, 0 errors (re-run).
- **After PR 5:** **1533 passed, 218 skipped, 0 errors** in `.venv-opencode` (+152 passed, +5 skipped; 1594 → 1751 collected). Commit 1
  (rename-only, `inference/{mn,bn,dbn}/tests` → `inference/tests/{mn,bn,dbn}/`) was checked with those tests (32 passed, 51 skipped,
  unchanged). Per file (before → after):
  - new: `inference/tests/test_map_query_dispatch.py` (123: normalisation of bare/constrained/pgmpy models, `TypeError` for non-models
    ×3 × 6 methods, unknown method, the method → kind table incl. every `ip_formulation` and `a_star` by family, structure params per
    (model, kind), one overload per representation class (17) and each overload solving, the registry errors for HMM-only methods on
    MN/BN/DBN and for unsupported families, `ip_formulation` on an MN, HMM evidence `None`/str/tuple ×6 methods → `InvalidInputError`,
    viterbi on `ConstrainedHiddenMarkovModel(hmm)`, a_star → recursive A* (list/dict, no solution), network flow lp/ip by constraints,
    DBN `stop=None` → 1 ×3 methods); `inference/tests/test_constraint_data_and_query_checks.py` (30: the two §9 regression groups);
  - `inference/oracle/tests/test_a_star_hmm.py` 17 → 21 (`test_pruning_reduces_the_search_without_changing_the_optimum` asserts
    `solution is None` instead of `IndexError`; new: no solution within `max_iterations`, infeasible model, fewer solutions than asked
    for, `map_query` without a solution);
  - `inference/mvr/tests/test_viterbi_mvr.py` +5 `mvr_viterbi` tests (torch; skipped in `.venv-opencode`);
  - no test removed; `test_public_api.py` unchanged (295: no public name added or removed; the dispatchers were private).
  In the `conin_scaling_experiments` venv: `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged); with smoek on `PYTHONPATH`, the two
  new files + `inference/tests/{mn,bn,dbn}` + `representations/factor` + `constrained_models` + `test_public_api.py`: 675 passed,
  9 skipped. In `/projects/.conin-scratch/venv` (torch): `inference/mvr/tests` + `test_public_api.py` 385 passed. Sphinx doctest: 67
  tests, 0 failures; HTML builds with 0 warnings. Parity with the old tree (`git archive 50f9eb1`, `conin_scaling_experiments` venv +
  smoek, highs + pytoulbar2 + pgmpy): 833 `map_query` cases (MN/BN/DBN/HMM × conin/pgmpy inputs × every family of the examples × the 5
  old methods × evidence none/dict/list/str, `variables`, `start`/`stop`, `ip_formulation`, `prune_network`/`create_MN`,
  `solution_with_evidence`, `num_solutions`, `max_iterations`, unknown options, the error cases): 542 identical (states, result keys,
  termination conditions, exception types and messages; `solvetime` ignored), 291 changed, every one by an agreed change (§11 "PR 5"
  rows): 120 HMM-only method on MN/BN/DBN (`TypeError` → registry `InvalidInputError`), 80 HMM evidence `None`/str (`None`,
  `AssertionError`, `TypeError`, `AttributeError` → `InvalidInputError`; 16 were already `InvalidInputError` with another message),
  28 viterbi on constrained HMMs (4 now run, 24 registry `InvalidInputError`), 34 toulbar2 query checks (14 partial `variables` now
  `RuntimeError`, 10 `prune_network`/`create_MN=False` now run, 10 `=True` `TypeError` → `RuntimeError`), 8 DBN `stop=None` (IP and
  toulbar2 now solve with `stop=1`), 6 a_star with recursive constraints (5 run recursive A*; 1 unknown option now raises
  recursive A*'s `TypeError`), 3 pgmpy VE with `solution_with_evidence=True` (now honoured), 2 a_star with no solution (`IndexError` →
  `solution=None`), 2 network flow on `ConstrainedHiddenMarkovModel(hmm)` (lp path: no `solution.variables`), 1 `ip_formulation`
  on an MN (now solved), 2 MN `integer_program` with `variables` and an unsupported family (the registry error now comes before the
  partial-query `RuntimeError`), 5 message texts (the unknown-method list adds `mvr_viterbi`; the not-a-model `TypeError` lists every
  model type for every method).
- **After the PR 5 follow-up** (same branch; HMM observations and query checks cleaned up, §11 "PR 5 follow-up" rows): **1600 passed,
  222 skipped, 0 errors** in `.venv-opencode` (+67 passed, +4 skipped; 1751 → 1822 collected). New:
  `graphical_models/tests/test_hmm_observations.py` (21) and `inference/tests/test_hmm_results_and_query_checks.py` (12);
  `test_map_query_dispatch.py` 123 → 161 (the HMM evidence test now uses a set instead of a tuple; new: `[]`/`{}`/sparse/badly keyed
  dicts ×5 methods, tuple evidence ×5 methods, `stop` limits the query ×3 methods × list/dict, `stop` past the last observation
  ×3 methods + ×3 kinds in `_structure_params`, VE `stop` row of `test_structure_params` changed 4 → 2 plus a default row). Changed
  expectations: `test_network_flow_rejects_unsupported_evidence` uses a sparse dict instead of a tuple (tuples are accepted now);
  `test_log_potential.py` matches the new type-error text. In the `conin_scaling_experiments` venv (+ smoek): the touched files +
  `representations/factor` + `constrained_models` + `test_log_potential.py` + `test_public_api.py`: 752 passed, 5 skipped;
  `test_MAP_Inference.py` 69 passed, 41 skipped (unchanged). Sphinx doctest/HTML clean. `test_public_api.py` unchanged (no public name).
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
│   ├── hidden_markov_model.py  # HiddenMarkovModel, random_hmm (4g: HMM_MatVecRepn removed)
│   └── conversions.py          # create_dbn_from_hmm, create_bn_from_dbn, create_mn_from_bn
├── constraints/                # simple containers: no backend logic, no model imports
│   ├── __init__.py
│   ├── base.py                 # Constraint ABC (renamed from ConstraintFunctor)
│   ├── pyomo.py                # PyomoConstraint, pyomo_constraint_fn
│   ├── toulbar2.py             # Toulbar2Constraint, toulbar2_constraint_fn
│   ├── algebraic/              # AlgebraicConstraint, algebraic_constraint_fn, bridge.py (smoek nodes)
│   ├── oracle.py               # OracleConstraint (sequence/assignment predicate), oracle_constraint_fn
│   ├── factor_oracle.py        # FactorOracleConstraint (predicate + node scope)
│   ├── recursive.py            # (PR 4c) RecursiveConstraint (formalizes the Num_Zeros protocol)
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
│   ├── feasibility.py          # (4e follow-up) is_feasible checks of total assignments
│   ├── hidden_markov_model.py  # ConstrainedHiddenMarkovModel (+ is_feasible, partial_is_feasible; 4g: generate_* removed)
│   ├── bayesian_network.py     # ConstrainedDiscreteBayesianNetwork
│   ├── markov_network.py       # ConstrainedDiscreteMarkovNetwork
│   └── dynamic_bayesian_network.py
├── representations/
│   ├── __init__.py
│   ├── base.py                 # Representation ABC, registry, build_representation()
│   ├── factor/                 # (PR 4e: done) factor_repn.py ((S,J,v,w), BN->MN skeleton, var-name map), materialise.py (FactorOracle -> factor/CPD),
│   │                           # common.py (apply_factor_constraints, FactorRepresentation), mn.py bn.py dbn.py hmm.py (Factor{MN,BN,DBN,HMM}Representation, "factor")
│   ├── pyomo/                  # (PR 4a: done)
│   │   ├── common.py           # ONE PyomoVarWrapper, apply_pyomo_constraints, IP-from-factor-repn, c5 constraints, smoek->pyomo, PyomoFactorRepresentation
│   │   ├── mn.py  bn.py  dbn.py  hmm.py   # Pyomo{MN,BN,DBN,HMM}Representation ("pyomo")
│   │   └── hmm_network_flow.py # PyomoNetworkFlowHMMRepresentation, from PyomoAlgebraic_CHMM (ip_formulation="network_flow")
│   ├── toulbar2/               # (PR 4b: done)
│   │   ├── common.py           # Toulbar2VarWrapper, apply_toulbar2_constraints, smoek->toulbar2 linear translator, Toulbar2FactorRepresentation (UAI text -> fresh CFN)
│   │   └── mn.py  bn.py  dbn.py  hmm.py   # Toulbar2{MN,BN,DBN,HMM}Representation ("toulbar2")
│   ├── mvr/                    # (PR 4d: done)
│   │   ├── hmm.py              # MVRHMMRepresentation (replaces MVR_CHMM): model + aligned MVRs; align_hidden_states
│   │   └── kernels.py          # torch kernels from mvr_common.py (static context built per query; used by inference/learning/sampling/queries)
│   ├── matvec/hmm.py           # (PR 4g: done) MatVecHMMRepresentation ("matvec"; replaces HMM_MatVecRepn / hmm.repn)
│   ├── oracle/hmm.py           # (PR 4c: done) OracleHMMRepresentation (replaces Oracle_CHMM); smoek->oracle stays in constraints/algebraic/
│   └── recursive/hmm.py        # (PR 4c: done) RecursiveHMMRepresentation
├── inference/
│   ├── __init__.py             # exports map_query
│   ├── map_query.py            # (PR 5: done) public entry point (normalise -> method -> repn kind -> ovld on repn type)
│   ├── hmm_results.py          # (PR 5 follow-up) HMM MAP result shaping; query_checks.py: the IP/toulbar2 query checks
│   ├── tests/                  # (PR 5) the cross-backend tests, incl. mn/ bn/ dbn/ (were inference/{mn,bn,dbn}/tests)
│   ├── unconstrained/          # viterbi.py (PR 4c; a_star_ stays in oracle/a_star_hmm.py, next to a_star)
│   ├── pyomo/                  # (PR 4a) solve.py (generic solve/parse); mn/bn/dbn/hmm.py (per-model post-processing); hmm_network_flow.py (lp_inference, ip_inference, from lp.py)
│   ├── toulbar2/               # (PR 4b) solve.py (solve_toulbar2_map_query_model + MN); bn/dbn/hmm.py (per-model post-processing)
│   ├── variable_elimination/   # (PR 4e) solve.py (pgmpy VE runner); mn/bn/dbn/hmm.py (inference_variable_elimination_map_query_*)
│   ├── mvr/                    # (PR 4d) viterbi.py, marginal_map.py; __init__ exports lazily (no torch import)
│   ├── oracle/                 # (PR 4c) a_star_hmm.py (a_star, a_star_)
│   └── recursive/              # (PR 4c) a_star_hmm.py (recursive_a_star)
├── learning/                   # (PR 4c) __init__ exports supervised_learning, add_unknowns, convert_to_simulations, mcem
│   ├── unconstrained/          # supervised.py, helpers.py
│   ├── oracle/                 # mcem.py
│   └── mvr/                    # (PR 4d) baum_welch.py (+ forward_backward), generalized_em.py
├── sampling/
│   ├── unconstrained/          # (PR 4g) hmm.py (generate_hidden, generate_observed, ...); (4g follow-up) dbn.py (DBNSampler)
│   ├── oracle/                 # (PR 4c) rejection_hmm.py (rejection_sample_hidden, from Oracle_CHMM.generate_hidden)
│   └── mvr/                    # (PR 4d) ffbs.py, stopped_sampling.py
├── queries/mvr/                # (PR 4d) sat_prob.py, sat_time.py (from other_queries/)
├── scoring/                    # (PR 5b) log_probability, log_potential (from common/log_potential.py); marginal/conditional log-probability query
├── examples/                   # markov_network.py, bayesian_network.py, dynamic_bayesian_network.py, hidden_markov_model.py (from */examples.py)
├── io/                         # (PR 4f) file IO: uai.py, cfn.py, pgmpy_files.py, unified load_model.py/save_model.py, convert_pgm.py CLI
├── common/                     # (PR 4f) non-IO helpers: is_polytree.py, pgmpy/ (pgmpy<->conin conversion, MapCPD); log_potential.py until PR 5b moves it to scoring/
├── mediation_variables/        # UNTOUCHED (research prototype)
└── util/                       # (PR 5) + constraint_data.py (merge_constraint_data)
```

Removed outright:
- the pgmax and pomegranate support: `to_pgmax`, `to_pomegranate`, the
  `pgmax/` and `pomegranate/` loaders (these just load with pgmpy and convert),
  the `model_type` branches in unified `load_model`, and the pyproject extras;
  (PR 4f) also the pyagrum loader, its `model_type="pyagrum"` branch and the `pyagrum`/`pgm-all` extras;
- `initialize_chmm`, `CHMM`, `Oracle_CHMM`, `Algebraic_CHMM`,
  `PyomoAlgebraic_CHMM` and `MVR_CHMM` (they become representations);
- `create_algebraic_chmm`, `Xgenerate_hidden`, and the duplicated `PyomoVarWrapper`s;
- (PR 4a, done) `Algebraic_CHMM`, `PyomoAlgebraic_CHMM`, `CallableDict`, `create_algebraic_chmm`, `Xgenerate_hidden`,
  `create_pyomo_map_query_model_{MN,BN,DDBN,HMM}` and the `add_constraints` dispatcher in `inference/mn/inference_pyomo.py`
  (replaced by `apply_pyomo_constraints`);
- (PR 4b, done) `create_toulbar2_map_query_model_{MN,BN,DDBN,HMM}`, the `add_constraints` dispatcher in
  `inference/mn/inference_toulbar2.py` (replaced by `apply_toulbar2_constraints`) and the `conin.inference.hmm` package;
- (PR 4c, done) `CHMM`, `Oracle_CHMM`, `HMMApplication` (and the `Num_Zeros` application classes in the examples and
  tests); no `applications/` package;
- (PR 4d, done) `MVR_CHMM` (`conin.hidden_markov_model.chmm_mvr`) and the whole `conin.hidden_markov_model` package;
  `ConstrainedGraphicalModel.supported_families`;
- (PR 4e, done) `_add_{bn,mn}_constraints_as_evidence`, `_run_map_query_helper`, `_execute_simple_model_query` and the other
  per-model helpers in `inference/VariableElimination.py` (replaced by `apply_factor_constraints`, the "factor" repns and
  `conin.inference.variable_elimination`).

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

| Constraint family           | pyomo | pyomo_network_flow | toulbar2 | factor (VE) | oracle (A*) | mvr | recursive | matvec |
|-----------------------------|:-----:|:------------------:|:--------:|:-----------:|:-----------:|:---:|:---------:|:------:|
| *(none)*                    |  ✓    |  ✓ (HMM)           |   ✓      |   ✓         |  ✓ (HMM)    | ✓ (HMM) | ✓ (HMM) | ✓ (HMM) |
| PyomoConstraint             |  ✓    |  ✓ (HMM)           |          |             |             |     |           |
| Toulbar2Constraint          |       |                    |   ✓      |             |             |     |           |
| AlgebraicConstraint         |  ✓    |  ✓ (HMM)           |   ✓      |   ✓ (via factor oracle) | ✓ (HMM) | |      |
| OracleConstraint            |       |                    |          |             |  ✓ (HMM)    |     |           |
| FactorOracleConstraint      |       |                    |          |   ✓         |             |     |           |
| MVR (MVRConstraint/BaseMVR) |       |                    |          |             |             | ✓ (HMM) |       |
| RecursiveConstraint         |       |                    |          |             |             |     | ✓ (HMM)   |

The builder registry (§4.3) is the single source of truth for this table. (PR 4c) Real now: oracle × {none, Oracle}
(`OracleHMMRepresentation`) and recursive × {none, Recursive}. Still open: oracle × Algebraic (not registered in 4c:
compiling algebraic constraints needs the horizon `T` and may refer to `("E", t)` nodes, which the oracle repn and
sampling don't have, and the compiled factor oracle has no partial function; revisit with PR 5).
(PR 4d) Real now: mvr × {none, MVR} (`MVRHMMRepresentation`). Since the end of PR 4 the registry also decides which families a
constrained model accepts (§4.2): a row with no ✓ for a model type is refused at construction (e.g. `OracleConstraint` and MVRs
on MN/BN/DBN).
(PR 4g) Real now: matvec × none (`MatVecHMMRepresentation`, HMM only; no constraint family by design: it is the
unconstrained HMM's parameters, used by Viterbi, A*, sampling and the MVR kernels).
Unsupported combinations raise a clear `InvalidInputError` that lists the supported options.
(PR 5) Unchanged registrations. Oracle × Algebraic stays unregistered (revisited: compiling needs `T` and `("E", t)` at build time,
i.e. a new structure param, and the compiled oracle has no partial function, so A* could not prune; nothing asks for it).
`map_query(method="a_star")` now picks the "recursive" column for recursive constraints (§4.5).

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
        self._data = data                     # passed to builders

    @property model / constraints / constraint_family / data   # read-only
    def representation(self, kind, **structure_params):    # builds a new repn on every call (no cache, after 4b)
    def with_constraints(self, constraints):               # new object
    def with_model(self, model):                           # new object (used by learning)
    @classmethod
    def ensure(cls, model):                                # (PR 4a) model if isinstance(model, cls) else cls(model)
```
- (PR 4a) `ensure(pgm)` replaces the repeated
  `pgm if isinstance(pgm, ConstrainedX) else ConstrainedX(pgm)` in the backend inference functions; use it in new
  code (4b onwards). Composition (not inheritance from the unconstrained class) is kept on purpose: a constrained
  model must not be silently accepted, with its constraints dropped, by code written for bare models (§11).
- The object is fully immutable: no `add_constraint`, no setters, no `initialize_chmm`.
  (PR 3) It uses `__slots__`, and `__setattr__`/`__delattr__` raise `AttributeError`. `copy`, `deepcopy` and
  pickle work (`__getstate__`/`__setstate__`).
- (PR 3) `constraints` is a single `Constraint` (wrapped into a 1-tuple) or an iterable (list, tuple, generator)
  of `Constraint`; a string or a non-`Constraint` element raises `InvalidInputError`, as do mixed families and
  a wrong model type.
  ~~Subclasses may set `supported_families`; `ConstrainedHiddenMarkovModel` accepts None, "oracle",
  "factor_oracle", "pyomo", "algebraic", "toulbar2", "mvr" and (PR 4c) "recursive".~~
  **(Decided after PR 3, done at the end of PR 4: PR 4d commit 3.)** `supported_families` is removed. The constructor derives
  the accepted families from the registry: a family is accepted for a model type iff
  `supported_representations(model_type)` has at least one `(kind, family)` pair for it. Then adding a constraint
  family or a model type means only registering builders. Families must not list model types, because
  support is a property of the (model type, family) pair, which the registry already records. This waits until
  4a/4b/4e have registered the HMM "pyomo", "algebraic", "toulbar2" and "factor_oracle" builders. Before that, the
  registry would reject HMMs that work today through the legacy `map_query` path.
  (PR 4d) Done as decided: `_derive_family` accepts a non-None family iff the private classmethod `_supported_families()`
  (the sorted non-None families of `supported_representations(model_type)`) contains it; the `InvalidInputError` names
  those families. Accepted now: HMM {algebraic, factor_oracle, mvr, oracle, pyomo, recursive, toulbar2} (unchanged);
  MN/BN/DBN {algebraic, factor_oracle, pyomo, toulbar2}. Newly rejected (before: any family): "oracle", "mvr" and
  "recursive" on MN/BN/DBN. Only "oracle" was used anywhere (5 tests, for `is_feasible`/`log_potential`), and those now
  use `FactorOracleConstraint`. No import cycle: `constrained_models.base` already imported `representations.base`, and
  `_load_builtins()` runs on the first constructor call with constraints.
- (PR 3) `representation()` raises `InvalidInputError` naming the supported `(kind, family)` pairs when no
  builder is registered.
- Every class has the same constructor shape:
  `ConstrainedHiddenMarkovModel(hmm, constraints=[...])`,
  `ConstrainedDiscreteBayesianNetwork(bn, constraints=[...])`, and so on.
- The thin per-model subclasses exist for three reasons:
  - the model type check;
  - as ovld/registry keys;
  - HMM-specific helpers (`generate_hidden`, `generate_observed`,
    `partial_is_feasible`), which delegate to a representation. (PR 4c) `generate_hidden(time_steps)` calls
    `conin.sampling.oracle.rejection_sample_hidden(self, time_steps)`; `generate_observed_from_hidden` and
    `partial_is_feasible` use `representation("oracle")` (internal indices).
    (PR 4g) The `generate_*` methods are removed (from `HiddenMarkovModel` too): use
    `conin.sampling.oracle.rejection_sample_hidden(cmodel, T)` and `conin.sampling.unconstrained.generate_*` (§4.8).
    `is_feasible` and `partial_is_feasible` stay.
- (PR 4c) `is_feasible` also checks `"recursive"` constraints (run `initial`/`update` along the sequence, then
  `feasible`). The oracle repn's checks share `feasibility.oracle_accepts`, so A*, rejection sampling and `is_feasible`
  call oracle predicates the same way (2-argument predicates get `data = Munch(hmm=Munch(T=list(range(T))))`; before 4c
  the oracle repn called them with one argument and raised `TypeError`).
- (4e follow-up) All four classes have `is_feasible(assignment)` (HMM: `is_feasible(seq, observed=None)`) for a
  total assignment, in `constrained_models/feasibility.py`. It evaluates the `"oracle"`, `"factor_oracle"`,
  `"algebraic"` (compiled as for VE; needs smoek) and `"mvr"` (run along the sequence) families, with the `data` the
  `"factor"` representation passes (`None`, `Munch(T)`, `Munch(hmm=Munch(T))`). `"pyomo"`/`"toulbar2"` raise
  `NotImplementedError`: their functions can add variables, so checking them is an IP/CFN feasibility problem.
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
  - `Representation` (ABC): class attribute `kind` (`get_copy()` was removed with the cache after 4b);
  - `@register_representation(kind, model_type, families=(None,))` registers `builder(cmodel, **structure_params)`
    under each `(kind, model_type, family)`; `None` means "no constraints"; re-registering a key with another
    builder raises `ValueError`; `unregister_representation(...)` is for tests;
  - `build_representation(cmodel, kind, **structure_params)` (`cmodel.representation()` calls it),
    `supported_representations(model_type)` (sorted `(kind, family)` pairs) and `get_representation_builder(...)`;
  - the built-in builder modules are imported lazily on first registry use, because they import
    `conin.hidden_markov_model`, which imports the constrained models.
- (PR 3) Registered HMM kinds: `"oracle"` (None, "oracle"), `"pyomo_network_flow"` (None, "pyomo") and
  `"mvr"` (None, "mvr"). `"algebraic"` is not registered for `"pyomo_network_flow"`: it never worked before
  PR 3 (initialize_chmm built no `.chmm` for it).
- (PR 4a) Registered: `"pyomo"` for MN, BN, DBN and HMM (None, "pyomo", "algebraic"); `"pyomo_network_flow"` for HMM
  now also accepts "algebraic" (smoek->pyomo on the network-flow `M.V`; tested). The Pyomo representations:
  - `PyomoFactorRepresentation` (base; `representations/pyomo/common.py`) holds `S, J, v, w`, `var_index_map`, `pgm`
    (the model the constraint applier sees: MN, BN, the unrolled BN for DBNs, the DBN for HMMs), `constraints`, `nodes`.
    `create_model(*, evidence=None, variables=None, omit_c5=False, timing=False)` builds a fresh `ConcreteModel`,
    fixes `model.V(k, State(v))` for the evidence, then applies the constraints with `constraint_data(evidence=,
    variables=)`, which is today's layout: MN/BN `Munch(variables, evidence)`; DBN `+start, stop, T`; HMM
    `Munch(start, stop, variables, evidence, hmm=Munch(T, hidden_to_internal))`. HMM evidence is keyed `("E", t)`.
    The constrained model's own `data` is **not** passed on these paths (unchanged from before 4a). **PR 5: it is (§9): the repns keep
    `cmodel.data` as `.data`, and `create_model` passes `merge_constraint_data(self.data, self.constraint_data(...))`.**
  - Structure params: MN, BN `()`; DBN `start=0, stop=1`; HMM `start=0, stop` (required).
    `omit_c5` (`use_cut_generation`), `variables`, `prune_network` and `create_MN` are **not** structure params.
  - The builders build on each other: HMM → `create_dbn_from_hmm` → DBN builder (`create_bn_from_dbn`) → BN builder
    (`skeleton_mn_of_bn`: one factor per CPD) → MN builder (`factor_representation_of_mn`).
  - `PyomoNetworkFlowHMMRepresentation` holds the HMM, constraints, application `data`, and the structure params
    `x_binary=True`, `y_binary=False`. `create_model(*, observed, Tmax=None)` returns `(model, data)`, where `data` is a
    shallow copy of the application data with `.hmm` = the index sets. `M.V` is the shared `PyomoVarWrapper`
    (keys `(("H", t), State(h))`).
- (PR 4b) Registered: `"toulbar2"` for MN, BN, DBN and HMM (None, "toulbar2", "algebraic"). The toulbar2 representations:
  - `Toulbar2FactorRepresentation` (base; `representations/toulbar2/common.py`) holds `uai_text` (the UAI text of
    `uai_model`, byte-identical to `conin.io.save_model`'s output), `uai_model` (the MN, the BN, or the unrolled BN for
    DBNs/HMMs; its `nodes`/`states_of` order fixes the CFN variable and value indices), `pgm` (the model the constraint
    applier sees: MN, BN, the unrolled BN for DBNs, the DBN for HMMs, as before), `constraints`, `nodes`.
    `create_model(*, evidence=None, variables=None, verbose=-1, write_uai_file=None, timing=False)` writes the text to a
    temp file (or exactly to `write_uai_file`), reads it into a fresh `pytoulbar2.CFN(verbose=)`, attaches a fresh
    `Toulbar2VarWrapper` as `M.V`, `M.states` and `M.V_evidence`, `Assign`s the evidence
    (`uai_model.states_of(k).index(v)`), then applies the constraints with `constraint_data(evidence=, variables=)`: the same
    layout as the Pyomo repns (MN/BN `Munch(variables, evidence)`; DBN `+start, stop, T`; HMM
    `Munch(start, stop, variables, evidence, hmm=Munch(T, hidden_to_internal))`; HMM evidence keyed `("E", t)`). The
    constrained model's own `data` is **not** passed (unchanged; §9 row for PR 5). **PR 5: passed, as for "pyomo".**
  - Structure params: MN, BN `()`; DBN `start=0, stop=1`; HMM `start=0, stop` (required). `verbose`,
    `write_uai_file`, `variables` and the solve options are per-query.
  - The builders build on each other: HMM → `create_dbn_from_hmm` → `build_toulbar2_dbn_representation`
    (`create_bn_from_dbn`) → `build_toulbar2_bn_representation`. The MN builder uses the MN directly.
  - The toulbar2 paths take no `prune_network`/`create_MN` (they reach `solve_toulbar2_map_query_model` and raise
    `TypeError`, as before), so 4a's `_validate_bn_query` is not shared. **PR 5: shared (`conin/inference/query_checks.py` since the
    PR 5 follow-up), §9.**
- (PR 4e) Registered: `"factor"` for MN, BN, DBN and HMM (None, "factor_oracle", "algebraic"). VE keeps running through
  pgmpy (§11). The factor representations:
  - `FactorRepresentation` (base; `representations/factor/common.py`) holds `pgm` (the model the constraint applier sees:
    the MN, the BN, or the unrolled BN of a DBN/HMM), `constraints`, `nodes` (of `pgm`), `ve_model` and
    `constraint_evidence`. The constraints are materialised **once, at build time**, by `apply_factor_constraints(pgm=,
    constraints=, data=)` (the single applier; "algebraic" goes through `create_oracle_constraints_from_algebraic` first):
    MN: one factor per constraint (`0` on infeasible assignments), no evidence; BN: one CPD of a binary node
    `(name, -1)` per constraint, with `constraint_evidence = {(name, -1): 1}`. This is possible because the VE constraint
    `data` holds no evidence. It is today's layout, unchanged: MN/BN `None`; DBN `Munch(T)`; HMM `Munch(hmm=Munch(T))`, with
    `T = list(range(start, stop + 1))` (the old HMM path used `range(len(evidence))`, the same since `start=0`, `stop =
    len(evidence) - 1`). Neither `variables`, the evidence nor `cmodel.data` is passed (§9 row for PR 5). **PR 5: `cmodel.data` is
    passed (the fields are set on a copy; MN/BN see the data itself); `variables` and the evidence still are not.**
  - Per query: `query_evidence(evidence)` returns a new dict (user evidence + `constraint_evidence`), and
    `create_model(*, write_uai_file=None)` converts `ve_model` into a fresh pgmpy model (`convert_conin_to_pgmpy_{bn,mn}`;
    `write_uai_file` saves `ve_model` with `conin.io.save_model`, as before). `variables`, `show_progress`, `timing` and
    `solution_with_evidence` are query-function options. Building the repn needs no pgmpy.
  - Structure params: MN, BN `()`; DBN `start=0, stop=1`; HMM `start=0, stop` (required), as for "pyomo"/"toulbar2".
  - The builders build on each other: HMM → `create_dbn_from_hmm` → `build_factor_dbn_representation`
    (`create_bn_from_dbn`) → `FactorRepresentation`.
  - `conin.inference.variable_elimination`: `inference_variable_elimination_map_query_{MN,BN,DDBN,HMM}(*, pgm, [start, stop,]
    variables=None, evidence=None, show_progress=False, timing=False, write_uai_file=None, [solution_with_evidence=False,]
    **options)` (`ensure` → `representation("factor", ...)` → `create_model` → `solve_variable_elimination_map_query_model`).
    The `VariableElimination.py` dispatcher calls them; its signature and the `map_query` behaviour are unchanged (the HMM
    overloads pass `stop=None`, because the old HMM path ignored `stop`; the pgmpy DBN overload converts, then calls the DDBN
    function).

- (PR 4c) Registered: `"oracle"` for HMM (None, "oracle"; unchanged pairs, a real class now) and `"recursive"` for HMM
  (None, "recursive"). No structure params (the horizon is per call). The representations:
  - `OracleHMMRepresentation` (`representations/oracle/hmm.py`) holds `model` (the HMM), `hmm` (`model.repn`, the
    `HMM_MatVecRepn` used by A* and sampling; **4g: a `MatVecHMMRepresentation`**) and `constraints` (the constrained model's tuple, not wrapped or copied).
    `is_feasible(seq)` and `partial_is_feasible(*, T, seq)` take internal indices (list or tuple) and call the user's
    predicates on `{t: external label}`, as `Oracle_CHMM`'s wrapped constraints did. The `.chmm` attribute and
    `generate_hidden` are gone. Build cost: 0.8 ms here, so rebuilding it per call is fine (no cache).
  - `RecursiveHMMRepresentation` (`representations/recursive/hmm.py`) holds `model` and `constraints` and combines them:
    `initial(h)`, `update(state, h)`, `feasible(state)`, `feasible_partial(state, t, T)` on the tuple of per-constraint
    states (`()` without constraints).
  - Neither holds per-query data, and their methods don't mutate them, so one repn serves several searches.
  - `conin.inference.oracle.a_star(*, observed, hmm, **kwargs)` and `recursive_a_star(*, hmm, observed, ...)` normalise
    with `ConstrainedHiddenMarkovModel.ensure` and call `representation("oracle")` / `representation("recursive")`;
    other families raise the registry's `InvalidInputError`. `a_star_` takes `oracle=` (was `chmm=`).
  - `map_query(method="a_star")` is unchanged: it does not route recursive constraints to `recursive_a_star` (PR 5). **PR 5: it does.**
  - (PR 5) `OracleHMMRepresentation.data` is `cmodel.data`; 2-argument predicates get `merge_constraint_data(data, Munch(hmm=Munch(T)))`
    (A*, rejection sampling and `is_feasible` agree).
- (PR 4d) `"mvr"` for HMM (None, "mvr"; unchanged pairs, a real class now). No structure params. The representation:
  - `MVRHMMRepresentation` (`representations/mvr/hmm.py`) holds `model` (the HMM) and `mvrs` (a tuple of `BaseMVR`, one per
    constraint, in order; `()` without constraints). Build: `MVRConstraint` factories are called with `(hmm, cmodel.data)`;
    prebuilt MVRs are deep-copied; a non-MVR result or hidden states that differ from the HMM's raise `InvalidInputError`;
    then `align_hidden_states(mvr, hmm.hidden_to_external)` permutes the hidden axis of the array representation (the MVR
    itself if already aligned). `.chmm` and `.hmm` are gone. Build cost here: about 2.5 ms for two prebuilt MVRs, 6 ms for a
    factory.
  - The static context (`_build_static_context`: per-MVR torch tensors, active sets, shapes, destination cache) depends on
    `T`, `dtype`, `device` and the semiring, which are per-query, and needs torch, so it is **not** in the repn: the kernels
    build it from `repn.mvrs` per query (§11 "MVR static context").
  - `kernels._model_parts(model)` is `ensure(model).representation("mvr")` → `(repn.model, list(repn.mvrs))`;
    `_build_sumprod_ctx(hmm, constraints, observed, ...)` takes the HMM and the aligned MVRs (was the `MVR_CHMM`).
  - The algorithms (`viterbi_torch_mvr_chmm`, `marginal_map_torch_mvr_chmm`, `forward_backward_mvr_chmm`, `baum_welch_mvr_chmm`,
    `generalized_em_mvr_chmm`, `ffbs_torch_mvr_chmm`, `stopped_sampling_torch_mvr_chmm`, `sat_prob_torch_mvr_chmm`,
    `sat_time_torch_mvr_chmm`) keep their names and signatures; `model` is a `ConstrainedHiddenMarkovModel` (or a bare HMM,
    via `ensure`). Other families raise the registry's `InvalidInputError`.
  - Learning (Baum-Welch, GEM) builds the repn **once per call** and passes the re-estimated HMM with the same MVRs to the
    kernels (the rebuilt HMMs keep the hidden-state order, so the alignment stays valid); no `deepcopy`, no
    `with_model` inside the loop (§11).
  - Stopped sampling: `ensure(model)`, the MVRs from the repn, then `model.with_constraints(...)` with the prefix-free target
    and with the target narrowed to `[a, tau]`. The window copies use a private `_with_time_range(mvr, range)` (a shallow
    copy that shares the read-only arrays), not `mvr_timerange(..., inplace=False)`, because that rebuilds and prunes the
    MVR, which would reorder the mediation axes and change the sampling draws.
  - Read-only arrays: `MVR_MatVecRepn` stores read-only copies of `ini_array`, `upd_array`, `evl_array`, `ini_idx`,
    `next_idx` (and re-freezes them in `__setstate__`, since `deepcopy`/pickle give writeable arrays); the kernels copy them
    (`np.array`) before `torch.as_tensor`.
- (PR 4g) `"matvec"` for HMM (None only). No structure params. The representation:
  - `MatVecHMMRepresentation(model)` (`representations/matvec/hmm.py`) holds `model`, `num_hidden_states`,
    `num_observed_states` and six read-only `float64` arrays on internal indices: `start_vec` (n,), `transition_mat` (n, n),
    `emission_mat` (n, m) and `log_start_vec`, `log_transition_mat`, `log_emission_mat` (`np.log`, `-inf` at zeros). They are
    copies of the model's lists (re-frozen in `__setstate__` after deepcopy/pickle). No `hidden_states`/`observed_states`
    (use `range(n)`; the model's properties are the external labels) and no sampling methods.
  - Built per call (`ensure(hmm).representation("matvec")`; no cache, §4.4); nothing in it changes after the build, so one repn
    serves several queries. `OracleHMMRepresentation.hmm` is one (built from `cmodel.model` directly, since the oracle repn's
    constrained model has constraints).
  - Users: `viterbi` (`ensure` → `representation("matvec")`; `viterbi_(observed, hmm=repn)` on internal indices, reading the
    log arrays), `a_star` / `a_star_` (via `oracle.hmm`; the arrays are read once into lists before the search loops),
    `conin.sampling.unconstrained.generate_*`, `rejection_sample_hidden` (through `oracle.hmm`) and
    `kernels._hmm_to_torch` (the plain or log arrays, copied before `torch.as_tensor`). Recursive A* still reads
    `hmm.get_*_probs()` (external labels, read once; no copy cost): unchanged.
  - Constraints are refused (the registry's `InvalidInputError`), so `viterbi` and the unconstrained samplers never drop
    constraints silently.
- (PR 4c) `RecursiveConstraint(*, initial, update, feasible, feasible_partial=None, name=None)` (`conin.constraints`),
  family `"recursive"`: `initial(h) -> state`, `update(state, h) -> state`, `feasible(state) -> bool`,
  `feasible_partial(state, t, T) -> bool` (`t` = prefix length; default always `True`; it must never reject a
  completable prefix). `h` is an external label; states must be hashable. No `data` argument. Several constraints
  combine into a tuple state; feasible iff all are.

### 4.4 Caching (removed after PR 4b)
- **Representations are not cached.** `cmodel.representation(kind, **structure_params)` calls the
  builder on every call and returns the new object; the caller owns it. There is no `get_copy()`.
- PR 3 added a per-object cache keyed by `(kind, structure_params)` with copy-on-get (`get_copy()`, default
  `deepcopy`, overridden per class), and 4a/4b relied on it. It was removed after 4b because nothing reused it:
  `map_query` on a bare model wraps it anew per call (`ensure`), learning calls `with_model` every iteration, and no
  code or notebook queried one constrained object repeatedly. Before PR 3 everything was rebuilt per query, so this
  is not a regression. The plain rule ("every call builds a new repn") removes the reasoning about mutation after
  build, evidence leaking into a cached object and which parameters are cache keys.
- What stays from 4a/4b: **structure params** (`start, stop`, `x_binary, y_binary`, ...) are build arguments;
  **per-query** inputs (evidence, `variables`, `omit_c5`, `verbose`, `write_uai_file`) are `create_model()` options,
  and the Pyomo/toulbar2/network-flow/factor repns are not mutated by `create_model()` (or, for "factor", `query_evidence()`), so one repn can serve several queries.
  If caching is needed later, it can be added inside `representation()` without an API change.
- The MVR repn deep-copies prebuilt MVR constraints at build time, so the user's MVR objects are never changed through
  a repn. (PR 4d: kept after `mvr_timerange` became non-mutating, since MVRs still have a `name` setter and
  `mvr_timerange(..., inplace=True)`; §11.)

### 4.5 Inference dispatch (keeps ovld + `map_query`)
```
map_query(pgm, method=..., evidence=..., ...)
  1. normalize: pgmpy -> conin; bare GraphicalModel -> Constrained*(gm, ())
  2. method -> representation kind:
       integer_program      -> "pyomo" | "pyomo_network_flow" (ip_formulation)
       toulbar2             -> "toulbar2"
       a_star               -> "oracle" | "recursive" (picked by constraint family)
       viterbi              -> "matvec" (4g; unconstrained only, §4.8)
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

(PR 5, done) The final design (`conin/inference/map_query.py`):
1. **Normalise** (`_normalise`): a `ConstrainedGraphicalModel` is used as it is; a pgmpy `DiscreteMarkovNetwork` /
   `DiscreteBayesianNetwork` / `DynamicBayesianNetwork` is converted with `convert_pgmpy_to_conin` (also for VE: the native pgmpy VE
   path is gone); a bare model is wrapped as `Constrained*(model)`. Anything else raises `TypeError("Unsupported model type: ...
   Expected one of: <the 4 model types, the 4 constrained classes, the 3 pgmpy classes>")`. For HMMs, the evidence must be a list,
   tuple or dict keyed by exactly `0..T-1` (`_hmm_observations` → `graphical_models.hmm_observations.as_sequence`,
   `InvalidInputError`; `mvr_viterbi` only checks the type, it reads a sparse `{t: label}` dict itself) before anything is built;
   the dense list is passed on as `query.observed`.
2. **Method → kind** (`_representation_kind`, a small function; `METHODS` lists the six methods, an unknown one raises `ValueError`):

   | method | kind | notes |
   |---|---|---|
   | `integer_program` | `"pyomo"` / `"pyomo_network_flow"` | pops `ip_formulation` for every model type: `None`/`"markov_network"` → pyomo, `"network_flow"` → network flow, else `ValueError` |
   | `toulbar2` | `"toulbar2"` | |
   | `variable_elimination` | `"factor"` | |
   | `a_star` | `"recursive"` if `constraint_family == "recursive"`, else `"oracle"` | |
   | `viterbi` | `"matvec"` | constraints → registry `InvalidInputError` |
   | `mvr_viterbi` | `"mvr"` | new |
3. **Structure params** (`_structure_params`): "pyomo"/"toulbar2"/"factor" on a DBN: `start`, `stop` (`None` → 1, every method); on
   an HMM: `start`, `stop` (`time_range`: `None` → `len(evidence) - 1`, a `stop` past the last observation → `InvalidInputError`;
   since the PR 5 follow-up "factor" honours `stop` too); no
   other kind has structure params here (`x_binary`/`y_binary` keep their defaults, `Tmax` is not used). Then
   `repn = cmodel.representation(kind, **params)`; unsupported (model type, family) pairs raise the registry's `InvalidInputError`
   (also for HMM-only methods on MN/BN/DBN).
4. **Dispatch**: `_solve_map_query(repn, query)` (`query` = Munch of variables, evidence, observed (the dense HMM observations, else
   `None`), show_progress, timing, start, stop, options), **one ovld overload per representation class, 17**: `Pyomo{MN,BN,DBN,HMM}Representation`,
   `PyomoNetworkFlowHMMRepresentation`, `Toulbar2{MN,BN,DBN,HMM}Representation`, `Factor{MN,BN,DBN,HMM}Representation`,
   `MatVecHMMRepresentation`, `OracleHMMRepresentation`, `RecursiveHMMRepresentation`, `MVRHMMRepresentation`. No fallback overload
   (the normalisation and the registry raise first).
- Each overload calls a **private repn-level core** in the backend module; the public function (`inference_pyomo_map_query_*`,
  `inference_toulbar2_map_query_*`, `inference_variable_elimination_map_query_*`, `lp_inference`, `ip_inference`, `viterbi`, `a_star`,
  `recursive_a_star`, `viterbi_torch_mvr_chmm`) is `ensure` + `representation(...)` + the same core (one implementation; public
  signatures unchanged). Per-option handling is unchanged: the pyomo cores pop `use_cut_generation` (and BN/DBN/HMM
  `prune_network`/`create_MN` via `reject_pruning_options`; partial `variables` via `full_query_variables`) and forward the rest to `solve_pyomo_map_query_model` (unknown options →
  `TypeError`); toulbar2 pops `verbose`, `write_uai_file` (+ the same checks) and forwards to `solve_toulbar2_map_query_model`; VE
  reads `write_uai_file`/`solution_with_evidence` and ignores the rest; viterbi ignores every option; a_star/recursive A* forward them
  (unknown → `TypeError`); mvr_viterbi forwards them to the kernel (`dtype`, `device`).
- Network flow (`_map_query_network_flow`): `_lp_inference` when the repn has no constraints, `_ip_inference` otherwise (the same rule
  in `inference_pyomo_map_query_HMM`); dict evidence is densified and the solutions keyed by time.
- **Returns per method** (unchanged except as listed in §5): integer_program/toulbar2/VE `solution` (`states`), `solutions`,
  `termination_condition` (not VE), `solvetime`; HMM IP/toulbar2/VE `states` a list (list evidence) or dict by time; network flow,
  viterbi, a_star: `observed`, `solution` (`states`, `hidden`, `log_likelihood`, + `variables` for ip), `solutions`,
  `termination_condition` (+ `solvetime` for network flow); a_star with recursive constraints: the same shape (`states` = `hidden`),
  plus `knowledge_state_times`, `objective_values`; no solution: `solution=None`, `solutions=[]`. **mvr_viterbi**:
  `Munch(observed, solution, solutions=[solution], termination_condition="ok")`, `solution = Munch(states, hidden, log_likelihood,
  augmented)` (`return_score=True`, `return_augmented=True`; float32 by default, `dtype=`/`device=` forwarded); list evidence is dense;
  dict evidence is sparse `{t: label}` and the states come back keyed by time; `time_horizon = stop + 1`, else `len(list)` or
  `max(dict) + 1`; `start != 0` and an empty dict without `stop` raise `InvalidInputError`; an infeasible lattice raises the kernel's
  `InvalidInputError`. torch is imported on the first `mvr_viterbi` call (`import conin` does not import it).
- Import layering: `map_query.py` imports the backend cores at the top (no cycle; nothing imports `map_query.py` but
  `conin.inference`); the only function-level import is `conin.inference.mvr.viterbi` (torch). `hmm_results.py` and
  `query_checks.py` import nothing; `graphical_models/hmm_observations.py` imports only `conin.exceptions`, so every layer above
  `graphical_models` (constrained_models, inference, scoring) can use it.

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
- (PR 4d, done) Differences from the above: `MVRRepresentation` is named `MVRHMMRepresentation` (as the oracle/recursive
  HMM repns); it does **not** hold the static context (per query, built by the kernels from `repn.mvrs`); learning builds
  the repn once per call instead of `with_model` per iteration (same MVRs, same results); stopped sampling uses
  `with_constraints` plus a private shallow window copy (not the rebuilding `mvr_timerange`, see §4.3). Notebooks use
  `ConstrainedHiddenMarkovModel(hmm, m)` (no brackets for one MVR), `ConstrainedHiddenMarkovModel(hmm)` for no constraints,
  `cmodel.model` instead of `.hidden_markov_model`, and the package imports (`from conin.inference.mvr import
  viterbi_torch_mvr_chmm`). No notebook used `chmm.chmm.constraints`.

### 4.7 Recursive A*
- `RecursiveConstraint` packages the four callbacks `Num_Zeros` uses today:
  - `initial(h)`,
  - `update(state, h)`,
  - `feasible_partial(state, t, T)`,
  - `feasible(state)`.
- `recursive_a_star` takes a `ConstrainedHiddenMarkovModel` with these constraints.
  `HMMApplication` no longer has to implement this protocol.
- Multiple `RecursiveConstraint`s combine automatically; the combined state is a tuple of the per-constraint states.
- (PR 4c, done) Final names: `feasible_partial(state, t, T)` with `t` the length of the prefix (the search checks the
  extended prefix: `feasible_partial(update(state, h), t + 1, T)`); `recursive_a_star(*, hmm, observed, num_solutions=1,
  debug=False, max_iterations=None, max_time=None)`; `HMMApplication` was deleted instead of decoupled (§11).
  Example: `conin.examples.hidden_markov_model.create_chmm1_recursive(lb=10, ub=12)`.

---

### 4.8 The unconstrained HMM representation (PR 4g)

`HiddenMarkovModel` stores `_start_vec`, `_transition_mat` and `_emission_mat` on internal indices.
`HMM_MatVecRepn` holds a validated copy of the same lists plus sampling methods, built lazily and cached on
the model (`initialize()`, `avoid_reinitialization`, `initialize=True`). That machinery dates from mutable
HMMs. Problems with it:
- It duplicates the model's data, and the same property name means different things:
  `HMM_MatVecRepn.hidden_states` is `range(n)`, `HiddenMarkovModel.hidden_states` is the external labels.
- `viterbi` and `a_star` accept an HMM, a constrained HMM or a raw `HMM_MatVecRepn`, and branch on
  `isinstance(hmm, HMM_MatVecRepn)`. Every other backend takes a representation.
- On the immutable branch, `transition_mat` / `emission_mat` return copies, and A* reads
  `hmm.transition_mat[h1][h2]` inside its loops, so each access copies the whole matrix (not yet timed).
- Viterbi is the only method in §4.5 with no representation kind ("unconstrained only").

Target (PR 4g, §6):
- A registered representation of the bare HMM (`HiddenMarkovModel`, family `None` only), built by
  `ConstrainedHiddenMarkovModel(hmm, ()).representation(<kind>)` like any other repn (bare models are
  already wrapped this way, §4.5). It holds read-only numpy arrays on internal indices, probably plus the
  log arrays, and is rebuilt per call like the other repns (§4.4).
- Viterbi, A*, recursive A*, rejection sampling and MVR's `_hmm_to_torch` read it. The oracle and
  recursive repns hold one instead of `model.repn`.
- `HMM_MatVecRepn`, `hmm.repn`, `initialize()` / `initialize=True` and the `isinstance` input branches
  are removed (no shims; the MVR notebooks use `hmm.repn` and `initialize=True`, and
  `HMM_MatVecRepn` is exported from `conin.graphical_models`, so §5 and the notebooks change too).
- Layering: `graphical_models` must not import `representations`, so `HiddenMarkovModel.generate_*`
  either samples from the model's own private arrays or moves to `sampling/`.
- It is a representation of the model alone, not of a (model, constraints) pair; that is a small
  stretch of the §4.3 definition, accepted for the uniform dispatch.

(PR 4g, done) Settled (§11 "PR 4g" rows): kind `"matvec"`, class `MatVecHMMRepresentation` with plain **and** log
arrays (§4.3); the `generate_*` samplers move to `conin.sampling.unconstrained` as functions (the methods are removed from
`HiddenMarkovModel` and `ConstrainedHiddenMarkovModel`); `hmm.repn`, `initialize()` and `initialize=` are removed outright
(no convenience); the probability checks moved from `HMM_MatVecRepn` into the `HiddenMarkovModel` constructor; the A*
matrix-copy cost was timed first (below). Recursive A* and the recursive repn are unchanged (they read
`get_*_probs()` once, so they never had the copy cost); the oracle repn holds a matvec repn in `.hmm`. The `HMM_MatVecRepn`
`map_query` overloads are removed with no replacement overload (repn-type dispatch is PR 5).
Timing of the copying properties in A* (best of 3, this emulated CPU; same results): 5 states T=30: 0.018 s with the old
copying reads → 0.008 s with list reads; 10 states T=30: 0.098 → 0.022 s; 20 states T=30, `num_solutions=5`: 0.77 → 0.097 s.

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
from conin.constraints import RecursiveConstraint
from conin.inference.unconstrained import viterbi
from conin.inference.oracle import a_star
from conin.inference.recursive import recursive_a_star
from conin.sampling.oracle import rejection_sample_hidden
from conin.sampling.unconstrained import generate_hidden, generate_hidden_until_state, generate_observed, generate_observed_from_hidden
from conin.representations.matvec import MatVecHMMRepresentation
from conin.representations.oracle import OracleHMMRepresentation
from conin.representations.recursive import RecursiveHMMRepresentation
from conin.sampling.mvr import ffbs_torch_mvr_chmm, stopped_sampling_torch_mvr_chmm
from conin.queries.mvr import sat_prob_torch_mvr_chmm, sat_time_torch_mvr_chmm
from conin.representations.mvr import MVRHMMRepresentation
from conin.examples.hidden_markov_model import create_hmm1, create_chmm1_pyomo
from conin.io import load_model, save_model
from conin.inference.pyomo import inference_pyomo_map_query_MN, ..., lp_inference, ip_inference, solve_pyomo_map_query_model
from conin.representations.pyomo import PyomoVarWrapper, PyomoNetworkFlowHMMRepresentation, ...
from conin.inference.toulbar2 import inference_toulbar2_map_query_MN, ..., solve_toulbar2_map_query_model
from conin.representations.toulbar2 import Toulbar2VarWrapper, Toulbar2MNRepresentation, ...
from conin.inference.variable_elimination import inference_variable_elimination_map_query_MN, ..., solve_variable_elimination_map_query_model
from conin.representations.factor import FactorRepresentation, FactorMNRepresentation, ..., apply_factor_constraints
from conin.common import log_potential, log_probability, is_polytree, convert_pgmpy_to_conin
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
- (PR 4f) pgmax, pomegranate and pyagrum support is removed: `convert_pgmpy_to_pgmax`, `convert_pgmpy_to_pomegranate`,
  `load_model(..., model_type="pgmax" | "pomegranate" | "pyagrum")` (now `RuntimeError: Unexpected model type`), and the
  `pgmax`, `pomegranate`, `pyagrum` and `pgm-all` extras.
- (PR 4f) File IO moves from `conin.common` to `conin.io`: `conin.common.load_model`/`save_model` →
  `conin.io.load_model`/`save_model` (`model_type` is `"conin"` or `"pgmpy"`); `conin.common.unified` is removed;
  `conin.common.conin.load_uai.load_conin_model_from_uai` → `conin.io.uai` (also `conin.io`);
  `conin.common.conin.load_cfn.load_conin_model_from_cfn` → `conin.io.cfn` (also `conin.io`);
  `conin.common.conin.save_model.save_model_uai` → `conin.io.uai` (also `conin.io`);
  `conin.common.conin.load_model`/`save_model` → `conin.io.load_model.load_conin_model` / `conin.io.save_model.save_conin_model`;
  `conin.common.pgmpy.load_model`/`save_model` → `conin.io.pgmpy_files.load_model`/`save_model`.
  The `convert_pgm` console script now points at `conin.io.convert_pgm:main` (same name and options).
- (PR 4f) `conin.common.conin` is removed: `convert_conin_to_pgmpy_bn`/`_mn` → `conin.common.pgmpy` (also `conin.common`);
  `is_polytree` → `conin.common.is_polytree` (also `conin.common`); the CONIN-native `log_potential` →
  `conin.common.log_potential.log_potential_conin`. `conin.common.log_potential` is now the unified dispatcher (it also
  accepts pgmpy models); for an unsupported model type it raises `TypeError` (the native function raised `RuntimeError`).
  `conin.common` also exports `convert_pgmpy_to_conin`, `convert_conin_to_pgmpy_bn` and `convert_conin_to_pgmpy_mn`.
- (PR 4a) The Pyomo inference code moves to `conin.inference.pyomo`: `inference_pyomo_map_query_{MN,BN,DDBN,HMM}`
  (were `conin.inference.{mn,bn,dbn,hmm}[.inference_pyomo]`), `solve_pyomo_map_query_model` (was
  `conin.inference.mn.inference_pyomo`), `lp_inference`/`ip_inference` (were `conin.hidden_markov_model[.inference]`
  and `conin.hidden_markov_model.inference.lp`). Names and signatures are unchanged.
- (PR 4a) `create_pyomo_map_query_model_{MN,BN,DDBN,HMM}` are removed. Use
  `Constrained*(model, constraints).representation("pyomo", [start=, stop=]).create_model(evidence=..., omit_c5=...)`.
  `create_MN_pyomo_map_query_model_from_factorial_repn`, `PyomoVarWrapper`, `add_c5_constraints`, `has_omitted_c5`,
  `C5LazyCutGenerator` and `add_algebraic_constraints_to_pyomo_model` move to `conin.representations.pyomo.common`
  (the last one was `conin.constraints.algebraic[.add_constraints_pyomo]`).
- (PR 4a, additive) `ConstrainedGraphicalModel.ensure(model)` (classmethod): returns `model` if it is already an
  instance of the class, else wraps it with no constraints.
- (PR 4a) `PyomoAlgebraic_CHMM`, `Algebraic_CHMM`, `CallableDict` and `create_algebraic_chmm` are removed (no longer
  exported by `conin.hidden_markov_model`); `representation("pyomo_network_flow")` is a
  `PyomoNetworkFlowHMMRepresentation`: `generate_model`/`generate_unconstrained_model` → `create_model(observed=)`,
  which returns `(model, data)`; `.chmm` and `.data` are gone.
- (PR 4a) On network-flow models, `M.V` is a `PyomoVarWrapper`: a missing variable raises `KeyError` (it returned
  `None`), a wrong number of arguments raises `ValueError` (it returned a `KeyError`), and direct indexing uses
  `M.V[("H", t), State(h)]`.
- (PR 4a) `map_query(hmm, method="integer_program", ip_formulation="network_flow")` raises `InvalidInputError` for
  evidence that is not a list or dict (it returned `None`). `prune_network=False`/`create_MN=False` are accepted
  (they raised `TypeError` from the solver).
- (PR 4b) The toulbar2 inference code moves to `conin.inference.toulbar2`: `inference_toulbar2_map_query_{MN,BN,DDBN,HMM}`
  (were `conin.inference.{mn,bn,dbn,hmm}[.inference_toulbar2]`) and `solve_toulbar2_map_query_model` (was
  `conin.inference.mn.inference_toulbar2`). Names and signatures are unchanged. `conin.inference.hmm` is removed, and
  `conin.inference.{mn,bn,dbn}` no longer export anything.
- (PR 4b) `create_toulbar2_map_query_model_{MN,BN,DDBN,HMM}` are removed. Use
  `Constrained*(model, constraints).representation("toulbar2", [start=, stop=]).create_model(evidence=..., verbose=...)`.
  `Toulbar2VarWrapper` (was `conin.inference.mn.inference_toulbar2`) moves to `conin.representations.toulbar2`;
  `add_algebraic_constraints_to_toulbar2_model` and `add_toulbar2_constraints` move to
  `conin.representations.toulbar2.common` (were `conin.constraints.algebraic[.add_constraints_toulbar2]`;
  `conin.constraints.algebraic` no longer exports `add_algebraic_constraints_to_toulbar2_model`).
- (PR 4b) `map_query(..., method="toulbar2", verbose=...)` works for every model type (it raised `TypeError` from
  `solve_toulbar2_map_query_model`). The DBN/HMM toulbar2 paths return the `"pytoulbar2 not available"` result when
  pytoulbar2 is missing (they raised `NameError`).
- (PR 4b) `write_uai_file=` writes exactly that file (it used `save_model`, which writes `name.split(".")[0] + ".uai"`, so a
  path with a dot in a directory name wrote elsewhere and the read failed, and a non-`.uai` name raised `RuntimeError`).
- (PR 4e) The variable-elimination code moves to `conin.inference.variable_elimination`: new public
  `inference_variable_elimination_map_query_{MN,BN,DDBN,HMM}` and `solve_variable_elimination_map_query_model`;
  `conin.common.pgmpy.custom_VE` is removed (see the 4e follow-up below).
  The private `_add_{bn,mn}_constraints_as_evidence` helpers of `conin.inference.VariableElimination` are removed; use
  `conin.representations.factor.apply_factor_constraints(pgm=, constraints=, data=)`, which returns `(new_pgm, evidence)`.
- (PR 4e) `conin.inference.mn.factor_repn` → `conin.representations.factor.factor_repn`. `factor_representation_of_mn`,
  `skeleton_mn_of_bn` and `tuple_var_index_map` move there from `conin.representations.pyomo.common`.
- (PR 4e, additive) `conin.representations.factor`: `FactorRepresentation`, `Factor{MN,BN,DBN,HMM}Representation`,
  `apply_factor_constraints`; `cmodel.representation("factor", [start=, stop=])`.
- (PR 4e) `map_query(..., method="variable_elimination")` on a constrained Markov network keeps the user `evidence` (it was
  replaced by `{}`) and no longer fixes the last node of each constraint's scope to the state `1` as evidence (so constraints
  that needed that node ≠ 1 were infeasible, and a partial query returned that node too). Results change only where these bugs
  applied.
- (PR 4e) `map_query(..., method="variable_elimination")` with a constraint family VE does not support ("pyomo",
  "toulbar2", "oracle", "mvr") raises `InvalidInputError` from the registry (it raised `TypeError`).
- (4e follow-up) Removed as dead, untested code: `conin.common.pgmpy.log_potential` (the module: `log_potential`,
  `log_probability`, `log_prob_v2`, `set_backend_`, `Log`) and `conin.common.pgmpy.custom_VE` (`logprob_VE`). Nothing used
  them; `log_prob_v2` always raised `TypeError` with current pgmpy and `logprob_VE.logprobs` crashed without evidence.
  `conin.common.log_potential` now accepts pgmpy models by converting them to conin.
- (4e follow-up) `conin.common.log_potential(model, assignment, *, observed=None)`: `evidence=` is removed (it was
  merged into the assignment, so it returned the joint, not the conditional; pass a total assignment instead), and
  `conin.common.log_potential.log_potential_conin` is removed (use `log_potential`). It now accepts DBNs, HMMs and
  constrained models (`-inf` for an infeasible assignment), a factor entry that is not given uses the factor's
  `default_value` (it raised `KeyError`), and an assignment that is not total raises `InvalidInputError` (was
  `AssertionError`). New: `conin.common.log_probability` (BN, DBN, HMM; refuses MNs and constrained models).
- (4e follow-up) `ConstrainedHiddenMarkovModel.is_feasible(seq, observed=None)` checks `"factor_oracle"`,
  `"algebraic"` and `"mvr"` constraints correctly (it called every constraint as an oracle with `{t: h}`) and raises
  `NotImplementedError` for `"pyomo"`/`"toulbar2"`; `seq` may be a dict keyed by time. New: `is_feasible` on the MN,
  BN and DBN constrained classes.
- (After 4b) `cmodel.representation(kind, ...)` builds a new representation on every call (no cache, no
  `Representation.get_copy()`; neither was in a release). The `"mvr"` representation holds deep copies of prebuilt MVR
  constraints, not the user's objects.
- (PR 4c) `viterbi` moves from `conin.hidden_markov_model.inference[.viterbi]` to `conin.inference.unconstrained[.viterbi]`;
  `a_star` from `conin.hidden_markov_model.inference[.a_star]` to `conin.inference.oracle[.a_star_hmm]`; `recursive_a_star`
  (and `Recursive_Heap_Item`, `Unique_Heapq`) from `conin.hidden_markov_model.inference.recursive_a_star` to
  `conin.inference.recursive[.a_star_hmm]`; `supervised_learning`, `mcem`, `add_unknowns`, `convert_to_simulations` from
  `conin.hidden_markov_model.learning` (and `conin.hidden_markov_model`) to `conin.learning`
  (`conin.learning.unconstrained.{supervised,helpers}`, `conin.learning.oracle.mcem`). `conin.hidden_markov_model` no longer
  exports `CHMM`, `Oracle_CHMM`, `HMMApplication`, `learning` or the learning functions.
- (PR 4c) `CHMM`, `Oracle_CHMM` (`conin.hidden_markov_model.chmm[_oracle]`) and `HMMApplication`
  (`conin.hidden_markov_model.hmm_application`) are removed. Use `ConstrainedHiddenMarkovModel(hmm, constraints, data=...)`
  and `representation("oracle")`; `Oracle_CHMM.generate_hidden` → `conin.sampling.oracle.rejection_sample_hidden(cmodel,
  time_steps)` (or `cmodel.generate_hidden`), which returns external labels. `HMMApplication.create_chmm(kind)` →
  `ConstrainedHiddenMarkovModel(hmm, constraints, data=app_data)`; `initialize_hmm_from_simulations` →
  `conin.learning.supervised_learning`.
- (PR 4c) `representation("oracle")` is an `OracleHMMRepresentation` with `model`, `hmm` and `constraints` (the user's
  tuple); `.chmm` and `.generate_hidden` are gone. Oracle predicates that take `(states, data)` now work with A* and
  sampling (they raised `TypeError`).
- (PR 4c) `recursive_a_star(*, hmm_app, observed, ...)` → `recursive_a_star(*, hmm, observed, ...)`: `hmm` is a
  `ConstrainedHiddenMarkovModel` with `RecursiveConstraint`s (new, `conin.constraints`) or a bare HMM. The
  `initialize_constraint_data` / `update_constraint_data` / `constraint_data_feasible[_partial]` protocol is replaced by
  `RecursiveConstraint(initial=, update=, feasible=, feasible_partial=)`. `a_star_(..., chmm=)` → `a_star_(..., oracle=)`.
- (PR 4c) `mcem(app=...)`: `app` is a `ConstrainedHiddenMarkovModel` (or a bare HMM); the `hidden_markov_model` duck typing
  is gone. Each iteration now samples from the re-estimated HMM and trains on the sequences it scores, so its results
  change (§9).
- (PR 4c) A* with oracle constraints prunes partial sequences with `partial_func` (it never did), so it needs fewer
  iterations; results within a `max_iterations`/`max_time` budget can change (the optimum does not).
- (PR 4c) `occurs_only_in_time_frame_constraint` and `occurs_at_least_once_in_time_frame_constraint` (oracle library):
  their `partial_func` no longer rejects prefixes that can still be completed (when `upper_t` is `None`, the window is
  relative to the full length `T`). The full predicates are unchanged.
- (PR 4d) The torch MVR code moves out of `conin.hidden_markov_model`, which is removed (`import conin` no longer imports it):
  `inference.viterbi_mvr` / `inference.marginal_map_mvr` → `conin.inference.mvr[.viterbi|.marginal_map]`;
  `learning.baum_welch_mvr` / `learning.generalized_em_mvr` → `conin.learning.mvr[.baum_welch|.generalized_em]`;
  `sampling.ffbs_mvr` / `sampling.stopped_sampling_mvr` → `conin.sampling.mvr[.ffbs|.stopped_sampling]`;
  `other_queries.sat_prob_mvr` / `other_queries.sat_time_mvr` → `conin.queries.mvr[.sat_prob|.sat_time]`;
  `mvr_common` → `conin.representations.mvr.kernels` (private helpers). The packages export the algorithms (imported on
  first access, so importing a package does not import torch). Names and signatures are unchanged.
- (PR 4d) `MVR_CHMM` (`conin.hidden_markov_model.chmm_mvr`) is removed. Use `ConstrainedHiddenMarkovModel(hmm, mvrs)` (a bare
  `HiddenMarkovModel` also works: no constraints); `MVR_CHMM(hidden_markov_model=hmm, constraints=[])` →
  `ConstrainedHiddenMarkovModel(hmm)`; `.hidden_markov_model` → `.model`; the aligned MVRs (`MVR_CHMM.constraints`) →
  `cmodel.representation("mvr").mvrs` (`cmodel.constraints` holds the user's constraints, not aligned). The MVR algorithms
  take a `ConstrainedHiddenMarkovModel` with MVR constraints (they read `model.hidden_markov_model`/`.constraints` duck-typed
  before); other constraint families raise `InvalidInputError`. The constructor errors move to `representation("mvr")` (a
  non-MVR constraint is refused by the constructor, as any family mix).
- (PR 4d) `representation("mvr")` is an `MVRHMMRepresentation` with `model` and `mvrs`; `.chmm` and `.hmm` are removed.
- (PR 4d) `mvr_timerange(mvr, time_range)` returns a new MVR and leaves the input unchanged; pass `inplace=True` for the
  old behaviour (set the range on the input and return it).
- (PR 4d) `MVR_MatVecRepn` arrays (`ini_array`, `upd_array`, `evl_array`, `ini_idx`, `next_idx`) are read-only copies of
  the inputs; writing to them raises `ValueError`.
- (PR 4d) `ConstrainedGraphicalModel.supported_families` is removed (a breaking change for subclasses that set it). The
  accepted constraint families are those with a registered representation builder for the model type
  (`supported_representations(model_type)`); register a builder to support a new family. Constrained Markov networks,
  Bayesian networks and dynamic Bayesian networks now refuse `"oracle"`, `"mvr"` and `"recursive"` constraints at
  construction (`InvalidInputError`; they were accepted but no inference method supported them; use
  `FactorOracleConstraint` for oracle constraints on these models). Constrained HMMs accept the same families as before.
- (PR 4g) `HMM_MatVecRepn` (`conin.graphical_models`) is removed, and so are `HiddenMarkovModel.repn`,
  `HiddenMarkovModel.initialize()` and the `initialize=` constructor argument (no replacement needed: drop `initialize=True`).
  The HMM parameters as arrays on internal indices are `ConstrainedHiddenMarkovModel(hmm).representation("matvec")`, a
  `MatVecHMMRepresentation` (`conin.representations.matvec`) with read-only `float64` NumPy arrays `start_vec`,
  `transition_mat`, `emission_mat` and their logs `log_start_vec`, `log_transition_mat`, `log_emission_mat` (the old
  `hmm.repn.start_vec` etc. were lists; `hmm.start_vec` etc. still return lists). It has no `hidden_states`/`observed_states`
  (they were `range(n)`; use `range(repn.num_hidden_states)`) and no sampling methods.
- (PR 4g) `HiddenMarkovModel.generate_hidden(T)`, `.generate_hidden_until_state(h)`, `.generate_observed_from_hidden(hidden)`
  and `.generate_observed(T)` are removed; use `conin.sampling.unconstrained.generate_hidden(hmm, T)`,
  `generate_hidden_until_state(hmm, h)`, `generate_observed_from_hidden(hmm, hidden)` and `generate_observed(hmm, T)` (same
  labels and the same random draws from NumPy's global state; `hmm.set_seed` stays). They also accept a
  `ConstrainedHiddenMarkovModel` without constraints and raise `InvalidInputError` if it has constraints.
- (PR 4g) `ConstrainedHiddenMarkovModel.generate_hidden(T)`, `.generate_observed_from_hidden(hidden)` and
  `.generate_observed(T)` are removed; use `conin.sampling.oracle.rejection_sample_hidden(cmodel, T)` (same draws) and
  `conin.sampling.unconstrained.generate_observed_from_hidden(cmodel.model, hidden)` (the old method also checked that
  `hidden` is feasible; check with `cmodel.is_feasible(hidden)`).
- (PR 4g) `HiddenMarkovModel(...)` checks its probabilities at construction (nonnegative; start probabilities and each row of
  transition/emission probabilities sum to 1) and raises `InvalidInputError`; before, they were checked only when the numeric
  representation was first built (`initialize=True` or the first Viterbi/A*/sampling/MVR call), so invalid HMMs could be
  created and used with the Pyomo, toulbar2 and VE backends.
- (PR 4g) `viterbi(observed=, hmm=)` and `a_star(observed=, hmm=)` no longer accept an `HMM_MatVecRepn` with internal indices
  (use `viterbi_(observed=, hmm=repn)` / `a_star_(observed=, hmm=repn)` with a `MatVecHMMRepresentation`); `viterbi` now also
  accepts a `ConstrainedHiddenMarkovModel` without constraints (constraints raise `InvalidInputError`).
  `map_query(..., method="viterbi" | "a_star")` no longer accepts an `HMM_MatVecRepn` (the overloads are removed).
  `OracleHMMRepresentation.hmm` is a `MatVecHMMRepresentation` (was the `HMM_MatVecRepn`).
- (4g follow-up) `conin.sampling.dbn.DBNSampler` → `conin.sampling.unconstrained.DBNSampler` (module
  `conin.sampling.unconstrained.dbn`); `conin.sampling.dbn` is removed. The class is unchanged.
- (PR 5, additive) `map_query(..., method="mvr_viterbi")`: the torch MVR Viterbi on the `"mvr"` representation. The solution has
  `states`, `hidden`, `log_likelihood` and `augmented` (the decoded augmented path); dict evidence is sparse and the states come back
  keyed by time; `stop + 1` is the time horizon; `dtype`/`device` are forwarded (default float32, "cpu"); `start` must be 0.
- (PR 5) `map_query` normalises every model to a constrained model and dispatches on the representation type. The private
  dispatcher modules `conin.inference.{OptimizationInference, CFNInference, AStarInference, ViterbiInference, VariableElimination}`
  (and their `_map_query_*` functions) are removed; use `map_query` or the public per-backend functions.
- (PR 5) `conin.inference.mn`, `conin.inference.bn` and `conin.inference.dbn` (empty packages) are removed.
- (PR 5) `map_query(..., method="viterbi" | "a_star")` on a Markov network, Bayesian network or DBN raises `InvalidInputError` (no
  such representation is registered) instead of `TypeError`. A non-model still raises `TypeError("Unsupported model type: ...")`.
- (PR 5) HMM queries need list or dict evidence: `evidence=None` (or any other type) raises `InvalidInputError` for every method
  (`viterbi`/`a_star` returned `None`, `integer_program`/`toulbar2` raised `AssertionError`, `variable_elimination` `TypeError`).
  The public `inference_{pyomo,toulbar2,variable_elimination}_map_query_HMM` functions raise the same error.
- (PR 5 follow-up) HMM observations are read one way everywhere (`map_query`, the public HMM functions, `lp_inference`/`ip_inference`
  via map_query, `log_probability`, `log_potential`, `is_feasible`): a non-empty list or tuple, or a dict keyed by exactly the times
  `0..T-1`. A sparse or badly keyed dict, or an empty sequence, raises `InvalidInputError` (it raised `KeyError`, `IndexError` or
  `AssertionError`); tuple evidence now works in `map_query` (it raised `InvalidInputError`) and returns list states. `mvr_viterbi`
  still takes a sparse `{t: label}` dict. The error messages changed ("... must be a list, a tuple or a dict keyed by the times
  0..T-1").
- (PR 5 follow-up) `map_query(hmm, method="variable_elimination", stop=s)` and `inference_variable_elimination_map_query_HMM(stop=s)`
  infer over `start..s` (`stop` was ignored, every observation was used). For integer_program, toulbar2 and variable_elimination, a
  `stop` past the last observation raises `InvalidInputError` (IP/toulbar2 raised `KeyError`).
- (PR 5 follow-up) The integer-program/toulbar2 query-check messages changed: a partial `variables` raises `RuntimeError("Partial MAP
  queries are only supported by method='variable_elimination'; ...")` (was "VariableElimination is not supported for CONIN models" or
  "CONIN does not currently support the generation of discrete Markov networks ..."); `prune_network=True`/`create_MN=True` raise
  `RuntimeError("<option>=True is not supported for CONIN models")`. On an MN, `variables` naming every node is now a full query (it
  raised `RuntimeError`).
- (PR 5) `map_query(ConstrainedHiddenMarkovModel(hmm), method="viterbi")` works (it raised `TypeError`); with constraints it raises
  `InvalidInputError` (was `TypeError`).
- (PR 5) `map_query(..., method="a_star")` with `RecursiveConstraint`s runs `recursive_a_star` (it raised `InvalidInputError`). Its
  solutions have `states`, `hidden` and `log_likelihood` (dicts keyed by time for dict evidence) and the result also has
  `knowledge_state_times` and `objective_values`.
- (PR 5) `a_star` (and `a_star_`, `map_query(method="a_star")`) returns `solution=None` with the solutions found so far (`[]`) and
  `termination_condition="error: no feasible solutions"` (or `"ok"` when `num_solutions > 1` and some were found) when it finds no
  solution (infeasible model, or `max_iterations`/`max_time` reached first); it raised `IndexError`.
- (PR 5) The constrained model's `data` (`Constrained*(model, constraints, data=d)`) reaches the constraint functions on every
  backend: "pyomo", "toulbar2", "factor" (VE), the oracle representation (A*, rejection sampling) and `is_feasible` (it was passed only
  by network flow and the MVR factories). Functions see a shallow copy of `d` with the backend's fields set (`T`, `hmm`, `start`,
  `stop`, `variables`, `evidence`, ...; as items for a dict/Munch, as attributes otherwise); a backend field replaces an application
  field of the same name; without `data` they see exactly what they saw before. The repns of these kinds have a `data` attribute.
- (PR 5) toulbar2 queries apply the integer-program query checks: a partial `variables` raises `RuntimeError` (it was ignored and every
  variable was returned; all nodes is still a full query), and `prune_network`/`create_MN` are accepted when `False` and raise
  `RuntimeError` when `True` (both raised `TypeError`).
- (PR 5) `map_query(dbn, method="integer_program" | "toulbar2")` with `stop=None` (the default) unrolls over `0..1`, as
  `variable_elimination` and the public `inference_*_DDBN` functions do (it raised `TypeError`).
- (PR 5) `ip_formulation` is understood for every model type: `"markov_network"` on an MN/BN/DBN is the default formulation (it
  raised `TypeError`), `"network_flow"` raises `InvalidInputError` (no such representation).
- (PR 5) `ip_formulation="network_flow"` on a `ConstrainedHiddenMarkovModel` without constraints uses `lp_inference`, as a bare HMM
  does (it used `ip_inference`, whose solutions also have `variables`; `solver="or_topas"` therefore needs constraints, or a direct
  `ip_inference` call). Applies to `map_query` and `inference_pyomo_map_query_HMM`.
- (PR 5) `map_query(pgmpy_model, method="variable_elimination")` converts the pgmpy MN/BN to CONIN like every other path (it ran pgmpy
  VE on the model directly): the MAP states are the same (checked on 20 cases), and `solution_with_evidence`/`write_uai_file` are now
  honoured for pgmpy inputs.
- (PR 5) The unknown-`method` `ValueError` lists `mvr_viterbi`; the not-a-model `TypeError` lists the same model types for every
  method.

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
| 4a | Pyomo repns + `inference/pyomo` | refactor | **DONE (branch `restructure-pr4a-pyomo`, off 4f).** One `PyomoVarWrapper` and one constraint applier. Split create from solve. Network-flow repn. Fix Pyomo-path bugs from §9. |
| 4b | toulbar2 repns + `inference/toulbar2` | refactor | **DONE (branch `restructure-pr4b-toulbar2`, off 4a).** Dedupe the MN/BN builders. Implement the rebuild-on-get strategy. Fix toulbar2 bugs. |
| 4c | Oracle + Recursive (A*), oracle sampling, mcem | refactor | **DONE (branch `restructure-pr4c-oracle`, off `restructure` 657c29f).** `Oracle_CHMM` becomes `OracleHMMRepresentation`; `CHMM` deleted. Add `RecursiveConstraint` and the `"recursive"` repn; `recursive_a_star` takes a constrained HMM. `HMMApplication` deleted (not moved; §11). Viterbi, A*, recursive A*, the learning helpers and rejection sampling move to `inference/{unconstrained,oracle,recursive}`, `learning/` and `sampling/oracle/`. Fix the A*, recursive A* and mcem bugs and two oracle-library partial functions. |
| 4c′ | Remove restructure references from the code | comments | **DONE (branch `restructure-comment-cleanup`, off 4c).** Comment, docstring and test-label changes only: the PR-number, §-number and `RESTRUCTURE_PLAN` references added by PRs 2–4c are reworded to describe the behaviour or the bug, `test_public_api.py`'s `RESTRUCTURE_PLAN_API` group becomes `PUBLIC_API` (test ids `public_api:...`; the `used_in` label `"RESTRUCTURE_PLAN.md §5"` becomes `"public API"`), and two `test_hmm_representations.py` tests named after the old `Oracle_CHMM` are renamed (`..._matches_the_reference...`). No test is added or removed (full suite in the `conin_scaling_experiments` venv, where torch now imports: 1694 passed / 123 skipped / 0 failed). The rule that stops new ones is in this section. |
| 4d | MVR | refactor | **DONE (branch `restructure-pr4d-mvr`, off `restructure` e5b7c04).** `chmm_mvr` and `mvr_common` become repn (`MVRHMMRepresentation`: model + aligned MVRs) plus kernels. Move inference, learning, sampling and queries to `conin.{inference,learning,sampling,queries}.mvr`; delete `MVR_CHMM` and `conin.hidden_markov_model`. `mvr_timerange` non-mutating by default; MVR arrays read-only. Update all MVR notebooks and cheat sheets. Final step of PR 4 (commit 3): accepted constraint families derived from the registry, `supported_families` removed. |
| 4g | Unconstrained HMM matrix/vector representation | refactor + delete | **DONE (branch `restructure-pr4g-hmm-repn`, off `restructure` b008502).** The `"matvec"` repn (`MatVecHMMRepresentation`: plain + log read-only arrays) replaces `HMM_MatVecRepn`/`hmm.repn`/`initialize`; sampling moves to `conin.sampling.unconstrained`; the HMM validates at construction; the `HMM_MatVecRepn` `map_query` overloads are removed (§4.3, §4.8, §5, §11). Original plan: After 4d (which rewrites `mvr_common` and the MVR notebooks, the heaviest `hmm.repn` users), before 5 (so Viterbi has a representation kind to dispatch on). Replace `HMM_MatVecRepn` and the cached `HiddenMarkovModel.repn` with a registered representation of the bare HMM (`HiddenMarkovModel`, family `None` only): read-only numpy arrays on internal indices, probably plus the log arrays that Viterbi and A* each compute now. Viterbi, A*, recursive A*, rejection sampling and MVR's `_hmm_to_torch` use it; the oracle and recursive repns hold one instead of `model.repn`. Remove `HMM_MatVecRepn`, `hmm.repn`, `initialize()` / `initialize=True` and the `isinstance(hmm, HMM_MatVecRepn)` input branches of `viterbi` and `a_star`. Rationale and open questions: §4.8, §12. |
| 4e | Factor / VE | refactor | **DONE (branch `restructure-pr4e-factor`, off `restructure` 3cf8d04).** VE constraint handling moves into the factor repn (`apply_factor_constraints`, run at build time). Move `factor_repn`. `conin.inference.variable_elimination`. Fix the VE evidence bugs. Follow-up (same branch): delete the dead pgmpy log-probability code (`custom_VE`, `log_prob_v2`); `log_probability`/`log_potential` for every model; `is_feasible` for every constrained model. Second follow-up: remove the `constraints.algebraic` ↔ `representations.pyomo` import cycle and move the function-level imports that no cycle needs to the top of their modules. |
| 4f | `common/` → `io/` + `common/`; remove pgmax/pomegranate | move + delete | **DONE (branch `restructure-pr4f-io`).** Rename-only first commit, then the folds and deletions; pyagrum removed too (§11). Independent of 3 and 4a–e. It can happen right after PR 1. |
| 5 | Dispatch rewrite | **design** | **DONE (branch `restructure-pr5-dispatch`, off `restructure` 50f9eb1).** `map_query` → repn → ovld on the repn type (§4.5). Remove the duplicated overloads (the five dispatcher modules are deleted). Add `method="mvr_viterbi"`. Also: a_star → recursive A* for recursive constraints, the three §9 rows (cmodel `data`, toulbar2 query checks, a_star no solution), the §12 test-location question (rename-only commit), §11 "PR 5" rows. |
| 5b | Scoring | move + feature | After 5, before 6 (so the 2.0 docs and CHANGELOG describe the final paths). New `conin/scoring/` package beside `inference/`. First commit, rename-only: `common/log_potential.py` → `scoring/` (`log_probability`, `log_potential`), with every importer updated (no `conin.common` re-export, per the no-shims rule; a breaking change in §5). Then the marginal and conditional log-probability query (`log P(X=x \| E=e)` with the other variables summed out, and the probability under constraints, `P(x) / P(C)`), dispatched like the PR 5 `map_query` (method → repn kind → ovld on the repn type), on the backends that can compute it (VE on the `"factor"` repn first). Restores `common/` as a bottom layer (it no longer imports `constrained_models`). Decide the query's API and backends at the start of the PR. |
| 6 | Docs + cleanup | docs | Sphinx API pages, prose docs, AGENTS.md (examples location), CHANGELOG, version 2.0. Delete this plan file. |

Dependencies: 0 → 1 → 2 → 3 → {4a, 4b, 4c, 4d, 4e} → 4g → 5 → 5b → 6 (all of PR 4 done after 4g). 4f can go anywhere after 1; 4g needs 4c and 4d.
4a–4e can be done in parallel.

Bugs are fixed inside the backend PR that touches the code (§9). Each fix gets its own
test and is called out in the PR description.

**No restructure references in the code.** Code, comments, docstrings, test names and
test ids must not mention PR numbers ("PR 4c", "before PR 4e", "moves out in PR 4d"),
sections of this plan ("§9", "RESTRUCTURE_PLAN") or the restructure itself. People
read the code long after this plan is deleted (PR 6). Say what the code does, or for a
regression test, what the bug was (e.g. "a_star used to check the parent prefix, so the
pruning never fired"). Put the PR number, the plan section and the "when" in the plan,
the commit message and the PR description instead. Notes about temporary states ("kept
until the MVR code moves") are fine if they don't name a PR, and must be removed when
the state ends. Before committing, check with
`rg -n -i '\bPR ?[0-9]|§[0-9]|RESTRUCTURE' conin doc --glob '!RESTRUCTURE_PLAN.md'`
(it should find nothing).

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
dependencies, no new top-level imports of optional backends, and no PR numbers or
plan-section references in code, comments or docstrings (§6).

---

## 8. Testing strategy
- PR 0 adds a public-API smoke test and records the baseline counts (959 passed / 195 skipped / 0 errors; see §2.3). After PR 1: 962 passed / 195 skipped. After PR 2: 1009 passed / 195 skipped (+47 new tests). After PR 3: 1074 passed / 195 skipped (+65). After the PR 3 follow-ups: 1075 passed / 195 skipped. After PR 4f: 1076 passed / 193 skipped (−4 passed and −2 skipped deleted loader tests, +5 `test_public_api.py` rows). After PR 4a: 1115 passed / 193 skipped (+30 Pyomo representation tests, +9 `test_public_api.py` rows); 1116 / 193 after the 4a follow-ups (+1, `test_ensure`). After PR 4b: 1151 passed / 193 skipped (+29 toulbar2 representation tests, +6 `test_public_api.py` rows). After removing the representation cache: 1150 passed / 193 skipped (−1: the two PR 3 cache/get_copy tests merged into `test_every_call_builds_a_new_representation`). After PR 4e: 1181 passed / 202 skipped (+25 factor representation tests, of which 9 skip without pgmpy; the 5 VE helper tests rewritten as 8; +12 `test_public_api.py` rows). After the 4e follow-up: 1269 passed / 204 skipped (`test_log_potential.py` rewritten, 2 → 62 tests; new `test_feasibility.py`, 26; +4 `test_public_api.py` rows). After the import-cycle follow-up: 1270 passed / 204 skipped (+1). After PR 4c: 1330 passed / 204 skipped (+33 new A* / recursive A* tests, +16 / +3 / +2 / +1 in rewritten files, −8 `test_application_chmm.py`, +13 `test_public_api.py` rows; §2.3). After PR 4d: 1349 passed / 213 skipped (+10 MVR repn tests, −1 in `test_mvr_representation.py`, +7 `test_base.py`, +2 `test_feasibility.py`, +1 `test_mvr_operators.py`, +9 `test_public_api.py` rows; §2.3). After PR 4g: 1380 passed / 213 skipped (+11 matvec repn tests, +16 in the new `sampling/unconstrained/tests/test_hmm.py` (8 moved), −3 net in `graphical_models/tests/test_hidden_markov_model.py`, −1 in `constrained_models/tests/test_hidden_markov_model.py`, +8 `test_public_api.py` rows; §2.3). After the 4g follow-up (`DBNSampler` moved): 1381 passed / 213 skipped (+1 `test_public_api.py` row). After PR 5: 1533 passed / 218 skipped (+123 `test_map_query_dispatch.py`, +30 `test_constraint_data_and_query_checks.py`, +4 in `test_a_star_hmm.py`, +5 torch `mvr_viterbi` tests (skipped here); none removed; §2.3). After the PR 5 follow-up: 1600 passed / 222 skipped (§2.3).
- Every PR must keep the pass count ≥ baseline, after adjusting for deliberately deleted tests.
- PR 3 adds tests for:
  - immutability (setattr raises; `with_*` returns a new object);
  - mixed-family rejection;
  - lazy building (nothing built at `__init__`);
  - cache hits (removed with the cache after 4b);
  - `get_copy` isolation (mutating a returned Pyomo model doesn't affect the next get; after 4b: the next `create_model()`).
  (PR 3: all done, using a test representation registered in the test; the Pyomo `get_copy` test moves to 4a.
  PR 4a: done, `test_mutating_a_returned_model_does_not_affect_the_next_one`. PR 4b: `test_two_gets_give_independent_models`.
  PR 4e: `test_two_create_model_calls_give_independent_models`. PR 4d: the "mvr" repn holds no per-query state;
  `test_mvr_prebuilt_mvrs_are_not_shared_with_the_user`, `test_mvr_arrays_are_read_only`,
  `test_mvr_representation_is_not_changed_by_mvr_timerange`, `test_stopped_sampling_leaves_the_caller_model_untouched`.
  PR 4g: `test_arrays_are_read_only` (built/deepcopy/pickle), `test_one_representation_serves_several_queries`,
  `test_every_call_builds_a_new_representation` in `representations/matvec/tests/test_matvec_representation.py`.)
- Each bug fix gets a regression test.
- Notebooks run in CI, including `MVR/`.

---

## 9. Bugs found during exploration (fix in the relevant backend PR)

| Bug | PR |
|-----|----|
| `a_star_`: the partial-feasibility pruning condition is always true, and it checks `seq` instead of `newSeq` | 4c (fixed: the extension `newSeq` is checked and pruned, and the initial states too; same optima on 90 parity cases, chmm1 T=20 needs 506 iterations instead of 983. `test_pruning_rejects_the_extended_prefix_not_its_parent`, `test_pruning_reduces_the_search_without_changing_the_optimum` in `inference/oracle/tests/test_a_star_hmm.py`) |
| `recursive_a_star`: partial feasibility uses the old `constraint_data`; `n_infeasible` is in the wrong branch | 4c (fixed: `feasible_partial(new_state, t + 1, T)`; `n_infeasible` counts infeasible leaves (debug output only). `test_partial_feasibility_uses_the_new_state`, `test_tight_bound_prunes_at_once`, `test_infeasible_leaves_are_skipped` in `inference/recursive/tests/test_a_star_hmm.py`. Changed results on 4 of 162 parity cases: `lb, ub = 3, 5`, 12 observations, `max_iterations=50` now finds the optimum (was "no feasible solutions"/none); one `num_solutions=4` case returns a different 4th solution with the same log-likelihood) |
| `mcem` replaces the HMM but the oracle keeps sampling from the old one; it also scores one sample and stores another | 4c (fixed: `cmodel = cmodel.with_model(new_hmm)` each iteration; one draw per sample, scored and trained on. `test_mcem_samples_from_the_reestimated_hmm`, `test_mcem_trains_on_the_samples_it_scores` in `learning/tests/test_learning.py`; mcem's numbers change) |
| The `CHMM` overloads in the AStar/Optimization/CFN dispatchers are broken | 3 (removed) |
| VE: the ConstrainedMN overload replaces user evidence with `{}`; the MN factor oracle forces an existing scope node to 1 | 4e (fixed: the MN applier adds the constraint factor and no evidence; the query keeps the user evidence. `test_constrained_mn_keeps_user_evidence`, `test_constrained_mn_does_not_force_a_scope_node_to_one`, `test_constrained_mn_partial_query_has_only_the_query_variables` in `representations/factor/tests/test_factor_representations.py`; `test_mn_factor_oracle_adds_a_factor_and_no_evidence` in `test_apply_factor_constraints.py`) |
| `ConstrainedHiddenMarkovModel.is_feasible` calls every constraint as an oracle with `{t: h}` (wrong for `"factor_oracle"`, `"mvr"`, `"pyomo"`, `"algebraic"`, `"toulbar2"`) | 4e follow-up (fixed: `constrained_models/feasibility.py`; `test_hmm_oracle_and_factor_oracle`, `test_hmm_mvr`, `test_hmm_pyomo_not_supported` in `constrained_models/tests/test_feasibility.py`) |
| `log_potential` on an MN with a sparse dict factor raises `KeyError` instead of using the factor's `default_value` | 4e follow-up (fixed; `test_mn_missing_factor_entries_use_default_value`). Other consumers (UAI writer, pgmpy MN converter, factor repn) still index missing entries directly | 
| `log_potential(..., evidence=e)` returned the joint `log P(x, e)` (it merged `e` into `x`), which reads like a conditional | 4e follow-up (removed `evidence`; `test_no_evidence_argument`) |
| `conin.common.pgmpy.log_prob_v2` always raised `TypeError` (pgmpy's `to_markov_model()` returns a `DiscreteMarkovNetwork`, not a `MarkovNetwork`); `logprob_VE.logprobs` without evidence raised `AttributeError` (a leftover tqdm call) and with evidence returned the unnormalised joint | 4e follow-up (deleted as dead code) |
| `HiddenMarkovModel.log_probability` raises `ValueError` (`math.log(0)`) for a zero-probability sequence | – (found in the 4e follow-up; `conin.common.log_probability` returns `-inf`; the method is unchanged) |
| `convert_pgmpy_to_conin` on a pgmpy `DynamicBayesianNetwork` turns the slice-0 and slice-1 CPDs of a node into two copies of one `(name, t)` template (e.g. `simple1_DDBN_pgmpy`'s `(B, 0)` and `(B, 1)`), so the unrolled BN has two CPDs for `(B, 0)` and `(B, 1)`; `check_model()` doesn't catch it | 6 (found in the 4e follow-up; `log_probability` raises `InvalidInputError` "more than one CPD"; `test_pgmpy_dbn_with_duplicate_cpds_is_refused`) |
| VE: `_add_{bn,mn}_constraints_as_evidence` return `None` for an empty constraint list, and their `TypeError` strings were missing the `f` prefix | 2 (fixed, regression tests in `inference/tests/test_variable_elimination_helpers.py`) |
| `solve_toulbar2_map_query_model` rejects extra `**options` (e.g. `verbose`) | 4b (fixed: the inference functions pop `verbose` and pass it to `create_model()`; `solve` keeps its signature and still rejects unknown options; `test_verbose_option_is_accepted` (MN/BN/DBN/HMM), `test_unknown_options_still_raise`) |
| The DBN/HMM toulbar2 paths don't check that pytoulbar2 is available | 4b (fixed: all four return the `"pytoulbar2 not available"` result; `test_pytoulbar2_not_available` (MN/BN/DBN/HMM)) |
| `save_model_uai` writes to `name.split(".")[0] + ".uai"`: a dot in a directory name (e.g. `/a.b/out.uai`, `./out.uai`) writes the wrong file. This is likely why ~66 toulbar2/UAI tests fail with `TMPDIR=/projects/.conin-scratch/tmp` (the old toulbar2 path wrote its temp UAI file through it; since 4b the toulbar2 path writes the cached text directly, but `save_model` and the IO tests still hit it) | 6 (found in 4b; use `os.path.splitext`) |
| pytoulbar2 keeps the input format in a global option: after `CFN.Read("x.cfn")`, later toulbar2 queries in the same process fail with "wrong problem file format" until `pytoulbar2.pytb2.option.cfn = False` (conin never reads `.cfn` with pytoulbar2 itself, only the tests do) | 6 (found in 4b; reset the option in `Toulbar2FactorRepresentation._read`) |
| The toulbar2 paths pass `variables` to the constraint functions but otherwise ignore it (the MN docstring said "Ignoring variables for now"); `prune_network`/`create_MN` reach `solve_toulbar2_map_query_model` and raise `TypeError` (the Pyomo paths raise `RuntimeError` via `_validate_bn_query`) | 5 (fixed: the checks are shared, `conin/inference/_query_checks.py`; `test_toulbar2_partial_variables_raise_runtime_error`, `test_partial_variables_raise_the_same_error_on_both_backends`, `test_toulbar2_all_variables_is_a_full_query`, `test_toulbar2_prune_network_and_create_MN` in `inference/tests/test_constraint_data_and_query_checks.py`) |
| The HMM `network_flow` branch can return `None` | 4a (fixed: `InvalidInputError`; `test_network_flow_rejects_unsupported_{evidence,model}` in `representations/pyomo/tests/test_pyomo_representations.py`) |
| `CallableDict.__getitem__` returns `KeyError` instead of raising it (it was `__call__`, for a bad arity; a missing key also returned `None`) | 4a (fixed: `CallableDict` replaced by `PyomoVarWrapper`; `test_network_flow_V_{lookups,bad_arity_raises,missing_key_raises}`) |
| BN/DBN/HMM Pyomo paths forwarded `prune_network=False`/`create_MN=False` to `solve_pyomo_map_query_model`, which raised `TypeError` | 4a (fixed; `test_bn_prune_network_false_is_accepted`) |
| `solve_pyomo_map_query_model` and `ip_inference` (or_topas) `pop("topas_method")` from the caller's `solver_options`, mutating it (a second call with the same dict uses `"balas"`) | 6 (found in 4a) |
| The MN/BN/DBN/HMM `"pyomo"` and (4b) `"toulbar2"` paths never pass the constrained model's `data` to constraint functions (only network flow does), so `Constrained*(..., data=d)` is ignored there | 5 (fixed for "pyomo", "toulbar2", "factor", the oracle repn and `is_feasible`: `merge_constraint_data` in `conin/util/constraint_data.py`; `test_application_data_reaches_the_constraint_functions` (5), `test_without_application_data_the_fields_are_unchanged`, `test_a_representation_field_replaces_an_application_field`, `test_attribute_style_application_data`, `test_factor_representation_passes_the_application_data`, `test_variable_elimination_uses_the_application_data`, `test_oracle_predicates_see_the_application_data`, `test_is_feasible_passes_the_application_data`, `test_is_feasible_dbn_sets_T_on_a_copy_of_the_application_data`, `test_integer_program_solves_with_the_application_data` in `inference/tests/test_constraint_data_and_query_checks.py`) |
| `ConstrainedDynamicDiscreteBayesianNetwork.nodes` is not a property | 3 (fixed, regression test in `constrained_models/tests/test_dynamic_bayesian_network.py`) |
| `a_star` on an unconstrained `ConstrainedHiddenMarkovModel` raised `AttributeError` (a bare `CHMM` has no `is_feasible`) | 3 (fixed as a side effect of the "oracle" repn) |
| `mcem` calls `app.generate_hidden` twice per sample (it scores one sequence and trains on another) | 4c (fixed with the row above) |
| `HMMApplication.create_chmm("pyomo")` attaches no `data`, so constraints that read application data (e.g. `data.num_zeros` in test_application_chmm) would fail if solved | 4c (moot: `HMMApplication` is deleted; `test_network_flow_constraint_reads_the_application_data` in `constrained_models/tests/test_hidden_markov_model.py` solves that constraint with `ConstrainedHiddenMarkovModel(..., data=Munch(num_zeros=3))`) |
| The examples' `Num_Zeros.run_simulations` used `self.oracle`, which was never set (`AttributeError`) | 4c (found; moot: the class is deleted; `create_chmm1_recursive` replaces its recursive-A* use) |
| `occurs_only_in_time_frame_constraint` (`same_partial_as_func=True`) and `occurs_at_least_once_in_time_frame_constraint` with `upper_t=None` have `partial_func`s that reject completable prefixes (the window end was taken from the prefix length, not `T`); harmless before 4c because A* never pruned | 4c (found and fixed: partials use `T`; a brute-force check over all prefixes of length ≤ 4 finds no unsound partial in the library. `test_time_frame_constraints_match_brute_force`, `test_time_frame_partial_functions_accept_completable_prefixes`; 4 expectations changed in `test_constraints.test_occurs_at_least_once_in_time_frame`) |
| `a_star` raises `IndexError` (`output[0]`, or `heappop` on an empty heap) when it finds no solution (infeasible model, or `max_iterations`/`max_time` reached first); `recursive_a_star` returns `termination_condition="error: no feasible solutions"` instead | 5 (fixed: `solution=None`, the solutions found, the termination condition the search computes; `test_no_solution_within_max_iterations_is_reported`, `test_infeasible_model_is_reported`, `test_fewer_solutions_than_asked_for_are_returned`, `test_map_query_a_star_without_a_solution` in `inference/oracle/tests/test_a_star_hmm.py`; `test_pruning_reduces_the_search_without_changing_the_optimum` asserts `solution is None`) |
| `recursive_a_star` with `num_solutions > 1` merges prefixes with equal `(last state, length, constraint state)`, so it returns at most one sequence per merged class, not the k best sequences (e.g. `RecursiveConstraint` "exactly two h0" on `["o0"] * 4` skips the 3rd best) | – (found in 4c; inherent to the algorithm; documented in `inference/recursive/tests/test_a_star_hmm.py`) |
| `MVR/formula.ipynb` prose says `MVR_HMM`/`MVRFunctor` (typos for `MVR_CHMM`/`MVRConstraint`) | 4d (fixed: `ConstrainedHiddenMarkovModel`/`MVRConstraint`; prose only, no test) |
| `DynamicDiscreteBayesianNetwork.t` is public and gets mutated by `create_bn_from_dbn` and `DBNSampler` (an immutability leak) | 1 (fixed: `create_bn_from_dbn` restores `t`; `DBNSampler` works on a deep copy) |
| `conin.io.save_model.save_conin_model` (was `common.conin.save_model`): the `.bif` branch writes the file but doesn't return, so it then raises "Unknown format" (`convert_pgm --bif` avoids it by saving with `model_type="pgmpy"`) | 6 (found in 4f) |
| `backends.rst` calls smoek optional although pyproject requires it; torch and pytoulbar2 aren't declared as extras | 6 |
| `create_bn_from_dbn(start=s)` with `s > 0` fails (`AttributeError: 'float' object has no attribute 'keys'`): `_unroll_cpds` skips the transition CPDs at `t = start` (their parents are at `start - 1`), so the time-`start` nodes keep the initial CPD whose values are not a dict. Every backend that unrolls DBNs/HMMs (pyomo, toulbar2, VE) is affected; `map_query(..., start=1)` has never worked | 6 (found in 4e; decide whether `start > 0` should be supported at all) |
| `HiddenMarkovModel` checked its probabilities (≥ 0, rows sum to 1) only when `HMM_MatVecRepn` was first built (`initialize=True`, or the first Viterbi/A*/sampling/MVR use), so an invalid HMM could be constructed and solved with the Pyomo, toulbar2 and VE backends unchecked | 4g (fixed: the constructor validates; `test_constructor_validates_probabilities` (4 cases) in `graphical_models/tests/test_hidden_markov_model.py`; no existing test or example had an invalid HMM) |
| VE on the unconstrained `weather` DBN (`stop=4`) runs out of memory under pgmpy (the VE weather tests in `test_MAP_Inference.py` are commented out) | – (found in 4e; pgmpy limitation, not a conin bug; PR 5 did not change the VE docs, so a note belongs to PR 6's docs) |
| `pgmpy_issue_1177_pgmpy` (`conin.examples.bayesian_network`) fails with the installed pgmpy 1.1.2: `model.fit(data, estimator=MaximumLikelihoodEstimator)` must now get an estimator instance ("Pass an initialized estimator, for example `DiscreteMLE()`") | 6 (found in 5; no test builds it) |
| `recursive_a_star` documents `solutions` with `hidden` only and returns no `solution`; `map_query(method="a_star")` adds `states` and `solution` for it, but the direct call keeps the old shape | – (found in 5; left as is, the public signature and result were kept) |

---

## 10. Critiques of the original sketch

- **[C1] Copy-on-get can't be one strategy.** Pyomo `clone()` is about as costly as a
  build, and pytoulbar2 CFNs can't be copied at all. Resolution:
  `Representation.get_copy()` lets each class choose, with `deepcopy` as the default (§4.4). (After 4b the cache
  and `get_copy()` are gone; the Pyomo/toulbar2 repns build a fresh model or CFN in `create_model()`.)
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
| Caching | ~~Cache representations; a getter returns `repn.get_copy()`~~. **Superseded after 4b: no cache, no `get_copy()`; `representation()` builds on every call (§4.4)** |
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
| HMMApplication | ~~Keep it, move it to `applications/`, decouple it via `RecursiveConstraint`~~. **Superseded in PR 4c: deleted** (nothing outside conin used it; inside, only two `Num_Zeros` subclasses, one with a broken `run_simulations`); no `applications/` package |
| `common/` | Split into `io/` (file IO) and `common/` (non-IO helpers) |
| pgmax/pomegranate | Remove the converters and the loaders (the loaders just load with pgmpy and convert); keep pyagrum (it reads files natively). **Superseded in PR 4f: pyagrum is removed too (unused)** |
| network_flow IP | Keep it as the `pyomo_network_flow` repn |
| Plan location | Root `RESTRUCTURE_PLAN.md`, deleted at the end |
| Constraint base class | Renamed `ConstraintFunctor` → `Constraint` |
| MVR kernels | `representations/mvr/kernels.py` |
| Wrapped-model property | `.model` only, no aliases |
| Tests | Stay next to the code |
| Unconstrained models in `map_query` | Wrapped as `Constrained*(gm, ())`; the wrapper is transparent to all methods |
| Cache size | (Moot after 4b: no cache) |
| `mvr_timerange` | Non-mutating by default |
| Multiple `RecursiveConstraint`s | Combined automatically (tuple of states) |
| DBN `.t` leak | Fixed in PR 1 |
| pgmpy conversion | Stays in `common/pgmpy/` |
| Version | 2.0.0 |
| Application `data` (PR 3) | Constructor kwarg: `Constrained*(model, constraints=(), data=None)`, read-only `.data`, passed to builders; not part of the cache key |
| PR 3 representation scope | Framework + the HMM kinds that replace `initialize_chmm`: `"oracle"` (Oracle_CHMM), `"pyomo_network_flow"` (PyomoAlgebraic_CHMM), `"mvr"` (MVR_CHMM). pyomo/toulbar2/factor builders come in 4a/4b/4e; `map_query` dispatch is unchanged until PR 5 |
| MVR in PR 3 | `MVR_CHMM` and the torch algorithms are untouched until 4d; the `"mvr"` repn wraps `MVR_CHMM` and already exposes `.mvrs` (**4d: `MVR_CHMM` deleted; the repn is real**) |
| Constrained-class location | PR 3 `git mv`s them to `constrained_models/` (rename-only first commit) and deletes the empty `markov_network/`, `bayesian_network/`, `dynamic_bayesian_network/` packages |
| Representation API (PR 3) | `Representation` ABC with `kind` and `get_copy()`; `@register_representation(kind, model_type, families=(None,))`; `build_representation(cmodel, kind, **params)`; `supported_representations(model_type)` (§4.3) |
| `with_model` / `with_constraints` cache (PR 3) | Always start with an empty cache (moot after 4b) |
| Single constraint (after PR 3) | `Constrained*(model, c)` is accepted (wrapped as `(c,)`); single-element lists dropped in examples/docs/tests |
| Accepted constraint families (after PR 3) | Derived from the representation registry at the end of PR 4; `supported_families` is removed then. Not stored on constraint classes (§4.2) |
| `ConstrainedHiddenMarkovModel` model type (PR 3) | Requires `HiddenMarkovModel`; `HMM_MatVecRepn` is rejected (no caller passed one) |
| HMM repn modules (PR 3) | `representations/oracle/hmm.py`, `representations/pyomo/hmm_network_flow.py`, `representations/mvr/hmm.py` |
| mcem (PR 3) | Keeps the model in a local variable and does not write it back to `app`; same results |
| `io/` and `common/` layout (PR 4f) | `conin.common.conin` is dissolved. `io/`: `uai.py`, `cfn.py`, `pgmpy_files.py`, `load_model.py`, `save_model.py`, `convert_pgm.py`. `common/`: `log_potential.py` (unified dispatcher), `is_polytree.py`, `pgmpy/` (now also `to_pgmpy_bn`/`to_pgmpy_mn`). Model data files live in `io/tests/`; the `common/tests` that need them point there |
| `convert_pgm` CLI (PR 4f) | Keeps its name; module `conin/io/convert_pgm.py` (the one-file `bin/` package is dropped) |
| `load_model`/`save_model` `model_type` (PR 4f) | Kept, with fewer values: `"conin"` (default) and `"pgmpy"` |
| Pyomo get_copy (PR 4a) | Rebuild per query, not `clone()`. Measured on this emulated CPU (best of 3): weather DBN stop=4 build 0.030 s / clone 0.049 s; stop=20 0.121 / 0.207 s; 5×4 HMM T=50 markov_network 0.172 / 0.329 s; network_flow 0.145 / 0.305 s. The repn caches the evidence-independent factor repn and `get_copy()` returns `self` |
| Pyomo cache key (PR 4a) | `()` for MN/BN, `(start, stop)` for DBN/HMM, `(x_binary, y_binary)` for network flow. `omit_c5`/`use_cut_generation` is a `create_model()` option; `variables`, `prune_network` and `create_MN` are validated in `inference/pyomo` (same errors as before) and are not structure params |
| `create_*_map_query_model_*` (PR 4a) | Removed (the "pyomo" repns' `create_model` replaces them); the primitives stay public in `conin.representations.pyomo.common` |
| `inference/pyomo/` layout (PR 4a) | `solve.py`, `mn.py`, `bn.py`, `dbn.py`, `hmm.py`, `hmm_network_flow.py`; the four `inference/*/inference_pyomo.py` and `hidden_markov_model/inference/lp.py` are gone; `conin.inference.{mn,bn,dbn,hmm}` keep only toulbar2 |
| Network-flow `M.V` (PR 4a) | The shared `PyomoVarWrapper` (`CallableDict` deleted) |
| Network-flow `x_binary`/`y_binary` (PR 4a) | Kept as structure params (defaults `True`/`False`); `cache_indices` dropped (index sets are per query) |
| Network flow + "algebraic" (PR 4a) | Registered and tested |
| Constrained model: composition vs inheritance (PR 4a) | Keep composition (`cmodel.model`). Inheritance from the unconstrained class would let bare-model code (Viterbi, `save_model`, converters, samplers, learning) silently drop the constraints, make inherited builders like `with_factor` return the base type, and turn `with_model` into copy-and-rebuild. The wrap-if-bare boilerplate is handled by the `ensure(pgm)` classmethod instead (and by PR 5's `map_query` normalisation) |
| toulbar2 cached intermediate (PR 4b) | The UAI text of the (unrolled) model (via the private `conin.io.uai._uai_text`, byte-identical to `save_model`) plus that model and the constraints; a fresh CFN is read per query; `get_copy()` returns `self`. Measured on this emulated CPU (best of 3), per query: weather DBN stop=4 old path 0.012 s / cached `create_model` 0.004 s; stop=20 0.042 / 0.012 s; 5×4 HMM T=50 0.060 / 0.015 s (the repn build, unroll + text, is 0.004 / 0.018 / 0.028 s, once) |
| toulbar2 evidence and per-query options (PR 4b) | Evidence is `Assign`ed on the fresh CFN, as before. `verbose` and `write_uai_file` are `create_model()` options (as `omit_c5` in 4a); the inference functions pop `verbose` from `**options` so `solve_toulbar2_map_query_model` keeps its signature and still rejects unknown options |
| toulbar2 cache key (PR 4b) | `()` for MN/BN, `(start, stop)` for DBN/HMM (HMM `stop` required), as for "pyomo" |
| `create_toulbar2_map_query_model_*` (PR 4b) | Removed (the "toulbar2" repns' `create_model` replaces them). `conin.representations.toulbar2.common` is public: `Toulbar2VarWrapper`, `apply_toulbar2_constraints`, `Toulbar2FactorRepresentation`, `add_algebraic_constraints_to_toulbar2_model`, `add_toulbar2_constraints` (its `__all__`, which also keeps Sphinx autodoc off the mocked `pytoulbar2`; `_scale_to_integers` stays private). `conin.constraints.algebraic` no longer re-exports the translator |
| `inference/toulbar2/` layout (PR 4b) | `solve.py` (`solve_toulbar2_map_query_model` + `inference_toulbar2_map_query_MN`, as the old MN module), `bn.py`, `dbn.py`, `hmm.py`; `conin/inference/hmm/` deleted; `conin.inference.{mn,bn,dbn}` keep their tests (and `mn/factor_repn.py` until 4e) |
| toulbar2 `write_uai_file` (PR 4b) | Writes the cached UAI text to exactly that path and reads the CFN from it |
| Representation cache (after 4b) | Removed: `_cache`, `_freeze`, `Representation.get_copy()` and its 5 overrides. Nothing reused cached repns (bare-model `map_query` wraps per call; learning uses `with_model` per iteration). Structure params vs `create_model()` options stay; a cache can be re-added inside `representation()` without an API change. The MVR repn deep-copies prebuilt MVRs at build time (took over from `get_copy`) |
| Bugs found in 4b outside its §9 rows (PR 4b) | `save_model_uai`'s path bug and the pytoulbar2 `option.cfn` leak are recorded in §9 for PR 6, not fixed in 4b |
| VE engine (PR 4e) | Keep pgmpy: the factor repn holds the materialised conin model (`ve_model`) and `create_model()` converts it with `convert_conin_to_pgmpy_{bn,mn}`; VE does not run on `(S, J, v, w)` (no behaviour risk) |
| Factor structure params (PR 4e) | `()` for MN/BN; `start=0, stop=1` for DBN; `start=0, stop` (required) for HMM, as for "pyomo"/"toulbar2". The HMM dispatcher keeps `stop = len(evidence) - 1` (it ignores `stop`, as before) (PR 5 follow-up: it honours `stop`) |
| VE constraint `data` (PR 4e) | Unchanged: MN/BN `None`; DBN `Munch(T)`; HMM `Munch(hmm=Munch(T))`. It holds no evidence, so constraints are materialised once at build time; the query evidence is merged per query (`query_evidence`). Aligning it with the Pyomo/toulbar2 layout is left to PR 5 (with the `cmodel.data` row in §9). **PR 5: the application data is merged in (see "Constraint `data` (PR 5)"); the fields stay the VE ones (no `variables`/`evidence`), since the constraints are materialised at build time** |
| VE public names (PR 4e) | `conin.inference.variable_elimination` (`solve.py`, `mn.py`, `bn.py`, `dbn.py`, `hmm.py`, `custom_VE.py`) with `inference_variable_elimination_map_query_{MN,BN,DDBN,HMM}` and `solve_variable_elimination_map_query_model`; `conin.representations.factor` with `FactorRepresentation`, `Factor{MN,BN,DBN,HMM}Representation`, `apply_factor_constraints`. `custom_VE.py` was moved here in 4e and deleted in the 4e follow-up (dead code; the "import cycle" that motivated a lazy import in `log_prob_v2` never broke imports, but it was a layering inversion: `common` importing `inference`) |
| Total-assignment scores (4e follow-up) | `conin.common.log_probability(model, x, *, observed=None)` (BN, DBN, HMM: exact `log P(x)`; refuses MN and constrained models) and `conin.common.log_potential` (same args; also MN = `log P + log Z`, and constrained models = `-inf` if infeasible, else the unconstrained value, not renormalised). DBN assignments are `{(name, t): s}` over `0..stop` (unrolled with `create_bn_from_dbn(start=0)`); HMM assignments are the hidden sequence (list or dict keyed by time, as in `map_query` solutions) plus `observed`. No `evidence` argument. pgmpy models are converted to conin. Both stay in `common/` (no inference) |
| Import layering (4e follow-up 2) | No import cycles between modules: every function-level import of a conin module that a cycle needed was removed or hoisted. `PyomoVarWrapper` lives in `conin/util/var_wrapper.py` (`conin.constraints.algebraic` needs it, and `conin.representations.pyomo` imports `conin.constraints`); its public name stays `conin.representations.pyomo.PyomoVarWrapper`. Still function-level by choice: the `conin.common.pgmpy` imports in `conin/examples/*` and `conin.inference.*` (inside `try_import`/pgmpy-only paths), ~~`representations/mvr/hmm.py` → `hidden_markov_model.chmm_mvr`~~ (4c removed the oracle one; 4d removed this one), and `decorators.py`'s doctest import of `conin`. `conin/__init__.py` imports `inference.map_query` eagerly, so importing any submodule loads all of inference (≈7 s here); making that lazy is left for later |
| `conin/scoring/` (decided after the 4e follow-ups) | A separate PR 5b, after PR 5 and before PR 6: `log_probability`/`log_potential` move from `common/` to a new `scoring/` package (rename-only commit, all importers updated, no re-export), and the marginal/conditional query is added there. Not next: PR 4 (4c, 4d) is unfinished, and the query depends on the PR 5 dispatch design. Until then they stay in `common/` (`common` → `constrained_models` is not a cycle) |
| Feasibility checks (4e follow-up) | `is_feasible` on every constrained model, in `constrained_models/feasibility.py`: `"oracle"`, `"factor_oracle"`, `"algebraic"` (smoek-compiled predicate, as for VE) and `"mvr"` (the torch algorithms' window semantics) only. `"pyomo"`/`"toulbar2"` raise `NotImplementedError` (a function may add variables, making the check an IP/CFN feasibility problem). The constraint `data` is the `"factor"` representation's |
| Factor-repn helpers (PR 4e) | `inference/mn/factor_repn.py` → `representations/factor/factor_repn.py`; `factor_representation_of_mn`, `skeleton_mn_of_bn`, `tuple_var_index_map` move there from `representations/pyomo/common.py` (the pyomo builders import them) |
| MN constraint factor (PR 4e) | The MN applier adds the materialised factor (0 on infeasible assignments) and no evidence. The old code also fixed the last scope node to `1`, which was the §9 bug |
| Oracle representation (PR 4c) | `OracleHMMRepresentation` holds `model`, `hmm` (`model.repn`; 4g: a `MatVecHMMRepresentation`), `constraints` (the user's tuple) and does feasibility only (`is_feasible(seq)`, `partial_is_feasible(*, T, seq)` on internal indices, sharing `feasibility.oracle_accepts`). Rebuilt per call (0.8 ms; no cache) |
| Oracle + algebraic (PR 4c) | Not registered: compiling needs `T` and may reference `("E", t)`, and the compiled oracle has no partial function. `create_oracle_constraints_from_algebraic` stays in `constraints/algebraic/` (the "factor" repn and `feasibility.py` use it; moving it into a representation module would invert the layering). **PR 5: revisited, still not registered** (it would need a `T` structure param and A* could not prune) |
| RecursiveConstraint (PR 4c) | `RecursiveConstraint(*, initial, update, feasible, feasible_partial=None, name=None)`, family `"recursive"`, no `data` argument; tuple states for several constraints; `"recursive"` added to `ConstrainedHiddenMarkovModel.supported_families` and `is_feasible` |
| recursive_a_star signature (PR 4c) | `recursive_a_star(*, hmm, observed, ...)` (same keyword as `a_star`); a bare HMM is wrapped with `ensure`. `map_query(method="a_star")` does not route to it (PR 5) (**PR 5: it does, see "a_star and recursive constraints (PR 5)"**) |
| Rejection sampling (PR 4c) | `conin.sampling.oracle.rejection_sample_hidden(hmm, time_steps, *, max_failures=1000)` → external labels, same RNG draws as `Oracle_CHMM.generate_hidden`; `ConstrainedHiddenMarkovModel.generate_hidden` delegates to it. A bare HMM is sampled with `hmm.generate_hidden` |
| mcem (PR 4c) | Keeps `app=`; `app` is a `ConstrainedHiddenMarkovModel` (or bare HMM, via `ensure`); one draw per sample; `cmodel.with_model(new_hmm)` each iteration; still returns the final average log-probability |
| PR 4c module layout | `inference/unconstrained/viterbi.py`; `inference/oracle/a_star_hmm.py` (`a_star` and its core `a_star_`, kept together); `inference/recursive/a_star_hmm.py`; `learning/{unconstrained,oracle}/`; `sampling/oracle/rejection_hmm.py`. `conin.learning` exports the four learning names. `conin.hidden_markov_model` keeps only the MVR code until 4d (**4d: deleted**) |
| MVR static context (PR 4d) | Not held by the repn: it depends on `T`, `dtype`, `device` and the semiring (per query) and needs torch (the repn module must import without torch). The kernels build it from `repn.mvrs` per call; learning keeps its per-`T` dict for one call |
| MVR learning loops (PR 4d) | Baum-Welch and GEM build `representation("mvr")` once per call and pass each re-estimated HMM with the same MVRs to the kernels (identical results; no deepcopy). `with_model` per iteration was rejected: it would re-call factories with each new HMM and rebuild the static contexts (+3–6 ms and a context rebuild per iteration) |
| MVR repn attributes (PR 4d) | `MVRHMMRepresentation` (class name kept, like `OracleHMMRepresentation`): `model` + `mvrs`; `.chmm` and `.hmm` removed, no aliases. `align_hidden_states` is public in `conin.representations.mvr` |
| MVR prebuilt deep copy (PR 4d) | Kept after `mvr_timerange` became non-mutating (MVRs still have a `name` setter and `inplace=True`; about 1 ms per MVR) |
| MVR algorithm inputs (PR 4d) | A `ConstrainedHiddenMarkovModel` or a bare `HiddenMarkovModel` (via `ensure`), as for `a_star`/`recursive_a_star`/`mcem` |
| `MVR_MatVecRepn` arrays (PR 4d) | Read-only copies (also after deepcopy/pickle); the kernels copy before `torch.as_tensor` (which warns on non-writable arrays). The HMM arrays are left to 4g |
| `conin.hidden_markov_model` (PR 4d) | Deleted: `test_util.py` → `conin/util/tests/`, the empty unused `tests/probabilities.json` deleted, `from . import hidden_markov_model` dropped from `conin/__init__.py` |
| MVR public paths (PR 4d) | `conin.inference.mvr`, `conin.learning.mvr`, `conin.sampling.mvr`, `conin.queries.mvr` export the algorithms through a lazy module `__getattr__` (importing the package or collecting its tests needs no torch); the module paths (`conin.inference.mvr.viterbi`, ...) work too. `conin.representations.mvr` does not import `kernels`. Notebooks and `test_public_api.py` use the package paths |
| MVR test placement (PR 4d) | Next to each module: `{inference,learning,sampling,queries}/mvr/tests/` (file names kept; cross-file helpers imported absolutely), `representations/mvr/tests/test_mvr_representation.py` (the MVR container/`MVR_MatVecRepn` tests and the repn construction tests); the non-torch repn tests are in `representations/tests/test_hmm_representations.py` |
| Stopped-sampling windows (PR 4d) | A private shallow copy with a new `_time_range` (`_with_time_range`), not `mvr_timerange(..., inplace=False)`: that rebuilds and prunes, which can reorder mediation axes and change the draws |
| Accepted families from the registry (PR 4d) | Done as decided (§4.2) with a private `_supported_families()` classmethod; oracle constraints on MN/BN/DBN are refused (the 5 tests that built them use `FactorOracleConstraint` now; no example, doc or notebook used them) |
| Unconstrained HMM representation (PR 4g) | Kind `"matvec"`, class `MatVecHMMRepresentation` in `representations/matvec/hmm.py`, registered for `HiddenMarkovModel` with family `None` only, no structure params. Attributes `model`, `num_hidden_states`, `num_observed_states`, read-only `float64` `start_vec`/`transition_mat`/`emission_mat` **and** `log_start_vec`/`log_transition_mat`/`log_emission_mat` (Viterbi and A* used to take the logs themselves; `np.log` on the array gives the same values as on each entry). No `hidden_states`/`observed_states`, no sampling methods. Rebuilt per call (§4.4) |
| HMM sampling (PR 4g) | The `generate_*` methods move to `conin.sampling.unconstrained` (`sampling/unconstrained/hmm.py`) as functions with the same names: `generate_hidden(hmm, T)`, `generate_hidden_until_state(hmm, h)`, `generate_observed_from_hidden(hmm, hidden)`, `generate_observed(hmm, T)` (external labels; NumPy's global RNG; identical draws; `hmm.set_seed` stays on the model). They take a bare HMM or an unconstrained `ConstrainedHiddenMarkovModel` (via `representation("matvec")`, so constraints raise `InvalidInputError`). Chosen over keeping the methods on the model (the recommendation): sampling is an algorithm, like the others in `sampling/` |
| Constrained HMM sampling methods (PR 4g) | `ConstrainedHiddenMarkovModel.generate_hidden` / `generate_observed_from_hidden` / `generate_observed` are removed too (consistency with the HMM; the recommendation was to keep them). `mcem` calls `rejection_sample_hidden` |
| `hmm.repn` / `initialize` (PR 4g) | Removed outright (no convenience property, no deprecation): `HMM_MatVecRepn`, `HiddenMarkovModel.repn`, `initialize()`, `initialize=`, `avoid_reinitialization`. `_rebuild_hmm` (Baum-Welch) no longer passes `initialize=True` |
| HMM probability validation (PR 4g) | In the `HiddenMarkovModel` constructor (it was in `HMM_MatVecRepn`, i.e. on `initialize=True` or the first numeric use). `_validate_*` are plain checks now (no `check_errors`, no copy). A full-suite probe before the change showed no test depends on lazy validation |
| `map_query` `HMM_MatVecRepn` overloads (PR 4g) | Removed from `ViterbiInference.py` and `AStarInference.py` with no replacement overload on `MatVecHMMRepresentation` (repn-type dispatch is PR 5); the `TypeError` messages no longer list `HMM_MatVecRepn`. `map_query` behaviour for HMMs and constrained HMMs is unchanged. (PR 5: `MatVecHMMRepresentation` has its overload; a user can't pass a repn to `map_query`, only a model) |
| `viterbi` / `a_star` inputs (PR 4g) | `viterbi(observed, hmm)` and `a_star(observed, hmm)` take a `HiddenMarkovModel` or `ConstrainedHiddenMarkovModel` (`ensure`); the `isinstance(hmm, HMM_MatVecRepn)` branches are gone. `viterbi_` and `a_star_(..., hmm=)` take a `MatVecHMMRepresentation` on internal indices. `a_star_` reports `np.float64(-val)` so result types match the old numpy-scalar scores |
| A* matrix reads (PR 4g) | Timed first (§4.8): A* read `hmm.transition_mat[h1][h2]` through copying properties inside its loops; it now reads the repn arrays once into lists (2–8× faster, identical results). Viterbi does the same |
| Recursive A* (PR 4g) | Unchanged: it reads `hmm.get_*_probs()` once (external labels), so it never had the copy cost; the `"recursive"` repn holds no arrays |
| Oracle repn `.hmm` (PR 4g) | A `MatVecHMMRepresentation(cmodel.model)` (built directly: `representation("matvec")` on the constrained model would refuse its constraints). `a_star` takes the arrays from `oracle.hmm`; rejection sampling draws from them with the same sampler as `conin.sampling.unconstrained` |
| `DBNSampler` location (4g follow-up) | `conin/sampling/dbn/sampling.py` → `conin/sampling/unconstrained/dbn.py` (next to `hmm.py`; file named after the model type, not `dbn_forward.py` as first sketched in §3); public name `conin.sampling.unconstrained.DBNSampler`; `conin.sampling.dbn` is removed (no shim). Tests → `sampling/unconstrained/tests/test_dbn.py` and `dbn_examples.py`. Class unchanged |
| Constructor names / CHMM overloads | First parameter is `model` (no `hmm=`/`pgm=` alias); the broken `CHMM` overloads in the AStar/Optimization/CFN dispatchers are removed in PR 3 |
| Dispatch layout (PR 5) | Everything in `conin/inference/map_query.py`: `map_query`, `_normalise`, `_representation_kind`, `_structure_params`, and one ovld function `_solve_map_query(repn, query)` with one overload per representation class (17). The five dispatcher modules are deleted (nothing moved, so no rename-only commit for them). Each backend's public function and the overload call one private repn-level core (`_map_query_<backend>_<model>`, `_map_query_network_flow`, `_lp_inference`, `_ip_inference`, `_viterbi`, `_a_star`, `_recursive_a_star`, `_viterbi_torch_mvr`); public signatures unchanged |
| Method → kind (PR 5) | A small function, not an ovld on the family: the table in §4.5. `ip_formulation` is popped for every model type (`"network_flow"` on MN/BN/DBN → registry `InvalidInputError`; unknown → `ValueError`) |
| Errors (PR 5) | A non-model: `TypeError("Unsupported model type: ...")` from the normalisation. A model type or family the method can't handle (incl. viterbi/a_star/mvr_viterbi on MN/BN/DBN): the registry's `InvalidInputError`. HMM evidence that is not a list or dict: `InvalidInputError` for every method (and in the public HMM functions); since the follow-up, tuples are accepted and dicts must be keyed `0..T-1` (row "HMM observations") |
| Viterbi on `ConstrainedHiddenMarkovModel(hmm)` (PR 5) | Allowed (falls out of the uniform dispatch; constraints → registry `InvalidInputError`) |
| `mvr_viterbi` (PR 5) | As in §4.5: `solution = Munch(states, hidden, log_likelihood, augmented)`, `solutions=[solution]`, `termination_condition="ok"`, `observed`; float32 default (`dtype`/`device` forwarded); list = dense, dict = sparse with time-keyed states; `time_horizon = stop + 1`, else `len(list)` / `max(dict) + 1`; `start != 0` → `InvalidInputError`; infeasible → the kernel's `InvalidInputError`; torch imported on first use |
| Toulbar2 query checks (PR 5) | Shared with the Pyomo paths (`conin/inference/query_checks.py` since the PR 5 follow-up: `reject_pruning_options`, `full_query_variables`; they replace `pyomo/bn._validate_bn_query`): `variables` must be `None`/empty or every node, for MN too (follow-up; an MN with all nodes named was a `RuntimeError`), else `RuntimeError` naming `method='variable_elimination'`; `prune_network`/`create_MN=True` → `RuntimeError`, `False` accepted |
| HMM observations (PR 5 follow-up) | One reading for every HMM query and for `hmm_sequences` (`log_probability`, `log_potential`, `is_feasible`): a non-empty list or tuple, or a dict keyed by exactly `0..T-1` (`conin/graphical_models/hmm_observations.py`, internal, no public export; it sits below constrained_models so scoring can use it too). Anything else → `InvalidInputError` (a sparse dict was a `KeyError`, `[]` an `IndexError`/`AssertionError`, a tuple an `InvalidInputError` in `map_query`). The torch MVR algorithms are the documented exception: a dict is sparse `{t: label}` (`require_sequence_type` checks only the type). The old `_hmm_evidence.py` (input half), the VE `_hmm_evidence_to_dbn`/`_hmm_states_from_map` copies and `feasibility._as_sequence` are gone |
| HMM `stop` (PR 5 follow-up) | integer_program, toulbar2 and variable_elimination infer over `start..stop` (`time_range`; `stop=None` → the last observation); a `stop` past the last observation → `InvalidInputError` (it was a `KeyError` for IP/toulbar2). VE no longer ignores `stop`. viterbi/a_star use every observation (unchanged) |
| Underscore modules (PR 5 follow-up) | No leading-underscore modules: `inference/_hmm_evidence.py` → `inference/hmm_results.py` (the result-shaping half; `hmm_states_from_dbn` → `hidden_states_from_unrolled(results, *, start, stop, keyed_by_time)`), `inference/_query_checks.py` → `inference/query_checks.py` (option popping and the partial-query check split into two functions; messages name the option / `method='variable_elimination'`). Neither is re-exported from `conin.inference` |
| Constraint `data` (PR 5) | "pyomo", "toulbar2", "factor", the oracle repn and `is_feasible` pass the application data: `merge_constraint_data(app_data, fields)` (`conin/util/constraint_data.py`) sets the representation's fields on a shallow copy (items for dicts/Munch, attributes otherwise; a field replaces an application field of the same name); `fields` itself when there is no application data (so the old layout exactly). The repns keep `cmodel.data` as `.data` (a new attribute of `PyomoFactorRepresentation`, `Toulbar2FactorRepresentation`, `FactorRepresentation`, `OracleHMMRepresentation`; not a structure param). The oracle repn was included so that A*, rejection sampling and `is_feasible` give 2-argument predicates the same `data` |
| a_star with no solution (PR 5) | `solution=None`, `solutions` = what was found, the termination condition the search computes (`"error: no feasible solutions"` for `num_solutions=1`, `"ok"` for more, as recursive A*); `a_star_` stops when the heap is empty |
| a_star and recursive constraints (PR 5) | `map_query(method="a_star")` with family `"recursive"` uses the `"recursive"` repn and `_recursive_a_star`; the result is shaped like A*'s (`solution`, solutions with `states` = `hidden`, dicts for dict evidence) and keeps `knowledge_state_times`/`objective_values`. `recursive_a_star` itself is unchanged |
| pgmpy inputs (PR 5) | Converted with `convert_pgmpy_to_conin` on every path, VE included (the native pgmpy VE overloads and `_execute_pgmpy_model_query` are deleted; identical states on 20 cases) |
| DBN `stop=None` (PR 5) | Means `1` for every method (IP and toulbar2 raised `TypeError`) |
| Network flow without constraints (PR 5) | `lp_inference` for `ConstrainedHiddenMarkovModel(hmm)` too (the empty wrapper is transparent), in `map_query` and in `inference_pyomo_map_query_HMM` |
| Cross-backend per-model tests (PR 5) | `conin/inference/{mn,bn,dbn}/tests/` → `conin/inference/tests/{mn,bn,dbn}/` (rename-only commit; the empty `conin.inference.{mn,bn,dbn}` packages are removed) |

---

## 12. Open questions

None blocking. These can be settled during PR 3 (the design PR):
- (Pyomo `get_copy()`: settled in PR 4a, rebuild; see §11. toulbar2: settled in PR 4b, UAI text + fresh CFN. The cache and `get_copy()` were removed after 4b.)
- (Where the cross-backend per-model inference tests go: settled in PR 5, `inference/tests/{mn,bn,dbn}/`; §11.)
- **The marginal/conditional log-probability query's API and backends** (PR 5b, §6): its name and signature, whether it
  shares `map_query`'s dispatch, and which backends compute it (VE on the `"factor"` repn first; the deleted
  `log_prob_v2`/`logprob_VE` were pgmpy-VE drafts of it). Decide at the start of 5b. PR 5's dispatch, to copy: `_normalise` (pgmpy →
  conin, bare → `Constrained*(model)`), a method → kind function, `_structure_params`, then one ovld overload per representation class
  calling a private repn-level core (§4.5); `conin/inference/map_query.py` is the template. `_normalise` and `_structure_params` are
  private to `map_query.py`; if 5b reuses them, move them to a shared private module rather than importing privates across packages.
- (The unconstrained HMM representation: settled in PR 4g, §11 "PR 4g" rows: `"matvec"`; sampling moves to
  `conin.sampling.unconstrained`; `hmm.repn` removed outright; the A* copy cost was timed, §4.8.)
- (`DBNSampler`'s location: settled in the 4g follow-up, `conin.sampling.unconstrained`, §11.)
- **Noted in 4g, not decided:** `conin.sampling.unconstrained.generate_*`, `DBNSampler` and `rejection_sample_hidden` use NumPy's
  global random state (`hmm.set_seed`), while the torch MVR samplers take a `generator`. Revisit with PR 5/6 if wanted.
- (`with_model` cache behaviour, the `Representation` signatures and the registration decorator were settled in PR 3; see §11.)
- (Settled in PR 4d, §11: the MVR static context, `MVR_MatVecRepn` read-only arrays, the MVR public paths, and the fate of
  `conin.hidden_markov_model`, which is deleted.)
- **Noted in 4d, not decided:** the MVR algorithms build `representation("mvr")` per call, so stopped sampling (one
  `sat_time` call plus one `ffbs` call per distinct stopping time) deep-copies the prebuilt MVRs each time (about 1 ms per
  MVR here). Measure before adding anything (a cache inside `representation()`, §4.4, or passing the built MVRs on
  internally).
