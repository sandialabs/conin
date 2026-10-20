# Conin Restructure Plan

Status: **In progress.** PR 0 is merged into `main` (#241). PR 1 is merged into `restructure` (#242). PR 2 is merged into `restructure` (#243). PR 3 is merged into `restructure` (#244). PR 4 is split into sub-PRs 4f, 4a, 4b, 4e, 4c, 4d (in that order, stacked branches); 4f is done (branch `restructure-pr4f-io`, off `restructure`, merged as #245); 4a is done (branch `restructure-pr4a-pyomo`, off 4f, merged as #246); 4b is done (branch `restructure-pr4b-toulbar2`, off 4a, merged as #247); the representation cache was removed after 4b (#248); 4e is done (branch `restructure-pr4e-factor`, off `restructure` 3cf8d04; its follow-up removes the dead pgmpy log-probability code and adds `log_probability` and `is_feasible`; a second follow-up removes the last import cycle and moves the unneeded function-level imports to the top of their modules). The later PRs target the `restructure` branch.
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
    "pyomo_network_flow" repn; **4a: replaced by `PyomoNetworkFlowHMMRepresentation.create_model`**);
  - inside inference functions: `create_*_map_query_model_*` (**4a: the Pyomo ones are removed; the
    "pyomo" representations build the models**; **4b: the toulbar2 ones too, the "toulbar2" representations build the CFNs**), and `lp_inference`
    builds a CHMM (PR 3: uses `representation("pyomo_network_flow")`; 4a: `create_model(observed=)`);
  - inside the dispatch layer: `VariableElimination._add_*_constraints_as_evidence` (**4e: replaced by
    `representations/factor/common.apply_factor_constraints`, run once when the "factor" repn is built**).
- **Almost nothing is cached.** Only `HiddenMarkovModel.repn` is. The
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
  New: `constrained_models/tests/test_feasibility.py` (27: every family, the HMM `is_feasible` §9 regression tests,
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
│   ├── feasibility.py          # (4e follow-up) is_feasible checks of total assignments
│   ├── hidden_markov_model.py  # ConstrainedHiddenMarkovModel (+ generate_hidden, is_feasible helpers)
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
│   ├── mvr/
│   │   ├── hmm.py              # MVRRepresentation (replaces MVR_CHMM): aligned MVRs + static context
│   │   └── kernels.py          # shared torch kernels from mvr_common.py (used by inference/learning/sampling/queries)
│   ├── oracle/hmm.py           # OracleRepresentation (replaces Oracle_CHMM); smoek->oracle
│   └── recursive/hmm.py
├── inference/
│   ├── __init__.py             # exports map_query
│   ├── map_query.py            # public entry point (method -> repn kind -> ovld on repn type)
│   ├── unconstrained/          # viterbi.py, a_star core search (a_star_)
│   ├── pyomo/                  # (PR 4a) solve.py (generic solve/parse); mn/bn/dbn/hmm.py (per-model post-processing); hmm_network_flow.py (lp_inference, ip_inference, from lp.py)
│   ├── toulbar2/               # (PR 4b) solve.py (solve_toulbar2_map_query_model + MN); bn/dbn/hmm.py (per-model post-processing)
│   ├── variable_elimination/   # (PR 4e) solve.py (pgmpy VE runner); mn/bn/dbn/hmm.py (inference_variable_elimination_map_query_*)
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
├── io/                         # (PR 4f) file IO: uai.py, cfn.py, pgmpy_files.py, unified load_model.py/save_model.py, convert_pgm.py CLI
├── common/                     # (PR 4f) non-IO helpers: log_potential.py (log_probability, log_potential), is_polytree.py, pgmpy/ (pgmpy<->conin conversion, MapCPD)
├── mediation_variables/        # UNTOUCHED (research prototype)
└── util/
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
    `partial_is_feasible`), which delegate to a representation.
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
    The constrained model's own `data` is **not** passed on these paths (unchanged from before 4a).
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
    constrained model's own `data` is **not** passed (unchanged; §9 row for PR 5).
  - Structure params: MN, BN `()`; DBN `start=0, stop=1`; HMM `start=0, stop` (required). `verbose`,
    `write_uai_file`, `variables` and the solve options are per-query.
  - The builders build on each other: HMM → `create_dbn_from_hmm` → `build_toulbar2_dbn_representation`
    (`create_bn_from_dbn`) → `build_toulbar2_bn_representation`. The MN builder uses the MN directly.
  - The toulbar2 paths take no `prune_network`/`create_MN` (they reach `solve_toulbar2_map_query_model` and raise
    `TypeError`, as before), so 4a's `_validate_bn_query` is not shared.
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
    len(evidence) - 1`). Neither `variables`, the evidence nor `cmodel.data` is passed (§9 row for PR 5).
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
- The MVR repn deep-copies prebuilt MVR constraints at build time (MVRs are mutable until 4d), so the user's MVR
  objects are never changed through a repn.

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
| 4a | Pyomo repns + `inference/pyomo` | refactor | **DONE (branch `restructure-pr4a-pyomo`, off 4f).** One `PyomoVarWrapper` and one constraint applier. Split create from solve. Network-flow repn. Fix Pyomo-path bugs from §9. |
| 4b | toulbar2 repns + `inference/toulbar2` | refactor | **DONE (branch `restructure-pr4b-toulbar2`, off 4a).** Dedupe the MN/BN builders. Implement the rebuild-on-get strategy. Fix toulbar2 bugs. |
| 4c | Oracle + Recursive (A*), oracle sampling, mcem | refactor | `Oracle_CHMM` becomes `OracleRepresentation`. Add `RecursiveConstraint`, decouple `HMMApplication` and move it to `applications/`. Fix the A* pruning bugs. |
| 4d | MVR | refactor | `chmm_mvr` and `mvr_common` become repn plus kernels. Move inference, learning, sampling and queries. Update all MVR notebooks. |
| 4e | Factor / VE | refactor | **DONE (branch `restructure-pr4e-factor`, off `restructure` 3cf8d04).** VE constraint handling moves into the factor repn (`apply_factor_constraints`, run at build time). Move `factor_repn`. `conin.inference.variable_elimination`. Fix the VE evidence bugs. Follow-up (same branch): delete the dead pgmpy log-probability code (`custom_VE`, `log_prob_v2`); `log_probability`/`log_potential` for every model; `is_feasible` for every constrained model. Second follow-up: remove the `constraints.algebraic` ↔ `representations.pyomo` import cycle and move the function-level imports that no cycle needs to the top of their modules. |
| 4f | `common/` → `io/` + `common/`; remove pgmax/pomegranate | move + delete | **DONE (branch `restructure-pr4f-io`).** Rename-only first commit, then the folds and deletions; pyagrum removed too (§11). Independent of 3 and 4a–e. It can happen right after PR 1. |
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
- PR 0 adds a public-API smoke test and records the baseline counts (959 passed / 195 skipped / 0 errors; see §2.3). After PR 1: 962 passed / 195 skipped. After PR 2: 1009 passed / 195 skipped (+47 new tests). After PR 3: 1074 passed / 195 skipped (+65). After the PR 3 follow-ups: 1075 passed / 195 skipped. After PR 4f: 1076 passed / 193 skipped (−4 passed and −2 skipped deleted loader tests, +5 `test_public_api.py` rows). After PR 4a: 1115 passed / 193 skipped (+30 Pyomo representation tests, +9 `test_public_api.py` rows); 1116 / 193 after the 4a follow-ups (+1, `test_ensure`). After PR 4b: 1151 passed / 193 skipped (+29 toulbar2 representation tests, +6 `test_public_api.py` rows). After removing the representation cache: 1150 passed / 193 skipped (−1: the two PR 3 cache/get_copy tests merged into `test_every_call_builds_a_new_representation`). After PR 4e: 1181 passed / 202 skipped (+25 factor representation tests, of which 9 skip without pgmpy; the 5 VE helper tests rewritten as 8; +12 `test_public_api.py` rows). After the 4e follow-up: 1269 passed / 204 skipped (`test_log_potential.py` rewritten, 2 → 62 tests; new `test_feasibility.py`, 26; +4 `test_public_api.py` rows). After the import-cycle follow-up: 1270 passed / 204 skipped (+1).
- Every PR must keep the pass count ≥ baseline, after adjusting for deliberately deleted tests.
- PR 3 adds tests for:
  - immutability (setattr raises; `with_*` returns a new object);
  - mixed-family rejection;
  - lazy building (nothing built at `__init__`);
  - cache hits (removed with the cache after 4b);
  - `get_copy` isolation (mutating a returned Pyomo model doesn't affect the next get; after 4b: the next `create_model()`).
  (PR 3: all done, using a test representation registered in the test; the Pyomo `get_copy` test moves to 4a.
  PR 4a: done, `test_mutating_a_returned_model_does_not_affect_the_next_one`. PR 4b: `test_two_gets_give_independent_models`.
  PR 4e: `test_two_create_model_calls_give_independent_models`.)
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
| The toulbar2 paths pass `variables` to the constraint functions but otherwise ignore it (the MN docstring said "Ignoring variables for now"); `prune_network`/`create_MN` reach `solve_toulbar2_map_query_model` and raise `TypeError` (the Pyomo paths raise `RuntimeError` via `_validate_bn_query`) | 5 (found in 4b; decide whether to share the Pyomo query checks) |
| The HMM `network_flow` branch can return `None` | 4a (fixed: `InvalidInputError`; `test_network_flow_rejects_unsupported_{evidence,model}` in `representations/pyomo/tests/test_pyomo_representations.py`) |
| `CallableDict.__getitem__` returns `KeyError` instead of raising it (it was `__call__`, for a bad arity; a missing key also returned `None`) | 4a (fixed: `CallableDict` replaced by `PyomoVarWrapper`; `test_network_flow_V_{lookups,bad_arity_raises,missing_key_raises}`) |
| BN/DBN/HMM Pyomo paths forwarded `prune_network=False`/`create_MN=False` to `solve_pyomo_map_query_model`, which raised `TypeError` | 4a (fixed; `test_bn_prune_network_false_is_accepted`) |
| `solve_pyomo_map_query_model` and `ip_inference` (or_topas) `pop("topas_method")` from the caller's `solver_options`, mutating it (a second call with the same dict uses `"balas"`) | 6 (found in 4a) |
| The MN/BN/DBN/HMM `"pyomo"` and (4b) `"toulbar2"` paths never pass the constrained model's `data` to constraint functions (only network flow does), so `Constrained*(..., data=d)` is ignored there | 5 (found in 4a; decide whether to merge `cmodel.data` into the per-query data) |
| `ConstrainedDynamicDiscreteBayesianNetwork.nodes` is not a property | 3 (fixed, regression test in `constrained_models/tests/test_dynamic_bayesian_network.py`) |
| `a_star` on an unconstrained `ConstrainedHiddenMarkovModel` raised `AttributeError` (a bare `CHMM` has no `is_feasible`) | 3 (fixed as a side effect of the "oracle" repn) |
| `mcem` calls `app.generate_hidden` twice per sample (it scores one sequence and trains on another) | 4c |
| `HMMApplication.create_chmm("pyomo")` attaches no `data`, so constraints that read application data (e.g. `data.num_zeros` in test_application_chmm) would fail if solved | 4c |
| `MVR/formula.ipynb` prose says `MVR_HMM`/`MVRFunctor` (typos for `MVR_CHMM`/`MVRConstraint`) | 4d |
| `DynamicDiscreteBayesianNetwork.t` is public and gets mutated by `create_bn_from_dbn` and `DBNSampler` (an immutability leak) | 1 (fixed: `create_bn_from_dbn` restores `t`; `DBNSampler` works on a deep copy) |
| `conin.io.save_model.save_conin_model` (was `common.conin.save_model`): the `.bif` branch writes the file but doesn't return, so it then raises "Unknown format" (`convert_pgm --bif` avoids it by saving with `model_type="pgmpy"`) | 6 (found in 4f) |
| `backends.rst` calls smoek optional although pyproject requires it; torch and pytoulbar2 aren't declared as extras | 6 |
| `create_bn_from_dbn(start=s)` with `s > 0` fails (`AttributeError: 'float' object has no attribute 'keys'`): `_unroll_cpds` skips the transition CPDs at `t = start` (their parents are at `start - 1`), so the time-`start` nodes keep the initial CPD whose values are not a dict. Every backend that unrolls DBNs/HMMs (pyomo, toulbar2, VE) is affected; `map_query(..., start=1)` has never worked | 6 (found in 4e; decide whether `start > 0` should be supported at all) |
| VE on the unconstrained `weather` DBN (`stop=4`) runs out of memory under pgmpy (the VE weather tests in `test_MAP_Inference.py` are commented out) | – (found in 4e; pgmpy limitation, not a conin bug; noted for PR 5's VE docs) |

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
| HMMApplication | Keep it, move it to `applications/`, decouple it via `RecursiveConstraint` |
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
| MVR in PR 3 | `MVR_CHMM` and the torch algorithms are untouched until 4d; the `"mvr"` repn wraps `MVR_CHMM` and already exposes `.mvrs` |
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
| Factor structure params (PR 4e) | `()` for MN/BN; `start=0, stop=1` for DBN; `start=0, stop` (required) for HMM, as for "pyomo"/"toulbar2". The HMM dispatcher keeps `stop = len(evidence) - 1` (it ignores `stop`, as before) |
| VE constraint `data` (PR 4e) | Unchanged: MN/BN `None`; DBN `Munch(T)`; HMM `Munch(hmm=Munch(T))`. It holds no evidence, so constraints are materialised once at build time; the query evidence is merged per query (`query_evidence`). Aligning it with the Pyomo/toulbar2 layout is left to PR 5 (with the `cmodel.data` row in §9) |
| VE public names (PR 4e) | `conin.inference.variable_elimination` (`solve.py`, `mn.py`, `bn.py`, `dbn.py`, `hmm.py`, `custom_VE.py`) with `inference_variable_elimination_map_query_{MN,BN,DDBN,HMM}` and `solve_variable_elimination_map_query_model`; `conin.representations.factor` with `FactorRepresentation`, `Factor{MN,BN,DBN,HMM}Representation`, `apply_factor_constraints`. `custom_VE.py` was moved here in 4e and deleted in the 4e follow-up (dead code; the "import cycle" that motivated a lazy import in `log_prob_v2` never broke imports, but it was a layering inversion: `common` importing `inference`) |
| Total-assignment scores (4e follow-up) | `conin.common.log_probability(model, x, *, observed=None)` (BN, DBN, HMM: exact `log P(x)`; refuses MN and constrained models) and `conin.common.log_potential` (same args; also MN = `log P + log Z`, and constrained models = `-inf` if infeasible, else the unconstrained value, not renormalised). DBN assignments are `{(name, t): s}` over `0..stop` (unrolled with `create_bn_from_dbn(start=0)`); HMM assignments are the hidden sequence (list or dict keyed by time, as in `map_query` solutions) plus `observed`. No `evidence` argument. pgmpy models are converted to conin. Both stay in `common/` (no inference) |
| Import layering (4e follow-up 2) | No import cycles between modules: every function-level import of a conin module that a cycle needed was removed or hoisted. `PyomoVarWrapper` lives in `conin/util/var_wrapper.py` (`conin.constraints.algebraic` needs it, and `conin.representations.pyomo` imports `conin.constraints`); its public name stays `conin.representations.pyomo.PyomoVarWrapper`. Still function-level by choice: the `conin.common.pgmpy` imports in `conin/examples/*` and `conin.inference.*` (inside `try_import`/pgmpy-only paths), `representations/{oracle,mvr}/hmm.py` → `hidden_markov_model.chmm_*`, and `decorators.py`'s doctest import of `conin`. `conin/__init__.py` imports `inference.map_query` eagerly, so importing any submodule loads all of inference (≈7 s here); making that lazy is left for later |
| Feasibility checks (4e follow-up) | `is_feasible` on every constrained model, in `constrained_models/feasibility.py`: `"oracle"`, `"factor_oracle"`, `"algebraic"` (smoek-compiled predicate, as for VE) and `"mvr"` (the torch algorithms' window semantics) only. `"pyomo"`/`"toulbar2"` raise `NotImplementedError` (a function may add variables, making the check an IP/CFN feasibility problem). The constraint `data` is the `"factor"` representation's |
| Factor-repn helpers (PR 4e) | `inference/mn/factor_repn.py` → `representations/factor/factor_repn.py`; `factor_representation_of_mn`, `skeleton_mn_of_bn`, `tuple_var_index_map` move there from `representations/pyomo/common.py` (the pyomo builders import them) |
| MN constraint factor (PR 4e) | The MN applier adds the materialised factor (0 on infeasible assignments) and no evidence. The old code also fixed the last scope node to `1`, which was the §9 bug |
| Constructor names / CHMM overloads | First parameter is `model` (no `hmm=`/`pgm=` alias); the broken `CHMM` overloads in the AStar/Optimization/CFN dispatchers are removed in PR 3 |

---

## 12. Open questions

None blocking. These can be settled during PR 3 (the design PR):
- (Pyomo `get_copy()`: settled in PR 4a, rebuild; see §11. toulbar2: settled in PR 4b, UAI text + fresh CFN. The cache and `get_copy()` were removed after 4b.)
- **Where the cross-backend per-model inference tests go once `inference/{mn,bn,dbn}/` hold no code** (from 4b; after 4e
  they now hold no code: empty `__init__` + `tests/`). `conin/inference/{mn,bn,dbn}/tests/` exercise pyomo, toulbar2, VE and `map_query`. Options:
  `inference/tests/{mn,bn,dbn}/`, flattened `inference/tests/test_<model>_*.py`, or keep the folders for tests only.
  Decide in a follow-up after 4e.
- **Marginal and conditional log-probabilities** (`log P(X=x | E=e)` with the other variables summed out, and the
  probability under constraints, `P(x) / P(C)`): a new inference query, in the spirit of `map_query`. Not started; the
  deleted `log_probability`/`log_prob_v2`/`logprob_VE` were drafts of it (pgmpy VE). Decide after PR 5 whether it goes
  through `map_query`-style dispatch and which backends support it.
- **Where `log_probability`/`log_potential` live.** They sit in `conin/common/`, which now imports `conin.constrained_models`
  (no cycle, but `common` is no longer a bottom layer). Option: a `conin/scoring/` package beside `inference/`, as a
  separate PR, together with the marginal/conditional query above. Not decided.
- (`with_model` cache behaviour, the `Representation` signatures and the registration decorator were settled in PR 3; see §11.)
