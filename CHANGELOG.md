Conin CHANGELOG
===============

-------------------------------------------------------------------------------
Conin 2.0
-------------------------------------------------------------------------------

Conin 2.0 reorganises the package. Unconstrained models, constraints,
constrained models, backend representations and algorithms now live in
separate packages, and every inference method is reached through one
dispatch. Old import paths were removed without compatibility aliases; the
table below lists the moves.

Highlights

- Unconstrained graphical models are immutable and live in
  conin.graphical_models; constrained models are immutable
  (model, constraints) pairs in conin.constrained_models.
- Backend representations (conin.representations) are built on demand with
  cmodel.representation(kind); a registry decides which constraint families
  each model type accepts.
- map_query normalises every model (bare, constrained or pgmpy) and dispatches
  on the representation, for every method; new method="mvr_viterbi".
- New conin.scoring.log_probability_query: marginal and conditional
  log-probabilities, also under constraints.
- New RecursiveConstraint for recursive A*; map_query(method="a_star") uses it.
- Algorithms are organised by backend under conin.inference, conin.learning
  and conin.sampling; the torch MVR code no longer loads with import conin.
- pgmax, pomegranate and pyagrum support is removed.

Migration table (old -> new)

  conin.markov_network, conin.bayesian_network,
  conin.dynamic_bayesian_network, conin.hidden_markov_model (models)
                                    -> conin.graphical_models
  Constrained* classes              -> conin.constrained_models
  */examples.py                     -> conin.examples.{markov_network,
                                       bayesian_network,
                                       dynamic_bayesian_network,
                                       hidden_markov_model}
  ConstraintFunctor                 -> conin.constraints.Constraint
  conin.constraints.oracle (library: all_diff, citation, ...)
                                    -> conin.constraints.oracle_library
  OracleConstraint(nodes=...)       -> conin.constraints.FactorOracleConstraint
  hidden_markov_model.mvr, mvr_constraints, mvr_operators, conin.operators,
  mvr_formula                       -> conin.constraints.mvr (base, primitives,
                                       operators, formula)
  CHMM, Oracle_CHMM, MVR_CHMM, PyomoAlgebraic_CHMM, Algebraic_CHMM,
  initialize_chmm(), chmm.chmm      -> ConstrainedHiddenMarkovModel(hmm,
                                       constraints, data=...) and
                                       cmodel.representation(kind)
  chmm.chmm.constraints (MVRs)      -> cmodel.representation("mvr").mvrs
  HMM_MatVecRepn, hmm.repn, hmm.initialize(), initialize=True
                                    -> ConstrainedHiddenMarkovModel(hmm)
                                       .representation("matvec")
  hmm.generate_hidden(T) etc.       -> conin.sampling.unconstrained
                                       .generate_hidden(hmm, T) etc.
  chmm.generate_hidden(T)           -> conin.sampling.oracle
                                       .rejection_sample_hidden(chmm, T)
  conin.sampling.dbn.DBNSampler     -> conin.sampling.unconstrained.DBNSampler
  viterbi                           -> conin.inference.unconstrained.viterbi
  a_star                            -> conin.inference.oracle.a_star
  recursive_a_star(hmm_app=...)     -> conin.inference.recursive
                                       .recursive_a_star(hmm=...)
  HMMApplication                    -> ConstrainedHiddenMarkovModel with
                                       RecursiveConstraint or data=...
  supervised_learning, mcem, add_unknowns, convert_to_simulations
                                    -> conin.learning
  inference.viterbi_mvr, inference.marginal_map_mvr,
  other_queries.sat_prob_mvr, other_queries.sat_time_mvr
                                    -> conin.inference.mvr
  learning.baum_welch_mvr, learning.generalized_em_mvr
                                    -> conin.learning.mvr
  sampling.ffbs_mvr, sampling.stopped_sampling_mvr
                                    -> conin.sampling.mvr
  hidden_markov_model lp_inference/ip_inference
                                    -> conin.inference.pyomo
  inference_pyomo_map_query_{MN,BN,DDBN,HMM}(pgm=m, ...)
                                    -> map_query(m, method="integer_program",
                                       ...)
  inference_toulbar2_map_query_{MN,BN,DDBN,HMM}(pgm=m, ...)
                                    -> map_query(m, method="toulbar2", ...)
  inference_variable_elimination_map_query_{MN,BN,DDBN,HMM}(pgm=m, ...)
                                    -> map_query(m,
                                       method="variable_elimination", ...)
  VariableElimination (VE dispatch) -> map_query(m,
                                       method="variable_elimination")
  create_pyomo_map_query_model_*    -> cmodel.representation("pyomo", ...)
                                       .create_model(...)
  create_toulbar2_map_query_model_* -> cmodel.representation("toulbar2", ...)
                                       .create_model(...)
  conin.markov_network.factor_repn  -> conin.representations.pyomo.factor_repn
  PyomoVarWrapper, add_c5_constraints, C5LazyCutGenerator, ...
                                    -> conin.representations.pyomo(.common)
  Toulbar2VarWrapper, add_toulbar2_constraints, ...
                                    -> conin.representations.toulbar2(.common)
  conin.common.load_model/save_model, conin.common.conin.* loaders/savers
                                    -> conin.io (load_model, save_model,
                                       uai, cfn, pgmpy_files)
  conin.common.pgmpy (convert_pgmpy_to_conin, MapCPD, is_polytree)
                                    -> conin.pgmpy
  log_probability, log_potential    -> conin.scoring
  MAP solution keys: hidden         -> states
                     log_likelihood, log_factor_sum
                                    -> log_potential (the same score for
                                       every method)
                     primal_bound, log_factor_sum (objective)
                                    -> solver_objective
  MAP result keys: observed         -> removed (the observations passed in)
  recursive_a_star knowledge_state_times, objective_values
                                    -> removed (use solutions[i].log_potential)
  conin.io.load_conin_model_from_uai, save_model_uai,
  load_conin_model_from_cfn         -> conin.io.load_uai, save_uai, load_cfn
  create_MN_pyomo_map_query_model_from_factorial_repn
                                    -> private (_create_pyomo_map_query_model);
                                       use cmodel.representation("pyomo")
                                       .create_model()
  conin.inference.methods           -> conin.representations.methods

Models

- The unconstrained model classes (HiddenMarkovModel, DiscreteBayesianNetwork,
  DiscreteMarkovNetwork, DynamicDiscreteBayesianNetwork, DiscreteCPD,
  DiscreteFactor) are immutable after construction. States, CPDs and factors
  are passed to the constructors; add_cpd/add_factor are replaced by the
  non-mutating with_cpd/with_factor. HiddenMarkovModel.load_model is removed:
  pass the required, non-empty start_probs, transition_probs and
  emission_probs to the constructor; internal state indices follow the sorted
  order of the state labels. DiscreteBayesianNetwork and
  DynamicDiscreteBayesianNetwork require a non-empty cpds argument and
  DiscreteMarkovNetwork a non-empty factors argument; a DBN also needs states
  or dynamic_states. Time-indexed DBN CPDs are written against an
  ExpressionVariable (conin.graphical_models) passed to the constructor as t.
  read_from_file is a classmethod returning a new instance; make_non_zero
  returns a new instance.
- HiddenMarkovModel checks its probabilities at construction (nonnegative,
  rows summing to 1) and raises InvalidInputError; before, they were checked
  only when the numeric representation was first built.
- HMM_MatVecRepn, HiddenMarkovModel.repn, initialize() and the initialize=
  argument are removed. The parameters as read-only float64 NumPy arrays on
  internal indices (plus their logarithms) are
  ConstrainedHiddenMarkovModel(hmm).representation("matvec"), a
  MatVecHMMRepresentation; hmm.start_vec etc. still return lists.
- create_bn_from_dbn and the code unrolling DBNs and HMMs use the DBN's .t
  without changing it. create_bn_from_dbn unrolls one time step
  (stop == start); start must be 0 (InvalidInputError otherwise).

Constrained models

- Constrained models are immutable and share one constructor,
  Constrained*(model, constraints=(), *, data=None): no hmm=/pgm= keywords, no
  constraints setters, no add_constraint, and .constraints is a tuple. Use
  with_constraints/with_model to derive new ones, and ensure(model) to wrap a
  bare model.
- .model replaces .pgm and .hidden_markov_model (no aliases).
- A single Constraint is accepted. Mixed constraint families, non-Constraint
  objects, a string and a wrong model type raise InvalidInputError at
  construction.
- initialize_chmm(), chmm.chmm, constraint_type and supported_families are
  removed. constraint_family is the constraints' family ("oracle_with_nodes"
  is now "factor_oracle"). The accepted families come from the representation
  registry: MN, BN and DBN accept "pyomo", "toulbar2", "algebraic" and
  "factor_oracle" (they now refuse "oracle", "mvr" and "recursive"); HMMs
  also accept "oracle", "recursive" and "mvr".
- The constrained model's data reaches the constraint functions on every
  backend, as a shallow copy with the backend's fields (T, hmm, start, stop,
  variables, evidence, ...) set. The time fields are the same for every
  backend and for is_feasible: start, stop and T for a DBN; start, stop and
  hmm = Munch(T, hidden_to_internal) for an HMM (the pgmpy, oracle and
  is_feasible data had only T or hmm.T, and network flow no start/stop).
- is_feasible on every constrained class checks oracle, factor-oracle,
  algebraic, recursive and MVR constraints (pyomo/toulbar2 raise
  NotImplementedError); the HMM version accepts a dict keyed by time.
- ConstrainedDynamicDiscreteBayesianNetwork.nodes is a property.
- ConstrainedHiddenMarkovModel.generate_hidden/generate_observed/
  generate_observed_from_hidden are removed (see Sampling).

Constraints

- ConstraintFunctor is renamed Constraint (no alias) and no longer requires
  __call__. Each constraint class has a family.
- OracleConstraint is a predicate on {key: state} (optional partial_func for
  A* pruning); OracleConstraint(nodes=...) becomes FactorOracleConstraint,
  whose call evaluates the predicate (it no longer materialises a factor).
  oracle_constraint_fn still returns the right class. The "all-nodes"
  materialisation mode is removed.
- New RecursiveConstraint(initial=, update=, feasible=, feasible_partial=)
  for recursive A*.
- The oracle library moves to conin.constraints.oracle_library and is no
  longer star-exported from conin.constraints. The partial functions of
  occurs_only_in_time_frame_constraint and
  occurs_at_least_once_in_time_frame_constraint no longer reject completable
  prefixes.
- The MVR modules move to conin.constraints.mvr. mvr_timerange returns a new
  MVR (inplace=True for the old behaviour). MVR_MatVecRepn arrays are
  read-only copies.

Representations (new)

- conin.representations: Representation, register_representation,
  build_representation, supported_representations; kinds "pyomo",
  "pyomo_network_flow", "toulbar2", "pgmpy", "matvec", "oracle",
  "recursive", "mvr". Every representation() call builds a new object;
  structure parameters (start, stop, ...) are build arguments and per-query
  inputs go to create_model().
- representation("oracle") is an OracleHMMRepresentation (model, hmm,
  constraints), representation("mvr") an MVRHMMRepresentation (model, mvrs:
  deep copies of prebuilt MVRs, aligned to the HMM) and
  representation("pyomo_network_flow") a PyomoNetworkFlowHMMRepresentation
  whose create_model(observed=) returns (model, data); their .chmm, .hmm,
  .data and generate_* attributes are gone.
- On network-flow models, M.V is a PyomoVarWrapper: a missing variable raises
  KeyError and a wrong number of arguments ValueError (they returned None and
  a KeyError).
- The DBN and HMM representations of the pyomo, toulbar2 and pgmpy backends
  share their time range (start, stop, time_steps, the constraint data) through
  the mixins of conin.representations.unrolled; the DBN builders default stop
  to 1 and the HMM builders require it. The backend appliers share their
  dispatch (conin.representations.base.apply_constraints): constraints of
  several families, or of a family the backend does not apply, raise TypeError
  with the same message for every backend.
- Algebraic constraints without smoek raise ImportError for every backend (the
  pyomo and toulbar2 translators raised TypeError).

Inference

- map_query normalises every model and dispatches on the representation; the
  private dispatcher modules (OptimizationInference, CFNInference,
  AStarInference, ViterbiInference, VariableElimination) and the empty
  conin.inference.{mn,bn,dbn,hmm} packages are removed. CHMM inputs are no
  longer accepted.
- New method="mvr_viterbi" (torch): the solution also has augmented; dict
  evidence is sparse; start must be 0.
- method="a_star" with RecursiveConstraints runs recursive_a_star.
  recursive_a_star takes hmm= (a constrained HMM with RecursiveConstraints or
  a bare HMM); a_star_(..., chmm=) is a_star_(..., oracle=).
- A* prunes with the oracle constraints' partial_func (it never did), so it
  needs fewer iterations; with no solution it returns solution=None and
  termination_condition="error: no feasible solutions" (it raised IndexError).
  Oracle predicates that take (states, data) work with A* and rejection
  sampling (they raised TypeError).
- HMM observations are read one way everywhere: a non-empty list or tuple, or
  a dict keyed by exactly 0..T-1 (InvalidInputError otherwise; evidence=None
  is refused for every method). integer_program, toulbar2 and
  variable_elimination infer over start..stop (VE ignored stop); a stop past
  the last observation raises InvalidInputError.
- viterbi accepts an unconstrained ConstrainedHiddenMarkovModel; viterbi and
  a_star no longer accept an HMM_MatVecRepn. viterbi/a_star on an MN, BN or
  DBN raise InvalidInputError (was TypeError).
- integer_program: ip_formulation is understood for every model type;
  prune_network=False/create_MN=False are accepted; network flow refuses
  unsupported evidence with InvalidInputError (it returned None) and uses
  lp_inference for an unconstrained ConstrainedHiddenMarkovModel.
- toulbar2: works for every model type with verbose=; without pytoulbar2 it
  raises ImportError for every model type, as the other backends do (it
  returned a result with termination_condition "pytoulbar2 not available"); a
  partial variables query raises RuntimeError (it was ignored);
  write_uai_file writes exactly that file.
- DBN queries with stop=None unroll over 0..1 for every method.
- variable_elimination: pgmpy inputs are converted to CONIN like on every
  other path; on a constrained Markov network it keeps the user evidence and
  no longer forces a scope node to 1; an unsupported constraint family raises
  InvalidInputError (was TypeError).
- The query-check messages changed: a partial variables query raises
  RuntimeError("Partial MAP queries are only supported by
  method='variable_elimination'; ..."), and prune_network/create_MN=True raise
  RuntimeError("<option>=True is not supported for CONIN models").
- integer_program, toulbar2 and variable_elimination have one solver per
  backend for MN, BN and DBN, and solve an HMM as its unrolled DBN, so the
  options mean the same for every model type: prune_network/create_MN are
  checked on Markov networks too (they raised TypeError), timing=True is
  honoured by the integer program on DBNs and HMMs (it was ignored), and
  solution_with_evidence=True works for an HMM integer program with list
  evidence (it raised AttributeError).
- The per-backend inference_pyomo_map_query_*, inference_toulbar2_map_query_*
  and inference_variable_elimination_map_query_* functions (MN, BN, DDBN,
  HMM) are removed; call map_query(pgm, method=...) instead (see the
  migration table). The model-solving functions solve_pyomo_map_query_model,
  solve_toulbar2_map_query_model, solve_variable_elimination_map_query_model,
  lp_inference and ip_inference stay.
- Every MAP result has the same keys for every method (conin.inference.results):
  solution, solutions, termination_condition and solvetime (viterbi, a_star,
  recursive_a_star and mvr_viterbi had no solvetime; variable_elimination had
  no solutions or termination_condition; the HMM methods returned observed;
  recursive_a_star had no solution). Every solution has states,
  log_potential and solver_objective. HMM solutions no longer have hidden (it
  was the same as states). log_potential is conin.scoring.log_potential of the
  assignment with the evidence on the model without its constraints (for an
  HMM, log P(hidden, observed)), so it is the same for every method; it
  replaces log_likelihood (viterbi, A*, network flow, mvr_viterbi) and
  log_factor_sum (integer program; it was the IP objective, whose factors are
  normalised per factor, so it was not log P(x): -10.21 instead of -3.97 for
  the same HMM path). The backend's objective is solver_objective (the IP
  objective, or toulbar2's primal bound, an integer cost). A partial MAP query
  (variable_elimination with variables=) gets log_potential None, as do the
  solve_*_map_query_model functions, which do not see the model.
- The or_topas solutions of an HMM integer program (ip_formulation=
  "markov_network") are all converted to hidden states (only the first was;
  the others kept the unrolled ("H", t)/("E", t) keys).

Scoring

- log_probability and log_potential move to conin.scoring. log_potential
  accepts DBNs, HMMs, pgmpy models and constrained models (-inf when
  infeasible); evidence= is removed (it returned the joint); an assignment
  that is not total raises InvalidInputError; missing factor entries use the
  factor's default_value. New log_probability (BN, DBN, HMM).
- New log_probability_query(pgm, assignment, *, evidence=None,
  method="variable_elimination", start=0, stop=None): log P(x | e), under the
  constraints for a constrained model (needs pgmpy).

Sampling and learning

- HMM sampling functions: conin.sampling.unconstrained.generate_hidden(hmm,
  T), generate_hidden_until_state, generate_observed_from_hidden,
  generate_observed (same draws from NumPy's global state; hmm.set_seed
  stays). Constrained HMMs: conin.sampling.oracle.rejection_sample_hidden.
- mcem(app=...) takes a constrained (or bare) HMM, samples from the
  re-estimated HMM each iteration and trains on the samples it scores (its
  results change), and no longer writes the HMM back to app.
- The torch MVR algorithms keep their names and signatures and take a
  ConstrainedHiddenMarkovModel with MVR constraints (or a bare HMM); other
  constraint families raise InvalidInputError.

I/O and conversion

- File I/O moves from conin.common to conin.io; load_model/save_model accept
  model_type="conin" or "pgmpy". The convert_pgm script is
  conin.io.convert_pgm:main (same name and options).
- pgmpy conversion (convert_pgmpy_to_conin, MapCPD, is_polytree for pgmpy
  models) moves from conin.common.pgmpy to conin.pgmpy, which also has the
  new convert_conin_to_pgmpy_bn and convert_conin_to_pgmpy_mn. The new
  conin.graphical_models.is_polytree checks CONIN Bayesian networks.
- pgmax, pomegranate and pyagrum support (converters, loaders, model_type
  values and the pgmax/pomegranate/pyagrum/pgm-all extras) is removed;
  load_model with those model_type values raises RuntimeError("Unexpected
  model type").
- conin.common.pgmpy.log_potential and conin.common.pgmpy.custom_VE (dead,
  untested code) are removed.
- conin/mediation_variables (an unused NumPy prototype of the mediation
  variable algorithms, with its notebooks) is removed; the MVR algorithms are
  in conin.inference.mvr, conin.learning.mvr and conin.sampling.mvr.
- The format loaders and saver are renamed to match: load_uai, save_uai and
  load_cfn (were load_conin_model_from_uai, save_model_uai and
  load_conin_model_from_cfn); load_model/save_model are unchanged.
- New extras: toulbar2 (pytoulbar2) and torch.

Bug fixes

- save_uai (then save_model_uai) replaces the file extension with os.path.splitext: a dot in
  a directory name, or a "./" prefix, wrote another file.
- Saving a CONIN model to a .bif file no longer raises "Unknown format" after
  writing it.
- toulbar2 queries work after reading a .cfn file with pytoulbar2 (its global
  input-format options are reset).
- solver="or_topas" no longer mutates the caller's solver_options.
- convert_pgmpy_to_conin keeps one CPD per template when pgmpy holds a CPD
  for several time slices (the unrolled network had two CPDs per node).
- map_query on a one-observation HMM (integer_program, toulbar2,
  variable_elimination) no longer raises AssertionError; create_bn_from_dbn
  with start > 0 raises InvalidInputError instead of AttributeError.
- The pgmpy_issue_1177_pgmpy example works with pgmpy 1.1 (DiscreteMLE()).
- The network-flow integer program with solver="or_topas" returns solutions
  with states (they had variable_value instead, so dict evidence failed).
- A* and recursive A* report "max_iterations: N" or "max_time: ..." in
  termination_condition when a limit stops the search (it was overwritten
  with "error: no feasible solutions").
- generate_hidden, generate_observed and rejection_sample_hidden with
  time_steps=0 return an empty list (they returned one state).
- Invalid user input raises an exception instead of failing an assert (the
  checks were skipped under python -O):
  - DiscreteMarkovNetwork.check_model and DiscreteBayesianNetwork.check_model
    raise InvalidInputError (was AssertionError, or RuntimeError for an edge
    to an unknown node);
  - the "pyomo", "toulbar2" and "pgmpy" HMM representations without stop,
    observations that no hidden state can emit in the network-flow model,
    lp_inference with num_solutions != 1, and supervised_learning/mcem without
    hidden or observable states raise InvalidInputError;
  - or_topas without solutions raises InsufficientSolutionsError;
  - a mixed list of pyomo or toulbar2 constraints raises TypeError;
  - a UAI file with an inconsistent function table raises ValueError, a
    missing UAI file RuntimeError (as for CFN files), and MapCPD argument
    errors ValueError.
- pgmpy save_model with an unknown file suffix raises RuntimeError (it
  returned without writing), and the load_model message names the file.
- The timing=True output of variable elimination prints its STOP line with
  the elapsed time.
- mcem stops when the average log-probability converges (it never updated
  the previous value, so it always ran max_iterations iterations), and no
  longer asserts that the Monte Carlo estimate never increases.
- The UAI reader joins every pair of nodes in a Markov-network factor (it
  joined only the last node to the others, so pgmpy rejected the converted
  model: saving such a model as .bif failed). A truncated UAI file raises
  ValueError (was StopIteration). Saving a DBN with dynamic nodes as UAI
  raises InvalidInputError asking to unroll it (was KeyError).
- The CFN reader rejects an object as a variable domain and a cost function
  whose costs are all infinite (it gave NaN factor values).
- mvr_precedence rejects MVRs that are neither HomMVR nor InhomMVR
  (InvalidInputError; was TypeError), and an MVR constraint function without
  arguments raises ValueError when it is defined (it failed when called).
- HomMVR and InhomMVR reject empty hidden or mediation state lists
  (InvalidInputError; an empty HomMVR failed in NumPy with "argmax of an
  empty sequence"). mvr_kfold_product, mvr_kfold_product_prefix and
  mvr_operator_fn(arity=...) reject True and False as integers.
- MVR_MatVecRepn accepts plain nested lists for a homogeneous MVR, and
  raises InvalidInputError naming the array for a ragged or non-numeric one
  (it raised a NumPy ValueError, or a misleading "must be a 3D array").

Known issues

- Variable elimination runs in pgmpy and can run out of memory on larger
  unrolled models (e.g. the unconstrained weather DBN with stop=4).
- recursive_a_star with num_solutions > 1 can skip sequences (prefixes with
  equal constraint state are merged); called directly, it returns solutions
  with hidden only and no solution.
- HiddenMarkovModel.log_probability raises ValueError for a zero-probability
  sequence (conin.scoring.log_probability returns -inf).
- DBNs and HMMs can only be unrolled from start=0.

-------------------------------------------------------------------------------
Conin 1.1.1
-------------------------------------------------------------------------------

- Fixed pypi README format.

-------------------------------------------------------------------------------
Conin 1.1
-------------------------------------------------------------------------------

- Fix issue where evidence was being ignored for MPE optimization
- Added the "V()" functor to simplify the expression of constraints
- Added converter from conin BN to pgmpy BN
- Added converter from HMM to DBN
- Documentation updates, including conin logo
- Added GitHub actions for testing
- Added ability to enumerate alternative MPE solutions using optimization 
- Integrated Toulbar2 solver for MPE inference
- Added constraint function decorators
- Added UAI writer

-------------------------------------------------------------------------------
Conin 1.0
-------------------------------------------------------------------------------

- Initial release.
