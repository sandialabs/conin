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
  conin.inference.{mn,bn,dbn,hmm} Pyomo functions,
  hidden_markov_model lp_inference/ip_inference
                                    -> conin.inference.pyomo
  conin.inference.{mn,bn,dbn,hmm} toulbar2 functions
                                    -> conin.inference.toulbar2
  VariableElimination (VE dispatch) -> conin.inference.variable_elimination
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
  variables, evidence, ...) set.
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

Inference

- map_query normalises every model and dispatches on the representation; the
  private dispatcher modules (OptimizationInference, CFNInference,
  AStarInference, ViterbiInference, VariableElimination) and the empty
  conin.inference.{mn,bn,dbn,hmm} packages are removed. CHMM inputs are no
  longer accepted.
- New method="mvr_viterbi" (torch): the solution also has log_likelihood and
  augmented; dict evidence is sparse; start must be 0.
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
- toulbar2: works for every model type with verbose=; the DBN/HMM paths
  return the "pytoulbar2 not available" result without pytoulbar2; a partial
  variables query raises RuntimeError (it was ignored); write_uai_file writes
  exactly that file.
- DBN queries with stop=None unroll over 0..1 for every method.
- variable_elimination: pgmpy inputs are converted to CONIN like on every
  other path; on a constrained Markov network it keeps the user evidence and
  no longer forces a scope node to 1; an unsupported constraint family raises
  InvalidInputError (was TypeError).
- The query-check messages changed: a partial variables query raises
  RuntimeError("Partial MAP queries are only supported by
  method='variable_elimination'; ..."), and prune_network/create_MN=True raise
  RuntimeError("<option>=True is not supported for CONIN models").

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
- New extras: toulbar2 (pytoulbar2) and torch.

Bug fixes

- save_model_uai replaces the file extension with os.path.splitext: a dot in
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
