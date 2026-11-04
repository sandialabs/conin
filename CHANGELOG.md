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
- map_query normalises every model (bare or constrained) and dispatches on
  the representation, for every method; new method="mvr_viterbi".
- Variable elimination is CONIN's own (numpy, in log space), so
  map_query(method="variable_elimination") and log_probability_query no
  longer need pgmpy. pgmpy is used only by conin.io.pgmpy (BIF files and
  model conversion); the algorithms take CONIN models only.
- New conin.algorithms.scoring.log_probability_query: marginal and conditional
  log-probabilities, also under constraints.
- New RecursiveConstraint for recursive A*; map_query(method="a_star") uses it.
- Algorithms are organised by backend under conin.algorithms.inference,
  conin.algorithms.learning and conin.algorithms.sampling, with scoring in
  conin.algorithms.scoring and the feasibility checks in
  conin.algorithms.feasibility; the torch MVR code no longer loads with
  import conin.
- The top-level conin package exports the everyday API (conin.__all__): the
  model and constrained model classes, the constraint helpers, map_query,
  log_probability_query, log_probability, log_potential, is_feasible,
  load_model, save_model and the exceptions, so
  "from conin import HiddenMarkovModel, map_query" works. The docs and
  notebooks import from conin.
- import conin is several times faster: it no longer imports pgmpy (and
  with it torch, scikit-learn and pandas) or looks for a MIP solver. pgmpy
  is imported the first time a pgmpy converter or file reader is used.
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
  ExpressionVariable (DBN time index)
                                    -> conin.graphical_models.TimeIndex
  ConstraintFunctor                 -> conin.constraints.Constraint
  algebraic_constraint_fn, AlgebraicConstraint, family "algebraic",
  conin.constraints.algebraic, *_algebraic* examples
                                    -> smoek_constraint_fn, SmoekConstraint,
                                       family "smoek", conin.constraints.smoek,
                                       *_smoek* examples
  conin.constraints.oracle (library: all_diff, citation, ...)
                                    -> conin.constraints.oracle.library
  OracleConstraint(nodes=...)       -> conin.constraints.FactorOracleConstraint
  conin.constraints.factor_oracle   -> conin.constraints.oracle
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
  hmm.generate_hidden(T) etc.       -> conin.algorithms.sampling.unconstrained
                                       .generate_hidden(hmm, T) etc.
  chmm.generate_hidden(T)           -> conin.algorithms.sampling.oracle
                                       .rejection_sample_hidden(chmm, T)
  chmm.is_feasible(seq), chmm.partial_is_feasible(T=T, seq=seq)
                                    -> conin.algorithms.feasibility
                                       .is_feasible(chmm, seq),
                                       .partial_is_feasible(chmm, T=T, seq=seq)
  conin.sampling.dbn.DBNSampler     -> conin.algorithms.sampling.unconstrained
                                       .DBNSampler
  viterbi                           -> conin.algorithms.inference.unconstrained
                                       .viterbi
  a_star                            -> conin.algorithms.inference.oracle.a_star
  recursive_a_star(hmm_app=...)     -> conin.algorithms.inference.recursive
                                       .recursive_a_star(hmm=...)
  HMMApplication                    -> ConstrainedHiddenMarkovModel with
                                       RecursiveConstraint or data=...
  supervised_learning, mcem, add_unknowns, convert_to_simulations
                                    -> conin.algorithms.learning
  inference.viterbi_mvr, inference.marginal_map_mvr,
  other_queries.sat_prob_mvr, other_queries.sat_time_mvr
                                    -> conin.algorithms.inference.mvr
  learning.baum_welch_mvr, learning.generalized_em_mvr
                                    -> conin.algorithms.learning.mvr
  sampling.ffbs_mvr, sampling.stopped_sampling_mvr
                                    -> conin.algorithms.sampling.mvr
  hidden_markov_model lp_inference/ip_inference
                                    -> conin.algorithms.inference.pyomo
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
                                       uai, cfn, pgmpy.files)
  conin.common.pgmpy.convert_pgmpy_to_conin
                                    -> conin.io.pgmpy.convert_pgmpy_to_conin
  conin.common.pgmpy.MapCPD, is_polytree (pgmpy models), *_pgmpy examples
                                    -> removed
  log_probability, log_potential    -> conin.algorithms.scoring
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
  emission_probs to the constructor (see below for the state order). DiscreteBayesianNetwork and
  DynamicDiscreteBayesianNetwork require a non-empty cpds argument and
  DiscreteMarkovNetwork a non-empty factors argument; a DBN also needs states
  or dynamic_states. Time-indexed DBN CPDs are written against an
  TimeIndex (conin.graphical_models) passed to the constructor as t.
  read_from_file is a classmethod returning a new instance; make_non_zero
  returns a new instance.
- The data of the MN, BN and DBN classes and of DiscreteFactor/DiscreteCPD
  is read-only: nodes, edges, states_of(node), the state lists in states and
  dynamic_states, factor and CPD values (also the nested CPD tables), factor
  nodes and CPD parents are FrozenList/FrozenDict (conin.util.frozen), list
  and dict subclasses whose mutating methods raise TypeError. They compare
  equal to and print like lists and dicts; copy them with list()/dict() to
  change them. The constructors copy their arguments, so changing the lists
  and dicts passed in no longer changes the model. states, dynamic_states,
  factors and cpds still return new dicts and lists. Before, e.g.
  mn.nodes.append(x) or mn.factors[0].values[key] = v silently changed the
  model. dbn.t cannot be reassigned. The inferred edges of a Markov network
  (and the edges of a Bayesian network) are computed at construction.
- HiddenMarkovModel checks its probabilities at construction (nonnegative,
  rows summing to 1) and raises InvalidInputError; before, they were checked
  only when the numeric representation was first built.
- HMM_MatVecRepn, HiddenMarkovModel.repn, initialize() and the initialize=
  argument are removed. The parameters as read-only float64 NumPy arrays on
  internal indices (plus their logarithms) are hmm.representation("matvec"),
  a MatVecHMMRepresentation.
- HiddenMarkovModel has no index view: start_vec, transition_mat,
  emission_mat, hidden_to_internal and observed_to_internal moved to the
  "matvec" representation (repn = hmm.representation("matvec");
  repn.start_vec, repn.hidden_to_internal, ...; the arrays are read-only
  float64 NumPy arrays, not lists), and hidden_to_external/
  observed_to_external are removed (use hidden_states/observable_states, the
  same lists). hidden_states and observable_states are read-only FrozenLists
  (no copy per access). Constraint functions still get
  data.hmm.hidden_to_internal.
- The "matvec" representation is the one place that converts between HMM
  state labels and indices: repn.hidden_states/observable_states (index ->
  label) and repn.hidden_to_internal/observable_to_internal (label -> index).
  The "mvr" representation aligns its MVRs to its .matvec, and the
  network-flow constraint data (data.hmm) also has hidden_states.
- HiddenMarkovModel's observable labels are "observable" everywhere ("observed"
  is kept for sequences of observations): observed_states,
  num_observed_states and get_observable_states() are observable_states and
  num_observable_states; matvec's observed_to_internal is
  observable_to_internal; random_hmm(observed_states=) is
  random_hmm(observable_states=). Error messages say "observable state".
- HiddenMarkovModel(..., hidden_states=None, observable_states=None): the
  state order (and so the "matvec" indices) is the given lists, or the sorted
  labels by default. observable_states may list states that are never
  emitted, and labels that cannot be sorted work with an explicit order. A
  label in the probability tables that is not one of the given states, an
  empty list or duplicate labels raise InvalidInputError. random_hmm,
  supervised_learning, mcem, make_non_zero and the MVR learning algorithms
  keep the order they are given (random_hmm and supervised_learning sorted
  the labels before).
- start_probs, transition_probs and emission_probs are read-only properties
  (FrozenDicts with every entry, zeros included) replacing
  get_start_probs()/get_transition_probs()/get_emission_probs() (which
  returned new dicts); use dict(hmm.start_probs) for a copy to change.
  get_hidden_states(), is_valid_hidden_state(h), is_valid_observed_state(o)
  and to_dict() are removed (use hidden_states, h in hmm.hidden_states and
  o in hmm.observable_states).
- HiddenMarkovModel.log_probability(observed, hidden) is removed: use
  conin.algorithms.scoring.log_probability(hmm, hidden, observed=observed)
  (note the order), which returns -inf instead of raising ValueError for a
  zero-probability sequence. mcem scores its samples with it.
- HiddenMarkovModel has check_model() (the constructor's checks; it raises
  InvalidInputError), __eq__/__hash__ (same states in the same order and the
  same probabilities) and a short __repr__. The validation messages name the
  table and the state ("transition_probs from 'h0' must sum to 1, got 0.9",
  "must be nonnegative"; they said "must be positive floats").
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
  registry: MN, BN and DBN accept "pyomo", "toulbar2", "smoek" and
  "factor_oracle" (they now refuse "oracle", "mvr" and "recursive"); HMMs
  also accept "oracle", "recursive" and "mvr".
- The constrained model's data reaches the constraint functions on every
  backend, as a shallow copy with the backend's fields (T, hmm, start, stop,
  variables, evidence, ...) set. The time fields are the same for every
  backend and for is_feasible: start, stop and T for a DBN; start, stop and
  hmm = Munch(T, hidden_to_internal) for an HMM (the pgmpy, oracle and
  is_feasible data had only T or hmm.T, and network flow no start/stop).
- The feasibility checks are functions, not methods:
  conin.algorithms.feasibility.is_feasible(cmodel, assignment) for every
  constrained model (and observed= for an HMM), and
  partial_is_feasible(cmodel, T=, seq=) for constrained HMMs only (other
  models raise TypeError). is_feasible checks oracle, factor-oracle,
  smoek, recursive and MVR constraints (pyomo/toulbar2 raise
  NotImplementedError); for an HMM it accepts a dict keyed by time.
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
- The oracle library moves to conin.constraints.oracle.library and is no
  longer star-exported from conin.constraints. The partial functions of
  occurs_only_in_time_frame_constraint and
  occurs_at_least_once_in_time_frame_constraint no longer reject completable
  prefixes.
- The misspelt "occurences" names are spelt "occurrences":
  has_minimum_number_of_occurrences(_constraint),
  has_maximum_number_of_occurrences(_constraint) and
  has_exact_number_of_occurrences_constraint (no aliases).
- The MVR modules move to conin.constraints.mvr. mvr_timerange returns a new
  MVR (inplace=True for the old behaviour). MVR_MatVecRepn arrays are
  read-only copies.

Representations (new)

- conin.representations: Representation, register_representation,
  build_representation, supported_representations; kinds "pyomo",
  "pyomo_network_flow", "toulbar2", "variable_elimination", "matvec", "oracle",
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
- The DBN and HMM representations of the pyomo, toulbar2 and
  variable_elimination backends share their time range (start, stop, time_steps, the constraint data) through
  the mixins of conin.representations.unrolled; the DBN builders default stop
  to 1 and the HMM builders require it. The backend appliers share their
  dispatch (conin.representations.base.apply_constraints): constraints of
  several families, or of a family the backend does not apply, raise TypeError
  with the same message for every backend.
- Smoek constraints without the smoek package raise ImportError for every
  backend (the pyomo and toulbar2 translators raised TypeError).
- Unconstrained models have representation(kind, **structure_params) too:
  model.representation(kind) is the representation of the model without
  constraints, e.g. hmm.representation("matvec") instead of
  ConstrainedHiddenMarkovModel(hmm).representation("matvec").
- Each Constrained* class registers itself, when it is defined, as the
  constrained class of its model_type (a subclass of a model class uses its
  base class's). map_query and the other queries wrap bare models with this
  registry; a second constrained class for the same model_type raises
  TypeError.

Inference

- map_query normalises every model and dispatches on the representation; the
  private dispatcher modules (OptimizationInference, CFNInference,
  AStarInference, ViterbiInference, VariableElimination) and the empty
  conin.inference.{mn,bn,dbn,hmm} packages are removed. CHMM inputs are no
  longer accepted.
- New method="mvr_viterbi" (torch): the solution also has augmented; dict
  evidence is sparse.
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
  is refused for every method).
- start and stop mean the same for every DBN and HMM method. start must be 0
  (InvalidInputError, with the same message for every method; viterbi,
  a_star and the network-flow integer program ignored it). For an HMM, every
  method infers over 0..stop and ignores the observations after stop, which
  defaults to the last observation; a stop past it raises InvalidInputError
  (viterbi, a_star and network flow ignored stop, variable_elimination
  ignored it too, and mvr_viterbi raised unless stop covered every
  observation, and with list evidence accepted a later stop). Only
  mvr_viterbi with sparse dict observations infers past the last
  observation. log_probability_query accepts stop == start (one time step),
  as map_query does (it raised InvalidInputError).
- viterbi accepts an unconstrained ConstrainedHiddenMarkovModel; viterbi and
  a_star no longer accept an HMM_MatVecRepn. viterbi/a_star on an MN, BN or
  DBN raise InvalidInputError (was TypeError).
- integer_program: ip_formulation is understood for every model type;
  network flow refuses
  unsupported evidence with InvalidInputError (it returned None) and uses
  lp_inference for an unconstrained ConstrainedHiddenMarkovModel.
- toulbar2: works for every model type with verbose=; without pytoulbar2 it
  raises ImportError for every model type, as the other backends do (it
  returned a result with termination_condition "pytoulbar2 not available"); a
  partial variables query raises RuntimeError (it was ignored);
  write_uai_file writes exactly that file.
- A query without a solution (the evidence or the constraints exclude every
  assignment) has the same result for every method: solution=None,
  solutions=[] and termination_condition "error: no feasible solutions"
  (conin.algorithms.inference.results.NO_FEASIBLE_SOLUTIONS,
  no_solution_results), as A* already returned. Before, viterbi and
  variable_elimination returned an arbitrary assignment with log_potential
  -inf (or a finite one, for constraints) and termination_condition "ok";
  toulbar2 raised TypeError ("cannot unpack non-iterable NoneType object"),
  or RuntimeError if it found the contradiction while a constraint was
  added; integer_program raised
  RuntimeError (with solver="highs"/"appsi_highs" a Pyomo error while
  loading the missing solution); the network-flow integer program
  (also lp_inference/ip_inference) and mvr_viterbi raised
  InvalidInputError. Only a proof of infeasibility gives this result: other
  solver failures (e.g. a time limit) still raise RuntimeError, and
  integer_program with solver="or_topas" still raises for an infeasible
  model. viterbi_torch_mvr_chmm and the network-flow representation's
  create_model still raise, now InfeasibleQueryError
  (conin.exceptions), a subclass of InvalidInputError.
- DBN queries with stop=None unroll over 0..1 for every method.
- variable_elimination: on a constrained Markov network it keeps the user
  evidence and no longer forces a scope node to 1; an unsupported constraint family raises
  InvalidInputError (was TypeError).
- The query-check messages changed: a partial variables query raises
  RuntimeError("Partial MAP queries are only supported by
  method='variable_elimination'; ...").
- The pgmpy options prune_network and create_MN are removed: like any other
  unknown option, they raise TypeError for every method (False was accepted
  and True raised RuntimeError).
- integer_program, toulbar2 and variable_elimination have one solver per
  backend for MN, BN and DBN, and solve an HMM as its unrolled DBN, so the
  options mean the same for every model type: timing=True is
  honoured by the integer program on DBNs and HMMs (it was ignored), and
  solution_with_evidence=True works for an HMM integer program with list
  evidence (it raised AttributeError).
- Every method raises TypeError for an option it does not take, so a
  misspelt option is not silently ignored: variable_elimination and viterbi
  ignored unknown options (e.g. solver=). variable_elimination also accepts
  solution_with_fixed (a synonym of solution_with_evidence, as for
  integer_program and toulbar2; it was ignored).
- lp_inference and ip_inference (the network-flow integer program) pass
  solver_options to the solver (they were ignored for every solver but
  or_topas), and with solver="or_topas" pass tee (quiet=False) to both
  topas methods. The integer program on the other models passes tee to the
  "balas" topas method too.
- The per-backend inference_pyomo_map_query_*, inference_toulbar2_map_query_*
  and inference_variable_elimination_map_query_* functions (MN, BN, DDBN,
  HMM) are removed; call map_query(pgm, method=...) instead (see the
  migration table). The model-solving functions solve_pyomo_map_query_model,
  solve_toulbar2_map_query_model, solve_variable_elimination_map_query_model,
  lp_inference and ip_inference stay.
- Every MAP result has the same keys for every method
  (conin.algorithms.inference.results): solution, solutions,
  termination_condition and solvetime (viterbi, a_star, recursive_a_star and
  mvr_viterbi had no solvetime; variable_elimination had no solutions or
  termination_condition; the HMM methods returned observed;
  recursive_a_star had no solution). Every solution has states,
  log_potential and solver_objective. HMM solutions no longer have hidden (it
  was the same as states). log_potential is
  conin.algorithms.scoring.log_potential of the assignment with the evidence
  on the model without its constraints (for an HMM, log P(hidden,
  observed)), so it is the same for every method; it replaces log_likelihood
  (viterbi, A*, network flow, mvr_viterbi) and log_factor_sum (integer
  program; it was the IP objective, whose factors are
  normalised per factor, so it was not log P(x): -10.21 instead of -3.97 for
  the same HMM path). The backend's objective is solver_objective (the IP
  objective, or toulbar2's primal bound, an integer cost). A partial MAP query
  (variable_elimination with variables=) gets log_potential None, as do the
  solve_*_map_query_model functions, which do not see the model.
- The or_topas solutions of an HMM integer program (ip_formulation=
  "markov_network") are all converted to hidden states (only the first was;
  the others kept the unrolled ("H", t)/("E", t) keys).

Scoring

- log_probability and log_potential move to conin.algorithms.scoring.
  log_potential accepts DBNs, HMMs and constrained models (-inf when
  infeasible); log_potential and log_probability raise TypeError for pgmpy
  models (convert them with conin.io.pgmpy.convert_pgmpy_to_conin);
  evidence= is removed (it returned the joint); an assignment that is not
  total raises InvalidInputError; missing factor entries use the factor's
  default_value. New log_probability (BN, DBN, HMM).
- New log_probability_query(pgm, assignment, *, evidence=None,
  method="variable_elimination", start=0, stop=None): log P(x | e), under the
  constraints for a constrained model. It raises TypeError for pgmpy
  models, as map_query does.

Sampling and learning

- HMM sampling functions:
  conin.algorithms.sampling.unconstrained.generate_hidden(hmm, T),
  generate_hidden_until_state, generate_observed_from_hidden,
  generate_observed. Constrained HMMs:
  conin.algorithms.sampling.oracle.rejection_sample_hidden.
- Every NumPy sampler takes rng (None, a seed or a numpy.random.Generator;
  conin.util.as_rng): the HMM sampling functions, rejection_sample_hidden,
  mcem and DBNSampler(dbn, rng=...). They no longer use NumPy's global random
  state, so numpy.random.seed has no effect on them and the draws for a given
  seed changed. HiddenMarkovModel.set_seed is removed (it only called
  numpy.random.seed), the conin.examples HMM builders no longer reset the
  global seed, and random_hmm(seed=...) is random_hmm(rng=...).
  Util.sample_from_vec takes the generator as a second argument.
- generate_hidden_until_state takes max_steps (default 10000) and raises
  RuntimeError if the target is not reached within that many states (it
  looped forever when the target became unreachable, e.g. after an absorbing
  state).
- DBNSampler.sample(T, N) is sample(time_steps, num_samples) and returns a
  list of traces (was a dict keyed 0..N-1); DBNSampler(G) is DBNSampler(dbn).
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
- pgmpy conversion moves from conin.common.pgmpy to conin.io.pgmpy
  (convert_pgmpy_to_conin, and the new convert_conin_to_pgmpy_bn and
  convert_conin_to_pgmpy_mn), with the pgmpy file readers and writers. MapCPD
  and is_polytree for pgmpy models are removed; the new
  conin.graphical_models.is_polytree checks CONIN Bayesian networks.
- conin.io.pgmpy.files.load_model no longer loads a pgmpy example model by
  name when the file is missing; it raises FileNotFoundError.
- pgmax, pomegranate and pyagrum support (converters, loaders, model_type
  values and the pgmax/pomegranate/pyagrum/pgm-all extras) is removed;
  load_model with those model_type values raises InvalidInputError
  ("Unexpected model type").
- conin.common.pgmpy.log_potential and conin.common.pgmpy.custom_VE (dead,
  untested code) are removed.
- conin/mediation_variables (an unused NumPy prototype of the mediation
  variable algorithms, with its notebooks) is removed; the MVR algorithms are
  in conin.algorithms.inference.mvr, conin.algorithms.learning.mvr and
  conin.algorithms.sampling.mvr.
- The format loaders and saver are renamed to match: load_uai, save_uai and
  load_cfn (were load_conin_model_from_uai, save_model_uai and
  load_conin_model_from_cfn); load_model/save_model are unchanged.
- New extras: toulbar2 (pytoulbar2) and torch.
- conin.io.pgmpy imports pgmpy when it is first used (the new
  conin.util.lazy_import), not when conin is imported.
- Hidden Markov models are saved and loaded with save_model/load_model as
  CONIN JSON (.json or .json.gz): a tagged document ("format":
  "conin.HiddenMarkovModel", "version": 1) with the state order and the
  nonzero probabilities, so a loaded model equals the saved one. State
  labels must be JSON values (strings, numbers, booleans, None).
  conin.io.save_json/load_json also work on JSON text. load_model still
  reads the files of the removed HiddenMarkovModel.write_to_file and
  read_from_file methods.
- conin.io errors follow one policy: a missing file raises
  FileNotFoundError (was RuntimeError); malformed or invalid file content
  raises the new conin.exceptions.ModelFileError, an InvalidInputError and
  a ValueError (UAI/CFN parse errors were ValueError, and int()/float()
  failures or bad variable indices escaped as bare ValueError/IndexError);
  an unknown file suffix or model_type, cost_scale <= 0 and load_cfn without
  a filename or string raise InvalidInputError (were RuntimeError or
  ValueError); a model a format or converter cannot hold raises TypeError
  (save_uai, BIF, convert_conin_to_pgmpy_bn/mn and convert_pgmpy_to_conin
  raised ValueError or RuntimeError); saving BIF or pgmpy files without
  pgmpy's readers and writers raises ImportError (was RuntimeError, or an
  "unknown format" error for .bif). NotImplementedError for unsupported CFN
  features is unchanged. load_model now passes its options to the UAI and
  JSON readers too, so an option they do not take raises TypeError.

Configuration

- conin.config.default_mip_solver is looked up the first time it is used,
  not when conin is imported. The Pyomo inference functions default to
  solver=None, which means the default solver; the new
  conin.config.mip_solver(solver) resolves it and raises RuntimeError if no
  MIP solver is available (gurobi, glpk, highs or appsi_highs).
- conin.config.conin_home (the unused conin_HOME environment variable) is
  removed.

Bug fixes

- DiscreteBayesianNetwork.check_model() now rejects a node without a CPD
  or with more than one, a distribution that does not sum to 1, a missing
  parent state and a cycle (it accepted all of them; inference then failed
  later or gave wrong results). DynamicDiscreteBayesianNetwork.check_model()
  was a placeholder that returned True: it now unrolls the DBN until every
  CPD template applies (plus one time step) and checks the unrolled
  network, so e.g. an initial CPD for a time where a template also applies
  raises InvalidInputError ("Invalid dynamic Bayesian network ..."). It
  returns None, as the other check_model methods. create_bn_from_dbn moved
  to conin.graphical_models.dynamic_bayesian_network (still exported by
  conin.graphical_models).
- map_query(method="variable_elimination") added the evidence to
  solution.states when the model had constraints, but not without them.
  It now adds it only with solution_with_evidence=True (or
  solution_with_fixed), as integer_program and toulbar2 do.

- map_query(method="toulbar2") returned an assignment of probability zero
  (log_potential -inf) when the evidence and the constraints left no
  assignment of positive probability: toulbar2 encodes a zero probability
  as a large finite cost. The scoring now drops solutions of probability
  zero, so every method returns no solution then. Found by the new
  randomized cross-method tests (conin/algorithms/tests), which compare
  every MAP method, A* top-k, partial MAP and log_probability_query with
  brute force on small random models (CONIN_RANDOM_CASES sets the number
  of cases).

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
- variable_elimination works with a factor-oracle or smoek constraint
  over one node (e.g. model.V("A", 0) == 1, or a constraint on one hidden
  state of an HMM); it raised KeyError.
- DiscreteMarkovNetwork.with_factor keeps the new factor's edges: once the
  inferred edges had been read, they were passed on as if they had been given,
  so the new model's edges missed those of the added factor.
- The network-flow integer program with solver="or_topas" returns solutions
  with states (they had variable_value instead, so dict evidence failed).
- recursive_a_star with num_solutions > 1 returns the num_solutions most
  likely feasible sequences. It merged the prefixes with equal (last state,
  length, constraint state) into one search node that kept only the best
  prefix, so it could skip sequences. A merged node now keeps its
  num_solutions best prefixes. A* and recursive A* share one search
  (conin.algorithms.inference.hmm_a_star); recursive A* also runs on the
  matvec arrays instead of dicts. When the search finds no solution, both report
  termination_condition="error: no feasible solutions" for every
  num_solutions (it was "ok" for num_solutions > 1, except when recursive
  A*'s first step was already infeasible).
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
  - the "pyomo", "toulbar2" and "variable_elimination" HMM representations
    without stop,
    observations that no hidden state can emit in the network-flow model,
    lp_inference with num_solutions != 1, and supervised_learning/mcem without
    hidden or observable states raise InvalidInputError;
  - or_topas without solutions raises InsufficientSolutionsError;
  - a mixed list of pyomo or toulbar2 constraints raises TypeError;
  - a UAI file with an inconsistent function table raises ModelFileError, a
    missing UAI file FileNotFoundError (as for CFN files), and MapCPD
    argument errors ValueError.
- pgmpy save_model with an unknown file suffix raises InvalidInputError (it
  returned without writing), and the load_model message names the file.
- The timing=True output of variable elimination prints its STOP line with
  the elapsed time.
- mcem stops when the average log-probability converges (it never updated
  the previous value, so it always ran max_iterations iterations), and no
  longer asserts that the Monte Carlo estimate never increases.
- The UAI reader joins every pair of nodes in a Markov-network factor (it
  joined only the last node to the others, so pgmpy rejected the converted
  model: saving such a model as .bif failed). A truncated UAI file raises
  ModelFileError (was StopIteration). Saving a DBN with dynamic nodes as UAI
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
- Factors whose dict values omit assignments (they get the factor's
  default_value) work with save_uai, toulbar2, the integer program and
  convert_conin_to_pgmpy_mn: save_uai, toulbar2 and the converter raised
  KeyError, and the integer program left the omitted assignments out of the
  model (as if their weight were 0), so it could return a wrong assignment or
  no solution.
- viterbi, a_star, recursive_a_star, lp_inference and ip_inference raise
  InvalidInputError for empty observations, as map_query does (the first
  three raised IndexError).
- rejection_sample_hidden no longer draws one more sample after the last
  rejected one (it was never checked, but it advanced NumPy's random
  state); the error says how many samples were rejected (max_failures + 1).

Known issues

- DBNs and HMMs can only be unrolled from start=0 (start is kept so that a
  later start can be supported without another API change).

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
