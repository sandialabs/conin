Conin CHANGELOG
===============

-------------------------------------------------------------------------------
Conin 1.2
-------------------------------------------------------------------------------

- Made the unconstrained graphical model classes (HiddenMarkovModel,
  DiscreteBayesianNetwork, DiscreteMarkovNetwork, DynamicDiscreteBayesianNetwork,
  DiscreteCPD, DiscreteFactor) immutable after construction. States, CPDs, and
  factors are now passed to constructors instead of assigned via setters;
  add_cpd/add_factor are replaced by non-mutating with_cpd/with_factor methods
  that return a new model instance. HiddenMarkovModel.load_model has been
  removed: HMMs are now built by passing the required, non-empty
  start_probs, transition_probs, and emission_probs dictionaries to the
  HiddenMarkovModel constructor (an empty HiddenMarkovModel() can no longer be
  created), and the internal state indices follow the sorted order of the
  state labels. DiscreteBayesianNetwork and DynamicDiscreteBayesianNetwork
  now require a non-empty cpds argument, and DiscreteMarkovNetwork requires
  a non-empty factors argument; states (and edges) remain optional and
  default to None. DynamicDiscreteBayesianNetwork also requires at least one
  of states or dynamic_states to be non-empty. Time-indexed DBN CPDs are now
  written against an ExpressionVariable (exported from
  conin.dynamic_bayesian_network) that is passed to the constructor as t,
  instead of the t of a throwaway network.
  read_from_file is now a classmethod that returns a new instance, and
  make_non_zero returns a new instance rather than modifying the model it's
  called on. The constrained wrapper classes
  (ConstrainedDiscreteBayesianNetwork, ConstrainedDiscreteMarkovNetwork,
  ConstrainedDynamicDiscreteBayesianNetwork, ConstrainedHiddenMarkovModel) and
  CHMM/MVR classes are unaffected and remain mutable.

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
