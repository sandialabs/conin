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
  that return a new model instance. HiddenMarkovModel.load_model and
  read_from_file are now classmethods that return a new instance instead of
  mutating in place, and make_non_zero returns a new instance rather than
  modifying the model it's called on. The constrained wrapper classes
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
