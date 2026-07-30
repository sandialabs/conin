from conin.exceptions import InvalidInputError
from . import chmm
from .mvr import BaseMVR


class MVR_CHMM(chmm.CHMM):
    """Constrained HMM variant based on mediation variable representations.

    Parameters
    ----------
    hidden_markov_model : HiddenMarkovModel
        Hidden Markov model with an initialized numeric representation.
    constraints : list of BaseMVR, optional
        Mediation variable representation constraints to enforce.
    data : optional
        Application-specific data passed through to the base constrained HMM.

    Raises
    ------
    InvalidInputError
        If ``hidden_markov_model`` is missing, lacks a numeric
        representation, or if any constraint is not a compatible ``BaseMVR``.
    """

    def __init__(self, *, hidden_markov_model=None, constraints=None, data=None):
        # Validation checks
        # Checking for missing arguments
        if hidden_markov_model is None:
            raise InvalidInputError("hidden_markov_model is a required argument")
        if hidden_markov_model.repn is None:
            raise InvalidInputError(
                "hidden_markov_model.repn is missing "
                "Please run Load_model() with the correct start/trans/emit probs"
            )
        # Constraint consistency checks
        if constraints:
            hidden_states = set(hidden_markov_model.hidden_to_external)
            for i, mvr in enumerate(constraints):
                if not isinstance(mvr, BaseMVR):
                    raise InvalidInputError(f"Constraint {i} is not an MVR object")
                if set(mvr.hidden_states) != hidden_states:
                    raise InvalidInputError(
                        f"Hidden states of constraint {i} do not match those of hidden_markov_model"
                    )

        super().__init__(
            hidden_markov_model=hidden_markov_model, data=data, constraints=constraints
        )
