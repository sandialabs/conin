from conin.exceptions import InvalidInputError
from . import chmm
from .mvr import BaseMVR

class MVR_CHMM(chmm.CHMM):
    """
    Class for MVR augmentation on a Hidden Markov Model (HMM)
    """

    def __init__(
        self,
        *,
        hidden_markov_model=None,
        constraints=None,
        data=None
    ):
        """
        Parameters:
            hidden_markov_model(HiddenMarkovModel):
                Requires a numeric representation.
            constraints (list, optional):
                A list of MVR objects.
        """
        #Validation checks
        #Checking for missing arguments
        if hidden_markov_model is None:
            raise InvalidInputError("hidden_markov_model is a required argument")
        if hidden_markov_model.repn is None:
            raise InvalidInputError("hidden_markov_model.repn is missing "
                                    "Please run Load_model() with the correct start/trans/emit probs"
                                 )
         #Constraint consistency checks   
        if constraints:
            hidden_states = set(hidden_markov_model.hidden_to_external)
            for i,mvr in enumerate(constraints):
                if not isinstance(mvr, BaseMVR):
                    raise InvalidInputError(f"Constraint {i} is not an MVR object")
                if set(mvr.hidden_states) != hidden_states:
                    raise InvalidInputError(f'Hidden states of constraint {i} do not match those of hidden_markov_model')
                
        super().__init__(hidden_markov_model=hidden_markov_model, data=data, constraints = constraints)
