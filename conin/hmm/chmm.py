from munch import Munch
from conin.hmm.hmm import HMM_MatVecRepn


class CHMM:

    def __init__(
        self, *, hmm=None, hidden_markov_model=None, constraints=None, data=None
    ):
        if hmm and not isinstance(hmm, HMM_MatVecRepn):
            raise ValueError(
                f"The hmm argument must be a HMM_MatVecRepn instance: {type(hmm)=}"
            )
        self.hmm = hmm
        if hidden_markov_model:
            from conin.hmm import HiddenMarkovModel

            if not isinstance(hidden_markov_model, HiddenMarkovModel):
                raise ValueError(
                    f"The hidden_markov_model argument must be a HiddenMarkovModel instance: {type(hidden_markov_model)=}"
                )

        self.hidden_markov_model = hidden_markov_model
        self.constraints = constraints
        self.data = Munch() if data is None else data
