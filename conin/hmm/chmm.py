from conin.hmm import HMM


class CHMM:

    def __init__(self, *, hmm, constraints=None):
        if not isinstance(hmm, HMM):
            raise ValueError("The hmm argument must be a HMM instance")
        self.hmm = hmm
        self.constraints = constraints
