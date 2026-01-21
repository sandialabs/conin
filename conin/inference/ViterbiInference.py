import warnings

from conin.util import try_import
from conin.hmm.hmm import HiddenMarkovModel, HMM_MatVecRepn
from conin.hmm.inference import viterbi


class ViterbiInference:

    def __init__(self, pgm):
        self.pgm = pgm

    def map_query(
        self,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        **options,
    ):
        """
        Computes the MAP Query over the variables given the evidence. Returns the
        highest probable state in the joint distribution of `variables`.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict or list
            a list of observed states or dict key, value pair as {var: state_of_var_observed}

        show_progress: boolean
            If True, shows search progress. (ignored)

        timing: boolean
            If True, shows timing information. (ignored)
        """
        pgm = self.pgm

        if isinstance(pgm, HiddenMarkovModel) or isinstance(pgm, HMM_MatVecRepn):
            if type(evidence) is dict:
                observed = [evidence[i] for i in range(len(evidence))]
                results = viterbi(observed=observed, hmm=pgm)
                solutions = results.solutions
                for soln in solutions:
                    soln.states = {i: v for i, v in enumerate(soln.states)}
                    soln.hidden = soln.states
                results.solutions = solutions
                return results
            elif type(evidence) is list:
                return viterbi(observed=evidence, hmm=pgm)
        else:
            raise TypeError("Unexpected model type: {type(pgm)}")
