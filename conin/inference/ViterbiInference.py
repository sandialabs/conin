from conin.hidden_markov_model.hmm import HiddenMarkovModel, HMM_MatVecRepn
from conin.hidden_markov_model.inference import viterbi


class ViterbiInference:
    """Run Viterbi MAP inference for hidden Markov models.

    This wrapper dispatches to :func:`conin.hidden_markov_model.inference.viterbi`
    for CONIN hidden Markov model representations.
    """

    def __init__(self, pgm):
        """Store the model used for subsequent Viterbi MAP queries.

        Parameters
        ----------
        pgm : HiddenMarkovModel or HMM_MatVecRepn
            Hidden Markov model representation passed to the Viterbi inference
            backend.
        """
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
        """Compute the most likely hidden-state sequence for the observations.

        Parameters
        ----------
        variables : list, optional
            Accepted for API compatibility with other inference wrappers and
            ignored by this implementation.
        evidence : list or dict, optional
            Observed emissions supplied either as a dense list ordered by time
            step or as a dictionary keyed by consecutive integer time indices.
            When a dictionary is provided, the returned solution states are also
            converted to a dictionary keyed by time index.
        show_progress : bool, optional
            Accepted for API compatibility and ignored.
        timing : bool, optional
            Accepted for API compatibility and ignored.
        **options : dict, optional
            Additional keyword arguments accepted for API compatibility. They
            are currently ignored because :func:`viterbi` does not consume extra
            options.

        Returns
        -------
        munch.Munch
            Result object returned by the Viterbi backend. The object contains a
            ``solution`` entry for the best sequence and a ``solutions`` list
            with the reported candidate solutions.

        Raises
        ------
        TypeError
            If ``self.pgm`` is not a supported hidden Markov model type.

        Notes
        -----
        Dictionary-valued evidence is converted with
        ``[evidence[i] for i in range(len(evidence))]`` before dispatch. The
        keys are therefore expected to be dense, zero-based time indices.
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
