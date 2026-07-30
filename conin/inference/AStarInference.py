from conin.hidden_markov_model.hmm import HiddenMarkovModel, HMM_MatVecRepn
from conin.hidden_markov_model import ConstrainedHiddenMarkovModel, CHMM
from conin.hidden_markov_model.inference import a_star


class AStarInference:
    """Run A* MAP inference for hidden Markov models.

    This wrapper dispatches to :func:`conin.hidden_markov_model.inference.a_star`
    for CONIN hidden Markov model representations.
    """

    def __init__(self, pgm):
        """Store the model used for subsequent A* MAP queries.

        Parameters
        ----------
        pgm : HiddenMarkovModel or HMM_MatVecRepn or ConstrainedHiddenMarkovModel or CHMM
            Hidden Markov model representation passed to the A* inference
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
        """Compute a MAP hidden-state sequence from observed evidence.

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
            Additional keyword arguments forwarded to
            :func:`conin.hidden_markov_model.inference.a_star`.

        Returns
        -------
        munch.Munch
            Result object returned by the A* backend. The object contains a
            ``solution`` entry for the best sequence and a ``solutions`` list
            with all reported candidate solutions.

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

        if (
            isinstance(pgm, HiddenMarkovModel)
            or isinstance(pgm, HMM_MatVecRepn)
            or isinstance(pgm, ConstrainedHiddenMarkovModel)
            or isinstance(pgm, CHMM)
        ):
            if type(evidence) is dict:
                observed = [evidence[i] for i in range(len(evidence))]
                results = a_star(observed=observed, hmm=pgm, **options)
                solutions = results.solutions
                for soln in solutions:
                    soln.states = {i: v for i, v in enumerate(soln.states)}
                    soln.hidden = soln.states
                results.solutions = solutions
                return results
            elif type(evidence) is list:
                return a_star(observed=evidence, hmm=pgm, **options)
        else:
            raise TypeError("Unexpected model type: {type(pgm)}")
