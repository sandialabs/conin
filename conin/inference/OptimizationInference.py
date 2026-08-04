from conin.util import try_import
from conin.hidden_markov_model import (
    HiddenMarkovModel,
    ConstrainedHiddenMarkovModel,
    CHMM,
)

from conin.markov_network import (
    DiscreteMarkovNetwork,
    ConstrainedDiscreteMarkovNetwork,
)
from .mn import (
    inference_pyomo_map_query_MN,
)
from conin.bayesian_network import (
    DiscreteBayesianNetwork,
    ConstrainedDiscreteBayesianNetwork,
)
from .bn import (
    inference_pyomo_map_query_BN,
)
from conin.dynamic_bayesian_network import (
    DynamicDiscreteBayesianNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
)
from .dbn import (
    inference_pyomo_map_query_DDBN,
)

from .hmm import (
    inference_pyomo_map_query_HMM,
)

with try_import() as pgmpy_available:
    import pgmpy.models
    from conin.common.pgmpy import convert_pgmpy_to_conin


class IntegerProgrammingInference:
    """Run MAP inference through the Pyomo mixed-integer optimization backend.

    This wrapper accepts static discrete Markov networks and Bayesian networks.
    When pgmpy is available, compatible pgmpy models are converted to their
    CONIN representation before solving.
    """

    def __init__(self, pgm):
        """Store a model for subsequent optimization-based MAP queries.

        Parameters
        ----------
        pgm : DiscreteMarkovNetwork or ConstrainedDiscreteMarkovNetwork or DiscreteBayesianNetwork or ConstrainedDiscreteBayesianNetwork or pgmpy.models.DiscreteMarkovNetwork or pgmpy.models.DiscreteBayesianNetwork
            Graphical model to solve. Compatible pgmpy models are converted to
            CONIN model objects when pgmpy is installed.
        """
        if pgmpy_available and (
            isinstance(pgm, pgmpy.models.DiscreteMarkovNetwork)
            or isinstance(pgm, pgmpy.models.DiscreteBayesianNetwork)
        ):
            pgm = convert_pgmpy_to_conin(pgm)
        self.pgm = pgm
        # self.variables = self.pgm.nodes

    def map_query(
        self,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        **options,
    ):
        """Compute a MAP assignment with a Pyomo-based optimization model.

        Parameters
        ----------
        variables : list, optional
            Variables included in the MAP query. Support for partial MAP queries
            depends on the selected backend helper.
        evidence : dict, optional
            Observed variable assignments as ``{variable: state}``. Use ``None``
            when no evidence is supplied.
        show_progress : bool, optional
            Whether to request solver progress reporting when supported by the
            backend.
        timing : bool, optional
            If ``True``, include timing information in the returned result.
        **options : dict, optional
            Additional keyword arguments forwarded to the selected Pyomo
            inference helper, such as solver options or output-file settings.

        Returns
        -------
        munch.Munch
            Result object produced by the selected Pyomo inference helper. The
            returned object typically contains ``solution.states`` and, when
            available, ``solvetime``.

        Raises
        ------
        TypeError
            If ``self.pgm`` is not a supported static Markov network or
            Bayesian network type.
        """
        pgm = self.pgm

        if isinstance(pgm, DiscreteMarkovNetwork) or isinstance(
            pgm, ConstrainedDiscreteMarkovNetwork
        ):
            return inference_pyomo_map_query_MN(
                pgm=pgm,
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )

        elif isinstance(pgm, DiscreteBayesianNetwork) or isinstance(
            pgm, ConstrainedDiscreteBayesianNetwork
        ):
            return inference_pyomo_map_query_BN(
                pgm=pgm,
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )

        else:
            raise TypeError(f"Unexpected model type: {type(pgm)}")


class DPGM_IntegerProgrammingInference:
    """Run Pyomo MAP inference for dynamic models and hidden Markov models.

    This wrapper handles dynamic Bayesian networks, hidden Markov models, and
    their constrained counterparts by dispatching to the appropriate Pyomo
    formulation.
    """

    def __init__(self, pgm):
        """Store a dynamic model for subsequent optimization-based MAP queries.

        Parameters
        ----------
        pgm : DynamicDiscreteBayesianNetwork or ConstrainedDynamicDiscreteBayesianNetwork or HiddenMarkovModel or ConstrainedHiddenMarkovModel or CHMM or pgmpy.models.DynamicBayesianNetwork
            Dynamic graphical model to solve. Compatible pgmpy dynamic Bayesian
            networks are converted to CONIN model objects when pgmpy is
            installed.
        """
        if pgmpy_available and isinstance(pgm, pgmpy.models.DynamicBayesianNetwork):
            pgm = convert_pgmpy_to_conin(pgm)
        self.pgm = pgm

    def map_query(
        self,
        *,
        start=0,
        stop=None,
        variables=None,
        evidence=None,
        show_progress=False,
        **options,
    ):
        """Compute a MAP assignment for a dynamic model with Pyomo.

        Parameters
        ----------
        start : int, optional
            Initial time index included in the inference horizon.
        stop : int, optional
            Final time index included in the inference horizon. When ``pgm`` is
            a hidden Markov model, the helper may infer this value from the
            supplied evidence.
        variables : list, optional
            Variables included in the MAP query. The interpretation depends on
            the selected dynamic-model backend.
        evidence : dict or list, optional
            Evidence used by the dynamic-model backend. Dynamic Bayesian network
            helpers expect a dictionary of variable assignments, while hidden
            Markov model helpers accept either a dense list of observations or a
            dictionary keyed by time index.
        show_progress : bool, optional
            Whether to request solver progress reporting when supported by the
            backend. Some helper implementations ignore this argument.
        **options : dict, optional
            Additional keyword arguments forwarded to the selected Pyomo
            inference helper, such as solver settings, formulation options, or
            output-file paths.

        Returns
        -------
        munch.Munch
            Result object produced by the selected Pyomo inference helper. The
            returned object typically contains ``solution.states`` and, when
            available, ``solvetime``.

        Raises
        ------
        TypeError
            If ``self.pgm`` is not a supported dynamic Bayesian network or
            hidden Markov model type.
        """

        pgm = self.pgm

        if isinstance(pgm, DynamicDiscreteBayesianNetwork) or isinstance(
            pgm, ConstrainedDynamicDiscreteBayesianNetwork
        ):
            # TODO: warning about specifying 'variables'
            # TODO: warning about specifying timing
            return inference_pyomo_map_query_DDBN(
                pgm=pgm,
                start=start,
                stop=stop,
                variables=variables,
                evidence=evidence,
                **options,
            )

        elif (
            isinstance(pgm, HiddenMarkovModel)
            or isinstance(pgm, ConstrainedHiddenMarkovModel)
            or isinstance(pgm, CHMM)
        ):
            # TODO: warning about specifying 'variables'
            # TODO: warning about specifying timing
            return inference_pyomo_map_query_HMM(
                pgm=pgm,
                start=start,
                stop=stop,
                variables=variables,
                evidence=evidence,
                **options,
            )

        else:
            raise TypeError(f"Unexpected model type: {type(pgm)}")
