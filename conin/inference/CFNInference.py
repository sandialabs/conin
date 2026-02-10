from conin.util import try_import

from conin.markov_network import (
    DiscreteMarkovNetwork,
    ConstrainedDiscreteMarkovNetwork,
)
from .mn import (
    inference_toulbar2_map_query_MN,
)
from conin.bayesian_network import (
    DiscreteBayesianNetwork,
    ConstrainedDiscreteBayesianNetwork,
)
from .bn import (
    inference_toulbar2_map_query_BN,
)
from conin.dynamic_bayesian_network import (
    DynamicDiscreteBayesianNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
)
from .dbn import (
    inference_toulbar2_map_query_DDBN,
)

with try_import() as pgmpy_available:
    import pgmpy.models
    from conin.common.pgmpy import convert_pgmpy_to_conin


class CFNInference:

    def __init__(self, pgm):
        if pgmpy_available and (
            isinstance(pgm, pgmpy.models.MarkovNetwork)
            or isinstance(pgm, pgmpy.models.DiscreteBayesianNetwork)
        ):
            pgm = convert_pgmpy_to_conin(pgm)
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

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from conin.inference import CFNInference
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model = model.fit(values)
        >>> inference = CFNInference(model)
        >>> phi_query = inference.map_query(variables=['A', 'B'])
        """
        pgm = self.pgm

        if isinstance(pgm, DiscreteMarkovNetwork) or isinstance(
            pgm, ConstrainedDiscreteMarkovNetwork
        ):
            return inference_toulbar2_map_query_MN(
                pgm=pgm,
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )

        elif isinstance(pgm, DiscreteBayesianNetwork) or isinstance(
            pgm, ConstrainedDiscreteBayesianNetwork
        ):
            return inference_toulbar2_map_query_BN(
                pgm=pgm,
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )

        #
        # TODO
        #
        # elif isinstance(pgm, HiddenMarkovModel):
        #    pass

        #
        # TODO
        #
        # elif isinstance(pgm, ConstrainedHiddenMarkovModel) or isinstance(pgm, CHMM):
        #    pass

        else:
            raise TypeError("Unexpected model type: {type(pgm)}")


class DDBN_CFNInference:

    def __init__(self, pgm):
        if pgmpy_available and isinstance(pgm, pgmpy.models.DynamicBayesianNetwork):
            pgm = convert_pgmpy_to_conin(pgm)
        self.pgm = pgm
        # self.variables = self.pgm.nodes

    def map_query(
        self,
        *,
        start=0,
        stop=1,
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

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from conin.inference import OptimizationInference
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model = model.fit(values)
        >>> inference = OptimizationInference(model)
        >>> phi_query = inference.map_query(variables=['A', 'B'])
        """

        pgm = self.pgm

        if isinstance(pgm, DynamicDiscreteBayesianNetwork) or isinstance(
            pgm, ConstrainedDynamicDiscreteBayesianNetwork
        ):
            return inference_toulbar2_map_query_DDBN(
                pgm=pgm,
                start=start,
                stop=stop,
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )
        else:
            raise TypeError("Unexpected model type: {type(pgm)}")
