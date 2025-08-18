import warnings

from conin.util import try_import
from conin.markov_network import (
    DiscreteMarkovNetwork,
    ConstrainedDiscreteMarkovNetwork,
    optimize_map_query_model,
    create_MN_map_query_model,
    convert_to_DiscreteMarkovNetwork,
)
from conin.bayesian_network import (
    DiscreteBayesianNetwork,
    ConstrainedDiscreteBayesianNetwork,
    create_BN_map_query_model,
    convert_to_DiscreteBayesianNetwork,
)
from conin.dynamic_bayesian_network import (
    ConstrainedDynamicBayesianNetwork,
    create_DBN_map_query_model,
)

with try_import() as pgmpy_available:
    import pgmpy.models


class IntegerProgrammingInference:

    def __init__(self, pgm):
        # pgm.check_model()
        self.pgm = pgm
        self.variables = self.pgm.nodes

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
        if (
            isinstance(self.pgm, DiscreteMarkovNetwork)
            or isinstance(self.pgm, ConstrainedDiscreteMarkovNetwork)
            or isinstance(self.pgm, DiscreteBayesianNetwork)
            or isinstance(self.pgm, ConstrainedDiscreteBayesianNetwork)
        ):
            model = self.pgm.create_map_query_model(
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )
            return optimize_map_query_model(model, timing=timing, **options)

        elif isinstance(self.pgm, pgmpy.models.MarkovNetwork):
            model = create_MN_map_query_model(
                pgm=convert_to_DiscreteMarkovNetwork(self.pgm),
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )
            return optimize_map_query_model(model, timing=timing, **options)

        elif isinstance(self.pgm, pgmpy.models.DiscreteBayesianNetwork):
            model = create_BN_map_query_model(
                pgm=convert_to_DiscreteBayesianNetwork(self.pgm),
                variables=variables,
                evidence=evidence,
                timing=timing,
                **options,
            )
            return optimize_map_query_model(model, timing=timing, **options)


class DBN_IntegerProgrammingInference:

    def __init__(self, pgm):
        self.pgm = pgm
        self.variables = self.pgm.nodes()

    def map_query(
        self,
        *,
        start=0,
        stop=1,
        variables=None,
        evidence=None,
        show_progress=False,
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

        if isinstance(self.pgm, ConstrainedDynamicBayesianNetwork):
            model = self.pgm.create_map_query_model(
                start=start, stop=stop, variables=variables, evidence=evidence
            )
            return optimize_map_query_model(model, **options)

        elif isinstance(self.pgm, pgmpy.models.DynamicBayesianNetwork):
            model = create_DBN_map_query_model(
                start=start,
                stop=stop,
                pgm=self.pgm,
                variables=variables,
                evidence=evidence,
            )
            return optimize_map_query_model(model, **options)
