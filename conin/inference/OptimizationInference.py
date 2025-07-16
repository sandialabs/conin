import warnings

try:
    import pgmpy.models
except Exception as e:
    warnings.warn(
        f"Warning: pgmpy not installed, so OptimizationInference.py will not work. Exception: {e}"
    )

from conin.markov_network import (
    ConstrainedMarkovNetwork,
    optimize_map_query_model,
    create_MN_map_query_model,
)
from conin.bayesian_network import (
    ConstrainedDiscreteBayesianNetwork,
    create_BN_map_query_model,
)
from conin.dynamic_bayesian_network import (
    ConstrainedDynamicBayesianNetwork,
    create_DBN_map_query_model,
)


class IntegerProgrammingInference:

    def __init__(self, pgm):
        pgm.check_model()
        self.pgm = pgm
        self.variables = self.pgm.nodes()

    def map_query(
        self, *, variables=None, evidence=None, show_progress=False, **options
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

        if isinstance(self.pgm, ConstrainedMarkovNetwork):
            model = self.pgm.create_map_query_model(
                variables=variables,
                evidence=evidence,
                **options,
            )
            return optimize_map_query_model(model, **options)

        elif isinstance(self.pgm, ConstrainedDiscreteBayesianNetwork):
            prune_network = options.pop("prune_network", True)
            model = self.pgm.create_map_query_model(
                variables=variables,
                evidence=evidence,
                prune_network=prune_network,
                **options,
            )
            return optimize_map_query_model(model, **options)

        elif isinstance(self.pgm, pgmpy.models.MarkovNetwork):
            model = create_MN_map_query_model(
                pgm=self.pgm,
                variables=variables,
                evidence=evidence,
                **options,
            )
            return optimize_map_query_model(model, **options)

        elif isinstance(self.pgm, pgmpy.models.DiscreteBayesianNetwork):
            model = create_BN_map_query_model(
                pgm=self.pgm,
                variables=variables,
                evidence=evidence,
                **options,
            )
            return optimize_map_query_model(model, **options)


class DBN_IntegerProgrammingInference:

    def __init__(self, pgm):
        pgm.check_model()
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
