from conin.markov_network import ConstrainedMarkovNetwork, optimize_map_query_model
from conin.bayesian_network import ConstrainedDiscreteBayesianNetwork
from conin.dynamic_bayesian_network import ConstrainedDynamicBayesianNetwork


class OptimizationInference:

    def __init__(self, model):
        self.model = model
        model.check_model()
        self.variables = self.model.nodes()

    def map_query(
        self, variables=None, evidence=None, show_progress=False, **solver_options
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
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = OptimizationInference(model)
        >>> phi_query = inference.map_query(['A', 'B'])
        """

        if isinstance(self.model, ConstrainedMarkovNetwork) or isinstance(
            self.model, ConstrainedDiscreteBayesianNetwork
        ):
            opt_model = self.model.create_map_query_model(
                X=variables, evidence=evidence
            )
            return optimize_map_query_model(opt_model, **solver_options)

        elif isinstance(self.model, ConstrainedDynamicBayesianNetwork):
            opt_model = self.model.create_map_query_model(
                variables=variables, evidence=evidence
            )
            return optimize_map_query_model(opt_model, **solver_options)
