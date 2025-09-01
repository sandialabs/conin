import copy
import itertools
from functools import reduce
from typing import Hashable, Optional

import networkx as nx
import numpy as np
from opt_einsum import contract
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Inference, VariableElimination
from pgmpy.inference.EliminationOrder import (
    MinFill,
    MinNeighbors,
    MinWeight,
    WeightedMinFill,
)
from pgmpy.models import (
    DiscreteBayesianNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    FunctionalBayesianNetwork,
    JunctionTree,
    LinearGaussianBayesianNetwork,
)
from pgmpy.utils import compat_fns


class logprob_VE(VariableElimination):
    def ve2(
        self,
        variables,
        operation,
        evidence=None,
        elimination_order="MinFill",
        joint=True,
        show_progress=True,
    ):
        """
        Implementation of a generalized variable elimination.

        Parameters
        ----------
        variables: list, array-like
            variables that are not to be eliminated.

        operation: str ('marginalize' | 'maximize')
            The operation to do for eliminating the variable.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        elimination_order: str or list (array-like)
            If str: Heuristic to use to find the elimination order.
            If array-like: The elimination order to use.
            If None: A random elimination order is used.
        """
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, str):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables are not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            if joint:
                return factor_product(*set(all_factors))
            else:
                return set(all_factors)

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors and elimination order
        working_factors = self._get_working_factors(evidence)
        elimination_order = self._get_elimination_order(
            variables, evidence, elimination_order, show_progress=show_progress
        )

        # Step 3: Run variable elimination
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        for var in pbar:
            if show_progress and config.SHOW_PROGRESS:
                pbar.set_description(f"Eliminating: {var}")
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            phi = factor_product(*factors)
            phi = getattr(phi, operation)([var], inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add((phi, var))
            eliminated_variables.add(var)

        # Step 4: Prepare variables to be returned.
        final_distribution = set()
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add((factor, origin))
        final_distribution = [factor for factor, _ in final_distribution]

        return final_distribution

    def logprobs(
        self,
        variables: list[Hashable],
        variable_values: list[list[Hashable]],
        evidence: Optional[dict[Hashable, int]] = None,
        elimination_order="MinFill",
        show_progress=True,
    ):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        elimination_order: str or list (default='greedy')
            Order in which to eliminate the variables in the algorithm. If list is provided,
            should contain all variables in the model except the ones in `variables`. str options
            are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
            refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
            for details.

        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.query(["A", "B"])
        """
        evidence = evidence if evidence is not None else dict()

        if isinstance(
            self.model, (LinearGaussianBayesianNetwork, FunctionalBayesianNetwork)
        ):
            raise NotImplementedError(
                f"Variable Elimination is not supported for {self.model.__class__.__name__}."
                f"Please use the 'predict' method of the {self.model.__class__.__name__} class instead."
            )

        # Step 1: Parameter Checks
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        if not variables:
            raise ValueError(
                "The `variables` argument to query() must contain at least one variable."
            )

        # Step 2: Do not allow virtual evidence

        # Step 3: Prune the network based on variables and evidence.
        if isinstance(self.model, DiscreteBayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
            factors = model_reduced.cpds
        else:
            model_reduced = self.model
            factors = self.model.factors

        # Step 4: Greedy EO not allowed since it has its own custom VE implementation.

        # Step 5.1: Initialize data structures for the reduced bn.
        reduced_ve = logprob_VE(model_reduced)
        reduced_ve._initialize_structures()

        # Step 5.2: Do the actual variable elimination
        final_factors = reduced_ve.ve2(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            elimination_order=elimination_order,
            show_progress=show_progress,
        )

        # Now, for each assignment of values, compute the log_prob.
        ret_val = []

        for assignment in variable_values:
            log_prob = 0
            var_val_dict = dict(zip(variables, assignment))
            for factor in final_factors:
                scope_assignment = {
                    k: v for k, v in var_val_dict.items() if k in factor.scope()
                }
                log_prob += np.log(factor.get_value(**scope_assignment))
            ret_val.append(log_prob)

        return ret_val
