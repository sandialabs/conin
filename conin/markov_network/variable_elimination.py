import pprint
from pgmpy.factors import factor_product
from pgmpy.inference.ExactInference import VariableElimination


def _variable_elimination(
    *,
    pgm,
    variables,
    operation="marginalize",
    evidence=None,
    elimination_order="MinFill",
):
    """
    Implementation of a generalized variable elimination.

    This is adapted from pgmpy.inference.ExactInference.VariableElimination._variable_elimination.
    Specifically, this (1) eliminates the variables outside of 'variables' and 'evidence', and
    (2) is tailored for DiscreteMarkovNetwork objects

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
    assert type(variables) is list, "variables must be a list of strings"
    assert (
        type(evidence) is dict
    ), "evidence must be a dictionary mapping variables to values"
    assert (
        variables or evidence
    ), "We don't need to call variable elimination if no variables or no evidence is specified"

    # Step 2: Prepare data structures to run the algorithm.
    eliminated_variables = set()

    # Get working factors and elimination order
    inf = VariableElimination(pgm)
    inf._initialize_structures()
    working_factors = inf._get_working_factors(evidence)
    elimination_order = inf._get_elimination_order(
        variables, evidence, elimination_order
    )

    if variables:

        # Step 3: Run variable elimination
        for var in elimination_order:
            # print("="*60)
            # print(f"ELIMINATION_VAR: {var}")
            # print("HERE")
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            # for f in factors:
            #    print(f)
            phi = factor_product(*factors)
            # print("RESULTS")
            # print(phi)
            phi = getattr(phi, operation)([var], inplace=False)
            # print(phi)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add((phi, var))
            eliminated_variables.add(var)

        # print("="*60)
        # print(f"ELIMINATION_VAR: {var}")
        # print("HERE")
        # for k,v in working_factors.items():
        #    print(f"WORKING KEY: {k}")
        #    for kk,vv in v:
        #        print(kk)
        #        print(f"ORIGIN: {vv}")
        #    print("")

    # Step 4: Prepare variables to be returned.
    final_distribution = {}
    for node in working_factors:
        for factor, origin in working_factors[node]:
            if not set(factor.variables).intersection(eliminated_variables):
                v = tuple(sorted(factor.variables))
                if v in final_distribution:
                    final_distribution[v] = factor_product(
                        factor, final_distribution[v]
                    )
                else:
                    final_distribution[v] = factor

    return [factor for factor in final_distribution.values()]
