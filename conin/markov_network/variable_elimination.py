"""
The MIT License (MIT)

Copyright (c) 2013-2024 pgmpy

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

try:
    from pgmpy.factors import factor_product
    from pgmpy.inference.ExactInference import VariableElimination
except:
    pass


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
            # 
            # Collect all factors containing 'var', ignoring all the 
            # factors that contain eliminated variables.
            #
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            #
            # Create the factor product of 'factors', and marginalize over 'var'
            #
            phi = factor_product(*factors)
            phi = getattr(phi, operation)([var], inplace=False)
            #
            # Update 'working_factors' and 'eliminated_variables'
            #
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add((phi, var))
            eliminated_variables.add(var)

    # Step 4: Prepare factors to be returned.
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
