"""Query checks shared by the integer-program and toulbar2 MAP queries.

Neither backend can run a partial MAP query (it would need variable
elimination to sum out the other variables), and neither supports network
pruning or Markov-network generation for CONIN models.
"""


def validate_mn_query(variables):
    """Reject a partial MAP query on a Markov network (``RuntimeError``)."""
    if variables:
        raise RuntimeError("VariableElimination is not supported for CONIN models")


def validate_bn_query(options, variables, nodes, *, evidence=None):
    """Pop ``prune_network``/``create_MN`` from ``options`` and reject unsupported queries.

    Neither option is supported for CONIN models, and neither is any
    non-empty ``variables`` other than all of ``nodes`` (that would need
    variable elimination).  These raise ``RuntimeError``.

    Returns ``variables`` (``None`` if it named every node).
    """
    prune_network = options.pop("prune_network", False)
    create_MN = options.pop("create_MN", False)
    if prune_network:
        """
        Apply the pruning methods described here:

        M. Baker, T. E. Boult. 1990. Pruning bayesian networks for efficient computation.
        In Proc. of the Sixth Annual Conf. on Uncertainty in Artificial Intelligence (UAI '90).
        Elsevier Science Inc., USA, 225–232.
        https://arxiv.org/abs/1304.1112

        Note: prune_network defaults to False because the implementation of this method in pgmpy is
        slow for large models.
        """
        raise RuntimeError("Pruning is not currently supported with CONIN models.")

    if variables and len(variables) == len(nodes):
        assert set(variables) == set(
            nodes
        ), "Mismatch in the specified variables and the nodes in the model"
        # We continue with 'variables' set to None
        variables = None

    if variables or create_MN:
        raise RuntimeError(
            f"CONIN does not currently support the generation of discrete Markov networks using variable elimination to prune unspecified variables. {variables=} {evidence=} {create_MN=}"
        )
    return variables
