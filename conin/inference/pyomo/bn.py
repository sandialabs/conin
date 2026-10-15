"""Pyomo MAP inference for (constrained) Bayesian networks."""

from conin.constrained_models import ConstrainedDiscreteBayesianNetwork

from .solve import solve_pyomo_map_query_model


def pop_bn_build_options(options):
    """Pop and check the BN build options (``prune_network``, ``create_MN``).

    Neither is supported for CONIN models; ``True`` raises ``RuntimeError``.
    Returns ``create_MN``.
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
    return create_MN


def check_bn_variables(variables, nodes, *, evidence=None, create_MN=False):
    """Check a BN MAP query's ``variables``.

    ``variables`` naming every node is the same as ``None``.  Any other
    non-empty ``variables`` (or ``create_MN=True``) would need variable
    elimination, which is not supported for CONIN models.

    Returns ``variables`` (``None`` if it named every node).
    """
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


def inference_pyomo_map_query_BN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    **options,
):
    """MAP query on a Bayesian network with the factor-repn integer program.

    The IP is built on the BN's Markov-network skeleton (one factor per CPD).

    Parameters
    ----------
    pgm : DiscreteBayesianNetwork or ConstrainedDiscreteBayesianNetwork
    variables : list, optional
        Only ``None`` or all nodes are supported.
    evidence : dict, optional
        ``{node: state}``.
    timing : bool, optional
        Print timing information.
    **options
        ``use_cut_generation``, ``prune_network`` and ``create_MN`` (both
        unsupported: ``True`` raises ``RuntimeError``), and the options of
        :func:`~conin.inference.pyomo.solve_pyomo_map_query_model`.
    """
    use_cut_generation = options.pop("use_cut_generation", False)
    create_MN = pop_bn_build_options(options)

    cmodel = (
        pgm
        if isinstance(pgm, ConstrainedDiscreteBayesianNetwork)
        else ConstrainedDiscreteBayesianNetwork(pgm)
    )
    repn = cmodel.representation("pyomo")
    variables = check_bn_variables(
        variables, repn.nodes, evidence=evidence, create_MN=create_MN
    )
    model = repn.create_model(
        evidence=evidence,
        variables=variables,
        omit_c5=use_cut_generation,
        timing=timing,
    )
    return solve_pyomo_map_query_model(
        model, timing=timing, evidence=evidence, **options
    )
