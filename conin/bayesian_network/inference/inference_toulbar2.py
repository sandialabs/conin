import warnings
import munch
from pyomo.common.timing import TicTocTimer

import conin.common
from .model import ConstrainedDiscreteBayesianNetwork


def create_toulbar2_map_query_model_BN(
    *,
    pgm,
    variables=None,
    evidence=None,
    var_index_map=None,
    timing=False,
    **options,
):
    """Create a MAP query model.

    Parameters
    ----------
    variables : list, optional
        Nodes for which to compute the MAP estimate.
    evidence : dict, optional
        Observed node assignments.
    timing : bool, optional
        Whether to collect timing information.
    var_index_map : dict, optional
        Dictionary mapping variable indices to Pyomo variable data objects
    **options : dict, optional
        Additional keyword arguments forwarded to
        :func:`create_BN_map_query_model`.

    Returns
    -------
    conin.bayesian_network.inference.BNMapQueryModel
        Constrained MAP query model.
    """
    if timing:
        timer = TicTocTimer()
        timer.tic("create_toulbar2_map_query_model_BN - START")
    prune_network = options.pop("prune_network", False)
    create_MN = options.pop("create_MN", False)

    pgm_ = pgm.pgm if isinstance(pgm, ConstrainedDiscreteBayesianNetwork) else pgm

    if variables and len(variables) == len(pgm_.nodes):
        assert set(variables) == set(
            pgm_.nodes
        ), "Mismatch in the specified variables and the nodes in the model"
        # We continue with 'variables' set to None, which is a special case recognized below
        variables = None

    #
    # WEH - This assumes that toulbar2 is smart enough to optimize UAI files for BAYES and MARKOV
    #       differently.
    #
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "model.uai")
        conin.common.save_model(pgm_, filename)

        model = pytoulbar2.CFN()
        model.Read(filename)

    model.X = {name: i for i, name in enumerate(pgm.nodes)}

    if isinstance(pgm, ConstrainedDiscreteBayesianNetwork) and pgm.constraints:
        data = munch.Munch(variables=variables, evidence=evidence)
        for func in pgm.constraints:
            model = func(model, data)

    if timing:
        timer.toc("create_toulbar2_map_query_model_BN - STOP")
    return model
