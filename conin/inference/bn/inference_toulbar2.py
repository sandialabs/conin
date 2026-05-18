import os.path
import tempfile
import munch
from pyomo.common.timing import TicTocTimer

from conin.util import try_import
from conin.inference.mn.inference_toulbar2 import (
    solve_toulbar2_map_query_model,
    VarWrapper,
)

with try_import() as pytoulbar2_available:
    import pytoulbar2

import conin.common
from conin.bayesian_network import ConstrainedDiscreteBayesianNetwork


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

    This assumes that toulbar2 is smart enough to optimize UAI files for BAYES and MARKOV
    differently.

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
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("create_toulbar2_map_query_model_BN - START")
    verbose = options.pop("verbose", -1)

    cpgm = pgm if isinstance(pgm, ConstrainedDiscreteBayesianNetwork) else None
    pgm = cpgm.pgm if cpgm is not None else pgm

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "model.uai")
        conin.common.save_model(pgm, filename)
        model = pytoulbar2.CFN(verbose=verbose)
        model.Read(filename)

    # TODO - do something different here?
    # if var_index_map:
    model.V = VarWrapper(pgm)
    model.states = {i: pgm.states_of(name) for i, name in enumerate(pgm.nodes)}

    model.V_evidence = set()
    if evidence:
        for k, v in evidence.items():
            model.Assign(model.V[k], pgm.states_of(k).index(v))
            model.V_evidence.add(k)

    if cpgm is not None and cpgm.constraints:
        data = munch.Munch(variables=variables, evidence=evidence)
        for func in cpgm.constraints:
            model = func(model, data)

    if timing:  # pragma:nocover
        timer.toc("create_toulbar2_map_query_model_BN - STOP")
    return model


def inference_toulbar2_map_query_BN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    **options,
):
    if not pytoulbar2_available:
        return munch.Munch(
            solution=None,
            solutions=[],
            termination_condition="pytoulbar2 not available",
            solvetime=0.0,
        )

    model = create_toulbar2_map_query_model_BN(
        pgm=pgm, variables=variables, evidence=evidence, timing=timing, **options
    )
    return solve_toulbar2_map_query_model(model, timing=timing, **options)
