"""Pyomo MAP inference for (constrained) HMMs.

``ip_formulation`` selects the formulation: ``None`` or ``"markov_network"``
(the factor-repn IP of the unrolled DBN, the ``"pyomo"`` representation) or
``"network_flow"`` (the ``"pyomo_network_flow"`` representation, solved by
:func:`~conin.inference.pyomo.lp_inference` /
:func:`~conin.inference.pyomo.ip_inference`).
"""

import logging

from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
from conin.constrained_models import ConstrainedHiddenMarkovModel

from .bn import check_bn_variables, pop_bn_build_options
from .hmm_network_flow import lp_inference, ip_inference
from .solve import solve_pyomo_map_query_model

logger = logging.getLogger("conin")


def _network_flow(pgm, evidence, options):
    if isinstance(pgm, HiddenMarkovModel):
        infer = lambda observed: lp_inference(hmm=pgm, observed=observed, **options)
    elif isinstance(pgm, ConstrainedHiddenMarkovModel):
        # TODO: warning about specifying 'variables'
        # TODO: warning about specifying timing
        infer = lambda observed: ip_inference(hmm=pgm, observed=observed, **options)
    else:
        raise InvalidInputError(
            "The 'network_flow' integer programming formulation expects a "
            f"HiddenMarkovModel or ConstrainedHiddenMarkovModel, got {type(pgm).__name__}"
        )

    if type(evidence) is list:
        return infer(evidence)

    if type(evidence) is dict:
        observed = [evidence[i] for i in range(len(evidence))]
        results = infer(observed)

        solutions = results.solutions
        for soln in solutions:
            soln.states = {i: v for i, v in enumerate(soln.states)}
            soln.hidden = soln.states
        results.solutions = solutions
        return results

    raise InvalidInputError(
        "The 'network_flow' integer programming formulation needs the "
        "observations as evidence: a list, or a dict keyed by time step "
        f"(got {type(evidence).__name__})"
    )


def inference_pyomo_map_query_HMM(
    *,
    pgm,
    start=0,
    stop=None,
    variables=None,
    evidence=None,
    **options,
):
    """MAP query on an HMM with an integer program.

    Parameters
    ----------
    pgm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
    start, stop : int, optional
        The time steps (``stop`` defaults to ``len(evidence) - 1``).
    variables : list, optional
        Only ``None`` or all unrolled nodes are supported.
    evidence : list or dict
        The observations: a list, or a dict keyed by time step.
    **options
        ``ip_formulation`` (``None``, ``"markov_network"`` or
        ``"network_flow"``), ``use_cut_generation`` (``"markov_network"``
        only), and the solver options.

    Raises
    ------
    InvalidInputError
        For ``ip_formulation="network_flow"`` without list or dict evidence.
    ValueError
        For an unknown ``ip_formulation``.
    """
    ip_formulation = options.pop("ip_formulation", None)
    use_cut_generation = options.pop("use_cut_generation", None)

    if ip_formulation == "network_flow":
        if use_cut_generation:
            logger.warning(
                "Cut generation is not supported for the HMM 'network_flow' integer programming formulation"
            )
        return _network_flow(pgm, evidence, options)

    elif ip_formulation is None or ip_formulation == "markov_network":
        if stop is None and evidence is not None:
            stop = len(evidence) - 1
        create_MN = pop_bn_build_options(options)

        cmodel = (
            pgm
            if isinstance(pgm, ConstrainedHiddenMarkovModel)
            else ConstrainedHiddenMarkovModel(pgm)
        )
        repn = cmodel.representation("pyomo", start=start, stop=stop)
        if type(evidence) is list:
            dbn_evidence = {("E", i): evidence[i] for i in range(start, stop + 1)}
        else:
            dbn_evidence = {("E", k): v for k, v in evidence.items()}
        check_bn_variables(
            variables, repn.nodes, evidence=dbn_evidence, create_MN=create_MN
        )
        model = repn.create_model(
            evidence=dbn_evidence, variables=variables, omit_c5=use_cut_generation
        )
        results = solve_pyomo_map_query_model(model, evidence=evidence, **options)
        if type(evidence) is list:
            results.solution.states = [
                results.solution.states["H", i] for i in range(start, stop + 1)
            ]
        else:
            results.solution.states = {
                i: results.solution.states["H", i] for i in range(start, stop + 1)
            }
        return results

    else:
        raise ValueError(f"Unexpected ip_formulation value: '{ip_formulation}'")
