"""Pyomo MAP inference for (constrained) HMMs.

``ip_formulation`` selects the formulation: ``None`` or ``"markov_network"``
(the factor-repn IP of the unrolled DBN, the ``"pyomo"`` representation) or
``"network_flow"`` (the ``"pyomo_network_flow"`` representation, solved by
:func:`~conin.inference.pyomo.lp_inference` /
:func:`~conin.inference.pyomo.ip_inference`).
"""

import logging

from conin.exceptions import InvalidInputError
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models.hmm_observations import (
    as_sequence,
    time_range,
    unrolled_evidence,
)
from conin.inference.hmm_results import (
    hidden_states_from_unrolled,
    key_solutions_by_time,
)
from conin.inference.query_checks import full_query_variables, reject_pruning_options

from .hmm_network_flow import _ip_inference, _lp_inference
from .solve import solve_pyomo_map_query_model

logger = logging.getLogger("conin")


def _map_query_pyomo_hmm(repn, *, variables, evidence, options):
    """Solve a MAP query on the ``"pyomo"`` representation of an HMM."""
    use_cut_generation = options.pop("use_cut_generation", None)
    start, stop = repn.start, repn.stop
    dbn_evidence = unrolled_evidence(as_sequence(evidence), start, stop)
    reject_pruning_options(options)
    variables = full_query_variables(variables, repn.nodes)
    model = repn.create_model(
        evidence=dbn_evidence, variables=variables, omit_c5=use_cut_generation
    )
    results = solve_pyomo_map_query_model(model, evidence=evidence, **options)
    return hidden_states_from_unrolled(
        results, start=start, stop=stop, keyed_by_time=isinstance(evidence, dict)
    )


def _map_query_network_flow(repn, *, evidence, options):
    """Solve a MAP query on the ``"pyomo_network_flow"`` representation of an HMM.

    Without constraints the model is solved with :func:`lp_inference`, with
    constraints with :func:`ip_inference`.
    """
    if options.pop("use_cut_generation", None):
        logger.warning(
            "Cut generation is not supported for the HMM 'network_flow' integer programming formulation"
        )
    observed = as_sequence(
        evidence,
        "evidence (the observations) of the 'network_flow' integer programming formulation",
    )
    infer = _ip_inference if repn.constraints else _lp_inference
    # TODO: warning about specifying 'variables'
    # TODO: warning about specifying timing
    results = infer(repn, observed=observed, **options)
    if isinstance(evidence, dict):
        key_solutions_by_time(results)
    return results


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
        The time steps (``stop`` defaults to ``len(evidence) - 1``, the last
        observation; a later ``stop`` raises ``InvalidInputError``).
    variables : list, optional
        Only ``None`` or all unrolled nodes are supported.
    evidence : list, tuple or dict
        The observations: a list or tuple, or a dict keyed by the times
        ``0..T-1``.
    **options
        ``ip_formulation`` (``None``, ``"markov_network"`` or
        ``"network_flow"``), ``use_cut_generation`` (``"markov_network"``
        only), and the solver options.  ``"network_flow"`` solves an HMM
        without constraints with :func:`~conin.inference.pyomo.lp_inference`
        and one with constraints with
        :func:`~conin.inference.pyomo.ip_inference`.

    Raises
    ------
    InvalidInputError
        For evidence that is not a list, tuple or dict keyed by ``0..T-1``,
        a ``stop`` past the last observation, or a model that is not an HMM.
    ValueError
        For an unknown ``ip_formulation``.
    """
    ip_formulation = options.pop("ip_formulation", None)

    if ip_formulation == "network_flow":
        try:
            cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
        except InvalidInputError:
            raise InvalidInputError(
                "The 'network_flow' integer programming formulation expects a "
                "HiddenMarkovModel or ConstrainedHiddenMarkovModel, got "
                f"{type(pgm).__name__}"
            ) from None
        return _map_query_network_flow(
            cmodel.representation("pyomo_network_flow"),
            evidence=evidence,
            options=options,
        )

    elif ip_formulation is None or ip_formulation == "markov_network":
        start, stop = time_range(as_sequence(evidence), start, stop)
        cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
        return _map_query_pyomo_hmm(
            cmodel.representation("pyomo", start=start, stop=stop),
            variables=variables,
            evidence=evidence,
            options=options,
        )

    else:
        raise ValueError(f"Unexpected ip_formulation value: '{ip_formulation}'")
