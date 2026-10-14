import logging
import munch
import conin
from conin.graphical_models import HiddenMarkovModel
from conin.hidden_markov_model.inference import lp_inference, ip_inference
from conin.graphical_models import create_dbn_from_hmm
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference.dbn.inference_pyomo import create_pyomo_map_query_model_DDBN

logger = logging.getLogger("conin")


def create_pyomo_map_query_model_HMM(
    *, pgm, start=0, stop=None, variables=None, evidence=None, **options
):
    pgm_ = (
        pgm.hidden_markov_model
        if isinstance(pgm, ConstrainedHiddenMarkovModel)
        else pgm
    )

    assert stop is not None, "Expecting evidence (observations) for an HMM"

    dbn = create_dbn_from_hmm(hmm=pgm_)
    if type(evidence) is list:
        evidence = {("E", i): evidence[i] for i in range(start, stop + 1)}
    else:
        evidence = {("E", k): v for k, v in evidence.items()}

    model = create_pyomo_map_query_model_DDBN(
        pgm=dbn,
        variables=variables,
        evidence=evidence,
        start=start,
        stop=stop,
        **options,
    )

    if isinstance(pgm, ConstrainedHiddenMarkovModel) and pgm.constraints:
        data = munch.Munch(
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            hmm=munch.Munch(
                T=list(range(start, stop + 1)),
                hidden_to_internal=pgm_.hidden_to_internal,
            ),
        )
        conin.inference.mn.inference_pyomo.add_constraints(
            pgm=dbn, constraints=pgm.constraints, model=model, data=data
        )

    return model


def inference_pyomo_map_query_HMM(
    *,
    pgm,
    start=0,
    stop=None,
    variables=None,
    evidence=None,
    **options,
):
    ip_formulation = options.pop("ip_formulation", None)
    use_cut_generation = options.pop("use_cut_generation", None)

    if ip_formulation == "network_flow":
        if use_cut_generation:
            logger.warning(
                "Cut generation is not supported for the HMM 'network_flow' integer programming formulation"
            )
        if isinstance(pgm, HiddenMarkovModel):
            if type(evidence) is list:
                return lp_inference(hmm=pgm, observed=evidence, **options)

            if type(evidence) is dict:
                observed = [evidence[i] for i in range(len(evidence))]
                results = lp_inference(hmm=pgm, observed=observed, **options)

                solutions = results.solutions
                for soln in solutions:
                    soln.states = {i: v for i, v in enumerate(soln.states)}
                    soln.hidden = soln.states
                results.solutions = solutions
                return results

        elif isinstance(pgm, ConstrainedHiddenMarkovModel):
            # TODO: warning about specifying 'variables'
            # TODO: warning about specifying timing
            if type(evidence) is list:
                return ip_inference(hmm=pgm, observed=evidence, **options)

            if type(evidence) is dict:
                observed = [evidence[i] for i in range(len(evidence))]
                results = ip_inference(hmm=pgm, observed=observed, **options)

                solutions = results.solutions
                for soln in solutions:
                    soln.states = {i: v for i, v in enumerate(soln.states)}
                    soln.hidden = soln.states
                results.solutions = solutions
                return results

    elif ip_formulation is None or ip_formulation == "markov_network":
        if stop is None and evidence is not None:
            stop = len(evidence) - 1
        model = create_pyomo_map_query_model_HMM(
            pgm=pgm,
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            omit_c5=use_cut_generation,
            **options,
        )
        results = conin.inference.mn.inference_pyomo.solve_pyomo_map_query_model(
            model, evidence=evidence, **options
        )
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
