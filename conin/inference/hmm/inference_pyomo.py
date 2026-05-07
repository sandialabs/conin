import munch
from conin.hmm.hmm_to_dbn import create_dbn_from_hmm
from conin.hmm import ConstrainedHiddenMarkovModel

from conin.inference.dbn.inference_pyomo import create_pyomo_map_query_model_DDBN


def create_pyomo_map_query_model_DDBN(
    *, pgm, start=0, stop=1, variables=None, evidence=None, **options
):
    pgm_ = (
        pgm.pgm if isinstance(pgm, ConstrainedHiddenMarkovModel) else pgm
    )

    bn = create_bn_from_dbn(dbn=pgm_, start=start, stop=stop)

    model = create_pyomo_map_query_model_DDBN(
        pgm=bn,
        variables=variables,
        evidence=evidence,
        var_index_map=bn._pyomo_index_names,
        **options,
    )

    if isinstance(pgm, ConstrainedHiddenMarkovModel) and pgm.constraints:
        data = munch.Munch(
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            T=list(range(start, stop + 1)),
        )
        for func in pgm.constraints:
            model = func(model, data)

    return model


def inference_pyomo_map_query_HMM(
    *,
    pgm,
    start=0,
    stop=1,
    variables=None,
    evidence=None,
    **options,
):
    ip_formulation = options.get('ip_formulation', None)

    if ip_formulation == "network_flow":
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

        elif isinstance(pgm, ConstrainedHiddenMarkovModel) or isinstance(pgm, CHMM):
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
        model = create_pyomo_map_query_model_HMM(
            pgm=pgm,
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            **options,
        )
        return conin.inference.mn.inference_pyomo.solve_pyomo_map_query_model(
            model, **options
        )

    else:
        raise ValueError(f"Unexpected ip_formulation value: '{ip_formulation}'")

