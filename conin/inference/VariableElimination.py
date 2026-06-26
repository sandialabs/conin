import munch
from conin.util import try_import

from conin.hidden_markov_model import (
    create_dbn_from_hmm,
    HiddenMarkovModel,
    ConstrainedHiddenMarkovModel,
    CHMM,
)

from conin.markov_network import (
    DiscreteMarkovNetwork,
    ConstrainedDiscreteMarkovNetwork,
)
from conin.bayesian_network import (
    DiscreteBayesianNetwork,
    ConstrainedDiscreteBayesianNetwork,
)
from conin.dynamic_bayesian_network import (
    create_bn_from_dbn,
    DynamicDiscreteBayesianNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
)

from conin.common.conin import convert_conin_to_pgmpy_mn, convert_conin_to_pgmpy_bn

with try_import() as pgmpy_available:
    import pgmpy.models
    import pgmpy.inference


class VariableEliminationInference:

    def __init__(self, pgm):
        assert (
            pgmpy_available
        ), "PGMPY must be installed to perform inference with VariableElimination"
        if isinstance(pgm, pgmpy.models.DiscreteMarkovNetwork) or isinstance(
            pgm, pgmpy.models.DiscreteBayesianNetwork
        ):
            self._pgm = pgm
            self.pgm = pgm
        elif isinstance(pgm, DiscreteBayesianNetwork):
            self._pgm = pgm
            self.pgm = convert_conin_to_pgmpy_bn(pgm)
        elif isinstance(pgm, DiscreteMarkovNetwork):
            self._pgm = pgm
            self.pgm = convert_conin_to_pgmpy_mn(pgm)
        else:
            raise TypeError(f"Unexpected model type: {type(pgm)}")

    def map_query(
        self,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        **options,
    ):
        """
        Computes the MAP Query over the variables given the evidence. Returns the
        highest probable state in the joint distribution of `variables`.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from conin.inference import VariableElimination
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model = model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.map_query(variables=['A', 'B'])
        """
        if variables is None:
            variables = self.pgm.nodes

        infer = pgmpy.inference.VariableElimination(self.pgm)
        map_states = infer.map_query(
            variables=variables, evidence=evidence, show_progress=show_progress
        )

        return munch.Munch(solution=munch.Munch(states=map_states))


class DPGM_VariableEliminationInference:

    def __init__(self, pgm):
        assert (
            pgmpy_available
        ), "PGMPY must be installed to perform inference with DPGM_VariableElimination"
        self.pgm = pgm

    def map_query(
        self,
        *,
        start=0,
        stop=None,
        variables=None,
        evidence=None,
        show_progress=False,
        solution_with_evidence=False,
        **options,
    ):
        """
        Computes the MAP Query over the variables given the evidence. Returns the
        highest probable state in the joint distribution of `variables`.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from conin.inference import DPGM_VariableElimination
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model = model.fit(values)
        >>> inference = DPGM_VariableElimination(model)
        >>> phi_query = inference.map_query(variables=['A', 'B'])
        """
        if isinstance(self.pgm, DynamicDiscreteBayesianNetwork):
            if stop is None:
                stop = 1
            conin_bn = create_bn_from_dbn(dbn=self.pgm, start=start, stop=stop)
            pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

            infer = pgmpy.inference.VariableElimination(pgmpy_bn)

            if variables is None:
                if evidence:
                    variables = [
                        node for node in pgmpy_bn.nodes if node not in evidence
                    ]
                else:
                    variables = [node for node in pgmpy_bn.nodes]

            map_states = infer.map_query(
                variables=variables, evidence=evidence, show_progress=show_progress
            )
            if solution_with_evidence and evidence:
                map_states.update(evidence)
            return munch.Munch(solution=munch.Munch(states=map_states))

        elif isinstance(self.pgm, HiddenMarkovModel):
            stop = len(evidence) - 1
            conin_dbn = create_dbn_from_hmm(self.pgm)
            conin_bn = create_bn_from_dbn(dbn=conin_dbn, start=start, stop=stop)
            pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

            infer = pgmpy.inference.VariableElimination(pgmpy_bn)

            if type(evidence) is list:
                evidence_ = {("E", i): v for i, v in enumerate(evidence)}
            elif type(evidence) is dict:
                evidence_ = {("E", i): v for i, v in evidence.items()}
            if variables is None:
                variables = [node for node in pgmpy_bn.nodes if node not in evidence_]

            map_states = infer.map_query(
                variables=variables, evidence=evidence_, show_progress=show_progress
            )
            if type(evidence) is list:
                states = [map_states["H", i] for i in range(len(map_states))]
            elif type(evidence) is dict:
                states = {i: map_states["H", i] for i in range(len(map_states))}
            else:
                states = map_states

            return munch.Munch(solution=munch.Munch(states=states))

        else:
            raise TypeError(f"Unexpected model type: {type(self.pgm)}")
