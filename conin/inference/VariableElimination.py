import copy
import munch
from conin.util import try_import
from conin.common.unified import save_model
from pyomo.common.timing import TicTocTimer

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
    from conin.common.pgmpy import convert_pgmpy_to_conin


class VariableEliminationInference:

    def __init__(self, pgm):
        assert (
            pgmpy_available
        ), "PGMPY must be installed to perform inference with VariableElimination"
        self.pgm = pgm

    def map_query(
        self,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        write_uai_file=None,
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
        if timing:
            timer = TicTocTimer()
            timer.tic("VariableEliminationInference.map_query - START")

        evidence = copy.copy(evidence) if evidence else {}

        if isinstance(
            self.pgm,
            (pgmpy.models.DiscreteMarkovNetwork, pgmpy.models.DiscreteBayesianNetwork),
        ):
            pgmpy_pgm = self.pgm

        elif isinstance(self.pgm, DiscreteBayesianNetwork):
            if write_uai_file:
                save_model(self.pgm, write_uai_file)
            pgmpy_pgm = convert_conin_to_pgmpy_bn(self.pgm)

        elif isinstance(self.pgm, DiscreteMarkovNetwork):
            if write_uai_file:
                save_model(self.pgm, write_uai_file)
            pgmpy_pgm = convert_conin_to_pgmpy_mn(self.pgm)

        elif isinstance(self.pgm, ConstrainedDiscreteMarkovNetwork):
            for con in self.pgm.constraints:
                factor = con(self.pgm.pgm)
                self.pgm.pgm._factors.append(factor)
            if write_uai_file:
                save_model(self.pgm.pgm, write_uai_file)
            pgmpy_pgm = convert_conin_to_pgmpy_mn(self.pgm.pgm)

        elif isinstance(self.pgm, ConstrainedDiscreteBayesianNetwork):
            for con in self.pgm.constraints:
                cpd = con(self.pgm.pgm)
                self.pgm.pgm.add_cpd(cpd)
                evidence[cpd.node] = 1
            if write_uai_file:
                save_model(self.pgm.pgm, write_uai_file)
            pgmpy_pgm = convert_conin_to_pgmpy_bn(self.pgm.pgm)

        else:
            raise TypeError(f"Unexpected model type: {type(self.pgm)}")

        if timing:
            timer.tic("Created PGMPY model")
        infer = pgmpy.inference.VariableElimination(pgmpy_pgm)
        if variables is None:
            if evidence:
                variables = [node for node in self.pgm.nodes if node not in evidence]
            else:
                variables = self.pgm.nodes

        solver_timer = TicTocTimer()
        solver_timer.tic(None)
        map_states = infer.map_query(
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            **options,
        )
        solvetime = solver_timer.toc(None)

        if timing:
            timer.tic("VariableEliminationInference.map_query - STOP")

        return munch.Munch(solution=munch.Munch(states=map_states), solvetime=solvetime)


class DPGM_VariableEliminationInference:

    def __init__(self, pgm):
        assert (
            pgmpy_available
        ), "PGMPY must be installed to perform inference with DPGM_VariableElimination"
        self.pgm = pgm

    def _run_map_query(self, pgmpy_bn, variables, evidence, show_progress, **options):
        infer = pgmpy.inference.VariableElimination(pgmpy_bn)
        if variables is None:
            if evidence:
                variables = [node for node in pgmpy_bn.nodes if node not in evidence]
            else:
                variables = [node for node in pgmpy_bn.nodes]

        solver_timer = TicTocTimer()
        solver_timer.tic(None)
        map_states = infer.map_query(
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            **options,
        )
        solvetime = solver_timer.toc(None)

        return map_states, solvetime

    def _hmm_evidence_to_pgmpy(self, evidence):
        if isinstance(evidence, list):
            return {("E", i): v for i, v in enumerate(evidence)}
        if isinstance(evidence, dict):
            return {("E", i): v for i, v in evidence.items()}
        return evidence

    def _hmm_states_from_map(self, map_states, evidence):
        if isinstance(evidence, list):
            return [map_states["H", i] for i in range(len(map_states))]
        if isinstance(evidence, dict):
            return {i: map_states["H", i] for i in range(len(map_states))}
        return map_states

    def _add_constraints_as_evidence(self, conin_bn, constraints, data, evidence):
        for con in constraints:
            cpd = con(conin_bn, data)
            cpd.node = (cpd.node, -1)
            conin_bn.add_cpd(cpd)
            evidence[cpd.node] = 1

    def map_query(
        self,
        *,
        start=0,
        stop=None,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        solution_with_evidence=False,
        write_uai_file=None,
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
        if timing:
            timer = TicTocTimer()
            timer.tic("DPGM_VariableEliminationInference.map_query - START")

        if isinstance(
            self.pgm,
            (DynamicDiscreteBayesianNetwork, pgmpy.models.DynamicBayesianNetwork),
        ):
            if isinstance(self.pgm, pgmpy.models.DynamicBayesianNetwork):
                pgm = convert_pgmpy_to_conin(self.pgm)
            else:
                pgm = self.pgm
            if stop is None:
                stop = 1
            conin_bn = create_bn_from_dbn(dbn=pgm, start=start, stop=stop)
            if write_uai_file:
                save_model(conin_bn, write_uai_file)
            pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

            if timing:
                timer.tic("Created PGMPY model")
            states, solvetime = self._run_map_query(
                pgmpy_bn=pgmpy_bn,
                variables=variables,
                evidence=evidence,
                show_progress=show_progress,
            )
            if solution_with_evidence and evidence:
                states.update(evidence)

        elif isinstance(self.pgm, ConstrainedDynamicDiscreteBayesianNetwork):
            evidence_ = copy.copy(evidence) if evidence else {}

            if stop is None:
                stop = 1
            conin_bn = create_bn_from_dbn(dbn=self.pgm.pgm, start=start, stop=stop)
            data = munch.Munch(T=list(range(start, stop + 1)))
            self._add_constraints_as_evidence(
                conin_bn=conin_bn,
                constraints=self.pgm.constraints,
                data=data,
                evidence=evidence_,
            )
            if write_uai_file:
                save_model(conin_bn, write_uai_file)
            pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

            if timing:
                timer.tic("Created PGMPY model")
            states, solvetime = self._run_map_query(
                pgmpy_bn=pgmpy_bn,
                variables=variables,
                evidence=evidence_,
                show_progress=show_progress,
            )
            if solution_with_evidence and evidence:
                states.update(evidence)

        elif isinstance(self.pgm, HiddenMarkovModel):
            stop = len(evidence) - 1
            conin_dbn = create_dbn_from_hmm(self.pgm)
            conin_bn = create_bn_from_dbn(dbn=conin_dbn, start=start, stop=stop)
            if write_uai_file:
                save_model(conin_bn, write_uai_file)
            pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

            if timing:
                timer.tic("Created PGMPY model")
            evidence_ = self._hmm_evidence_to_pgmpy(evidence)
            map_states, solvetime = self._run_map_query(
                pgmpy_bn=pgmpy_bn,
                variables=variables,
                evidence=evidence_,
                show_progress=show_progress,
            )
            states = self._hmm_states_from_map(map_states, evidence)

        elif isinstance(self.pgm, ConstrainedHiddenMarkovModel):
            evidence_ = self._hmm_evidence_to_pgmpy(evidence)

            stop = len(evidence) - 1
            conin_dbn = create_dbn_from_hmm(self.pgm.hidden_markov_model)
            conin_bn = create_bn_from_dbn(dbn=conin_dbn, start=start, stop=stop)
            data = munch.Munch(hmm=munch.Munch(T=list(range(len(evidence)))))
            self._add_constraints_as_evidence(
                conin_bn=conin_bn,
                constraints=self.pgm.constraints,
                data=data,
                evidence=evidence_,
            )
            if write_uai_file:
                save_model(conin_bn, write_uai_file)
            pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

            if timing:
                timer.tic("Created PGMPY model")
            map_states, solvetime = self._run_map_query(
                pgmpy_bn=pgmpy_bn,
                variables=variables,
                evidence=evidence_,
                show_progress=show_progress,
            )
            states = self._hmm_states_from_map(map_states, evidence)

        else:
            raise TypeError(f"Unexpected model type: {type(self.pgm)}")

        if timing:
            timer.tic("DPGM_VariableEliminationInference.map_query - STOP")
        return munch.Munch(solution=munch.Munch(states=states), solvetime=solvetime)
