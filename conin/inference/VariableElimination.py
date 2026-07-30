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
    """Run MAP inference with pgmpy's variable elimination implementation.

    This wrapper supports static discrete Bayesian networks and Markov
    networks, including constrained CONIN variants that are converted into a
    pgmpy-compatible form before solving.
    """

    def __init__(self, pgm):
        """Store a model for subsequent variable elimination MAP queries.

        Parameters
        ----------
        pgm : DiscreteMarkovNetwork or ConstrainedDiscreteMarkovNetwork or DiscreteBayesianNetwork or ConstrainedDiscreteBayesianNetwork or pgmpy.models.DiscreteMarkovNetwork or pgmpy.models.DiscreteBayesianNetwork
            Graphical model to solve with pgmpy's variable elimination backend.

        Raises
        ------
        AssertionError
            If pgmpy is not installed.
        """
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
        """Compute a MAP assignment using pgmpy's variable elimination solver.

        Parameters
        ----------
        variables : list, optional
            Variables included in the MAP query. When omitted, the solver uses
            all model variables not fixed by ``evidence``.
        evidence : dict, optional
            Observed variable assignments as ``{variable: state}``. Use ``None``
            when no evidence is supplied.
        show_progress : bool, optional
            Whether to request progress reporting from pgmpy.
        timing : bool, optional
            If ``True``, collect setup timing information and include solver
            runtime in the returned result.
        write_uai_file : str, optional
            Path where the converted CONIN model should be written in UAI format
            before solving.
        **options : dict, optional
            Additional keyword arguments forwarded to
            ``pgmpy.inference.VariableElimination.map_query``.

        Returns
        -------
        munch.Munch
            Result object with ``solution.states`` containing the MAP
            assignment and ``solvetime`` containing the measured pgmpy solver
            runtime.

        Raises
        ------
        TypeError
            If ``self.pgm`` is not a supported static graphical model type.

        Notes
        -----
        Constrained CONIN models are materialized by injecting their generated
        factors or CPDs into the underlying unconstrained model prior to
        conversion to pgmpy.
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
    """Run pgmpy variable elimination for dynamic models and HMM-derived forms.

    This wrapper expands dynamic Bayesian networks and hidden Markov models into
    static Bayesian networks before solving them with pgmpy's variable
    elimination backend.
    """

    def __init__(self, pgm):
        """Store a dynamic model for subsequent variable elimination MAP queries.

        Parameters
        ----------
        pgm : DynamicDiscreteBayesianNetwork or ConstrainedDynamicDiscreteBayesianNetwork or HiddenMarkovModel or ConstrainedHiddenMarkovModel or pgmpy.models.DynamicBayesianNetwork
            Dynamic graphical model to solve.

        Raises
        ------
        AssertionError
            If pgmpy is not installed.
        """
        assert (
            pgmpy_available
        ), "PGMPY must be installed to perform inference with DPGM_VariableElimination"
        self.pgm = pgm

    def _run_map_query(self, pgmpy_bn, variables, evidence, show_progress, **options):
        """Solve a pgmpy Bayesian network MAP query and time the solve phase.

        Parameters
        ----------
        pgmpy_bn : pgmpy.models.DiscreteBayesianNetwork
            Static Bayesian network supplied to pgmpy's variable elimination
            solver.
        variables : list, optional
            Variables included in the MAP query. When ``None``, all nodes not
            fixed by ``evidence`` are queried.
        evidence : dict, optional
            Observed node assignments in pgmpy naming conventions.
        show_progress : bool
            Whether to request progress reporting from pgmpy.
        **options : dict, optional
            Additional keyword arguments forwarded to
            ``pgmpy.inference.VariableElimination.map_query``.

        Returns
        -------
        tuple
            Two-element tuple ``(map_states, solvetime)`` containing the MAP
            assignment returned by pgmpy and the measured solver runtime.
        """
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
        """Convert HMM evidence into the pgmpy-expanded DBN naming scheme.

        Parameters
        ----------
        evidence : list or dict or None
            HMM evidence supplied as a dense list of observations or a
            dictionary keyed by time index.

        Returns
        -------
        dict or None
            Evidence keyed by pgmpy DBN variable names such as ``("E", t)``.
            Inputs that are already ``None`` or in another format are returned
            unchanged.
        """
        if isinstance(evidence, list):
            return {("E", i): v for i, v in enumerate(evidence)}
        if isinstance(evidence, dict):
            return {("E", i): v for i, v in evidence.items()}
        return evidence

    def _hmm_states_from_map(self, map_states, evidence):
        """Project pgmpy MAP assignments back to HMM hidden-state outputs.

        Parameters
        ----------
        map_states : dict
            MAP assignment returned by the pgmpy solver on the expanded dynamic
            Bayesian network.
        evidence : list or dict or None
            Original HMM evidence format used to decide whether to return a list
            or dictionary of hidden states.

        Returns
        -------
        list or dict
            Hidden-state sequence formatted to match the evidence style when
            possible.
        """
        if isinstance(evidence, list):
            return [map_states["H", i] for i in range(len(map_states))]
        if isinstance(evidence, dict):
            return {i: map_states["H", i] for i in range(len(map_states))}
        return map_states

    def _add_constraints_as_evidence(self, conin_bn, constraints, data, evidence):
        """Inject generated constraint CPDs into a Bayesian network as evidence.

        Parameters
        ----------
        conin_bn : DiscreteBayesianNetwork
            Static Bayesian network that will receive additional CPDs encoding
            the constraints.
        constraints : iterable
            Constraint callbacks that generate CPDs from ``conin_bn`` and
            ``data``.
        data : munch.Munch
            Auxiliary data structure passed to each constraint callback.
        evidence : dict
            Evidence dictionary that is updated so each generated constraint CPD
            node is fixed to state ``1``.
        """
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
        """Compute a MAP assignment for a dynamic model with variable elimination.

        Parameters
        ----------
        start : int, optional
            Initial time index included in the unrolled dynamic model.
        stop : int, optional
            Final time index included in the unrolled dynamic model. For hidden
            Markov models, the implementation derives this from the evidence
            length when needed.
        variables : list, optional
            Variables included in the MAP query on the expanded static Bayesian
            network. When omitted, all non-evidence nodes are queried.
        evidence : dict or list, optional
            Evidence for the selected model family. Dynamic Bayesian network
            inputs use a dictionary keyed by time-expanded variable names, while
            hidden Markov models accept either a dense list of observations or a
            dictionary keyed by time index.
        show_progress : bool, optional
            Whether to request progress reporting from pgmpy.
        timing : bool, optional
            If ``True``, collect setup timing information and include solver
            runtime in the returned result.
        solution_with_evidence : bool, optional
            If ``True`` and dictionary-valued evidence is supplied for dynamic
            Bayesian network models, merge the evidence assignments into the
            returned state dictionary.
        write_uai_file : str, optional
            Path where the expanded static Bayesian network should be written in
            UAI format before solving.
        **options : dict, optional
            Additional keyword arguments forwarded to
            ``pgmpy.inference.VariableElimination.map_query``.

        Returns
        -------
        munch.Munch
            Result object with ``solution.states`` containing the MAP
            assignment and ``solvetime`` containing the measured pgmpy solver
            runtime.

        Raises
        ------
        TypeError
            If ``self.pgm`` is not a supported dynamic Bayesian network or
            hidden Markov model type.

        Notes
        -----
        Hidden Markov models are first converted to dynamic Bayesian networks
        and then unrolled into static Bayesian networks before inference.
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
