import copy
import munch
from ovld import ovld
from pyomo.common.timing import TicTocTimer

from conin.util import try_import
from conin.common.unified import save_model

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

from conin.constraints import FactorConstraint
from conin.constraints.algebraic import (
    create_factor_constraints_from_algebraic,
    AlgebraicConstraint,
)
from conin.common.conin import convert_conin_to_pgmpy_mn, convert_conin_to_pgmpy_bn

with try_import() as pgmpy_available:
    import pgmpy.models
    import pgmpy.inference
    from conin.common.pgmpy import convert_pgmpy_to_conin


def _run_map_query_helper(pgmpy_bn, variables, evidence, show_progress):
    """Helper to run MAP query on a pgmpy Bayesian network."""
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
    )
    solvetime = solver_timer.toc(None)
    return map_states, solvetime


def _hmm_evidence_to_pgmpy(evidence):
    """Convert HMM evidence to pgmpy DBN naming scheme."""
    if isinstance(evidence, list):
        return {("E", i): v for i, v in enumerate(evidence)}
    if isinstance(evidence, dict):
        return {("E", i): v for i, v in evidence.items()}
    return evidence


def _hmm_states_from_map(map_states, evidence):
    """Project pgmpy MAP assignments back to HMM hidden-state outputs."""
    if isinstance(evidence, list):
        return [map_states["H", i] for i in range(len(map_states))]
    if isinstance(evidence, dict):
        return {i: map_states["H", i] for i in range(len(map_states))}
    return map_states


def _add_bn_constraints_as_evidence(pgm, constraints, data, evidence, copy_pgm=False):
    """Inject generated constraint CPDs into a Bayesian network as evidence."""
    if len(constraints) == 0:
        return

    if copy_pgm:
        pgm = copy.deepcopy(pgm)
    if isinstance(constraints[0], AlgebraicConstraint):
        constraints = create_factor_constraints_from_algebraic(
            pgm=pgm, constraints=constraints, data=data
        )
    for con in constraints:
        if isinstance(con, FactorConstraint):
            cpd = con(pgm, data)
            cpd.node = (cpd.node, -1)
            pgm.add_cpd(cpd)
            evidence[cpd.node] = 1
        else:
            raise TypeError(
                "Unexpected constraint type {type(con)} for VariableElimination"
            )
    return pgm


def _add_mn_constraints_as_evidence(pgm, constraints, data, evidence, copy_pgm=False):
    """Inject generated constraint CPDs into a Bayesian network as evidence."""
    if len(constraints) == 0:
        return

    if copy_pgm:
        pgm = copy.deepcopy(pgm)
    if isinstance(constraints[0], AlgebraicConstraint):
        constraints = create_factor_constraints_from_algebraic(
            pgm=pgm, constraints=constraints, data=data
        )
    for con in constraints:
        if isinstance(con, FactorConstraint):
            factor = con(pgm, data)
            pgm._factors.append(factor)
            evidence[factor.nodes[-1]] = 1
        else:
            raise TypeError(
                "Unexpected constraint type {type(con)} for VariableElimination"
            )
    return pgm


def _prepare_evidence(evidence):
    """Prepare evidence dictionary for query."""
    return copy.copy(evidence) if evidence else {}


def _determine_variables(pgm, variables, evidence):
    """Determine query variables if not explicitly provided."""
    if variables is None:
        if evidence:
            return [node for node in pgm.nodes if node not in evidence]
        else:
            return pgm.nodes
    return variables


def _create_result(states, solvetime):
    """Create standardized result structure."""
    return munch.Munch(solution=munch.Munch(states=states), solvetime=solvetime)


def _with_timing(timing, func):
    """Execute function with optional timing."""
    if timing:
        timer = TicTocTimer()
        timer.tic("map_query - START")

    result = func()

    if timing:
        timer.tic("map_query - STOP")

    return result


def _execute_simple_model_query(
    pgm,
    conversion_func,
    variables,
    evidence,
    show_progress,
    timing,
    write_uai_file,
    solution_with_evidence,
):
    """Execute MAP query for unconstrained Markov/Bayesian networks."""

    def _execute():
        evidence_prepared = _prepare_evidence(evidence)

        if write_uai_file:
            save_model(pgm, write_uai_file)

        pgmpy_model = conversion_func(pgm)

        if timing:
            TicTocTimer().tic("Created PGMPY model")

        vars_determined = _determine_variables(pgm, variables, evidence_prepared)
        map_states, solvetime = _run_map_query_helper(
            pgmpy_model, vars_determined, evidence_prepared, show_progress
        )
        if solution_with_evidence and evidence:
            map_states.update(evidence)

        return _create_result(map_states, solvetime)

    return _with_timing(timing, _execute)


def _execute_pgmpy_model_query(
    pgm,
    variables,
    evidence,
    show_progress,
    timing,
):
    """Execute MAP query for native pgmpy models (no conversion needed)."""

    def _execute():
        evidence_prepared = _prepare_evidence(evidence)
        vars_determined = _determine_variables(pgm, variables, evidence_prepared)

        if timing:
            TicTocTimer().tic("Created PGMPY model")

        map_states, solvetime = _run_map_query_helper(
            pgm, vars_determined, evidence_prepared, show_progress
        )

        return _create_result(map_states, solvetime)

    return _with_timing(timing, _execute)


# -----------------------------------------------------------------------------------------
# map_query functions
# -----------------------------------------------------------------------------------------


@ovld
def _map_query_VariableElimination(
    pgm,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    """Compute a MAP assignment using pgmpy's Variable Elimination with multiple dispatch."""
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. "
        f"Expected one of: DiscreteMarkovNetwork, ConstrainedDiscreteMarkovNetwork, "
        f"DiscreteBayesianNetwork, ConstrainedDiscreteBayesianNetwork, "
        f"DynamicDiscreteBayesianNetwork, ConstrainedDynamicDiscreteBayesianNetwork, "
        f"HiddenMarkovModel, ConstrainedHiddenMarkovModel, "
        f"pgmpy.models.DiscreteMarkovNetwork, pgmpy.models.DiscreteBayesianNetwork, "
        f"pgmpy.models.DynamicBayesianNetwork."
    )


@ovld
def _map_query_VariableElimination(
    pgm: DiscreteMarkovNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return _execute_simple_model_query(
        pgm,
        convert_conin_to_pgmpy_mn,
        variables,
        evidence,
        show_progress,
        timing,
        options.get("write_uai_file"),
        options.get("solution_with_evidence", False),
    )


@ovld
def _map_query_VariableElimination(
    pgm: ConstrainedDiscreteMarkovNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    # Apply constraints as factors
    if pgm.constraints:
        evidence = {}
        newpgm = _add_mn_constraints_as_evidence(
            pgm.pgm, pgm.constraints, None, evidence, copy_pgm=True
        )
    else:
        newpgm = pgm.pgm

    return _execute_simple_model_query(
        newpgm,
        convert_conin_to_pgmpy_mn,
        variables,
        evidence,
        show_progress,
        timing,
        options.get("write_uai_file"),
        options.get("solution_with_evidence", False) or (len(pgm.constraints) > 0),
    )


@ovld
def _map_query_VariableElimination(
    pgm: DiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return _execute_simple_model_query(
        pgm,
        convert_conin_to_pgmpy_bn,
        variables,
        evidence,
        show_progress,
        timing,
        options.get("write_uai_file"),
        options.get("solution_with_evidence", False),
    )


@ovld
def _map_query_VariableElimination(
    pgm: ConstrainedDiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    prepared_evidence = _prepare_evidence(evidence)

    # Apply constraints as CPDs with evidence
    if pgm.constraints:
        newpgm = _add_bn_constraints_as_evidence(
            pgm.pgm, pgm.constraints, None, prepared_evidence, copy_pgm=True
        )
    else:
        newpgm = pgm.pgm

    # Execute with prepared evidence - pass None for variables to re-determine after constraint
    write_uai_file = options.get("write_uai_file")
    if write_uai_file:
        save_model(newpgm, write_uai_file)

    pgmpy_model = convert_conin_to_pgmpy_bn(newpgm)
    solution_with_evidence = (
        options.get("solution_with_evidence", False) or (len(pgm.constraints) > 0),
    )

    def _execute():
        if timing:
            TicTocTimer().tic("Created PGMPY model")

        vars_determined = _determine_variables(newpgm, variables, prepared_evidence)
        map_states, solvetime = _run_map_query_helper(
            pgmpy_model, vars_determined, prepared_evidence, show_progress
        )
        if (solution_with_evidence or (len(pgm.constraints) > 0)) and evidence:
            map_states.update(evidence)

        return _create_result(map_states, solvetime)

    return _with_timing(timing, _execute)


@ovld
def _map_query_VariableElimination(
    pgm: DynamicDiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    write_uai_file = options.get("write_uai_file")
    solution_with_evidence = options.get("solution_with_evidence", False)

    def _execute():
        if stop is None:
            stop_val = 1
        else:
            stop_val = stop

        conin_bn = create_bn_from_dbn(dbn=pgm, start=start, stop=stop_val)
        if write_uai_file:
            save_model(conin_bn, write_uai_file)
        pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

        if timing:
            TicTocTimer().tic("Created PGMPY model")

        states, solvetime = _run_map_query_helper(
            pgmpy_bn=pgmpy_bn,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
        )
        if solution_with_evidence and evidence:
            states.update(evidence)

        return _create_result(states, solvetime)

    return _with_timing(timing, _execute)


@ovld
def _map_query_VariableElimination(
    pgm: ConstrainedDynamicDiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    write_uai_file = options.get("write_uai_file")
    solution_with_evidence = options.get("solution_with_evidence", False)

    def _execute():
        evidence_ = _prepare_evidence(evidence)

        if stop is None:
            stop_val = 1
        else:
            stop_val = stop

        conin_bn = create_bn_from_dbn(dbn=pgm.pgm, start=start, stop=stop_val)
        data = munch.Munch(T=list(range(start, stop_val + 1)))
        _add_bn_constraints_as_evidence(
            pgm=conin_bn,
            constraints=pgm.constraints,
            data=data,
            evidence=evidence_,
        )
        if write_uai_file:
            save_model(conin_bn, write_uai_file)
        pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

        if timing:
            TicTocTimer().tic("Created PGMPY model")

        states, solvetime = _run_map_query_helper(
            pgmpy_bn=pgmpy_bn,
            variables=variables,
            evidence=evidence_,
            show_progress=show_progress,
        )
        if (solution_with_evidence or (len(pgm.constraints) > 0)) and evidence:
            states.update(evidence)

        return _create_result(states, solvetime)

    return _with_timing(timing, _execute)


@ovld
def _map_query_VariableElimination(
    pgm: HiddenMarkovModel,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    write_uai_file = options.get("write_uai_file")

    def _execute():
        stop_val = len(evidence) - 1
        conin_dbn = create_dbn_from_hmm(pgm)
        conin_bn = create_bn_from_dbn(dbn=conin_dbn, start=start, stop=stop_val)
        if write_uai_file:
            save_model(conin_bn, write_uai_file)
        pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

        if timing:
            TicTocTimer().tic("Created PGMPY model")

        evidence_ = _hmm_evidence_to_pgmpy(evidence)
        map_states, solvetime = _run_map_query_helper(
            pgmpy_bn=pgmpy_bn,
            variables=variables,
            evidence=evidence_,
            show_progress=show_progress,
        )
        states = _hmm_states_from_map(map_states, evidence)

        return _create_result(states, solvetime)

    return _with_timing(timing, _execute)


@ovld
def _map_query_VariableElimination(
    pgm: ConstrainedHiddenMarkovModel,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    write_uai_file = options.get("write_uai_file")

    def _execute():
        evidence_ = _hmm_evidence_to_pgmpy(evidence)

        stop_val = len(evidence) - 1
        conin_dbn = create_dbn_from_hmm(pgm.hidden_markov_model)
        conin_bn = create_bn_from_dbn(dbn=conin_dbn, start=start, stop=stop_val)
        data = munch.Munch(hmm=munch.Munch(T=list(range(len(evidence)))))
        _add_bn_constraints_as_evidence(
            pgm=conin_bn,
            constraints=pgm.constraints,
            data=data,
            evidence=evidence_,
        )
        if write_uai_file:
            save_model(conin_bn, write_uai_file)
        pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

        if timing:
            TicTocTimer().tic("Created PGMPY model")

        map_states, solvetime = _run_map_query_helper(
            pgmpy_bn=pgmpy_bn,
            variables=variables,
            evidence=evidence_,
            show_progress=show_progress,
        )
        states = _hmm_states_from_map(map_states, evidence)

        return _create_result(states, solvetime)

    return _with_timing(timing, _execute)


if pgmpy_available:

    @ovld
    def _map_query_VariableElimination(
        pgm: pgmpy.models.DiscreteMarkovNetwork,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
        options={},
    ):
        return _execute_pgmpy_model_query(
            pgm,
            variables,
            evidence,
            show_progress,
            timing,
        )

    @ovld
    def _map_query_VariableElimination(
        pgm: pgmpy.models.DiscreteBayesianNetwork,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
        options={},
    ):
        return _execute_pgmpy_model_query(
            pgm,
            variables,
            evidence,
            show_progress,
            timing,
        )

    @ovld
    def _map_query_VariableElimination(
        pgm: pgmpy.models.DynamicBayesianNetwork,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
        options={},
    ):
        write_uai_file = options.get("write_uai_file")
        solution_with_evidence = options.get("solution_with_evidence", False)

        def _execute():
            pgm_converted = convert_pgmpy_to_conin(pgm)
            if stop is None:
                stop_val = 1
            else:
                stop_val = stop

            conin_bn = create_bn_from_dbn(dbn=pgm_converted, start=start, stop=stop_val)
            if write_uai_file:
                save_model(conin_bn, write_uai_file)
            pgmpy_bn = convert_conin_to_pgmpy_bn(conin_bn)

            if timing:
                TicTocTimer().tic("Created PGMPY model")

            states, solvetime = _run_map_query_helper(
                pgmpy_bn=pgmpy_bn,
                variables=variables,
                evidence=evidence,
                show_progress=show_progress,
            )
            if solution_with_evidence and evidence:
                states.update(evidence)

            return _create_result(states, solvetime)

        return _with_timing(timing, _execute)
