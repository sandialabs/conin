import os.path
import tempfile
import munch
from pyomo.common.timing import TicTocTimer

from conin.util import try_import

with try_import() as pytoulbar2_available:
    import pytoulbar2

import conin.common
from conin.markov_network import ConstrainedDiscreteMarkovNetwork


class VarWrapper(object):
    def __init__(self, pgm):
        self._V = {name: i for i, name in enumerate(pgm.nodes)}
        self._V_state = {
            (name, state): i
            for name in pgm.nodes
            for i, state in enumerate(pgm.states_of(name))
        }

    def __call__(self, *args, coef=1):
        if len(args) == 2:
            r, s = args
        elif len(args) == 3:
            r, i, s = args
            r = (r, i)
        else:
            raise ValueError("There must be either 2 or 3 arguments")

        return (self._V[r], self._V_state[r, s], coef)

    def __getitem__(self, r):
        return self._V[r]

    def items(self):
        for k, v in self._V.items():
            yield k, v


def create_toulbar2_map_query_model_MN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    **options,
):
    """
    Ignoring variables for now
    """
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("create_toulbar2_map_query_model_MN - START")
    verbose = options.pop("verbose", -1)

    cpgm = pgm if isinstance(pgm, ConstrainedDiscreteMarkovNetwork) else None
    pgm = cpgm.pgm if cpgm is not None else pgm

    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "model.uai")
        conin.common.save_model(pgm, filename)
        model = pytoulbar2.CFN(verbose=verbose)
        model.Read(filename)

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
        timer.toc("create_toulbar2_map_query_model_MN - STOP")
    return model


def solve_toulbar2_map_query_model(
    model,
    *,
    solution_with_fixed=False,
    solution_with_evidence=False,
    timing=False,
):
    solution_with_evidence = solution_with_fixed or solution_with_evidence
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("CFN_map_query - START")

    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    res = model.Solve()
    solvetime = solver_timer.toc(None)
    solution, primal_bound, num_solutions = res
    var = {
        name: model.states[i][solution[i]]
        for name, i in model.V.items()
        if solution_with_evidence or name not in model.V_evidence
    }
    soln = munch.Munch(states=var, log_factor_sum=None, primal_bound=primal_bound)

    if timing:  # pragma:nocover
        timer.toc("Completed optimization")

    return munch.Munch(
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
        solvetime=solvetime,
    )


def inference_toulbar2_map_query_MN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    **options,
):
    if not pytoulbar2_available:  # pragma:nocover
        return munch.Munch(
            solution=None,
            solutions=[],
            termination_condition="pytoulbar2 not available",
            solvetime=0.0,
        )

    model = create_toulbar2_map_query_model_MN(
        pgm=pgm, variables=variables, evidence=evidence, timing=timing, **options
    )
    return solve_toulbar2_map_query_model(model, timing=timing, **options)
