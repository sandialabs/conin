"""Every example in conin.examples builds, and solves to its documented solution.

Some example builders were never called by a test or a doc page.  The model
examples return ``Munch(pgm=..., solutions=[MPESolution, ...])``; their first
solution is checked with the method that matches the constraint family (the
weather DBNs are documented for ``stop=4``; their variable-elimination examples
are only built, since VE on them needs a lot of memory).
"""

import importlib
import inspect

import pytest

from conin.config import default_mip_solver
from conin.constrained_models import ConstrainedGraphicalModel
from conin.graphical_models import GraphicalModel
from conin.inference import map_query
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

with try_import() as smoek_available:
    import smoek  # noqa: F401

MODULES = ["markov_network", "bayesian_network", "dynamic_bayesian_network"]
HMM_MODULE = "hidden_markov_model"


def _builders(module_name):
    module = importlib.import_module(f"conin.examples.{module_name}")
    return [
        pytest.param(f, id=name)
        for name, f in inspect.getmembers(module, inspect.isfunction)
        if f.__module__ == module.__name__ and not name.startswith("_")
    ]


def _needs(name):
    """The optional packages an example needs, from its name."""
    needs = []
    if "pgmpy" in name:
        needs.append(("pgmpy", pgmpy_available))
    if "algebraic" in name:
        needs.append(("smoek", smoek_available))
    return needs


def _skip_unless_available(name):
    for package, available in _needs(name):
        if not available:
            pytest.skip(f"{name} needs {package}")


METHOD = {
    None: "integer_program",
    "pyomo": "integer_program",
    "algebraic": "integer_program",
    "toulbar2": "toulbar2",
    "factor_oracle": "variable_elimination",
}


@pytest.mark.parametrize("builder", [p for m in MODULES for p in _builders(m)])
def test_model_example_builds(builder):
    _skip_unless_available(builder.__name__)
    example = builder()
    pgm = getattr(example, "pgm", example)
    if type(pgm).__module__.startswith("pgmpy"):
        # The *_pgmpy examples of unconstrained models are pgmpy models
        # (pgmpy_issue_1177_pgmpy returns one directly).
        pgm.check_model()
        return
    assert isinstance(pgm, (GraphicalModel, ConstrainedGraphicalModel))
    if hasattr(example, "solutions"):
        assert example.solutions


def _solvable():
    """(builder, method, structure kwargs) for the examples with a solution."""
    cases = []
    for module in MODULES:
        for param in _builders(module):
            name = param.id
            if name in ("example6_conin", "example6_pgmpy", "pgmpy_issue_1177_pgmpy"):
                continue  # no documented solution
            kwargs = {}
            if "weather" in name:
                kwargs = dict(stop=4)
                if "oracle" in name:
                    continue  # variable elimination on weather needs a lot of memory
            elif module == "dynamic_bayesian_network":
                kwargs = dict(stop=1)
            cases.append(pytest.param(param.values[0], kwargs, id=name))
    return cases


@pytest.mark.parametrize("builder,kwargs", _solvable())
def test_model_example_solution(builder, kwargs):
    _skip_unless_available(builder.__name__)
    example = builder()
    family = getattr(example.pgm, "constraint_family", None)
    method = METHOD[family]
    if method == "integer_program" and not default_mip_solver:
        pytest.skip("No mip solver installed")
    if method == "toulbar2" and not pytoulbar2_available:
        pytest.skip("pytoulbar2 not installed")
    if method == "variable_elimination" and not pgmpy_available:
        pytest.skip("pgmpy not installed")
    results = map_query(example.pgm, method=method, **kwargs)
    assert results.solution.states == example.solutions[0].states


@pytest.mark.parametrize("builder", _builders(HMM_MODULE))
def test_hmm_example_builds(builder):
    _skip_unless_available(builder.__name__)
    from conin.constrained_models import ConstrainedHiddenMarkovModel
    from conin.graphical_models import HiddenMarkovModel

    model = builder()
    assert isinstance(model, (HiddenMarkovModel, ConstrainedHiddenMarkovModel))
    hmm = getattr(model, "model", model)
    assert hmm.hidden_states and hmm.observed_states
