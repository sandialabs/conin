import os
import pyomo.opt
from conin.util import try_import
from conin.inference.map_query import map_query
import conin.markov_network.examples
import conin.bayesian_network.examples
import conin.hidden_markov_model.examples
import conin.dynamic_bayesian_network.examples

with try_import() as pgmpy_available:
    import pgmpy

with try_import() as pytoulbar2_available:
    import pytoulbar2

with try_import() as smoek_available:
    import smoek

import pytest

# Check for available MIP solver
mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs")
mip_solver = mip_solver[0] if mip_solver else None

# Skip conditions
skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)
skipif_smoek_not_available = pytest.mark.skipif(
    not smoek_available, reason="smoek not installed"
)
skipif_no_mip_solver = pytest.mark.skipif(
    not mip_solver, reason="No mip solver installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)

# Test file paths
cwd = os.path.dirname(__file__)
testfile_uai = os.path.join(cwd, "test.uai")
testfile_lp = os.path.join(cwd, "test.lp")


#
# DiscreteMarkovNetwork tests
#


@pytest.mark.parametrize(
    "method,solver,write_file",
    [
        pytest.param(
            "toulbar2", None, testfile_uai, marks=skipif_toulbar2_not_available
        ),
        pytest.param(
            "integer_program", mip_solver, testfile_lp, marks=skipif_no_mip_solver
        ),
        pytest.param(
            "variable_elimination", None, testfile_uai, marks=skipif_pgmpy_not_available
        ),
    ],
)
def test_ABC_conin(method, solver, write_file):
    example = conin.markov_network.examples.ABC_conin()

    # Test without file writing
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


@skipif_pgmpy_not_available
@pytest.mark.parametrize(
    "method,solver",
    [
        pytest.param("toulbar2", None, marks=skipif_toulbar2_not_available),
        pytest.param("integer_program", mip_solver, marks=skipif_no_mip_solver),
        pytest.param("variable_elimination", None),
    ],
)
def test_ABC_pgmpy(method, solver):
    example = conin.markov_network.examples.ABC_pgmpy()
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states


#
# ConstrainedDiscreteMarkovNetwork tests
#


@pytest.mark.parametrize(
    "method,solver,write_file,example_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.markov_network.examples.ABC_constrained_toulbar2_conin,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.markov_network.examples.ABC_constrained_algebraic_conin,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.markov_network.examples.ABC_constrained_pyomo_conin,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.markov_network.examples.ABC_constrained_algebraic_conin,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.markov_network.examples.ABC_constrained_oracle_conin,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.markov_network.examples.ABC_constrained_algebraic_conin,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test_ABC_constrained_conin(method, solver, write_file, example_factory):
    example = example_factory()

    # Test without file writing
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


@skipif_pgmpy_not_available
def test_ABC2_constrained_oracle_conin():
    """Additional test case only available for variable elimination."""
    example = conin.markov_network.examples.ABC2_constrained_oracle_conin()
    results = map_query(example.pgm, method="variable_elimination")
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float


#
# DiscreteBayesianNetwork tests
#


@pytest.mark.parametrize(
    "method,solver,write_file",
    [
        pytest.param(
            "toulbar2", None, testfile_uai, marks=skipif_toulbar2_not_available
        ),
        pytest.param(
            "integer_program", mip_solver, testfile_lp, marks=skipif_no_mip_solver
        ),
        pytest.param(
            "variable_elimination", None, testfile_uai, marks=skipif_pgmpy_not_available
        ),
    ],
)
def test_cancer1_BN_conin(method, solver, write_file):
    example = conin.bayesian_network.examples.cancer1_BN_conin()

    # Test without file writing
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


@skipif_pgmpy_not_available
@pytest.mark.parametrize(
    "method,solver",
    [
        pytest.param("toulbar2", None, marks=skipif_toulbar2_not_available),
        pytest.param("integer_program", mip_solver, marks=skipif_no_mip_solver),
        pytest.param("variable_elimination", None),
    ],
)
def test_cancer1_BN_pgmpy(method, solver):
    example = conin.bayesian_network.examples.cancer1_BN_pgmpy()
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states


@pytest.mark.parametrize(
    "method,solver,write_file",
    [
        pytest.param(
            "toulbar2", None, testfile_uai, marks=skipif_toulbar2_not_available
        ),
        pytest.param(
            "integer_program", mip_solver, testfile_lp, marks=skipif_no_mip_solver
        ),
        pytest.param(
            "variable_elimination", None, testfile_uai, marks=skipif_pgmpy_not_available
        ),
    ],
)
def test_wide_BN_conin(method, solver, write_file):
    example = conin.bayesian_network.examples.wide_BN_conin()

    # Test without file writing
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


#
# ConstrainedBayesianNetwork tests
#


@pytest.mark.parametrize(
    "method,solver,write_file,example_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.bayesian_network.examples.cancer1_BN_constrained_toulbar2_conin,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.bayesian_network.examples.cancer1_BN_constrained_algebraic_conin,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.bayesian_network.examples.cancer1_BN_constrained_pyomo_conin,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.bayesian_network.examples.cancer1_BN_constrained_algebraic_conin,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.bayesian_network.examples.cancer1_BN_constrained_oracle_conin,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.bayesian_network.examples.cancer1_BN_constrained_algebraic_conin,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test_cancer1_BN_constrained_conin(method, solver, write_file, example_factory):
    example = example_factory()

    # Test without file writing
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


@skipif_pgmpy_not_available
def test_cancer1_BN_constrained_oracle_pgmpy():
    """Test cancer1 constrained BN with pgmpy - only for variable elimination."""
    example = conin.bayesian_network.examples.cancer1_BN_constrained_oracle_pgmpy()
    results = map_query(example.pgm, method="variable_elimination")
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float


@pytest.mark.parametrize(
    "method,solver,write_file,example_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained1_conin_toulbar2,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained1_conin_algebraic,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.bayesian_network.examples.wide_BN_constrained1_conin_pyomo,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.bayesian_network.examples.wide_BN_constrained1_conin_algebraic,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained1_conin_oracle,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained1_conin_algebraic,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test_wide_BN_constrained1_conin(method, solver, write_file, example_factory):
    example = example_factory()

    # Test without file writing
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


@pytest.mark.parametrize(
    "method,solver,write_file,example_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained2_conin_toulbar2,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained2_conin_algebraic,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.bayesian_network.examples.wide_BN_constrained2_conin_pyomo,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.bayesian_network.examples.wide_BN_constrained2_conin_algebraic,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained2_conin_oracle,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.bayesian_network.examples.wide_BN_constrained2_conin_algebraic,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test_wide_BN_constrained2_conin(method, solver, write_file, example_factory):
    example = example_factory()

    # Test without file writing
    kwargs = {"method": method}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


#
# HiddenMarkovModel tests
#


@pytest.mark.parametrize(
    "method,solver,write_file,ip_formulation",
    [
        pytest.param(
            "toulbar2", None, testfile_uai, None, marks=skipif_toulbar2_not_available
        ),
        pytest.param(
            "integer_program", mip_solver, testfile_lp, None, marks=skipif_no_mip_solver
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            "markov_network",
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            "network_flow",
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            None,
            marks=skipif_pgmpy_not_available,
        ),
    ],
)
def test0_hmm1(method, solver, write_file, ip_formulation):
    pgm = conin.hidden_markov_model.examples.create_hmm1()
    observed = ["o0", "o0", "o1", "o0", "o0"]

    # Test without file writing
    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    if ip_formulation:
        kwargs["ip_formulation"] = ip_formulation
    results = map_query(pgm, **kwargs)
    assert results.solution.states == ["h0", "h0", "h0", "h0", "h0"]
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing (only for default ip_formulation to avoid redundancy)
    if ip_formulation is None:
        if method == "integer_program":
            kwargs["write_lp_file"] = write_file
        else:
            kwargs["write_uai_file"] = write_file
        results = map_query(pgm, **kwargs)
        assert os.path.exists(write_file)
        os.remove(write_file)


@pytest.mark.parametrize(
    "method,solver,ip_formulation",
    [
        pytest.param("toulbar2", None, None, marks=skipif_toulbar2_not_available),
        pytest.param("integer_program", mip_solver, None, marks=skipif_no_mip_solver),
        pytest.param(
            "integer_program", mip_solver, "markov_network", marks=skipif_no_mip_solver
        ),
        pytest.param(
            "integer_program", mip_solver, "network_flow", marks=skipif_no_mip_solver
        ),
        pytest.param(
            "variable_elimination", None, None, marks=skipif_pgmpy_not_available
        ),
    ],
)
def test1_hmm1(method, solver, ip_formulation):
    pgm = conin.hidden_markov_model.examples.create_hmm1()
    observed = ["o0", "o1", "o1", "o1", "o1"]

    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    if ip_formulation:
        kwargs["ip_formulation"] = ip_formulation
    results = map_query(pgm, **kwargs)
    assert results.solution.states == ["h1", "h1", "h1", "h1", "h1"]


@pytest.mark.parametrize(
    "method,solver",
    [
        pytest.param("toulbar2", None, marks=skipif_toulbar2_not_available),
        pytest.param("integer_program", mip_solver, marks=skipif_no_mip_solver),
        pytest.param("variable_elimination", None, marks=skipif_pgmpy_not_available),
    ],
)
def test2_hmm1(method, solver):
    pgm = conin.hidden_markov_model.examples.create_hmm1()
    observed = {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"}

    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    results = map_query(pgm, **kwargs)
    assert results.solution.states == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
    }


@pytest.mark.parametrize(
    "method,solver",
    [
        pytest.param("toulbar2", None, marks=skipif_toulbar2_not_available),
        pytest.param("integer_program", mip_solver, marks=skipif_no_mip_solver),
        pytest.param("variable_elimination", None, marks=skipif_pgmpy_not_available),
    ],
)
def test3_hmm1(method, solver):
    pgm = conin.hidden_markov_model.examples.create_hmm1()
    observed = {0: "o0", 1: "o1", 2: "o1", 3: "o1", 4: "o1"}

    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    results = map_query(pgm, **kwargs)
    assert results.solution.states == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h1",
        4: "h1",
    }


#
# ConstrainedHiddenMarkovModel tests
#


@pytest.mark.parametrize(
    "method,solver,write_file,pgm_factory,ip_formulation",
    [
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.hidden_markov_model.examples.create_chmm1_toulbar2,
            None,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            None,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            None,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            "markov_network",
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            "network_flow",
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            None,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.hidden_markov_model.examples.create_chmm1_oracle_ve,
            None,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            testfile_uai,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            None,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test0_chmm1(method, solver, write_file, pgm_factory, ip_formulation):
    pgm = pgm_factory()
    observed = ["o0"] * 15

    # Test without file writing
    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    if ip_formulation:
        kwargs["ip_formulation"] = ip_formulation
    results = map_query(pgm, **kwargs)
    assert results.solution.states == [
        "h1",
        "h1",
        "h1",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
    ]
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with file writing (only for default ip_formulation to avoid redundancy)
    if ip_formulation is None:
        if method == "integer_program":
            kwargs["write_lp_file"] = write_file
        else:
            kwargs["write_uai_file"] = write_file
        results = map_query(pgm, **kwargs)
        assert os.path.exists(write_file)
        os.remove(write_file)


@pytest.mark.parametrize(
    "method,solver,pgm_factory,ip_formulation",
    [
        pytest.param(
            "toulbar2",
            None,
            conin.hidden_markov_model.examples.create_chmm1_toulbar2,
            None,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            None,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            None,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            "markov_network",
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            "network_flow",
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            None,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            conin.hidden_markov_model.examples.create_chmm1_oracle_ve,
            None,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            None,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test1_chmm1(method, solver, pgm_factory, ip_formulation):
    pgm = pgm_factory()
    observed = ["o0"] + ["o1"] * 14

    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    if ip_formulation:
        kwargs["ip_formulation"] = ip_formulation
    results = map_query(pgm, **kwargs)
    assert results.solution.states == [
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h1",
        "h1",
        "h1",
        "h1",
        "h1",
    ]


@pytest.mark.parametrize(
    "method,solver,pgm_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            conin.hidden_markov_model.examples.create_chmm1_toulbar2,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            conin.hidden_markov_model.examples.create_chmm1_oracle_ve,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test2_chmm1(method, solver, pgm_factory):
    pgm = pgm_factory()
    observed = {i: "o0" for i in range(15)}

    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    results = map_query(pgm, **kwargs)
    assert results.solution.states == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h0",
        4: "h0",
        5: "h0",
        6: "h0",
        7: "h0",
        8: "h0",
        9: "h0",
        10: "h0",
        11: "h0",
        12: "h0",
        13: "h0",
        14: "h0",
    }


@pytest.mark.parametrize(
    "method,solver,pgm_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            conin.hidden_markov_model.examples.create_chmm1_toulbar2,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_pyomo,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        pytest.param(
            "variable_elimination",
            None,
            conin.hidden_markov_model.examples.create_chmm1_oracle_ve,
            marks=skipif_pgmpy_not_available,
        ),
        pytest.param(
            "variable_elimination",
            None,
            conin.hidden_markov_model.examples.create_chmm1_algebraic,
            marks=[skipif_pgmpy_not_available, skipif_smoek_not_available],
        ),
    ],
)
def test3_chmm1(method, solver, pgm_factory):
    pgm = pgm_factory()
    observed = {0: "o0"}
    for i in range(14):
        observed[i + 1] = "o1"

    kwargs = {"method": method, "evidence": observed}
    if solver:
        kwargs["solver"] = solver
    results = map_query(pgm, **kwargs)
    assert results.solution.states == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
        5: "h0",
        6: "h0",
        7: "h0",
        8: "h0",
        9: "h0",
        10: "h1",
        11: "h1",
        12: "h1",
        13: "h1",
        14: "h1",
    }


#
# DynamicBayesianNetwork tests
#

weather_evidence = {
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
}

q_unconstrained = {
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Hot",
    ("T", 2): "Mild",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Cloudy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}

q_constrained = {
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Mild",
    ("T", 2): "Cold",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Rainy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}


@pytest.mark.parametrize(
    "method,solver,write_file",
    [
        pytest.param(
            "toulbar2", None, testfile_uai, marks=skipif_toulbar2_not_available
        ),
        pytest.param(
            "integer_program", mip_solver, testfile_lp, marks=skipif_no_mip_solver
        ),
        # pytest.param(
        #    "variable_elimination", None, testfile_uai, marks=skipif_pgmpy_not_available
        # ),
    ],
)
def test_DPGM_weather_conin(method, solver, write_file):
    example = conin.dynamic_bayesian_network.examples.weather_conin()

    # Test without evidence
    kwargs = {"method": method, "stop": 4}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with evidence
    kwargs["evidence"] = weather_evidence
    kwargs["solution_with_evidence"] = True
    results = map_query(example.pgm, **kwargs)
    assert q_unconstrained == results.solution.states

    # Test with file writing
    kwargs = {"method": method, "stop": 4}
    if solver:
        kwargs["solver"] = solver
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


@skipif_pgmpy_not_available
@pytest.mark.parametrize(
    "method,solver",
    [
        pytest.param("toulbar2", None, marks=skipif_toulbar2_not_available),
        pytest.param("integer_program", mip_solver, marks=skipif_no_mip_solver),
        # pytest.param("variable_elimination", None),
    ],
)
def test_DPGM_weather_pgmpy(method, solver):
    example = conin.dynamic_bayesian_network.examples.weather2_pgmpy()

    # Test without evidence
    kwargs = {"method": method, "stop": 4}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states

    # Test with evidence
    kwargs["evidence"] = weather_evidence
    kwargs["solution_with_evidence"] = True
    results = map_query(example.pgm, **kwargs)
    assert q_unconstrained == results.solution.states


#
# ConstrainedDynamicBayesianNetwork tests
#


@pytest.mark.parametrize(
    "method,solver,write_file,example_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.dynamic_bayesian_network.examples.weather_constrained_toulbar2_conin,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            testfile_uai,
            conin.dynamic_bayesian_network.examples.weather_constrained_algebraic_conin,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.dynamic_bayesian_network.examples.weather_constrained_pyomo_conin,
            marks=skipif_no_mip_solver,
        ),
        pytest.param(
            "integer_program",
            mip_solver,
            testfile_lp,
            conin.dynamic_bayesian_network.examples.weather_constrained_algebraic_conin,
            marks=[skipif_no_mip_solver, skipif_smoek_not_available],
        ),
        # pytest.param(
        #    "variable_elimination",
        #    None,
        #    testfile_uai,
        #    conin.dynamic_bayesian_network.examples.weather_constrained_oracle_conin,
        #    marks=skipif_pgmpy_not_available,
        # ),
    ],
)
def test_DPGM_weather_constrained_conin(method, solver, write_file, example_factory):
    example = example_factory()

    # Test without evidence
    kwargs = {"method": method, "stop": 4}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # Test with evidence
    kwargs["evidence"] = weather_evidence
    kwargs["solution_with_evidence"] = True
    results = map_query(example.pgm, **kwargs)
    assert q_constrained == results.solution.states

    # Test with file writing
    kwargs = {"method": method, "stop": 4}
    if solver:
        kwargs["solver"] = solver
    if method == "integer_program":
        kwargs["write_lp_file"] = write_file
    else:
        kwargs["write_uai_file"] = write_file
    results = map_query(example.pgm, **kwargs)
    assert os.path.exists(write_file)
    os.remove(write_file)


@skipif_pgmpy_not_available
@pytest.mark.parametrize(
    "method,solver,example_factory",
    [
        pytest.param(
            "toulbar2",
            None,
            conin.dynamic_bayesian_network.examples.weather_constrained_toulbar2_pgmpy,
            marks=skipif_toulbar2_not_available,
        ),
        pytest.param(
            "toulbar2",
            None,
            conin.dynamic_bayesian_network.examples.weather_constrained_algebraic_pgmpy,
            marks=[skipif_toulbar2_not_available, skipif_smoek_not_available],
        ),
        # pytest.param(
        #    "variable_elimination",
        #    None,
        #    conin.dynamic_bayesian_network.examples.weather_constrained_oracle_pgmpy,
        # ),
    ],
)
def test_DPGM_weather_constrained_pgmpy(method, solver, example_factory):
    example = example_factory()

    # Test without evidence
    kwargs = {"method": method, "stop": 4}
    if solver:
        kwargs["solver"] = solver
    results = map_query(example.pgm, **kwargs)
    assert results.solution.states == example.solutions[0].states

    # Test with evidence
    kwargs["evidence"] = weather_evidence
    kwargs["solution_with_evidence"] = True
    results = map_query(example.pgm, **kwargs)
    assert q_constrained == results.solution.states


#
# Test for unsupported types
#


def test_map_query_unsupported_type():
    """Test that map_query raises TypeError for unsupported types."""
    with pytest.raises(TypeError, match="Unsupported model type"):
        map_query("not_a_model", method="integer_program")
