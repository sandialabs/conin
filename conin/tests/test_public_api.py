"""The top-level public API, ``conin.__all__`` (doc/sphinx/api_conin.rst)."""

import conin
import conin.algorithms.feasibility
import conin.algorithms.inference
import conin.algorithms.scoring
import conin.constrained_models
import conin.constraints
import conin.exceptions
import conin.graphical_models
import conin.io

# Each public name and the subpackage it comes from.
PUBLIC = {
    conin.graphical_models: [
        "DiscreteFactor",
        "DiscreteMarkovNetwork",
        "DiscreteCPD",
        "DiscreteBayesianNetwork",
        "DynamicDiscreteBayesianNetwork",
        "HiddenMarkovModel",
    ],
    conin.constrained_models: [
        "ConstrainedDiscreteMarkovNetwork",
        "ConstrainedDiscreteBayesianNetwork",
        "ConstrainedDynamicDiscreteBayesianNetwork",
        "ConstrainedHiddenMarkovModel",
    ],
    conin.constraints: [
        "Constraint",
        "mvr_constraint_fn",
        "oracle_constraint_fn",
        "pyomo_constraint_fn",
        "smoek_constraint_fn",
        "toulbar2_constraint_fn",
    ],
    conin.algorithms.inference: ["map_query"],
    conin.algorithms.scoring: [
        "log_probability_query",
        "log_probability",
        "log_potential",
    ],
    conin.algorithms.feasibility: ["is_feasible"],
    conin.io: ["load_model", "save_model"],
    conin.exceptions: [
        "InvalidInputError",
        "InsufficientSolutionsError",
        "ModelFileError",
    ],
}


def test_all_is_the_public_api():
    expected = [name for names in PUBLIC.values() for name in names]
    assert conin.__all__ == expected


def test_public_names_are_the_subpackage_objects():
    for module, names in PUBLIC.items():
        for name in names:
            assert getattr(conin, name) is getattr(module, name), name


def test_star_import():
    namespace = {}
    exec("from conin import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(conin.__all__)


def test_subpackages_are_loaded():
    for name in [
        "algorithms",
        "config",
        "constrained_models",
        "constraints",
        "exceptions",
        "graphical_models",
        "io",
        "representations",
        "util",
    ]:
        assert hasattr(conin, name), name
    for name in ["feasibility", "inference", "learning", "sampling", "scoring"]:
        assert hasattr(conin.algorithms, name), name
