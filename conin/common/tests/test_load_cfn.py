import itertools
import math
import os.path
import tempfile

import pytest

from conin.util import try_import
from conin.common import load_model
from conin.common.conin import load_conin_model_from_cfn
from conin.common.conin.log_potential import log_potential
from conin.constraints import Toulbar2Constraint
from conin.inference import map_query
from conin.markov_network import (
    ConstrainedDiscreteMarkovNetwork,
    DiscreteMarkovNetwork,
)

with try_import() as pytoulbar2_available:
    import pytoulbar2

skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)

cwd = os.path.dirname(__file__)


def factor_values(pgm):
    return {tuple(f.nodes): f.values for f in pgm.factors}


#
# Parsing
#


def test_small_cfn():
    pgm = load_model(os.path.join(cwd, "small.cfn"))
    assert isinstance(pgm, DiscreteMarkovNetwork)
    assert pgm.nodes == ["a", "b"]
    assert pgm.states == {"a": [0, 1], "b": ["lo", "mid", "hi"]}

    values = factor_values(pgm)
    assert set(values) == {("a", "b"), ("b",), ("a",)}

    # Full table, costs shifted so the minimum cost (0) has value 1
    f1 = values["a", "b"]
    assert f1[0, "lo"] == pytest.approx(math.exp(-0.5))
    assert f1[0, "mid"] == pytest.approx(math.exp(-1.25))
    assert f1[0, "hi"] == pytest.approx(1.0)
    assert f1[1, "lo"] == pytest.approx(math.exp(-2))
    assert f1[1, "mid"] == pytest.approx(math.exp(-3.5))
    # Cost 10 violates the upper bound of 10, so the entry is forbidden
    assert f1[1, "hi"] == 0.0

    # Sparse table with a value name
    f2 = values["b",]
    assert f2["lo"] == pytest.approx(math.exp(-0.65))
    assert f2["mid"] == pytest.approx(math.exp(-0.65))
    assert f2["hi"] == pytest.approx(1.0)

    # Scope given by variable index
    f3 = values["a",]
    assert f3[0] == pytest.approx(math.exp(-0.3))
    assert f3[1] == pytest.approx(1.0)


def test_relaxed_syntax():
    pgm = load_conin_model_from_cfn(
        string="""
# A comment
{ problem { name t2 mustbe <5 }
  variables { x [a b] y 2 }
  functions { f { scope [x y] costs [0 1 2 5] } }
}
"""
    )
    assert pgm.states == {"x": ["a", "b"], "y": [0, 1]}
    f = factor_values(pgm)["x", "y"]
    assert f["a", 0] == pytest.approx(1.0)
    assert f["a", 1] == pytest.approx(math.exp(-1))
    assert f["b", 0] == pytest.approx(math.exp(-2))
    assert f["b", 1] == 0.0


def test_quoted_punctuation_in_value_names():
    pgm = load_conin_model_from_cfn(
        string="""{"problem":{"name":"p","mustbe":"<10"},
        "variables":{"x":["{","]"]},
        "functions":{"f":{"scope":["x"],"defaultcost":0,"costs":["]",1]}}}"""
    )
    assert pgm.states == {"x": ["{", "]"]}
    assert factor_values(pgm)["x",] == {"{": 1.0, "]": pytest.approx(math.exp(-1))}


def test_function_list_and_unused_variable():
    pgm = load_conin_model_from_cfn(
        string="""{"problem":{"name":"p","mustbe":"<10"},
        "variables":{"x":2, "u":3},
        "functions":[{"scope":["x"],"costs":[1,0]}]}"""
    )
    values = factor_values(pgm)
    assert values["x",] == {0: pytest.approx(math.exp(-1)), 1: 1.0}
    assert values["u",] == {0: 1.0, 1: 1.0, 2: 1.0}


def test_maximization():
    pgm = load_conin_model_from_cfn(
        string="""{"problem":{"name":"p","mustbe":">-50"},
        "variables":{"x":3},
        "functions":{"f":{"scope":["x"],"costs":[1,7,3]}}}"""
    )
    f = factor_values(pgm)["x",]
    assert f[1] == pytest.approx(1.0)
    assert f[0] == pytest.approx(math.exp(-6))
    assert f[2] == pytest.approx(math.exp(-4))


def test_negative_costs_and_bound():
    # The bound applies to the total cost: x=0 has cost 60 >= 50 alone, but
    # the minimum total cost is -30 + 1 = -29, so 60 - 1 + (-29) = 30 < 50.
    pgm = load_conin_model_from_cfn(
        string="""{"problem":{"name":"p","mustbe":"<50"},
        "variables":{"x":2, "y":2},
        "functions":{"f":{"scope":["x"],"costs":[60,1]},
                     "g":{"scope":["y"],"costs":[-30,-30]}}}"""
    )
    f = factor_values(pgm)["x",]
    assert f[0] == pytest.approx(math.exp(-59))
    assert f[1] == pytest.approx(1.0)


def test_cost_scale():
    pgm = load_conin_model_from_cfn(
        string="""{"problem":{"name":"p","mustbe":"<100"},
        "variables":{"x":2},
        "functions":{"f":{"scope":["x"],"costs":[0,20]}}}""",
        cost_scale=10,
    )
    assert factor_values(pgm)["x",][1] == pytest.approx(math.exp(-2))


def test_cancer_cfn_matches_uai():
    """cancer_mn.cfn was written by Toulbar2 from cancer_mn.uai"""
    uai = load_model(os.path.join(cwd, "cancer_mn.uai"))
    for name in ["cancer_mn.cfn", "cancer_mn_compressed.cfn.gz"]:
        cfn = load_model(os.path.join(cwd, name), cost_scale=1e7)
        assert cfn.nodes == ["x0", "x1", "x2", "x3", "x4"]

        # Compare the normalized log-potentials of all assignments.
        uai_lp = {}
        cfn_lp = {}
        for values in itertools.product([0, 1], repeat=5):
            uai_lp[values] = log_potential(
                uai, {f"var{i}": v for i, v in enumerate(values)}
            )
            cfn_lp[values] = log_potential(
                cfn, {f"x{i}": f"v{v}" for i, v in enumerate(values)}
            )
        uai_max = max(uai_lp.values())
        cfn_max = max(cfn_lp.values())
        for values in uai_lp:
            assert cfn_lp[values] - cfn_max == pytest.approx(
                uai_lp[values] - uai_max, abs=1e-5
            )


def test_cancer_cfn_without_cost_scale_warns():
    with pytest.warns(UserWarning, match="cost_scale"):
        load_model(os.path.join(cwd, "cancer_mn.cfn"))


#
# Global cost functions
#

KNAPSACK_TEMPLATE = """{"problem":{"name":"p","mustbe":"<1000"},
 "variables":{"w":2, "x":["a","b","c"], "y":3},
 "functions":{"fx":{"scope":["x"],"costs":[0,3,6]},
              "fy":{"scope":["y"],"costs":[0,3,6]},
              "k":{"scope":%s,"type":"%s","params":%s}}}"""


def test_abc_constrained_cfn():
    pgm = load_model(os.path.join(cwd, "abc_constrained.cfn"))
    assert isinstance(pgm, ConstrainedDiscreteMarkovNetwork)

    # The unconstrained model
    assert isinstance(pgm.pgm, DiscreteMarkovNetwork)
    assert pgm.nodes == ["A", "B", "C"]
    assert set(factor_values(pgm.pgm)) == {("A", "B"), ("B", "C"), ("A", "C"), ("A",)}

    # The constraints are Toulbar2 constraints with a portable description
    assert len(pgm.constraints) == 3
    for i, con in enumerate(pgm.constraints):
        assert isinstance(con, Toulbar2Constraint)
        assert con.name == "F_0_1_2"
        assert con.cfn_type == "knapsackv"
        assert con.scope == ["A", "B", "C"]
        assert con.terms == [("A", f"s{i}", -1), ("B", f"s{i}", -1), ("C", f"s{i}", -1)]
        assert con.operator == ">="
        assert con.rhs == -1


def test_knapsackv_variable_names_and_indices():
    # weightedvalues refer to variables by name or by their index in the
    # problem's variable list (not the function scope).
    for weightedvalues in ['[["x",2,1],["y",1,1]]', "[[1,2,1],[2,1,1]]"]:
        pgm = load_conin_model_from_cfn(
            string=KNAPSACK_TEMPLATE
            % (
                '["x","y"]',
                "knapsackv",
                '{"capacity":1,"weightedvalues":%s}' % weightedvalues,
            )
        )
        (con,) = pgm.constraints
        assert con.terms == [("x", "c", 1), ("y", 1, 1)]
        assert con.rhs == 1


def test_knapsack_boolean():
    pgm = load_conin_model_from_cfn(
        string="""{"problem":{"name":"p","mustbe":"<1000"},
        "variables":{"x":["no","yes"], "y":2},
        "functions":{"k":{"scope":["x","y"],"type":"knapsack",
                          "params":{"capacity":3,"weights":[1,2]}}}}"""
    )
    (con,) = pgm.constraints
    assert con.cfn_type == "knapsack"
    assert con.terms == [("x", "yes", 1), ("y", 1, 2)]
    assert con.rhs == 3
    # Variables that only appear in constraints get uniform factors
    assert factor_values(pgm.pgm) == {("x",): {"no": 1.0, "yes": 1.0}, ("y",): {0: 1.0, 1: 1.0}}


def test_knapsack_non_boolean_error():
    with pytest.raises(NotImplementedError, match="Boolean"):
        load_conin_model_from_cfn(
            string=KNAPSACK_TEMPLATE
            % ('["x","y"]', "knapsack", '{"capacity":1,"weights":[1,1]}')
        )


@pytest.mark.parametrize(
    "ftype,params",
    [
        ("salldiff", '{"metric":"var","cost":10}'),
        ("knapsackv", '{"capacity":1,"weightedvalues":[[1,2,1]],"Nbconstr":1}'),
    ],
)
def test_unsupported_global_cost_function_error(ftype, params):
    with pytest.raises(NotImplementedError):
        load_conin_model_from_cfn(
            string=KNAPSACK_TEMPLATE % ('["x","y"]', ftype, params)
        )


#
# Errors
#

@pytest.mark.parametrize(
    "text",
    [
        # Wrong table size
        """{"problem":{"name":"p","mustbe":"<3"},"variables":{"x":2},
           "functions":{"f":{"scope":["x"],"costs":[0,1,2]}}}""",
        # Unknown variable
        """{"problem":{"name":"p","mustbe":"<3"},"variables":{"x":2},
           "functions":{"f":{"scope":["z"],"costs":[0,1]}}}""",
        # Unknown value name
        """{"problem":{"name":"p","mustbe":"<3"},"variables":{"x":["a","b"]},
           "functions":{"f":{"scope":["x"],"defaultcost":0,"costs":["c",1]}}}""",
        # Bad mustbe
        """{"problem":{"name":"p","mustbe":"=3"},"variables":{"x":2},
           "functions":{"f":{"scope":["x"],"costs":[0,1]}}}""",
        # Unterminated
        """{"problem":{"name":"p","mustbe":"<3"},"variables":{"x":2""",
    ],
)
def test_parse_errors(text):
    with pytest.raises(ValueError):
        load_conin_model_from_cfn(string=text)


def test_missing_file():
    with pytest.raises(RuntimeError):
        load_model(os.path.join(cwd, "unknown.cfn"))


#
# Agreement with Toulbar2
#


def _toulbar2_solution(filename=None, string=None):
    """Solve a CFN file directly with pytoulbar2."""
    if string is not None:
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "model.cfn")
            with open(filename, "w") as OUTPUT:
                OUTPUT.write(string)
            return _toulbar2_solution(filename=filename)

    m = pytoulbar2.CFN(verbose=-1)
    m.Read(filename)
    res = m.Solve()
    # pytoulbar2 keeps the input format in a global option, which prevents
    # reading UAI files (as done by map_query) after reading a CFN file.
    pytoulbar2.pytb2.option.cfn = False
    return None if res is None else list(res[0])


def _conin_solution(pgm):
    states = map_query(pgm, method="toulbar2").solution.states
    return [pgm.states_of(v).index(states[v]) for v in pgm.nodes]


@skipif_toulbar2_not_available
@pytest.mark.parametrize("name", ["small.cfn", "cancer_mn.cfn", "abc_constrained.cfn"])
def test_map_matches_toulbar2(name):
    filename = os.path.join(cwd, name)
    tb2_solution = _toulbar2_solution(filename)

    pgm = load_model(filename, cost_scale=1e7 if name == "cancer_mn.cfn" else 1.0)
    assert _conin_solution(pgm) == tb2_solution


@skipif_toulbar2_not_available
def test_abc_constrained_vs_unconstrained():
    pgm = load_model(os.path.join(cwd, "abc_constrained.cfn"))
    states = map_query(pgm, method="toulbar2").solution.states
    assert states == {"A": "s0", "B": "s2", "C": "s1"}

    # The extracted unconstrained model has a different MAP solution
    states = map_query(pgm.pgm, method="toulbar2").solution.states
    assert states == {"A": "s1", "B": "s1", "C": "s1"}


@skipif_toulbar2_not_available
@pytest.mark.parametrize(
    "scope,ftype,params",
    [
        # x == c or y == 1
        ('["x","y"]', "knapsackv", '{"capacity":1,"weightedvalues":[["x",2,1],["y",1,1]]}'),
        # 2*(x == b) + 3*(y == 2) + (w == 1) >= 4, using problem variable indices
        (
            '["w","x","y"]',
            "knapsackv",
            '{"capacity":4,"weightedvalues":[[1,1,2],[2,2,3],[0,1,1]]}',
        ),
        # (x == a) + (y == 0) <= 0, i.e. -(x == a) - (y == 0) >= 0
        ('["x","y"]', "knapsackv", '{"capacity":0,"weightedvalues":[[1,0,-1],[2,0,-1]]}'),
    ],
)
def test_knapsackv_matches_toulbar2(scope, ftype, params):
    text = KNAPSACK_TEMPLATE % (scope, ftype, params)
    tb2_solution = _toulbar2_solution(string=text)
    pgm = load_conin_model_from_cfn(string=text)
    assert _conin_solution(pgm) == tb2_solution


@skipif_toulbar2_not_available
def test_knapsack_matches_toulbar2():
    text = """{"problem":{"name":"p","mustbe":"<1000"},
        "variables":{"x":2, "y":2, "z":2},
        "functions":{"fx":{"scope":["x"],"costs":[0,5]},
                     "fy":{"scope":["y"],"costs":[0,1]},
                     "fz":{"scope":["z"],"costs":[0,2]},
                     "k":{"scope":["x","y","z"],"type":"knapsack",
                          "params":{"capacity":3,"weights":[1,2,1]}}}}"""
    tb2_solution = _toulbar2_solution(string=text)
    assert tb2_solution == [0, 1, 1]
    pgm = load_conin_model_from_cfn(string=text)
    assert _conin_solution(pgm) == tb2_solution


@skipif_toulbar2_not_available
def test_dumped_linear_constraints_match_toulbar2():
    """Round trip constraints written by pytoulbar2's linear constraint API."""
    builders = [
        lambda m: m.AddGeneralizedLinearConstraint(
            [("x", 1, 2), ("y", 2, 3), ("z", 1, 1)], ">=", 4
        ),
        lambda m: m.AddGeneralizedLinearConstraint([("x", 2, 1), ("z", 0, 1)], "==", 0),
        lambda m: m.AddLinearConstraint([-1, -2], ["y", "z"], "<", -2),
        lambda m: m.AddSumConstraint(["x", "y", "z"], ">", 2),
    ]
    with tempfile.TemporaryDirectory() as tempdir:
        for i, build in enumerate(builders):
            m = pytoulbar2.CFN(verbose=-1)
            m.AddVariable("u", ["p", "q"])
            m.AddVariable("x", ["a", "b", "c"])
            m.AddVariable("y", range(3))
            m.AddVariable("z", ["r", "s"])
            m.AddFunction(["x", "y"], [0, 1, 2, 1, 2, 3, 2, 3, 4])
            m.AddFunction(["u"], [0, 1])
            m.AddFunction(["z"], [0, 1])
            build(m)
            filename = os.path.join(tempdir, f"model{i}.cfn")
            m.Dump(filename)

            tb2_solution = _toulbar2_solution(filename)
            pgm = load_model(filename)
            assert isinstance(pgm, ConstrainedDiscreteMarkovNetwork)
            assert _conin_solution(pgm) == tb2_solution
            # Each constraint is binding
            assert _conin_solution(pgm.pgm) != tb2_solution
