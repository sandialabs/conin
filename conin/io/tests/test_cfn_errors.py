"""Error paths and less common valid inputs of the CFN reader (conin.io.cfn)."""

import gzip
import math

import pytest

from conin.io import load_model, load_cfn
from conin.graphical_models import DiscreteMarkovNetwork


def factor_values(pgm):
    return {tuple(f.nodes): f.values for f in pgm.factors}


def cfn(variables='{"x":2}', functions='{"f":{"scope":["x"],"costs":[0,1]}}'):
    return '{"problem":{"name":"p","mustbe":"<100"},"variables":%s,"functions":%s}' % (
        variables,
        functions,
    )


KNAPSACKV = cfn(
    variables='{"x":["a","b","c"], "y":3}',
    functions='{"k":{"scope":["x","y"],"type":"knapsackv","params":%s}}',
)
KNAPSACK = cfn(
    variables='{"x":2, "y":2}',
    functions='{"k":{"scope":["x","y"],"type":"knapsack","params":%s}}',
)


#
# Arguments
#


@pytest.mark.parametrize("cost_scale", [0, -1.0])
def test_non_positive_cost_scale(cost_scale):
    with pytest.raises(ValueError, match="cost_scale must be positive"):
        load_cfn(string=cfn(), cost_scale=cost_scale)


def test_missing_cfn_file(tmp_path):
    name = str(tmp_path / "missing.cfn")
    with pytest.raises(RuntimeError, match="Cannot read missing CFN file"):
        load_cfn(filename=name)


@pytest.mark.parametrize("filename", [None, ""])
def test_neither_filename_nor_string(filename):
    with pytest.raises(ValueError, match="Either a filename or a string"):
        load_cfn(filename=filename)


def test_filename_takes_precedence_over_string(tmp_path):
    name = tmp_path / "model.cfn"
    name.write_text(cfn(variables='{"z":3}', functions="{}"))
    pgm = load_cfn(filename=str(name), string=cfn())
    assert pgm.states == {"z": [0, 1, 2]}


def test_load_model_cfn_gz(tmp_path):
    name = tmp_path / "model.cfn.gz"
    with gzip.open(name, "wt") as OUTPUT:
        OUTPUT.write(cfn(functions='{"f":{"scope":["x"],"costs":[0,20]}}'))
    pgm = load_model(str(name), cost_scale=10)
    assert isinstance(pgm, DiscreteMarkovNetwork)
    assert factor_values(pgm)["x",] == {0: 1.0, 1: pytest.approx(math.exp(-2))}


def test_load_model_unknown_gz_suffix(tmp_path):
    name = tmp_path / "model.txt.gz"
    with gzip.open(name, "wt") as OUTPUT:
        OUTPUT.write(cfn())
    with pytest.raises(RuntimeError, match="unexpected file.*model.txt.gz"):
        load_model(str(name))


def test_load_model_missing_cfn_file(tmp_path):
    with pytest.raises(RuntimeError, match="Missing file.*missing.cfn"):
        load_model(str(tmp_path / "missing.cfn"))


#
# Tokenizing and parsing
#


def test_unterminated_string():
    with pytest.raises(ValueError, match="Unterminated string"):
        load_cfn(string='{"variables":{"x:2}}')


@pytest.mark.parametrize("text", ["", "# only a comment\n", '{"variables":{"x":2}'])
def test_unexpected_end(text):
    with pytest.raises(ValueError, match="Unexpected end of CFN file"):
        load_cfn(string=text)


@pytest.mark.parametrize("key", ["{", "[", "]"])
def test_unexpected_token_as_object_key(key):
    text = '{"variables":{%s "x":2}}' % key
    with pytest.raises(ValueError, match="Unexpected token .* in CFN object"):
        load_cfn(string=text)


@pytest.mark.parametrize("text", ["]", "}", '{"variables" ]}', "[1 }]"])
def test_unexpected_closing_token(text):
    with pytest.raises(ValueError, match="Unexpected token .* in CFN file"):
        load_cfn(string=text)


@pytest.mark.parametrize("text", ["[1, 2]", "[[1]]", "42", "name"])
def test_missing_top_level_object(text):
    with pytest.raises(ValueError, match="must contain a top-level object"):
        load_cfn(string=text)


def test_hash_inside_a_line_is_not_a_comment():
    # '#' only starts a comment at the beginning of a line; elsewhere it is
    # part of a bare word.
    pgm = load_cfn(string="""# comment
{ variables { x [a#1 b] }
  functions { f { scope [x] costs [0 1] } } }""")
    assert pgm.states == {"x": ["a#1", "b"]}


#
# Problem section
#


def test_no_problem_section_minimizes_without_bound():
    pgm = load_cfn(
        string='{"variables":{"x":2},"functions":{"f":{"scope":["x"],"costs":[0,5]}}}'
    )
    # No upper bound, so the large cost is not forbidden
    assert factor_values(pgm)["x",] == {0: 1.0, 1: pytest.approx(math.exp(-5))}


def test_problem_without_mustbe():
    pgm = load_cfn(
        string='{"problem":{"name":"p"},"variables":{"x":2},'
        '"functions":{"f":{"scope":["x"],"costs":[3,0]}}}'
    )
    assert factor_values(pgm)["x",] == {0: pytest.approx(math.exp(-3)), 1: 1.0}


def test_mustbe_with_space_and_numeric_bound():
    pgm = load_cfn(
        string='{"problem":{"name":"p","mustbe":"< 5"},"variables":{"x":3},'
        '"functions":{"f":{"scope":["x"],"costs":[0,4,5]}}}'
    )
    assert factor_values(pgm)["x",] == {
        0: 1.0,
        1: pytest.approx(math.exp(-4)),
        2: 0.0,
    }


@pytest.mark.parametrize("mustbe", ["=3", "3", "<", "<abc", ">1.2.3", "<5x"])
def test_bad_mustbe(mustbe):
    text = (
        '{"problem":{"name":"p","mustbe":"%s"},"variables":{"x":2},'
        '"functions":{"f":{"scope":["x"],"costs":[0,1]}}}' % mustbe
    )
    with pytest.raises(ValueError, match="Unexpected CFN 'mustbe' value"):
        load_cfn(string=text)


#
# Variables
#


@pytest.mark.parametrize("variables", ["5", "[1, 2]", "[[1]]"])
def test_variables_not_an_object(variables):
    with pytest.raises(ValueError, match="'variables' entry must be an object"):
        load_cfn(string=cfn(variables=variables))


def test_duplicate_variable():
    with pytest.raises(ValueError, match="Duplicate CFN variable: x"):
        load_cfn(string=cfn(variables='{"x":2, "x":3}'))


@pytest.mark.parametrize("size", [0, -2])
def test_non_positive_domain_size(size):
    with pytest.raises(ValueError, match="Variable x must have a positive domain size"):
        load_cfn(string=cfn(variables='{"x":%d}' % size))


@pytest.mark.parametrize("domain", ['["a","b","a"]', "[1, 1]", "[a a]"])
def test_duplicate_value_names(domain):
    with pytest.raises(ValueError, match="Variable x has duplicate value names"):
        load_cfn(string=cfn(variables='{"x":%s}' % domain))


@pytest.mark.parametrize("domain", ["2.5", "[]", '"two"'])
def test_bad_domain_type(domain):
    with pytest.raises(ValueError, match="Unexpected domain for CFN variable x"):
        load_cfn(string=cfn(variables='{"x":%s}' % domain))


def test_numeric_value_names():
    # A list of numbers is a list of value names, not a domain size.  Sparse
    # costs refer to values by index, so 10 is not a valid value.
    pgm = load_cfn(
        string=cfn(
            variables='{"x":[10, 20]}',
            functions='{"f":{"scope":["x"],"defaultcost":0,"costs":[1,2]}}',
        )
    )
    assert pgm.states == {"x": [10, 20]}
    assert factor_values(pgm)["x",] == {10: 1.0, 20: pytest.approx(math.exp(-2))}

    with pytest.raises(ValueError, match="Invalid value 10 for CFN variable x"):
        load_cfn(
            string=cfn(
                variables='{"x":[10, 20]}',
                functions='{"f":{"scope":["x"],"defaultcost":0,"costs":[10,2]}}',
            )
        )


#
# Cost functions in extension
#


@pytest.mark.parametrize("functions", ["5", '"f"'])
def test_functions_not_object_or_list(functions):
    with pytest.raises(ValueError, match="'functions' entry must be an object or list"):
        load_cfn(string=cfn(functions=functions))


@pytest.mark.parametrize(
    "functions,fname",
    [
        ('{"f":5}', "f"),
        ('{"f":[0, 1]}', "f"),
        ("[5]", "F0"),
        ('[{"scope":[],"costs":[0]}, 7]', "F1"),
    ],
)
def test_function_not_an_object(functions, fname):
    with pytest.raises(ValueError, match=f"CFN function {fname} must be an object"):
        load_cfn(string=cfn(functions=functions))


@pytest.mark.parametrize("index", [2, -1])
def test_invalid_variable_index_in_scope(index):
    functions = '{"f":{"scope":[%d],"costs":[0,1]}}' % index
    with pytest.raises(ValueError, match=f"Invalid variable index {index} in CFN"):
        load_cfn(string=cfn(variables='{"x":2,"y":2}', functions=functions))


def test_unknown_variable_name_in_scope():
    functions = '{"g":{"scope":["x","nope"],"costs":[0,1,2,3]}}'
    with pytest.raises(ValueError, match="Unknown variable nope in CFN function g"):
        load_cfn(string=cfn(functions=functions))


@pytest.mark.parametrize("scope", ['["x","x"]', '["x",0]'])
def test_repeated_variable_in_scope(scope):
    functions = '{"f":{"scope":%s,"costs":[0,1,2,3]}}' % scope
    with pytest.raises(ValueError, match="CFN function f has a repeated variable"):
        load_cfn(string=cfn(functions=functions))


@pytest.mark.parametrize(
    "fdef",
    [
        '{"scope":["x"]}',
        '{"scope":["x"],"costs":5}',
        '{"scope":["x"],"costs":"0 1"}',
        '{"scope":["x"],"function":"x+1"}',
    ],
)
def test_cost_function_not_a_table(fdef):
    with pytest.raises(NotImplementedError, match="does not define an explicit cost"):
        load_cfn(string=cfn(functions='{"f":%s}' % fdef))


def test_full_table_size_mismatch_message():
    functions = '{"f":{"scope":["x","y"],"costs":[0,1,2]}}'
    with pytest.raises(ValueError, match="has 3 costs but 6 were expected"):
        load_cfn(string=cfn(variables='{"x":2,"y":3}', functions=functions))


@pytest.mark.parametrize(
    "fdef",
    [
        '{"scope":["x"],"costs":["a",1]}',
        '{"scope":["x"],"defaultcost":0,"costs":[1,"z"]}',
        '{"scope":["x"],"defaultcost":"high","costs":[]}',
    ],
)
def test_non_numeric_costs(fdef):
    with pytest.raises(ValueError, match="could not convert string to float"):
        load_cfn(string=cfn(functions='{"f":%s}' % fdef))


def test_sparse_costs_not_a_multiple_of_width():
    functions = '{"f":{"scope":["x","y"],"defaultcost":0,"costs":[0,1,5,1]}}'
    with pytest.raises(ValueError, match="not a multiple of 3"):
        load_cfn(string=cfn(variables='{"x":2,"y":2}', functions=functions))


@pytest.mark.parametrize("value", ["2", "-1", "1.0"])
def test_invalid_value_index_in_sparse_costs(value):
    functions = '{"f":{"scope":["x"],"defaultcost":0,"costs":[%s,1]}}' % value
    with pytest.raises(ValueError, match=f"Invalid value {value} for CFN variable x"):
        load_cfn(string=cfn(functions=functions))


def test_sparse_two_variable_table_with_names_and_indices():
    # Sparse tuples may mix value names and value indices, and the last
    # variable changes fastest in the table.
    pgm = load_cfn(
        string=cfn(
            variables='{"x":["a","b"], "y":3}',
            functions='{"f":{"scope":["x","y"],"defaultcost":4,'
            '"costs":["a",2,0, 1,0,1, "b",1,3]}}',
        )
    )
    f = factor_values(pgm)["x", "y"]
    expected = {
        ("a", 0): 4,
        ("a", 1): 4,
        ("a", 2): 0,
        ("b", 0): 1,
        ("b", 1): 3,
        ("b", 2): 4,
    }
    assert f == {k: pytest.approx(math.exp(-c)) for k, c in expected.items()}


def test_sparse_table_with_no_tuples_is_uniform():
    pgm = load_cfn(
        string=cfn(functions='{"f":{"scope":["x"],"defaultcost":7,"costs":[]}}')
    )
    assert factor_values(pgm)["x",] == {0: 1.0, 1: 1.0}


def test_sparse_default_cost_over_bound_is_forbidden():
    pgm = load_cfn(
        string=cfn(functions='{"f":{"scope":["x"],"defaultcost":100,"costs":[1,0]}}')
    )
    assert factor_values(pgm)["x",] == {0: 0.0, 1: 1.0}


def test_maximization_with_sparse_costs():
    pgm = load_cfn(
        string='{"problem":{"name":"p","mustbe":">0"},"variables":{"x":3},'
        '"functions":{"f":{"scope":["x"],"defaultcost":1,"costs":[2,5]}}}'
    )
    assert factor_values(pgm)["x",] == {
        0: pytest.approx(math.exp(-4)),
        1: pytest.approx(math.exp(-4)),
        2: 1.0,
    }


def test_zero_arity_function_is_ignored():
    pgm = load_cfn(
        string=cfn(
            functions='{"c":{"scope":[],"costs":[42]},'
            '"f":{"scope":["x"],"costs":[0,1]}}'
        )
    )
    assert factor_values(pgm) == {("x",): {0: 1.0, 1: pytest.approx(math.exp(-1))}}


def test_duplicate_function_names_are_kept():
    pgm = load_cfn(
        string=cfn(
            functions='{"f":{"scope":["x"],"costs":[0,1]},'
            '"f":{"scope":[0],"costs":[0,2]}}'
        )
    )
    assert [f.values[1] for f in pgm.factors] == [
        pytest.approx(math.exp(-1)),
        pytest.approx(math.exp(-2)),
    ]


#
# Global cost functions
#


@pytest.mark.parametrize("params", ["5", "[1, 2]", '"capacity"'])
def test_knapsack_params_not_an_object(params):
    with pytest.raises(ValueError, match="must define 'params' as an object"):
        load_cfn(string=KNAPSACKV % params)


def test_knapsack_missing_params():
    text = cfn(functions='{"k":{"scope":["x"],"type":"knapsack"}}')
    with pytest.raises(ValueError, match="must define 'params' as an object"):
        load_cfn(string=text)


@pytest.mark.parametrize(
    "template,params",
    [
        (KNAPSACK, '{"capacity":1}'),
        (KNAPSACK, '{"capacity":1,"weightedvalues":[]}'),
        (KNAPSACKV, '{"capacity":1,"weights":[1,1]}'),
        (KNAPSACKV, '{"weightedvalues":[]}'),
    ],
)
def test_knapsack_unexpected_parameters(template, params):
    with pytest.raises(NotImplementedError, match="but only .* are supported"):
        load_cfn(string=template % params)


@pytest.mark.parametrize("capacity", ['"a"', "[1]", "big"])
def test_knapsack_non_numeric_capacity(capacity):
    params = '{"capacity":%s,"weights":[1,1]}' % capacity
    with pytest.raises(ValueError, match="Expected a number for capacity"):
        load_cfn(string=KNAPSACK % params)


@pytest.mark.parametrize("weights", ["[1]", "[1,2,3]", "2"])
def test_knapsack_wrong_number_of_weights(weights):
    params = '{"capacity":1,"weights":%s}' % weights
    with pytest.raises(ValueError, match="must have one weight per scope variable"):
        load_cfn(string=KNAPSACK % params)


def test_knapsack_non_numeric_weight():
    params = '{"capacity":1,"weights":[1,"w"]}'
    with pytest.raises(ValueError, match="Expected a number for weights"):
        load_cfn(string=KNAPSACK % params)


@pytest.mark.parametrize("weightedvalues", ["5", '"x"', "{}"])
def test_knapsackv_weightedvalues_not_a_list(weightedvalues):
    # An empty object is parsed as an empty list of pairs, so it is accepted
    params = '{"capacity":1,"weightedvalues":%s}' % weightedvalues
    if weightedvalues == "{}":
        pgm = load_cfn(string=KNAPSACKV % params)
        assert pgm.constraints[0].terms == []
        return
    with pytest.raises(ValueError, match="must define 'weightedvalues'"):
        load_cfn(string=KNAPSACKV % params)


@pytest.mark.parametrize("entry", ['["x",1]', '["x",1,1,1]', "5", '"x"'])
def test_knapsackv_invalid_weighted_value(entry):
    params = '{"capacity":1,"weightedvalues":[%s]}' % entry
    with pytest.raises(ValueError, match="has an invalid weighted value"):
        load_cfn(string=KNAPSACKV % params)


@pytest.mark.parametrize("index", [2, -1])
def test_knapsackv_invalid_variable_index(index):
    params = '{"capacity":1,"weightedvalues":[[%d,0,1]]}' % index
    with pytest.raises(ValueError, match=f"Invalid variable index {index} in CFN"):
        load_cfn(string=KNAPSACKV % params)


def test_knapsackv_unknown_variable():
    params = '{"capacity":1,"weightedvalues":[["z",0,1]]}'
    with pytest.raises(ValueError, match="Unknown variable z in CFN function k"):
        load_cfn(string=KNAPSACKV % params)


@pytest.mark.parametrize(
    "value,match",
    [("3", "Invalid value 3 for CFN variable x"), ('"d"', "Unknown value 'd'")],
)
def test_knapsackv_invalid_value(value, match):
    params = '{"capacity":1,"weightedvalues":[["x",%s,1]]}' % value
    with pytest.raises(ValueError, match=match):
        load_cfn(string=KNAPSACKV % params)


def test_knapsackv_value_name_and_non_numeric_weight():
    params = '{"capacity":1,"weightedvalues":[["x","b",2]]}'
    (con,) = load_cfn(string=KNAPSACKV % params).constraints
    assert con.terms == [("x", "b", 2)]

    params = '{"capacity":1,"weightedvalues":[["x","b","w"]]}'
    with pytest.raises(ValueError, match="Expected a number for weight in"):
        load_cfn(string=KNAPSACKV % params)


def test_knapsack_scope_errors():
    text = cfn(
        functions='{"k":{"scope":["x","x"],"type":"knapsack",'
        '"params":{"capacity":1,"weights":[1,1]}}}'
    )
    with pytest.raises(ValueError, match="repeated variable"):
        load_cfn(string=text)


def test_unsupported_global_type_message():
    text = cfn(functions='{"g":{"scope":["x"],"type":"wregular","params":{}}}')
    with pytest.raises(
        NotImplementedError,
        match="global cost function type 'wregular'.*knapsack, knapsackv",
    ):
        load_cfn(string=text)


#
# Fixed bugs
#


def test_object_domain_is_rejected():
    # An object domain is parsed as a list of (key, value) pairs; it passed
    # _parse_variables and its pairs became states.
    with pytest.raises(ValueError, match="Unexpected domain for CFN variable x"):
        load_cfn(string=cfn(variables='{"x":{"a":1}}'))


def test_all_infinite_costs_raise():
    # A table whose costs are all infinite gave NaN factor values (inf - inf).
    with pytest.raises(ValueError, match="CFN function f forbids every assignment"):
        load_cfn(
            string='{"variables":{"x":2},'
            '"functions":{"f":{"scope":["x"],"costs":[1e400,1e400]}}}'
        )
