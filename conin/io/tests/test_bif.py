"""The BIF reader and writer (``conin.io.bif``).

The 24 bnlearn networks in ``conin_scaling_experiments``
(``data/bnlearn/*.bif.gz``) are a larger check: each loads and round-trips
through ``save_bif``.
"""

import gzip
import itertools
import os
import textwrap

import numpy as np
import pytest

from conin import DiscreteBayesianNetwork, DiscreteCPD
from conin.exceptions import InvalidInputError, ModelFileError
from conin.io import load_bif, load_model, save_bif, save_model

cwd = os.path.dirname(__file__)

BIF_FILES = ["asia.bif", "cancer_bn.bif"]


def canonical(bn):
    """``{node: (parents, {parent states: {state: p}})}``, labels as strings."""
    result = {}
    for cpd in bn.cpds:
        parents = tuple(str(p) for p in cpd.parents or [])
        if parents:
            rows = {
                tuple(str(s) for s in (k if isinstance(k, tuple) else (k,))): v
                for k, v in cpd.values.items()
            }
        else:
            rows = {(): cpd.values}
        result[str(cpd.node)] = (
            parents,
            {k: {str(s): float(p) for s, p in v.items()} for k, v in rows.items()},
        )
    return result


def state_lists(bn):
    return {str(n): [str(s) for s in bn.states_of(n)] for n in bn.nodes}


def random_bn(rng, labels):
    """A BN over 2-6 nodes with up to 3 parents and 1-4 states per node."""
    n = int(rng.integers(2, 7))
    nodes = [labels(rng, f"N{i}", "node") for i in range(n)]
    states = {
        node: [labels(rng, f"s{j}", "state") for j in range(int(rng.integers(1, 5)))]
        for node in nodes
    }
    cpds = []
    for i, node in enumerate(nodes):
        k = int(rng.integers(0, min(i, 3) + 1))
        parents = [nodes[j] for j in sorted(rng.choice(i, size=k, replace=False))]

        def dist():
            w = rng.uniform(0.0, 1.0, size=len(states[node]))
            w[rng.random(len(w)) < 0.2] = 0.0
            if not w.any():
                w[0] = 1.0
            return [float(p) for p in w / w.sum()]

        if parents:
            keys = list(itertools.product(*(states[p] for p in parents)))
            values = {key[0] if k == 1 else key: dist() for key in keys}
            cpds.append(DiscreteCPD(node=node, parents=parents, values=values))
        else:
            cpds.append(DiscreteCPD(node=node, values=dist()))
    return DiscreteBayesianNetwork(states=states, cpds=cpds)


# Label generators: ``labels(rng, base, kind)`` is a unique label for
# ``base`` (kind is "node" or "state").


def plain_labels(rng, base, kind):
    return base


def spaced_labels(rng, base, kind):
    # Words, digits, '_', '-', and (for most states) spaces.
    if kind == "node" or base == "s0":
        return str(rng.choice([base, f"{base}-y_z", f"{base}.1"]))
    return str(rng.choice([base, f"{base} x", f"{base}-y_z", f"{base} a b"]))


def odd_labels(rng, base, kind):
    # Names that need quoting in BIF; the first label of each kind is a
    # keyword.
    if base in ("N0", "s0"):
        return str(rng.choice([base, "table", "probability", "property"]))
    return str(
        rng.choice([base, f"{base} x", f"{base}(a, b)", f"{base};{{}}", f"//{base}"])
    )


# ---------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------


def test_asia_values():
    bn = load_model(os.path.join(cwd, "asia.bif"))
    assert state_lists(bn) == {
        n: ["yes", "no"]
        for n in ["asia", "bronc", "dysp", "either", "lung", "smoke", "tub", "xray"]
    }
    cpds = canonical(bn)
    assert cpds["asia"] == ((), {(): {"yes": 0.01, "no": 0.99}})
    assert cpds["tub"] == (
        ("asia",),
        {("yes",): {"yes": 0.05, "no": 0.95}, ("no",): {"yes": 0.01, "no": 0.99}},
    )
    assert cpds["dysp"] == (
        ("bronc", "either"),
        {
            ("yes", "yes"): {"yes": 0.9, "no": 0.1},
            ("no", "yes"): {"yes": 0.7, "no": 0.3},
            ("yes", "no"): {"yes": 0.8, "no": 0.2},
            ("no", "no"): {"yes": 0.1, "no": 0.9},
        },
    )


def test_conditional_table_order():
    # A "table" entry for a node with parents: the node's state varies
    # slowest, the last parent's state fastest (the JavaBayes order).
    text = textwrap.dedent("""
        variable A { type discrete [ 2 ] { a0, a1 }; }
        variable B { type discrete [ 3 ] { b0, b1, b2 }; }
        variable C { type discrete [ 2 ] { c0, c1 }; }
        probability ( A ) { table 0.3 0.7; }
        probability ( B ) { table 0.2, 0.3, 0.5; }
        probability ( C | A, B ) {
          table 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4;
        }
        """)
    rows = canonical(load_bif(string=text))["C"][1]
    assert rows == {
        ("a0", "b0"): {"c0": 0.1, "c1": 0.9},
        ("a0", "b1"): {"c0": 0.2, "c1": 0.8},
        ("a0", "b2"): {"c0": 0.3, "c1": 0.7},
        ("a1", "b0"): {"c0": 0.4, "c1": 0.6},
        ("a1", "b1"): {"c0": 0.5, "c1": 0.5},
        ("a1", "b2"): {"c0": 0.6, "c1": 0.4},
    }


# ---------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------


@pytest.mark.parametrize("labels", [plain_labels, spaced_labels, odd_labels])
@pytest.mark.parametrize("seed", range(40))
def test_random_bn_round_trip(seed, labels):
    bn = random_bn(np.random.default_rng(seed), labels)
    loaded = load_bif(string=save_bif(bn))
    # repr keeps every probability exact.
    assert canonical(loaded) == canonical(bn)
    assert state_lists(loaded) == state_lists(bn)


@pytest.mark.parametrize("name", BIF_FILES)
def test_file_round_trip(name):
    bn = load_bif(os.path.join(cwd, name))
    assert canonical(load_bif(string=save_bif(bn))) == canonical(bn)


def test_non_string_labels_are_written_as_strings(tmp_path):
    bn = DiscreteBayesianNetwork(
        states={("X", 0): [0, 1], ("X", 1): [True, False]},
        cpds=[
            DiscreteCPD(node=("X", 0), values=[0.25, 0.75]),
            DiscreteCPD(
                node=("X", 1),
                parents=[("X", 0)],
                values={0: [1.0, 0.0], 1: [0.5, 0.5]},
            ),
        ],
    )
    fname = str(tmp_path / "labels.bif")
    assert save_bif(bn, fname) is None
    loaded = load_model(fname)
    assert loaded.states == {"('X', 0)": ["0", "1"], "('X', 1)": ["True", "False"]}
    assert canonical(loaded) == canonical(bn)


def test_save_and_load_model_gzip(tmp_path):
    bn = load_bif(os.path.join(cwd, "asia.bif"))
    fname = str(tmp_path / "asia.bif.gz")
    save_model(bn, fname)
    with gzip.open(fname, "rt") as f:
        assert f.read() == save_bif(bn)
    assert canonical(load_model(fname)) == canonical(bn)


def test_compressed_file_matches_plain_file():
    plain = load_model(os.path.join(cwd, "asia.bif"))
    compressed = load_model(os.path.join(cwd, "asia_compressed.bif.gz"))
    assert compressed.cpds == plain.cpds


# ---------------------------------------------------------------------
# The grammar
# ---------------------------------------------------------------------


def test_comments_properties_and_quotes():
    text = textwrap.dedent("""
        // A comment
        network "my net" {
          property author = "someone" ;
        }
        /* A block
           comment */
        variable "Rain today" {
          type discrete [ 2 ] { "yes, really", no };  // trailing comment
          property position = (1, 2) ;
        }
        variable wet {
          type discrete [ 2 ] { true false };
        }
        probability ( "Rain today" ) {
          property note = "x" ;
          table 2e-1, 8E-1;
        }
        probability ( wet | "Rain today" ) {
          ("yes, really") 0.9 0.1;
          (no) .2, .8;
        }
        """)
    bn = load_bif(string=text)
    assert state_lists(bn) == {
        "Rain today": ["yes, really", "no"],
        "wet": ["true", "false"],
    }
    assert canonical(bn)["wet"][1] == {
        ("yes, really",): {"true": 0.9, "false": 0.1},
        ("no",): {"true": 0.2, "false": 0.8},
    }


def test_multi_word_names_and_case_insensitive_references():
    text = textwrap.dedent("""
        variable Heavy Rain { type discrete [ 2 ] { very high, low }; }
        variable Wind { type discrete [ 1 ] { calm }; }
        probability ( wind ) { table 1; }
        variable Wet { type discrete [ 2 ] { yes, no }; }
        probability ( heavy rain ) { table 0.5, 0.5; }
        probability ( WET | Heavy  Rain ) {
          (very high) 0.9, 0.1;
          (low) 0.2, 0.8;
        }
        """)
    bn = load_bif(string=text)
    assert canonical(bn)["Wet"] == (
        ("Heavy Rain",),
        {
            ("very high",): {"yes": 0.9, "no": 0.1},
            ("low",): {"yes": 0.2, "no": 0.8},
        },
    )


def test_old_style_probability_header():
    # "probability ( node parent1 parent2 )", without '|'.
    text = textwrap.dedent("""
        variable a { type discrete [ 2 ] { 0, 1 }; }
        variable b { type discrete [ 2 ] { 0, 1 }; }
        variable c { type discrete [ 2 ] { 0, 1 }; }
        probability ( a ) { table 0.5, 0.5; }
        probability ( b ) { table 0.5, 0.5; }
        probability ( c a b ) {
          (0, 0) 1, 0; (0, 1) 0, 1; (1, 0) 0, 1; (1, 1) 1, 0;
        }
        """)
    bn = load_bif(string=text)
    assert canonical(bn)["c"][0] == ("a", "b")
    assert canonical(bn)["c"][1][("1", "0")] == {"0": 0.0, "1": 1.0}


def test_default_entry():
    text = textwrap.dedent("""
        variable a { type discrete [ 3 ] { x, y, z }; }
        variable b { type discrete [ 2 ] { 0, 1 }; }
        probability ( a ) { table 0.2, 0.3, 0.5; }
        probability ( b | a ) {
          (y) 0.9, 0.1;
          default 0.5, 0.5;
        }
        """)
    rows = canonical(load_bif(string=text))["b"][1]
    assert rows == {
        ("x",): {"0": 0.5, "1": 0.5},
        ("y",): {"0": 0.9, "1": 0.1},
        ("z",): {"0": 0.5, "1": 0.5},
    }


def test_root_node_row_without_parent_states():
    text = "variable a { type discrete [ 2 ] { 0, 1 }; }\n"
    text += "probability ( a ) { () 0.25, 0.75; }\n"
    assert canonical(load_bif(string=text))["a"][1] == {(): {"0": 0.25, "1": 0.75}}


def test_writer_quotes_names_that_need_it():
    bn = DiscreteBayesianNetwork(
        states={"a b": ["x", "y(1)"], "table": ["", "-1.5e3"]},
        cpds=[
            DiscreteCPD(node="a b", values=[0.5, 0.5]),
            DiscreteCPD(
                node="table",
                parents=["a b"],
                values={"x": [0.1, 0.9], "y(1)": [1.0, 0.0]},
            ),
        ],
    )
    text = save_bif(bn, name="net 1")
    assert text.startswith('network "net 1" {\n}\n')
    assert 'variable "a b" {\n  type discrete [ 2 ] { x, "y(1)" };' in text
    assert 'variable "table" {\n  type discrete [ 2 ] { "", -1.5e3 };' in text
    assert 'probability ( "table" | "a b" ) {\n  (x) 0.1, 0.9;' in text
    assert canonical(load_bif(string=text)) == canonical(bn)


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------

VARS = "variable a { type discrete [ 2 ] { 0, 1 }; }\n"
VARS += "variable b { type discrete [ 2 ] { 0, 1 }; }\n"
PA = "probability ( a ) { table 0.5, 0.5; }\n"


@pytest.mark.parametrize(
    "text, match",
    [
        ("", "declares no variables"),
        ("foo", "Expected 'network', 'variable' or 'probability', got 'foo' at line 1"),
        ("variable a {", "Expected '}' at the end"),
        ('variable "a', "Unterminated quoted string on line 1"),
        ("/* x\n", "Unterminated comment on line 1"),
        ("variable a { }", "Variable 'a' has no type"),
        (
            "variable a { type discrete [ 3 ] { 0, 1 }; }",
            "declares 3 states but lists 2",
        ),
        ("variable a { type discrete [ x ] { 0 }; }", "the number of states, got 'x'"),
        ("variable a { type discrete [ 2 ] { 0, 0 }; }", "repeated states"),
        ("variable a { type discrete [ 1 ] { 0 } }", "Expected ';', got '}'"),
        (VARS + VARS, "Variable 'a' is declared twice at line 3"),
        (VARS + PA, r"No probability block for the variables \['b'\]"),
        (VARS + PA + PA, "two probability blocks"),
        (VARS + "probability ( c ) { table 1; }", "Undeclared variable 'c' at line 3"),
        (VARS + "probability ( a ) { table 0.5; }", "has 1 values, expected 2"),
        (
            VARS + "probability ( a ) { table 0.5, x; }",
            "Expected a probability, got 'x'",
        ),
        (VARS + "probability ( a ) { }", "probability block of 'a' is empty"),
        (VARS + "probability ( a ) { table 1, 0; (0) 1, 0; }", "mixes 'table'"),
        (VARS + "probability ( a ) { default 1, 0; default 1, 0; }", "two defaults"),
        (
            VARS + PA + "probability ( b | a ) { (0) 1, 0; }",
            r"no entry for the parent states \('1',\)",
        ),
        (VARS + PA + "probability ( b | a ) { (0) 1, 0; (0) 1, 0; }", "repeats"),
        (VARS + PA + "probability ( b | a ) { (2) 1, 0; }", "Unknown state '2' of 'a'"),
        (
            VARS + PA + "probability ( b | a ) { (0, 1) 1, 0; }",
            "2 parent states, expected 1",
        ),
        (
            VARS + PA + "probability ( b | a ) { (0) 1, 0, 0; }",
            "has 3 values, expected 2",
        ),
        (VARS + PA + "probability ( b | ) { table 1, 0; }", "Expected a parent name"),
        (
            VARS + "probability ( a | b ) { table 0.5, 0.5, 0.5, 0.5; }\n"
            "probability ( b | a ) { table 0.5, 0.5, 0.5, 0.5; }",
            "cycle",
        ),
        (VARS + PA + "probability ( b ) { table 0.7, 0.7; }", "sum"),
    ],
)
def test_malformed_data(text, match):
    with pytest.raises(ModelFileError, match=match):
        load_bif(string=text)


def test_continuous_variable_is_not_implemented():
    with pytest.raises(NotImplementedError, match="type 'continuous'"):
        load_bif(string="variable a { type continuous [ 1 ] { 0 }; }")


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_model(os.path.join(cwd, "missing.bif"))


def test_load_bif_needs_input():
    with pytest.raises(InvalidInputError):
        load_bif()


def test_save_bif_rejects_colliding_labels():
    bn = DiscreteBayesianNetwork(
        states={"a": [1, "1"]}, cpds=[DiscreteCPD(node="a", values=[0.5, 0.5])]
    )
    with pytest.raises(InvalidInputError, match="string forms repeat"):
        save_bif(bn)


def test_save_bif_rejects_double_quotes():
    bn = DiscreteBayesianNetwork(
        states={'a"b': [0, 1]}, cpds=[DiscreteCPD(node='a"b', values=[0.5, 0.5])]
    )
    with pytest.raises(InvalidInputError, match="double quote|'\"'"):
        save_bif(bn)
