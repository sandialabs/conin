import pytest
import os.path
import importlib

import conin.io
from conin.io import load_model
from conin.exceptions import InvalidInputError
from conin.tests.optional import pgmpy_installed

pgmpy_available = pgmpy_installed
pgmpy_readwrite_available = pgmpy_installed


cwd = os.path.dirname(__file__)

#
# errors
#


def test_load_model_error1():
    with pytest.raises(FileNotFoundError):
        load_model(os.path.join(cwd, "unknown.uai"))


def test_load_model_error2():
    with pytest.raises(InvalidInputError, match="unknown file suffix"):
        load_model(os.path.join(cwd, "test_load_model.py"))


def test_load_model_error3():
    with pytest.raises(InvalidInputError, match="Unexpected model type"):
        load_model(os.path.join(cwd, "asia.uai"), model_type="unknown")


def test_load_model_documented_io_namespace():
    pgm = conin.io.load_model(os.path.join(cwd, "asia.uai"), model_type="conin")
    assert pgm.nodes


@pytest.mark.skipif(
    pgmpy_available, reason="Testing an error when pgmpy is not installed"
)
def test_load_model_error4():
    with pytest.raises(ImportError):
        load_model(os.path.join(cwd, "asia.uai"), model_type="pgmpy")


def test_pgmpy_load_model_requires_pgmpy(monkeypatch):
    module = importlib.import_module("conin.io.pgmpy.files")
    monkeypatch.setattr(module, "pgmpy_available", False)

    with pytest.raises(ImportError, match="pgmpy"):
        module.load_model(os.path.join(cwd, "asia.uai"))


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.parametrize("model_type", ["conin", "pgmpy"])
def test_load_model_missing_bif_file(tmp_path, model_type):
    # A missing file used to be looked up as a pgmpy example model by name.
    with pytest.raises(FileNotFoundError, match="Missing file"):
        load_model(str(tmp_path / "missing.bif"), model_type=model_type)


def test_pgmpy_load_model_file_requires_readwrite(monkeypatch):
    module = importlib.import_module("conin.io.pgmpy.files")
    monkeypatch.setattr(module, "pgmpy_available", True)
    monkeypatch.setattr(module, "pgmpy_readwrite_available", False)

    with pytest.raises(ImportError, match="pgmpy.readwrite"):
        module.load_model(os.path.join(cwd, "asia.uai"))


#
# Compressed files load the same model as their uncompressed versions
#


@pytest.mark.parametrize(
    "plain,compressed",
    [
        ("asia.uai", "asia_compressed.uai.gz"),
        pytest.param(
            "asia.bif",
            "asia_compressed.bif.gz",
            marks=pytest.mark.skipif(
                not pgmpy_readwrite_available, reason="pgmpy not installed"
            ),
        ),
        ("barley.uai", "barley_compressed.uai.gz"),
        ("deer.uai", "deer_compressed.uai.gz"),
    ],
)
def test_load_model_compressed_matches_uncompressed(
    plain, compressed, assert_same_model
):
    model = load_model(os.path.join(cwd, plain))
    assert model.nodes
    assert_same_model(load_model(os.path.join(cwd, compressed)), model)


def test_load_model_uai_cardinalities():
    # The cardinalities of the first variables of the barley.uai header.
    pgm = load_model(os.path.join(cwd, "barley.uai"))
    assert len(pgm.nodes) == 48
    assert [len(pgm.states_of(f"var{i}")) for i in range(5)] == [5, 9, 5, 3, 9]


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
@pytest.mark.parametrize(
    "filename",
    ["asia.uai", "asia_compressed.uai.gz", "asia.bif", "asia_compressed.bif.gz"],
)
def test_load_model_pgmpy(filename):
    pgm = load_model(os.path.join(cwd, filename), model_type="pgmpy")
    assert type(pgm).__module__.startswith("pgmpy.")
    assert len(pgm.nodes()) == len(load_model(os.path.join(cwd, filename)).nodes)


@pytest.mark.skipif(not pgmpy_readwrite_available, reason="pgmpy not installed")
def test_load_model_bif_conin_matches_pgmpy(assert_same_model):
    from conin.io.pgmpy import convert_pgmpy_to_conin

    filename = os.path.join(cwd, "asia.bif")
    assert_same_model(
        load_model(filename),
        convert_pgmpy_to_conin(load_model(filename, model_type="pgmpy")),
    )


#
# cancer
#


def test_load_bn_model_cancer_conin():
    pgm = load_model(os.path.join(cwd, "cancer_bn.uai"))
    cpds = {cpd.node: cpd for cpd in pgm.cpds}
    assert cpds["var0"].values == {
        (0, 0): {0: 0.03, 1: 0.97},
        (0, 1): {0: 0.05, 1: 0.95},
        (1, 0): {0: 0.001, 1: 0.999},
        (1, 1): {0: 0.02, 1: 0.98},
    }
    assert cpds["var1"].values == {0: {0: 0.65, 1: 0.35}, 1: {0: 0.3, 1: 0.7}}
    assert cpds["var2"].values == {0: 0.9, 1: 0.1}
    assert cpds["var3"].values == {0: 0.3, 1: 0.7}
    assert cpds["var4"].values == {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.8}}


def test_load_mn_model_cancer_conin():
    pgm = load_model(os.path.join(cwd, "cancer_mn.uai"))
    factors = {
        tuple(f.nodes) if len(f.nodes) > 1 else f.nodes[0]: f.values
        for f in pgm.factors
    }

    assert factors["var0", "var1"] == {
        (0, 0): 0.65,
        (0, 1): 0.35,
        (1, 0): 0.3,
        (1, 1): 0.7,
    }
    assert factors["var2"] == {0: 0.9, 1: 0.1}
    assert factors["var3"] == {0: 0.3, 1: 0.7}
    assert factors["var0", "var4"] == {
        (0, 0): 0.9,
        (0, 1): 0.1,
        (1, 0): 0.2,
        (1, 1): 0.8,
    }
    assert factors["var2", "var3", "var0"] == {
        (0, 0, 0): 0.03,
        (0, 0, 1): 0.97,
        (0, 1, 0): 0.05,
        (0, 1, 1): 0.95,
        (1, 0, 0): 0.001,
        (1, 0, 1): 0.999,
        (1, 1, 0): 0.02,
        (1, 1, 1): 0.98,
    }


#
# toulbar2
#


def test_load_bn_model_toulbar2_conin():
    pgm = load_model(os.path.join(cwd, "toulbar2_bn.uai"))
    cpds = {cpd.node: cpd for cpd in pgm.cpds}

    assert cpds["var0"].values == {0: 0.436, 1: 0.564}
    assert cpds["var1"].values == {0: {0: 0.128, 1: 0.872}, 1: {0: 0.92, 1: 0.08}}
    assert cpds["var2"].values == {
        0: {0: 0.21, 1: 0.333, 2: 0.457},
        1: {0: 0.811, 1: 0.0, 2: 0.189},
    }


@pytest.mark.skipif(
    not pgmpy_readwrite_available, reason="pgmpy.readwrite is not available"
)
def test_pgmpy_load_model_unknown_suffix_names_the_file(tmp_path):
    # The message was not an f-string, so it printed "{name}".
    module = importlib.import_module("conin.io.pgmpy.files")
    name = tmp_path / "model.txt"
    name.write_text("")
    with pytest.raises(InvalidInputError, match="model.txt"):
        module.load_model(str(name))


def test_uai_markov_network_edges_join_every_pair_of_factor_nodes():
    # The UAI reader joined only the last node of each factor to the others.
    import itertools

    pgm = load_model(os.path.join(cwd, "cancer_mn.uai"))
    edges = {frozenset(e) for e in pgm.edges}
    for factor in pgm.factors:
        for a, b in itertools.combinations(factor.nodes, 2):
            assert frozenset((a, b)) in edges
