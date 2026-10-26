"""The public variable-elimination wrappers (inference_variable_elimination_map_query_*).

They are documented on the API page but were only reached through map_query.
Each one must give the same result as map_query(method="variable_elimination").
"""

import munch
import pytest

import conin.examples.bayesian_network as bex
import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.exceptions import InvalidInputError
from conin.inference import map_query
from conin.inference.variable_elimination import (
    inference_variable_elimination_map_query_BN,
    inference_variable_elimination_map_query_DDBN,
    inference_variable_elimination_map_query_HMM,
    inference_variable_elimination_map_query_MN,
)
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401

pytestmark = pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")


def _states(results):
    return results.solution.states


def _same(wrapper_results, map_query_results):
    assert _states(wrapper_results) == _states(map_query_results)
    assert "solvetime" in wrapper_results


# ---------------------------------------------------------------------
# Markov and Bayesian networks
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "example", [mex.ABC_conin, mex.ABC_constrained_oracle_conin], ids=["mn", "cmn"]
)
@pytest.mark.parametrize("evidence", [None, {"B": 0}])
def test_mn_wrapper_matches_map_query(example, evidence):
    pgm = example().pgm
    _same(
        inference_variable_elimination_map_query_MN(pgm=pgm, evidence=evidence),
        map_query(pgm, method="variable_elimination", evidence=evidence),
    )


def test_mn_wrapper_solution():
    example = mex.ABC_conin()
    results = inference_variable_elimination_map_query_MN(pgm=example.pgm)
    assert _states(results) == example.solutions[0].states


def test_mn_wrapper_partial_query_and_evidence_in_the_solution():
    pgm = mex.ABC_conin().pgm
    results = inference_variable_elimination_map_query_MN(
        pgm=pgm, variables=["A"], evidence={"B": 0}, solution_with_evidence=True
    )
    assert _states(results) == {"A": 2, "B": 0}


@pytest.mark.parametrize(
    "example",
    [bex.cancer1_BN_conin, bex.cancer1_BN_constrained_oracle_conin],
    ids=["bn", "cbn"],
)
@pytest.mark.parametrize("evidence", [None, {"Smoker": 1}])
def test_bn_wrapper_matches_map_query(example, evidence):
    pgm = example().pgm
    _same(
        inference_variable_elimination_map_query_BN(pgm=pgm, evidence=evidence),
        map_query(pgm, method="variable_elimination", evidence=evidence),
    )


def test_bn_wrapper_solution():
    example = bex.cancer1_BN_constrained_oracle_conin()
    results = inference_variable_elimination_map_query_BN(pgm=example.pgm)
    assert _states(results) == example.solutions[0].states


def test_wrappers_ignore_unknown_options():
    pgm = mex.ABC_conin().pgm
    results = inference_variable_elimination_map_query_MN(pgm=pgm, solver="glpk")
    assert _states(results) == mex.ABC_conin().solutions[0].states


def test_mn_wrapper_writes_the_factor_model(tmp_path):
    filename = tmp_path / "model.uai"
    inference_variable_elimination_map_query_MN(
        pgm=mex.ABC_constrained_oracle_conin().pgm, write_uai_file=str(filename)
    )
    assert filename.read_text().startswith("MARKOV")


def test_wrapper_rejects_the_wrong_model_type():
    with pytest.raises(InvalidInputError):
        inference_variable_elimination_map_query_MN(pgm=bex.cancer1_BN_conin().pgm)


# ---------------------------------------------------------------------
# Dynamic Bayesian networks
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "example",
    [dex.simple1_DDBN_conin, dex.simple1_DDBN_constrained_oracle_conin],
    ids=["dbn", "cdbn"],
)
def test_dbn_wrapper_matches_map_query(example):
    pgm = example().pgm
    _same(
        inference_variable_elimination_map_query_DDBN(pgm=pgm, stop=2),
        map_query(pgm, method="variable_elimination", stop=2),
    )


def test_dbn_wrapper_stop_none_means_one():
    example = dex.simple1_DDBN_constrained_oracle_conin()
    results = inference_variable_elimination_map_query_DDBN(pgm=example.pgm, stop=None)
    assert _states(results) == example.solutions[0].states
    assert _states(results) == _states(
        inference_variable_elimination_map_query_DDBN(pgm=example.pgm)
    )


# ---------------------------------------------------------------------
# HMMs
# ---------------------------------------------------------------------

OBSERVED = ["o0", "o1", "o1", "o0", "o1"]


@pytest.mark.parametrize(
    "make", [hex_.create_hmm1, hex_.create_chmm1_oracle_ve], ids=["hmm", "chmm"]
)
@pytest.mark.parametrize("observed", [OBSERVED, dict(enumerate(OBSERVED))])
def test_hmm_wrapper_matches_map_query(make, observed):
    pgm = make()
    _same(
        inference_variable_elimination_map_query_HMM(pgm=pgm, evidence=observed),
        map_query(pgm, method="variable_elimination", evidence=observed),
    )


def test_hmm_wrapper_shapes_the_hidden_states():
    hmm = hex_.create_hmm1()
    as_list = inference_variable_elimination_map_query_HMM(pgm=hmm, evidence=OBSERVED)
    as_dict = inference_variable_elimination_map_query_HMM(
        pgm=hmm, evidence=dict(enumerate(OBSERVED))
    )
    assert isinstance(_states(as_list), list) and len(_states(as_list)) == 5
    assert _states(as_dict) == dict(enumerate(_states(as_list)))


def test_hmm_wrapper_start_and_stop():
    hmm = hex_.create_hmm1()
    results = inference_variable_elimination_map_query_HMM(
        pgm=hmm, evidence=OBSERVED, start=0, stop=2
    )
    assert len(_states(results)) == 3


@pytest.mark.parametrize(
    "evidence,match",
    [(None, "observed sequence must be a list"), (OBSERVED, "stop")],
)
def test_hmm_wrapper_bad_observations(evidence, match):
    kwargs = dict(stop=10) if evidence is not None else {}
    with pytest.raises(InvalidInputError, match=match):
        inference_variable_elimination_map_query_HMM(
            pgm=hex_.create_hmm1(), evidence=evidence, **kwargs
        )


def test_hmm_wrapper_timing(capsys):
    inference_variable_elimination_map_query_HMM(
        pgm=hex_.create_hmm1(), evidence=OBSERVED, timing=True
    )
    out = capsys.readouterr().out
    assert "map_query - START" in out and "map_query - STOP" in out


def test_munch_results():
    results = inference_variable_elimination_map_query_MN(pgm=mex.ABC_conin().pgm)
    assert isinstance(results, munch.Munch)
