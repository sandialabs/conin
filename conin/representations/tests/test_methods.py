"""The method table shared by map_query and log_probability_query."""

import importlib

import pytest

from conin.representations.methods import (
    LOG_PROBABILITY_METHODS,
    MAP_METHODS,
    REPRESENTATION_KINDS,
    check_method,
    representation_kind,
)

mq = importlib.import_module("conin.inference.map_query")
lpq = importlib.import_module("conin.scoring.log_probability_query")


def test_every_method_has_a_representation_kind():
    assert set(MAP_METHODS) | set(LOG_PROBABILITY_METHODS) == set(REPRESENTATION_KINDS)


def test_the_entry_points_use_the_table():
    assert mq.METHODS is MAP_METHODS
    assert lpq.METHODS is LOG_PROBABILITY_METHODS


def test_a_method_of_both_queries_runs_on_the_same_kind():
    for method in set(MAP_METHODS) & set(LOG_PROBABILITY_METHODS):
        assert lpq._representation_kind(method) == representation_kind(method, None, {})


def test_check_method():
    check_method("viterbi", MAP_METHODS, "inference type")
    with pytest.raises(
        ValueError, match="Unsupported thing: bogus. Expected one of: a, b"
    ):
        check_method("bogus", ("a", "b"), "thing")
