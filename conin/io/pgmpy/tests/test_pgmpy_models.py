"""convert_pgmpy_to_conin on pgmpy models that have a CONIN twin.

The converted model must give the same (log-)probability as the CONIN
example on every total assignment (a sample for the larger DBNs, unrolled
over ``0..2``), and the same MAP solution.
"""

import importlib
import itertools
import math
import random

import pytest

from conin.algorithms.inference import map_query
from conin.algorithms.scoring import (
    log_potential,
    log_probability,
    log_probability_query,
)
from conin.graphical_models import (
    DynamicDiscreteBayesianNetwork,
    create_bn_from_dbn,
)
from conin.io.pgmpy import convert_pgmpy_to_conin
from conin.tests.optional import needs_pgmpy

from . import pgmpy_models

pytestmark = needs_pgmpy


def _twin(name):
    module, function = pgmpy_models.CONIN_TWINS[name].split(".")
    return getattr(importlib.import_module(f"conin.examples.{module}"), function)()


def _assignments(pgm, rng, limit=500):
    nodes = list(pgm.nodes)
    total = math.prod(len(pgm.states[n]) for n in nodes)
    if total <= limit:
        for values in itertools.product(*(pgm.states[n] for n in nodes)):
            yield dict(zip(nodes, values))
    else:
        for _ in range(limit):
            yield {n: rng.choice(pgm.states[n]) for n in nodes}


@pytest.mark.parametrize("name", sorted(pgmpy_models.CONIN_TWINS))
def test_converted_model_matches_its_conin_twin(name):
    example = getattr(pgmpy_models, name)()
    converted = convert_pgmpy_to_conin(example.pgm)
    expected = _twin(name).pgm
    assert type(converted) is type(expected)
    if isinstance(expected, DynamicDiscreteBayesianNetwork):
        converted = create_bn_from_dbn(dbn=converted, start=0, stop=2)
        expected = create_bn_from_dbn(dbn=expected, start=0, stop=2)
    assert set(converted.nodes) == set(expected.nodes)
    for node in expected.nodes:
        assert list(converted.states[node]) == list(expected.states[node])
    rng = random.Random(name)
    for assignment in _assignments(expected, rng):
        assert log_potential(converted, assignment) == pytest.approx(
            log_potential(expected, assignment), abs=1e-9
        )


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in sorted(pgmpy_models.CONIN_TWINS)
        if "DDBN" not in name and "weather" not in name
    ],
)
def test_converted_model_has_the_map_solution(name):
    example = getattr(pgmpy_models, name)()
    if example.solutions[0].states is None:
        pytest.skip(f"{name} has no recorded MAP solution")
    results = map_query(
        convert_pgmpy_to_conin(example.pgm), method="variable_elimination"
    )
    assert results.solution.states == example.solutions[0].states


@pytest.mark.parametrize(
    "query",
    [
        lambda pgm: map_query(pgm, method="variable_elimination"),
        lambda pgm: log_probability_query(pgm, {"Cancer": 1}),
        lambda pgm: log_potential(pgm, {"Cancer": 1}),
        lambda pgm: log_probability(pgm, {"Cancer": 1}),
    ],
    ids=["map_query", "log_probability_query", "log_potential", "log_probability"],
)
def test_pgmpy_models_are_rejected(query):
    pgm = pgmpy_models.cancer1_BN_pgmpy().pgm
    with pytest.raises(TypeError, match="convert them with conin.io.pgmpy"):
        query(pgm)
