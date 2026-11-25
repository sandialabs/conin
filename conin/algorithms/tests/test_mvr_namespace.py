"""``conin.algorithms.mvr`` gathers the MVR algorithms of every subpackage."""

import importlib

import pytest

import conin.algorithms.mvr as mvr_namespace
from conin.tests.optional import needs_torch

HOMES = [
    "conin.algorithms.inference.mvr",
    "conin.algorithms.other_queries.mvr",
    "conin.algorithms.sampling.mvr",
    "conin.algorithms.learning.mvr",
]


def test_namespace_has_every_mvr_algorithm():
    names = set()
    for home in HOMES:
        names |= set(importlib.import_module(home).__all__)
    assert set(mvr_namespace.__all__) == names


@needs_torch
@pytest.mark.parametrize("home", HOMES)
def test_namespace_gives_the_same_functions(home):
    module = importlib.import_module(home)
    for name in module.__all__:
        assert getattr(mvr_namespace, name) is getattr(module, name)


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError):
        mvr_namespace.mvr_unknown
