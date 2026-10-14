import importlib

import pytest

from conin.constraints import Constraint, MVRConstraint
from conin.constraints.mvr.mvr import BaseMVR, HomMVR, InhomMVR
from conin.constraints.mvr.primitives import mvr_forbid_state
from conin.constraints.mvr.tests.test_mvr_constraints import make_hmm


@pytest.mark.parametrize("cls", [BaseMVR, HomMVR, InhomMVR])
def test_mvr_classes_are_mvr_family_constraints(cls):
    assert issubclass(cls, Constraint)
    assert cls.family == "mvr"


def test_built_mvr_is_mvr_family_constraint():
    hmm = make_hmm(hidden_states=("A", "B"))
    mvr = mvr_forbid_state(hmm, {"A"})

    assert isinstance(mvr, HomMVR)
    assert isinstance(mvr, Constraint)
    assert mvr.family == "mvr"


def test_mvr_constraint_factory_is_mvr_family():
    assert MVRConstraint.family == "mvr"
    assert MVRConstraint.family == BaseMVR.family


def test_public_mvr_names_import_from_package():
    from conin.constraints.mvr import (  # noqa: F401
        HomMVR,
        mvr_forbid_state,
        mvr_not_yet,
        build_mvr,
        build_mvr_functor,
    )


def test_conin_operators_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("conin.operators")
