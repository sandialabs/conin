"""Tests for the code the backends share.

* :func:`conin.representations.base.apply_constraints`, the family dispatch of
  the constraint appliers;
* :mod:`conin.representations.unrolled`, the time range of the DBN and HMM
  representations;
* the constraint ``data`` of the DBN and HMM representations, which is the same
  for every backend and for ``is_feasible``.
"""

import importlib

import munch
import pytest

import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
from conin.constrained_models import (
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constraints import (
    FactorOracleConstraint,
    OracleConstraint,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.constraints.base import constraint_families
from conin.exceptions import InvalidInputError
from conin.representations.base import apply_constraints
from conin.representations.unrolled import (
    DEFAULT_DBN_STOP,
    UnrolledDBNMixin,
    UnrolledHMMMixin,
    dbn_time_range,
    hmm_time_range,
)
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

# ---------------------------------------------------------------------
# apply_constraints
# ---------------------------------------------------------------------


@pyomo_constraint_fn()
def _pyomo_con(model):
    return model


@toulbar2_constraint_fn()
def _tb2_con(model):
    return model


def test_constraint_families_keeps_the_order_of_first_appearance():
    assert constraint_families([]) == []
    assert constraint_families([_tb2_con, _pyomo_con, _tb2_con]) == [
        "toulbar2",
        "pyomo",
    ]


def test_apply_constraints_without_constraints_calls_nothing():
    def fail(constraints):
        raise AssertionError("called")

    assert apply_constraints((), {"pyomo": fail}, backend="x") is None


def test_apply_constraints_calls_the_applier_of_the_family():
    seen = []
    result = apply_constraints(
        (_pyomo_con, _pyomo_con),
        {
            "pyomo": lambda cons: seen.append(cons) or "applied",
            "toulbar2": lambda cons: pytest.fail("wrong family"),
        },
        backend="x",
    )
    assert result == "applied"
    assert seen == [(_pyomo_con, _pyomo_con)]


def test_apply_constraints_rejects_mixed_families():
    with pytest.raises(
        TypeError,
        match=r"The demo backend applies constraints of one family, "
        r"found \['pyomo', 'toulbar2'\]",
    ):
        apply_constraints(
            (_pyomo_con, _tb2_con),
            {"pyomo": lambda c: c, "toulbar2": lambda c: c},
            backend="demo",
        )


def test_apply_constraints_rejects_an_unhandled_family():
    with pytest.raises(
        TypeError,
        match=r"The demo backend does not apply constraints of family 'toulbar2' "
        r"\(Toulbar2Constraint\); supported families: \['algebraic', 'pyomo'\]",
    ):
        apply_constraints(
            (_tb2_con,),
            {"pyomo": lambda c: c, "algebraic": lambda c: c},
            backend="demo",
        )


def test_pyomo_applier_rejects_other_families():
    common = importlib.import_module("conin.representations.pyomo.common")
    with pytest.raises(TypeError, match="pyomo backend does not apply .* 'toulbar2'"):
        common.apply_pyomo_constraints(
            pgm=None, constraints=(_tb2_con,), model=object(), data=None
        )
    with pytest.raises(TypeError, match="pyomo backend applies .* one family"):
        common.apply_pyomo_constraints(
            pgm=None, constraints=(_pyomo_con, _tb2_con), model=object(), data=None
        )


def test_pyomo_applier_without_constraints_returns_the_model():
    common = importlib.import_module("conin.representations.pyomo.common")
    model = object()
    assert (
        common.apply_pyomo_constraints(pgm=None, constraints=(), model=model, data=1)
        is model
    )


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_pgmpy_applier_rejects_mixed_families():
    from conin.representations.pgmpy import apply_pgmpy_constraints

    import conin.examples.markov_network as mex

    factor = FactorOracleConstraint(func=lambda s: True, nodes=["A"])
    with pytest.raises(TypeError, match="pgmpy backend applies .* one family"):
        apply_pgmpy_constraints(
            pgm=mex.ABC_conin().pgm, constraints=[factor, _pyomo_con]
        )


# ---------------------------------------------------------------------
# Time ranges
# ---------------------------------------------------------------------


def test_dbn_time_range_defaults_stop():
    assert DEFAULT_DBN_STOP == 1
    assert dbn_time_range(0, None) == (0, 1)
    assert dbn_time_range(0, 4) == (0, 4)


def test_hmm_time_range_needs_a_stop():
    assert hmm_time_range(0, 3) == (0, 3)
    with pytest.raises(InvalidInputError, match="needs a stop time"):
        hmm_time_range(0, None)


class _Base:
    def __init__(self, *, data=None):
        # The mixins set their fields before the base __init__ runs.
        self.fields_at_init = self.structure_fields()
        self.data = data


class _DBN(UnrolledDBNMixin, _Base):
    pass


class _HMM(UnrolledHMMMixin, _Base):
    pass


def test_dbn_mixin():
    repn = _DBN(start=0, stop=2, data="D")
    assert (repn.start, repn.stop, repn.data) == (0, 2, "D")
    assert repn.time_steps == [0, 1, 2]
    assert repn.structure_fields() == munch.Munch(start=0, stop=2, T=[0, 1, 2])
    assert repn.fields_at_init == repn.structure_fields()


def test_hmm_mixin_copies_hidden_to_internal():
    h2i = {"a": 0, "b": 1}
    repn = _HMM(start=0, stop=1, hidden_to_internal=h2i)
    assert repn.time_steps == [0, 1]
    fields = repn.structure_fields()
    assert fields == munch.Munch(
        start=0, stop=1, hmm=munch.Munch(T=[0, 1], hidden_to_internal=h2i)
    )
    fields.hmm.hidden_to_internal["c"] = 2
    assert h2i == {"a": 0, "b": 1}
    assert repn.fields_at_init == repn.structure_fields()


# ---------------------------------------------------------------------
# Every DBN/HMM representation gives the constraints the same time fields
# ---------------------------------------------------------------------

DBN_KINDS = ["pyomo"]
if pytoulbar2_available:
    DBN_KINDS.append("toulbar2")
if pgmpy_available:
    DBN_KINDS.append("pgmpy")


def _dbn_fields(dbn_stop):
    return munch.Munch(start=0, stop=dbn_stop, T=list(range(dbn_stop + 1)))


def _hmm_fields(hmm, stop):
    return munch.Munch(
        start=0,
        stop=stop,
        hmm=munch.Munch(
            T=list(range(stop + 1)), hidden_to_internal=hmm.hidden_to_internal
        ),
    )


def _time_fields(data):
    return munch.Munch({k: data[k] for k in ("start", "stop", "T", "hmm") if k in data})


def _structure_fields_of(repn):
    data = (
        repn.constraint_data()
        if repn.kind == "pgmpy"
        else repn.constraint_data(evidence=None, variables=None)
    )
    return _time_fields(data)


@pytest.mark.parametrize("kind", DBN_KINDS)
@pytest.mark.parametrize("stop", [None, 0, 3])
def test_dbn_representations_share_the_time_fields(kind, stop):
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(dex.weather_conin().pgm)
    params = {} if stop is None else dict(stop=stop)
    repn = cmodel.representation(kind, **params)
    expected_stop = DEFAULT_DBN_STOP if stop is None else stop
    assert (repn.start, repn.stop) == (0, expected_stop)
    assert repn.time_steps == list(range(expected_stop + 1))
    assert _structure_fields_of(repn) == _dbn_fields(expected_stop)


@pytest.mark.parametrize("kind", DBN_KINDS)
def test_hmm_representations_share_the_time_fields(kind):
    hmm = hex_.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm)
    with pytest.raises(InvalidInputError, match="needs a stop time"):
        cmodel.representation(kind)
    repn = cmodel.representation(kind, stop=2)
    assert repn.hidden_to_internal == hmm.hidden_to_internal
    assert _structure_fields_of(repn) == _hmm_fields(hmm, 2)


def test_hmm_is_feasible_and_oracle_see_the_same_time_fields():
    seen = []

    def record(states, data):
        seen.append(_time_fields(data))
        return True

    hmm = hex_.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm, OracleConstraint(func=record))
    cmodel.is_feasible(["h0", "h1", "h0"])
    cmodel.representation("oracle").is_feasible([0, 1, 0])
    assert seen == [_hmm_fields(hmm, 2)] * 2


def test_dbn_is_feasible_sees_the_dbn_time_fields():
    seen = []

    def scope(data):
        seen.append(_time_fields(data))
        return [("A", 0)]

    dbn = dex.simple1_DDBN_conin().pgm
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(
        dbn, FactorOracleConstraint(func=lambda s: True, nodes=scope)
    )
    assignment = {(n, t): 0 for n in ("A", "B") for t in range(3)}
    assert cmodel.is_feasible(assignment)
    assert seen == [_dbn_fields(2)]


def test_network_flow_data_has_the_hmm_time_fields_and_keeps_app_data():
    hmm = hex_.create_hmm1()
    app = munch.Munch(limit=3, hmm="replaced")
    cmodel = ConstrainedHiddenMarkovModel(hmm, data=app)
    _, data = cmodel.representation("pyomo_network_flow").create_model(
        observed=["o0", "o1", "o0"]
    )
    assert (data.start, data.stop, data.limit) == (0, 2, 3)
    assert data.hmm.T == [0, 1, 2]
    assert data.hmm.hidden_to_internal == hmm.hidden_to_internal
    # The application data is not changed.
    assert app == munch.Munch(limit=3, hmm="replaced")


def test_network_flow_data_accepts_non_dict_app_data():
    class AppData:
        limit = 3

    cmodel = ConstrainedHiddenMarkovModel(hex_.create_hmm1(), data=AppData())
    _, data = cmodel.representation("pyomo_network_flow").create_model(
        observed=["o0", "o1"]
    )
    assert data.limit == 3 and data.hmm.T == [0, 1] and data.stop == 1
