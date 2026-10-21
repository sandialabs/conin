"""Tests for ConstrainedGraphicalModel and the representation registry."""

import copy
import pickle

import pytest

from conin import pyomo_constraint_fn, toulbar2_constraint_fn
from conin.constrained_models import (
    ConstrainedGraphicalModel,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constraints import OracleConstraint
from conin.exceptions import InvalidInputError
from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
from conin.representations import (
    Representation,
    register_representation,
    unregister_representation,
    supported_representations,
)
import conin.examples.hidden_markov_model as tc


def _mn():
    return DiscreteMarkovNetwork(
        states={"A": [0, 1]}, factors=[DiscreteFactor(nodes=["A"], values=[1, 2])]
    )


@pyomo_constraint_fn()
def pyomo_c(model):
    pass


@toulbar2_constraint_fn()
def tb2_c(model):
    pass


class _Counting(Representation):
    """Test representation that counts its builds."""

    builds = 0

    def __init__(self, cmodel, **params):
        type(self).builds += 1
        self.params = params
        self.payload = {"values": [1, 2, 3]}


@pytest.fixture
def counting_kind():
    _Counting.builds = 0
    families = (None, "pyomo")

    @register_representation("_test_counting", DiscreteMarkovNetwork, families)
    def _build(cmodel, **params):
        return _Counting(cmodel, **params)

    yield "_test_counting"
    unregister_representation("_test_counting", DiscreteMarkovNetwork, families)


# --- construction and validation ---------------------------------------------


def test_properties():
    mn = _mn()
    data = {"x": 1}
    c = ConstrainedDiscreteMarkovNetwork(mn, [pyomo_c], data=data)
    assert c.model is mn
    assert c.constraints == (pyomo_c,)
    assert c.constraint_family == "pyomo"
    assert c.data is data


def test_empty_constraints_family_none():
    c = ConstrainedDiscreteMarkovNetwork(_mn())
    assert c.constraints == ()
    assert c.constraint_family is None
    assert c.data is None


def test_generator_constraints_accepted():
    c = ConstrainedDiscreteMarkovNetwork(_mn(), (x for x in [pyomo_c]))
    assert c.constraints == (pyomo_c,)


def test_single_constraint_accepted():
    c = ConstrainedDiscreteMarkovNetwork(_mn(), pyomo_c)
    assert c.constraints == (pyomo_c,)
    assert c.constraint_family == "pyomo"
    assert c.with_constraints(pyomo_c).constraints == (pyomo_c,)


def test_model_keyword():
    mn = _mn()
    assert ConstrainedDiscreteMarkovNetwork(model=mn).model is mn


def test_wrong_model_type_rejected():
    with pytest.raises(InvalidInputError, match="expects a HiddenMarkovModel"):
        ConstrainedHiddenMarkovModel(_mn())


def test_hmm_matvec_repn_rejected():
    with pytest.raises(InvalidInputError):
        ConstrainedHiddenMarkovModel(tc.create_hmm1().repn)


def test_non_constraint_rejected():
    with pytest.raises(InvalidInputError, match="constraint 1 is not"):
        ConstrainedDiscreteMarkovNetwork(_mn(), [pyomo_c, object()])


def test_mixed_families_rejected_and_named():
    with pytest.raises(InvalidInputError, match="one constraint family") as exc:
        ConstrainedDiscreteMarkovNetwork(_mn(), [pyomo_c, tb2_c])
    assert "pyomo" in str(exc.value) and "toulbar2" in str(exc.value)


def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ConstrainedGraphicalModel(_mn())


def test_repr():
    r = repr(ConstrainedDiscreteMarkovNetwork(_mn(), [pyomo_c]))
    assert "ConstrainedDiscreteMarkovNetwork" in r and "'pyomo'" in r


# --- immutability -------------------------------------------------------------


@pytest.mark.parametrize("name", ["model", "constraints", "data", "_data", "new"])
def test_setattr_raises(name):
    c = ConstrainedDiscreteMarkovNetwork(_mn())
    with pytest.raises(AttributeError):
        setattr(c, name, 1)


@pytest.mark.parametrize("name", ["model", "_model", "_data"])
def test_delattr_raises(name):
    c = ConstrainedDiscreteMarkovNetwork(_mn())
    with pytest.raises(AttributeError):
        delattr(c, name)
    assert c.model is not None


def test_with_constraints_and_with_model(counting_kind):
    mn, mn2 = _mn(), _mn()
    data = {"x": 1}
    c = ConstrainedDiscreteMarkovNetwork(mn, [pyomo_c], data=data)
    c2 = c.with_constraints([])
    assert type(c2) is ConstrainedDiscreteMarkovNetwork
    assert c2.constraints == () and c2.model is mn and c2.data is data
    c3 = c.with_model(mn2)
    assert c3.model is mn2 and c3.constraints == (pyomo_c,) and c3.data is data

    # original unchanged
    assert c.constraints == (pyomo_c,) and c.model is mn


def test_ensure():
    mn = _mn()
    c = ConstrainedDiscreteMarkovNetwork(mn, [pyomo_c])
    # an instance of the class is returned as is (constraints and data kept)
    assert ConstrainedDiscreteMarkovNetwork.ensure(c) is c
    # a bare model is wrapped with no constraints
    w = ConstrainedDiscreteMarkovNetwork.ensure(mn)
    assert type(w) is ConstrainedDiscreteMarkovNetwork
    assert w.model is mn and w.constraints == () and w.data is None
    # anything else is rejected by the constructor
    with pytest.raises(InvalidInputError, match="expects a DiscreteMarkovNetwork"):
        ConstrainedDiscreteMarkovNetwork.ensure(tc.create_hmm1())
    with pytest.raises(InvalidInputError, match="expects a HiddenMarkovModel"):
        ConstrainedHiddenMarkovModel.ensure(c)
    with pytest.raises(TypeError, match="does not define model_type"):
        ConstrainedGraphicalModel.ensure(mn)


# --- registry and building -----------------------------------------------------


def test_lazy_build(counting_kind):
    ConstrainedDiscreteMarkovNetwork(_mn(), [pyomo_c])
    assert _Counting.builds == 0


def test_every_call_builds_a_new_representation(counting_kind):
    c = ConstrainedDiscreteMarkovNetwork(_mn())
    r1 = c.representation(counting_kind, T=3)
    r2 = c.representation(counting_kind, T=3)
    assert _Counting.builds == 2
    assert r1 is not r2
    assert r1.params == {"T": 3}
    # structure params may be unhashable (nothing is used as a cache key)
    r3 = c.representation(counting_kind, T=[1, 2], opts={"a": 1})
    assert r3.params == {"T": [1, 2], "opts": {"a": 1}}
    # the caller owns the result
    r1.payload["values"].append(99)
    assert r2.payload == {"values": [1, 2, 3]}


def test_unsupported_kind_lists_options(counting_kind):
    c = ConstrainedDiscreteMarkovNetwork(_mn(), [tb2_c])
    with pytest.raises(InvalidInputError) as exc:
        c.representation(counting_kind)
    msg = str(exc.value)
    assert "'toulbar2'" in msg
    assert "('_test_counting', None)" in msg and "('_test_counting', 'pyomo')" in msg


def test_supported_representations_hmm():
    from conin.graphical_models import HiddenMarkovModel

    pairs = supported_representations(HiddenMarkovModel)
    for pair in [
        ("factor", None),
        ("factor", "algebraic"),
        ("factor", "factor_oracle"),
        ("oracle", None),
        ("oracle", "oracle"),
        ("recursive", None),
        ("recursive", "recursive"),
        ("pyomo", None),
        ("pyomo", "algebraic"),
        ("pyomo", "pyomo"),
        ("pyomo_network_flow", None),
        ("pyomo_network_flow", "algebraic"),
        ("pyomo_network_flow", "pyomo"),
        ("toulbar2", None),
        ("toulbar2", "algebraic"),
        ("toulbar2", "toulbar2"),
        ("mvr", None),
        ("mvr", "mvr"),
    ]:
        assert pair in pairs
    # "oracle" is not registered for "algebraic" (its compilation needs the
    # horizon, which the oracle representation does not have).
    assert ("oracle", "algebraic") not in pairs


def test_duplicate_registration_rejected(counting_kind):
    with pytest.raises(ValueError, match="already registered"):

        @register_representation(counting_kind, DiscreteMarkovNetwork, (None,))
        def _other(cmodel):
            return None


# --- copy / deepcopy / pickle --------------------------------------------------


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_copy_and_deepcopy(copier, counting_kind):
    c = ConstrainedDiscreteMarkovNetwork(_mn(), [pyomo_c], data={"x": 1})
    c.representation(counting_kind)
    c2 = copier(c)
    assert type(c2) is type(c)
    assert c2.constraint_family == "pyomo"
    assert len(c2.constraints) == 1
    assert c2.data == {"x": 1}
    with pytest.raises(AttributeError):
        c2.model = None
    c2.representation(counting_kind)
    assert _Counting.builds == 2


def test_deepcopy_constrained_hmm_keeps_behaviour():
    chmm = tc.create_chmm1_oracle()
    c2 = copy.deepcopy(chmm)
    seq = ["h0"] * 11
    assert c2.is_feasible(seq) and chmm.is_feasible(seq)
    assert c2.model is not chmm.model


def test_pickle_roundtrip():
    c = ConstrainedDiscreteMarkovNetwork(_mn(), data={"x": 1})
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.data == {"x": 1} and c2.constraints == ()


# --- HMM helpers delegate to the "oracle" representation ------------------------


def test_hmm_helpers_require_oracle_family():
    hmm = tc.create_hmm1()
    oc = OracleConstraint(func=lambda s: True)
    assert ConstrainedHiddenMarkovModel(hmm, [oc]).generate_hidden(3)
    with pytest.raises(InvalidInputError):
        ConstrainedHiddenMarkovModel(hmm, [pyomo_c]).generate_hidden(3)
