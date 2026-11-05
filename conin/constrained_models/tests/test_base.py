"""Tests for ConstrainedGraphicalModel and the representation registry."""

import copy
import pickle

import pytest

from conin.algorithms.feasibility import is_feasible
from conin import pyomo_constraint_fn, toulbar2_constraint_fn
from conin.constrained_models import (
    ConstrainedGraphicalModel,
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constraints import OracleConstraint
from conin.exceptions import InvalidInputError
from conin.constrained_models.normalise import as_constrained_model
from conin.graphical_models import (
    DiscreteFactor,
    DiscreteMarkovNetwork,
    GraphicalModel,
    HiddenMarkovModel,
)
from conin.graphical_models import base as gm_base
from conin.representations import (
    Representation,
    register_representation,
    unregister_representation,
    supported_representations,
)
from conin.algorithms.sampling.oracle import rejection_sample_hidden
import conin.examples.bayesian_network
import conin.examples.dynamic_bayesian_network
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
        self.cmodel = cmodel
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
    with pytest.raises(TypeError, match="expects a HiddenMarkovModel"):
        ConstrainedHiddenMarkovModel(_mn())


def _bn():
    return conin.examples.bayesian_network.simple1_BN_conin().pgm


def _dbn():
    return conin.examples.dynamic_bayesian_network.simple1_DDBN_conin().pgm


DELEGATING = [
    pytest.param(ConstrainedDiscreteBayesianNetwork, _bn, _mn, id="bn"),
    pytest.param(ConstrainedDiscreteMarkovNetwork, _mn, _bn, id="mn"),
    pytest.param(ConstrainedDynamicDiscreteBayesianNetwork, _dbn, _bn, id="dbn"),
]


@pytest.mark.parametrize("cls,make,make_other", DELEGATING)
def test_nodes_and_states_of_delegate_to_the_model(cls, make, make_other):
    model = make()
    cmodel = cls(model)
    assert cmodel.model is model
    # Regression test: ``nodes`` used to be a method on the DBN class.
    assert isinstance(cls.__dict__["nodes"], property)
    assert cmodel.nodes == model.nodes
    for node in model.nodes:
        assert cmodel.states_of(node) == model.states_of(node)
    with pytest.raises(TypeError, match="expects a"):
        cls(make_other())


def test_dbn_states_of_accepts_timed_nodes():
    dbn = _dbn()
    cdbn = ConstrainedDynamicDiscreteBayesianNetwork(dbn)
    for node in dbn.dynamic_nodes:
        assert cdbn.states_of((node, 0)) == dbn.states_of((node, 0))


def test_string_constraints_rejected():
    with pytest.raises(InvalidInputError):
        ConstrainedDiscreteMarkovNetwork(_mn(), constraints="not-a-list")


def test_hmm_matvec_repn_rejected():
    hmm = tc.create_hmm1()
    repn = ConstrainedHiddenMarkovModel(hmm).representation("matvec")
    with pytest.raises(TypeError):
        ConstrainedHiddenMarkovModel(repn)


def test_non_constraint_rejected():
    with pytest.raises(InvalidInputError, match="constraint 1 is not"):
        ConstrainedDiscreteMarkovNetwork(_mn(), [pyomo_c, object()])


def test_mixed_families_rejected_and_named():
    with pytest.raises(InvalidInputError, match="one constraint family") as exc:
        ConstrainedDiscreteMarkovNetwork(_mn(), [pyomo_c, tb2_c])
    assert "pyomo" in str(exc.value) and "toulbar2" in str(exc.value)


@pytest.mark.parametrize(
    "cls, expected",
    [
        (
            ConstrainedDiscreteMarkovNetwork,
            ["factor_oracle", "pyomo", "smoek", "toulbar2"],
        ),
        (
            ConstrainedDiscreteBayesianNetwork,
            ["factor_oracle", "pyomo", "smoek", "toulbar2"],
        ),
        (
            ConstrainedDynamicDiscreteBayesianNetwork,
            ["factor_oracle", "pyomo", "smoek", "toulbar2"],
        ),
        (
            ConstrainedHiddenMarkovModel,
            [
                "factor_oracle",
                "mvr",
                "oracle",
                "pyomo",
                "recursive",
                "smoek",
                "toulbar2",
            ],
        ),
    ],
)
def test_accepted_families_come_from_the_registry(cls, expected):
    assert cls._supported_families() == expected
    assert expected == sorted(
        {f for _, f in supported_representations(cls.model_type) if f is not None}
    )


def test_family_without_a_builder_is_rejected_and_named():
    with pytest.raises(InvalidInputError) as exc:
        ConstrainedDiscreteMarkovNetwork(_mn(), OracleConstraint(func=lambda s: True))
    msg = str(exc.value)
    assert "does not support constraint family 'oracle'" in msg
    assert "'factor_oracle'" in msg and "'pyomo'" in msg


def test_registering_a_builder_makes_a_family_accepted():
    oracle_c = OracleConstraint(func=lambda s: True)
    with pytest.raises(InvalidInputError):
        ConstrainedDiscreteMarkovNetwork(_mn(), oracle_c)

    @register_representation("_test_oracle_mn", DiscreteMarkovNetwork, ("oracle",))
    def _build(cmodel):
        return _Counting(cmodel)

    try:
        c = ConstrainedDiscreteMarkovNetwork(_mn(), oracle_c)
        assert c.constraint_family == "oracle"
        assert isinstance(c.representation("_test_oracle_mn"), _Counting)
    finally:
        unregister_representation("_test_oracle_mn", DiscreteMarkovNetwork, ("oracle",))
    with pytest.raises(InvalidInputError):
        ConstrainedDiscreteMarkovNetwork(_mn(), oracle_c)


def test_no_supported_families_attribute():
    assert not hasattr(ConstrainedGraphicalModel, "supported_families")
    assert not hasattr(ConstrainedHiddenMarkovModel, "supported_families")


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
    with pytest.raises(TypeError, match="expects a DiscreteMarkovNetwork"):
        ConstrainedDiscreteMarkovNetwork.ensure(tc.create_hmm1())
    with pytest.raises(TypeError, match="expects a HiddenMarkovModel"):
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


# --- representations of bare models -------------------------------------------


def test_bare_model_representation_wraps_without_constraints(counting_kind):
    mn = _mn()
    repn = mn.representation(counting_kind, T=3)
    assert _Counting.builds == 1
    assert type(repn.cmodel) is ConstrainedDiscreteMarkovNetwork
    assert repn.cmodel.model is mn and repn.cmodel.constraints == ()
    assert repn.params == {"T": 3}
    assert mn.representation(counting_kind) is not repn


def test_bare_hmm_matvec_matches_the_constrained_one():
    hmm = tc.create_hmm1()
    bare = hmm.representation("matvec")
    wrapped = ConstrainedHiddenMarkovModel(hmm).representation("matvec")
    assert bare.model is hmm
    for name in bare._ARRAYS:
        assert (getattr(bare, name) == getattr(wrapped, name)).all()


def test_bare_model_unsupported_kind_lists_options():
    with pytest.raises(InvalidInputError, match="No 'matvec' representation"):
        _mn().representation("matvec")


def test_constrained_classes_are_registered_by_model_type():
    assert gm_base.constrained_classes() == [
        ConstrainedDiscreteMarkovNetwork,
        ConstrainedDiscreteBayesianNetwork,
        ConstrainedDynamicDiscreteBayesianNetwork,
        ConstrainedHiddenMarkovModel,
    ]
    for cls in gm_base.constrained_classes():
        assert gm_base.constrained_class(cls.model_type) is cls


class _SubHMM(HiddenMarkovModel):
    pass


def test_model_subclass_uses_the_constrained_class_of_its_base():
    hmm = tc.create_hmm1()
    sub = _SubHMM(
        start_probs=hmm.start_probs,
        transition_probs=hmm.transition_probs,
        emission_probs=hmm.emission_probs,
    )
    assert gm_base.constrained_class(_SubHMM) is ConstrainedHiddenMarkovModel
    assert type(as_constrained_model(sub)) is ConstrainedHiddenMarkovModel
    assert sub.representation("matvec").model is sub


class _Unregistered(GraphicalModel):
    pass


def test_unregistered_model_class_raises_type_error():
    assert gm_base.constrained_class(_Unregistered) is None
    with pytest.raises(TypeError, match="No constrained model class .* _Unregistered"):
        _Unregistered().representation("matvec")
    with pytest.raises(TypeError, match="Unsupported model type"):
        as_constrained_model(_Unregistered())


def test_constrained_subclass_without_model_type_is_not_registered():
    class _Sub(ConstrainedHiddenMarkovModel):
        pass

    assert gm_base.constrained_class(HiddenMarkovModel) is ConstrainedHiddenMarkovModel
    assert type(_Sub(tc.create_hmm1())) is _Sub


def test_second_constrained_class_for_a_model_type_is_rejected():
    with pytest.raises(TypeError, match="ConstrainedHiddenMarkovModel is already"):

        class _Other(ConstrainedGraphicalModel):
            model_type = HiddenMarkovModel

    assert gm_base.constrained_class(HiddenMarkovModel) is ConstrainedHiddenMarkovModel


def test_new_model_type_gets_representations():
    class _Model(GraphicalModel):
        pass

    try:

        class _Constrained(ConstrainedGraphicalModel):
            model_type = _Model

        assert gm_base.constrained_class(_Model) is _Constrained
        assert type(as_constrained_model(_Model())) is _Constrained
        with pytest.raises(InvalidInputError, match="Supported .* none"):
            _Model().representation("matvec")
    finally:
        gm_base._CONSTRAINED_CLASSES.pop(_Model, None)


def test_supported_representations_hmm():
    from conin.graphical_models import HiddenMarkovModel

    pairs = supported_representations(HiddenMarkovModel)
    for pair in [
        ("variable_elimination", None),
        ("variable_elimination", "smoek"),
        ("variable_elimination", "factor_oracle"),
        ("oracle", None),
        ("oracle", "oracle"),
        ("recursive", None),
        ("recursive", "recursive"),
        ("pyomo", None),
        ("pyomo", "smoek"),
        ("pyomo", "pyomo"),
        ("pyomo_network_flow", None),
        ("pyomo_network_flow", "smoek"),
        ("pyomo_network_flow", "pyomo"),
        ("toulbar2", None),
        ("toulbar2", "smoek"),
        ("toulbar2", "toulbar2"),
        ("mvr", None),
        ("mvr", "mvr"),
        ("matvec", None),
    ]:
        assert pair in pairs
    # "matvec" holds the HMM parameters only: no constraint family.
    assert [p for p in pairs if p[0] == "matvec"] == [("matvec", None)]
    # "oracle" is not registered for "smoek" (its compilation needs the
    # horizon, which the oracle representation does not have).
    assert ("oracle", "smoek") not in pairs


def test_duplicate_registration_rejected(counting_kind):
    with pytest.raises(InvalidInputError, match="already registered"):

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
    assert is_feasible(c2, seq) and is_feasible(chmm, seq)
    assert c2.model is not chmm.model


def test_pickle_roundtrip():
    c = ConstrainedDiscreteMarkovNetwork(_mn(), data={"x": 1})
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.data == {"x": 1} and c2.constraints == ()


# --- HMM rejection sampling uses the "oracle" representation --------------------


def test_hmm_rejection_sampling_requires_oracle_family():
    hmm = tc.create_hmm1()
    oc = OracleConstraint(func=lambda s: True)
    assert rejection_sample_hidden(ConstrainedHiddenMarkovModel(hmm, oc), 3)
    with pytest.raises(InvalidInputError):
        rejection_sample_hidden(ConstrainedHiddenMarkovModel(hmm, pyomo_c), 3)
    # The sampling methods moved to conin.algorithms.sampling.
    for name in (
        "generate_hidden",
        "generate_observed",
        "generate_observed_from_hidden",
    ):
        assert not hasattr(ConstrainedHiddenMarkovModel, name)
