"""Tests for ConstrainedGraphicalModel and the representation registry."""

import copy
import pickle

import pytest

from conin.algorithms.feasibility import is_feasible
from conin import pyomo_constraint_fn, toulbar2_constraint_fn
from conin.constrained_models import ConstrainedGraphicalModel
from conin.constraints import OracleConstraint
from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteFactor,
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    GraphicalModel,
    HiddenMarkovModel,
)
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
    c = ConstrainedGraphicalModel(mn, [pyomo_c], data=data)
    assert c.model is mn
    assert c.constraints == (pyomo_c,)
    assert c.constraint_family == "pyomo"
    assert c.data is data


def test_empty_constraints_family_none():
    c = ConstrainedGraphicalModel(_mn())
    assert c.constraints == ()
    assert c.constraint_family is None
    assert c.data is None


def test_generator_constraints_accepted():
    c = ConstrainedGraphicalModel(_mn(), (x for x in [pyomo_c]))
    assert c.constraints == (pyomo_c,)


def test_single_constraint_accepted():
    c = ConstrainedGraphicalModel(_mn(), pyomo_c)
    assert c.constraints == (pyomo_c,)
    assert c.constraint_family == "pyomo"
    assert c.with_constraints(pyomo_c).constraints == (pyomo_c,)


def test_model_keyword():
    mn = _mn()
    assert ConstrainedGraphicalModel(model=mn).model is mn


def test_non_model_rejected():
    with pytest.raises(TypeError, match="expects a conin graphical model"):
        ConstrainedGraphicalModel({"A": [0, 1]})


def _bn():
    return conin.examples.bayesian_network.simple1_BN_conin().pgm


def _dbn():
    return conin.examples.dynamic_bayesian_network.simple1_DDBN_conin().pgm


MODELS = [
    pytest.param(_bn, DiscreteBayesianNetwork, id="bn"),
    pytest.param(_mn, DiscreteMarkovNetwork, id="mn"),
    pytest.param(_dbn, DynamicDiscreteBayesianNetwork, id="dbn"),
]


@pytest.mark.parametrize("make,model_type", MODELS)
def test_nodes_and_states_of_delegate_to_the_model(make, model_type):
    model = make()
    cmodel = ConstrainedGraphicalModel(model)
    assert cmodel.model is model
    assert cmodel.model_type is model_type
    # Regression test: ``nodes`` used to be a method on the DBN class.
    assert isinstance(ConstrainedGraphicalModel.__dict__["nodes"], property)
    assert cmodel.nodes == model.nodes
    for node in model.nodes:
        assert cmodel.states_of(node) == model.states_of(node)


def test_dbn_states_of_accepts_timed_nodes():
    dbn = _dbn()
    cdbn = ConstrainedGraphicalModel(dbn)
    for node in dbn.dynamic_nodes:
        assert cdbn.states_of((node, 0)) == dbn.states_of((node, 0))


def test_string_constraints_rejected():
    with pytest.raises(InvalidInputError):
        ConstrainedGraphicalModel(_mn(), constraints="not-a-list")


def test_hmm_matvec_repn_rejected():
    hmm = tc.create_hmm1()
    repn = ConstrainedGraphicalModel(hmm).representation("matvec")
    with pytest.raises(TypeError):
        ConstrainedGraphicalModel(repn)


def test_non_constraint_rejected():
    with pytest.raises(InvalidInputError, match="constraint 1 is not"):
        ConstrainedGraphicalModel(_mn(), [pyomo_c, object()])


def test_families_without_a_common_representation_are_rejected_and_named():
    with pytest.raises(
        InvalidInputError,
        match=r"no representation of DiscreteMarkovNetwork accepts the "
        r"constraint families \['pyomo', 'toulbar2'\] together",
    ) as exc:
        ConstrainedGraphicalModel(_mn(), [pyomo_c, tb2_c])
    # The message lists the combinations that work.
    assert "'pyomo': ['pyomo', 'smoek']" in str(exc.value)
    assert "'toulbar2': ['smoek', 'toulbar2']" in str(exc.value)


def test_families_with_a_common_representation_can_be_combined():
    smoek = pytest.importorskip("smoek")  # noqa: F841
    from conin import smoek_constraint_fn

    @smoek_constraint_fn()
    def smoek_c(model, data):
        return model.V("A", 1) >= 0

    c = ConstrainedGraphicalModel(_mn(), [pyomo_c, smoek_c, pyomo_c])
    assert c.constraint_families == ("pyomo", "smoek")
    assert c.constraint_family is None
    assert "constraint_families=['pyomo', 'smoek']" in repr(c)
    c.representation("pyomo")
    with pytest.raises(
        InvalidInputError,
        match=r"No 'toulbar2' representation is available for "
        r"DiscreteMarkovNetwork with constraint families \['pyomo', 'smoek'\]",
    ):
        c.representation("toulbar2")


@pytest.mark.parametrize(
    "make, expected",
    [
        (_mn, ["factor_oracle", "pyomo", "smoek", "toulbar2"]),
        (_bn, ["factor_oracle", "pyomo", "smoek", "toulbar2"]),
        (_dbn, ["factor_oracle", "pyomo", "smoek", "toulbar2"]),
        (
            tc.create_hmm1,
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
def test_accepted_families_come_from_the_registry(make, expected):
    cmodel = ConstrainedGraphicalModel(make())
    assert cmodel._supported_families() == expected
    assert expected == sorted(
        {f for _, f in supported_representations(cmodel.model_type) if f is not None}
    )


def test_family_without_a_builder_is_rejected_and_named():
    with pytest.raises(InvalidInputError) as exc:
        ConstrainedGraphicalModel(_mn(), OracleConstraint(func=lambda s: True))
    msg = str(exc.value)
    assert "does not support constraint family 'oracle'" in msg
    assert "'factor_oracle'" in msg and "'pyomo'" in msg


def test_registering_a_builder_makes_a_family_accepted():
    oracle_c = OracleConstraint(func=lambda s: True)
    with pytest.raises(InvalidInputError):
        ConstrainedGraphicalModel(_mn(), oracle_c)

    @register_representation("_test_oracle_mn", DiscreteMarkovNetwork, ("oracle",))
    def _build(cmodel):
        return _Counting(cmodel)

    try:
        c = ConstrainedGraphicalModel(_mn(), oracle_c)
        assert c.constraint_family == "oracle"
        assert isinstance(c.representation("_test_oracle_mn"), _Counting)
    finally:
        unregister_representation("_test_oracle_mn", DiscreteMarkovNetwork, ("oracle",))
    with pytest.raises(InvalidInputError):
        ConstrainedGraphicalModel(_mn(), oracle_c)


def test_no_supported_families_attribute():
    assert not hasattr(ConstrainedGraphicalModel, "supported_families")


def test_repr():
    r = repr(ConstrainedGraphicalModel(_mn(), [pyomo_c]))
    assert "ConstrainedGraphicalModel" in r and "'pyomo'" in r


# --- immutability -------------------------------------------------------------


@pytest.mark.parametrize("name", ["model", "constraints", "data", "_data", "new"])
def test_setattr_raises(name):
    c = ConstrainedGraphicalModel(_mn())
    with pytest.raises(AttributeError):
        setattr(c, name, 1)


@pytest.mark.parametrize("name", ["model", "_model", "_data"])
def test_delattr_raises(name):
    c = ConstrainedGraphicalModel(_mn())
    with pytest.raises(AttributeError):
        delattr(c, name)
    assert c.model is not None


def test_with_constraints_and_with_model(counting_kind):
    mn, mn2 = _mn(), _mn()
    data = {"x": 1}
    c = ConstrainedGraphicalModel(mn, [pyomo_c], data=data)
    c2 = c.with_constraints([])
    assert type(c2) is ConstrainedGraphicalModel
    assert c2.constraints == () and c2.model is mn and c2.data is data
    c3 = c.with_model(mn2)
    assert c3.model is mn2 and c3.constraints == (pyomo_c,) and c3.data is data

    # original unchanged
    assert c.constraints == (pyomo_c,) and c.model is mn


def test_ensure():
    mn = _mn()
    c = ConstrainedGraphicalModel(mn, [pyomo_c])
    # an instance of the class is returned as is (constraints and data kept)
    assert ConstrainedGraphicalModel.ensure(c) is c
    # a bare model is wrapped with no constraints
    w = ConstrainedGraphicalModel.ensure(mn)
    assert type(w) is ConstrainedGraphicalModel
    assert w.model is mn and w.constraints == () and w.data is None
    # anything else is rejected by the constructor
    with pytest.raises(TypeError, match="expects a conin graphical model"):
        ConstrainedGraphicalModel.ensure(object())
    # model_type restricts the wrapped model
    hmm = tc.create_hmm1()
    assert ConstrainedGraphicalModel.ensure(hmm, HiddenMarkovModel).model is hmm
    assert ConstrainedGraphicalModel.ensure(c, DiscreteMarkovNetwork) is c
    with pytest.raises(TypeError, match="Expected a HiddenMarkovModel .* got Disc"):
        ConstrainedGraphicalModel.ensure(mn, HiddenMarkovModel)
    with pytest.raises(
        TypeError,
        match=r"got ConstrainedGraphicalModel\(DiscreteMarkovNetwork\)",
    ):
        ConstrainedGraphicalModel.ensure(c, HiddenMarkovModel)


# --- registry and building -----------------------------------------------------


def test_lazy_build(counting_kind):
    ConstrainedGraphicalModel(_mn(), [pyomo_c])
    assert _Counting.builds == 0


def test_every_call_builds_a_new_representation(counting_kind):
    c = ConstrainedGraphicalModel(_mn())
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
    c = ConstrainedGraphicalModel(_mn(), [tb2_c])
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
    assert type(repn.cmodel) is ConstrainedGraphicalModel
    assert repn.cmodel.model is mn and repn.cmodel.constraints == ()
    assert repn.params == {"T": 3}
    assert mn.representation(counting_kind) is not repn


def test_bare_hmm_matvec_matches_the_constrained_one():
    hmm = tc.create_hmm1()
    bare = hmm.representation("matvec")
    wrapped = ConstrainedGraphicalModel(hmm).representation("matvec")
    assert bare.model is hmm
    for name in bare._ARRAYS:
        assert (getattr(bare, name) == getattr(wrapped, name)).all()


def test_bare_model_unsupported_kind_lists_options():
    with pytest.raises(InvalidInputError, match="No 'matvec' representation"):
        _mn().representation("matvec")


class _SubHMM(HiddenMarkovModel):
    pass


def test_model_subclass_uses_the_builders_of_its_base():
    hmm = tc.create_hmm1()
    sub = _SubHMM(
        start_probs=hmm.start_probs,
        transition_probs=hmm.transition_probs,
        emission_probs=hmm.emission_probs,
    )
    cmodel = ConstrainedGraphicalModel.ensure(sub)
    assert cmodel.model is sub and cmodel.model_type is HiddenMarkovModel
    assert sub.representation("matvec").model is sub


class _Unregistered(GraphicalModel):
    pass


def test_model_class_without_builders_raises_type_error():
    with pytest.raises(TypeError, match="no representations .* _Unregistered"):
        _Unregistered().representation("matvec")
    with pytest.raises(TypeError, match="no representations .* _Unregistered"):
        ConstrainedGraphicalModel(_Unregistered())


def test_registering_a_builder_supports_a_new_model_type():
    class _Model(GraphicalModel):
        pass

    @register_representation("_test_new_model", _Model)
    def _build(cmodel):
        return _Counting(cmodel)

    try:
        cmodel = ConstrainedGraphicalModel(_Model())
        assert cmodel.model_type is _Model
        assert isinstance(_Model().representation("_test_new_model"), _Counting)
        with pytest.raises(InvalidInputError, match="Supported .*'_test_new_model'"):
            _Model().representation("matvec")
    finally:
        unregister_representation("_test_new_model", _Model)
    with pytest.raises(TypeError, match="no representations"):
        ConstrainedGraphicalModel(_Model())


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
    c = ConstrainedGraphicalModel(_mn(), [pyomo_c], data={"x": 1})
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
    c = ConstrainedGraphicalModel(_mn(), data={"x": 1})
    c2 = pickle.loads(pickle.dumps(c))
    assert c2.data == {"x": 1} and c2.constraints == ()
    assert c2.model_type is DiscreteMarkovNetwork


# --- HMM rejection sampling uses the "oracle" representation --------------------


def test_hmm_rejection_sampling_requires_oracle_family():
    hmm = tc.create_hmm1()
    oc = OracleConstraint(func=lambda s: True)
    assert rejection_sample_hidden(ConstrainedGraphicalModel(hmm, oc), 3)
    with pytest.raises(InvalidInputError):
        rejection_sample_hidden(ConstrainedGraphicalModel(hmm, pyomo_c), 3)
    # The sampling methods moved to conin.algorithms.sampling.
    for name in (
        "generate_hidden",
        "generate_observed",
        "generate_observed_from_hidden",
    ):
        assert not hasattr(ConstrainedGraphicalModel, name)
