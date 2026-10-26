"""Argument-validation and error paths of the MVR containers, constraint
factory, operator extension point and primitives."""

import copy
import importlib
import pickle

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.constraints.mvr.base import BaseMVR, HomMVR, InhomMVR, MVR_MatVecRepn
from conin.constraints.mvr.constraint import MVRConstraint, mvr_constraint_fn
from conin.constraints.mvr.operators import (
    MVROperator,
    mvr_and,
    mvr_concatenate,
    mvr_concatenate_prefix,
    mvr_count,
    mvr_kfold_product,
    mvr_kfold_product_prefix,
    mvr_kleene_closure,
    mvr_kleene_closure_prefix,
    mvr_not,
    mvr_not_yet,
    mvr_operator_fn,
    mvr_or,
    mvr_precedence,
    mvr_reverse,
    mvr_sattime,
    mvr_setdiff,
    mvr_timerange,
)
from conin.constraints.mvr.primitives import mvr_current_transition, mvr_regex
from conin.constraints.mvr.tests.test_mvr_constraints import make_hmm
from conin.constraints.mvr.tests.test_mvr_operators import (
    ALPHABET,
    contains_symbol_mvr,
    eval_mvr,
    orphan_state_mvr,
    trivial_inhom_mvr,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def hom_kwargs():
    """Valid HomMVR keyword arguments for "contains a" over ALPHABET."""
    m = [False, True]
    return dict(
        hidden_states=list(ALPHABET),
        mediation_states=list(m),
        ini={h: h == "a" for h in ALPHABET},
        upd={(s, h): s or h == "a" for s in m for h in ALPHABET},
        evl={False: False, True: True},
    )


def inhom_kwargs(time_horizon=2):
    """Valid InhomMVR keyword arguments for "contains a" over ALPHABET."""
    m = [False, True]
    return dict(
        hidden_states=list(ALPHABET),
        mediation_states=[list(m) for _ in range(time_horizon + 1)],
        ini={h: h == "a" for h in ALPHABET},
        upd=[
            {(s, h): s or h == "a" for s in m for h in ALPHABET}
            for _ in range(time_horizon)
        ],
        evl=[{False: False, True: True} for _ in range(time_horizon + 1)],
    )


class OpaqueMVR(BaseMVR):
    """A BaseMVR that is neither a HomMVR nor an InhomMVR."""

    def __init__(self):
        super().__init__()
        self.hidden_states = list(ALPHABET)


def valid_hom_arrays():
    ini = np.array([[1.0, 0.0], [0.0, 1.0]])
    upd = np.zeros((2, 2, 2))
    upd[:, 0, :] = 1.0
    evl = np.array([0.0, 1.0])
    return ini, upd, evl


def valid_inhom_arrays():
    ini, upd, evl = valid_hom_arrays()
    return ini, [upd, upd.copy()], [evl, evl.copy(), evl.copy()]


# ---------------------------------------------------------------------
# BaseMVR
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad_name", [1, b"x", ["x"]])
def test_name_setter_rejects_non_strings(bad_name):
    mvr = HomMVR(**hom_kwargs())

    with pytest.raises(InvalidInputError, match="name must be a string or None"):
        mvr.name = bad_name

    with pytest.raises(InvalidInputError, match=type(bad_name).__name__):
        HomMVR(**hom_kwargs(), name=bad_name)


def test_name_setter_accepts_none_and_strings():
    mvr = HomMVR(**hom_kwargs(), name="x")
    assert mvr.name == "x"
    mvr.name = None
    assert mvr.name is None


def test_base_mvr_hooks_must_be_implemented():
    mvr = OpaqueMVR()

    with pytest.raises(NotImplementedError, match="_build_array_repn must be"):
        mvr.repn

    with pytest.raises(NotImplementedError, match="prune must be implemented"):
        mvr.prune()


def test_repn_is_rebuilt_lazily_and_initialize_is_idempotent():
    mvr = HomMVR(**hom_kwargs())
    repn = mvr.repn

    assert mvr.initialize() is repn

    mvr._repn = None
    rebuilt = mvr.repn

    assert rebuilt is not repn
    assert np.array_equal(rebuilt.ini_array, repn.ini_array)
    assert mvr.repn is rebuilt


# ---------------------------------------------------------------------
# HomMVR validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda kw: kw.update(hidden_states=["a", "b", "a"]),
            "hidden_states must not contain duplicates",
        ),
        (
            lambda kw: kw.update(mediation_states=[False, True, True]),
            "mediation_states must not contain duplicates",
        ),
        (
            lambda kw: kw["ini"].pop("b"),
            r"domain\(keys\) of ini must match hidden_states",
        ),
        (
            lambda kw: kw["ini"].update(b="z"),
            r"range\(values\) of ini must be contained in mediation_states",
        ),
        (
            lambda kw: kw["upd"].pop((False, "a")),
            r"domain\(keys\) of upd must match mediation_states x hidden_states",
        ),
        (
            lambda kw: kw["upd"].update({(False, "a"): "z"}),
            r"range\(values\) of upd must be contained in mediation_states",
        ),
        (
            lambda kw: kw["evl"].pop(True),
            r"domain\(keys\) of evl must match mediation_states",
        ),
        (
            lambda kw: kw["evl"].update({True: 1}),
            r"range\(values\) of evl must be boolean",
        ),
    ],
)
def test_hom_mvr_rejects_invalid_input(mutate, match):
    kwargs = hom_kwargs()
    mutate(kwargs)

    with pytest.raises(InvalidInputError, match=match):
        HomMVR(**kwargs)


def test_hom_mvr_rejects_upd_with_extra_keys():
    kwargs = hom_kwargs()
    kwargs["upd"][("z", "a")] = False

    with pytest.raises(InvalidInputError, match=r"domain\(keys\) of upd must match"):
        HomMVR(**kwargs)


# ---------------------------------------------------------------------
# InhomMVR validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda kw: kw.update(hidden_states=["a", "a"]),
            "hidden_states must not contain duplicates",
        ),
        (
            lambda kw: kw.update(mediation_states=[], upd=[], evl=[]),
            "mediation_states must be nonempty",
        ),
        (
            lambda kw: kw["evl"].pop(),
            "evl and mediation_states must be the same length",
        ),
        (
            lambda kw: kw["upd"].pop(),
            "upd length 1 must be one less than",
        ),
        (
            lambda kw: kw["mediation_states"].__setitem__(1, [False, True, False]),
            "mediation_states at time 1 must not contain duplicates",
        ),
        (
            lambda kw: kw["ini"].pop("a"),
            r"domain\(keys\) of ini must match hidden_states",
        ),
        (
            lambda kw: kw["ini"].update(a="z"),
            r"range\(values\) of ini must be contained in mediation_states at time 0",
        ),
        (
            lambda kw: kw["upd"][1].pop((True, "b")),
            r"domain\(keys\) of upd at time 1 must match mediation_states x hidden_states",
        ),
        (
            lambda kw: kw["upd"][0].update({(True, "b"): "z"}),
            r"range\(values\) of upd at time 0 must be contained in mediation_states",
        ),
        (
            lambda kw: kw["evl"][2].pop(False),
            r"domain\(keys\) of evl at time 2 must match mediation_states",
        ),
        (
            lambda kw: kw["evl"][1].update({False: 0}),
            r"range\(values\) of evl at time 1 must be boolean",
        ),
    ],
)
def test_inhom_mvr_rejects_invalid_input(mutate, match):
    kwargs = inhom_kwargs(time_horizon=2)
    mutate(kwargs)

    with pytest.raises(InvalidInputError, match=match):
        InhomMVR(**kwargs)


def test_inhom_mvr_upd_values_must_lie_in_the_next_slice():
    # Valid against slice t but not against slice t + 1.
    kwargs = inhom_kwargs(time_horizon=1)
    kwargs["mediation_states"] = [[False, True, "extra"], [False, True]]
    kwargs["upd"] = [
        {(s, h): "extra" for s in [False, True, "extra"] for h in ALPHABET}
    ]
    kwargs["evl"] = [
        {False: False, True: True, "extra": False},
        {False: False, True: True},
    ]

    with pytest.raises(
        InvalidInputError,
        match=r"range\(values\) of upd at time 0 must be contained",
    ):
        InhomMVR(**kwargs)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda kw: kw.update(hidden_states=[], ini={}, upd={}),
            "hidden_states must be nonempty",
        ),
        (
            lambda kw: kw.update(mediation_states=[], upd={}, evl={}),
            "mediation_states must be nonempty",
        ),
    ],
)
def test_hom_mvr_rejects_empty_state_spaces(mutate, match):
    # These used to pass validation and fail in NumPy ("argmax of an empty
    # sequence") or build an MVR over no hidden states.
    kwargs = hom_kwargs()
    mutate(kwargs)

    with pytest.raises(InvalidInputError, match=match):
        HomMVR(**kwargs)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda kw: kw.update(hidden_states=[], ini={}, upd=[{}, {}]),
            "hidden_states must be nonempty",
        ),
        (
            lambda kw: (
                kw["mediation_states"].__setitem__(1, []),
                kw["evl"].__setitem__(1, {}),
            ),
            "mediation_states at time 1 must be nonempty",
        ),
    ],
)
def test_inhom_mvr_rejects_empty_state_spaces(mutate, match):
    kwargs = inhom_kwargs(time_horizon=2)
    mutate(kwargs)

    with pytest.raises(InvalidInputError, match=match):
        InhomMVR(**kwargs)


def test_inhom_mvr_with_horizon_zero_is_valid():
    kwargs = inhom_kwargs(time_horizon=0)
    mvr = InhomMVR(**kwargs)

    assert mvr.time_horizon == 0
    assert mvr.repn.upd_array == []
    assert mvr.repn.num_mediation_states == [2]
    assert eval_mvr(mvr, ["a"]) is True
    assert eval_mvr(mvr, ["b"]) is False


# ---------------------------------------------------------------------
# MVR_MatVecRepn
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "ini,match",
    [
        (np.array([1.0, 0.0]), "ini_array must be a 2D array"),
        (np.array([[1.0, 0.0], [-1.0, 2.0]]), "ini_array entries must be nonnegative"),
        (np.array([[0.5, 0.5], [0.0, 1.0]]), "ini_array entries must be binary"),
        (np.array([[0.0, 0.0], [0.0, 1.0]]), "ini_array rows must sum to 1"),
    ],
)
def test_matvec_repn_rejects_bad_ini_array(ini, match):
    _, upd, evl = valid_hom_arrays()

    with pytest.raises(InvalidInputError, match=match):
        MVR_MatVecRepn(ini_array=ini, upd_array=upd, evl_array=evl)


def _bad_upd(kind):
    _, upd, _ = valid_hom_arrays()
    if kind == "rank":
        return upd[0]
    if kind == "negative":
        upd = upd.copy()
        upd[0, 1, 0] = -1.0
        return upd
    if kind == "binary":
        upd = upd.copy()
        upd[0, :, 0] = 0.5
        return upd
    raise ValueError(kind)


@pytest.mark.parametrize(
    "kind,match",
    [
        ("rank", "upd_array at index 0 must be a 3D array"),
        ("negative", "upd_array at index 0 entries must be nonnegative"),
        ("binary", "upd_array at index 0 entries must be binary"),
    ],
)
def test_matvec_repn_rejects_bad_hom_upd_array(kind, match):
    ini, _, evl = valid_hom_arrays()

    with pytest.raises(InvalidInputError, match=match):
        MVR_MatVecRepn(ini_array=ini, upd_array=_bad_upd(kind), evl_array=evl)


def test_matvec_repn_reports_the_index_of_a_bad_inhom_upd_array():
    ini, upd, evl = valid_inhom_arrays()
    upd[1] = upd[1].copy()
    upd[1][:] = 0.0

    with pytest.raises(
        InvalidInputError,
        match="upd_array at index 1 must sum to 1 over the current mediation axis",
    ):
        MVR_MatVecRepn(ini_array=ini, upd_array=upd, evl_array=evl)


@pytest.mark.parametrize(
    "bad_evl,match",
    [
        (np.array([-1.0, 1.0]), "evl_array at index 2 entries must be nonnegative"),
        (np.array([0.5, 1.0]), "evl_array at index 2 entries must be binary"),
        (np.array([[0.0, 1.0]]), "evl_array at index 2 must be a 1D array"),
    ],
)
def test_matvec_repn_rejects_bad_inhom_evl_array(bad_evl, match):
    ini, upd, evl = valid_inhom_arrays()
    evl[2] = bad_evl

    with pytest.raises(InvalidInputError, match=match):
        MVR_MatVecRepn(ini_array=ini, upd_array=upd, evl_array=evl)


def test_matvec_repn_rejects_negative_hom_evl_array():
    ini, upd, _ = valid_hom_arrays()

    with pytest.raises(
        InvalidInputError, match="evl_array at index 0 entries must be nonnegative"
    ):
        MVR_MatVecRepn(ini_array=ini, upd_array=upd, evl_array=np.array([-1.0, 1.0]))


@pytest.mark.parametrize(
    "case,match",
    [
        ("evl_list_upd_array", "must either both be lists or both be arrays"),
        ("upd_list_evl_array", "must either both be lists or both be arrays"),
        ("hom_hidden", "upd_array hidden dimension must match ini_array"),
        ("hom_evl", "evl_array mediation dimension must match ini_array"),
        ("hom_mediation", "homogeneous upd_array mediation dimensions must both"),
        ("inhom_empty", "evl_array list must be nonempty"),
        ("inhom_length", "upd_array list length must be one less than"),
        ("inhom_evl0", r"evl_array\[0\] mediation dimension must match"),
        ("inhom_hidden", r"upd_array\[1\] hidden dimension must match"),
        ("inhom_prev", r"upd_array\[1\] previous mediation dimension must match"),
        ("inhom_curr", r"upd_array\[0\] current mediation dimension must match"),
    ],
)
def test_matvec_repn_check_dimensions(case, match):
    ini, upd, evl = valid_hom_arrays()
    i_ini, i_upd, i_evl = valid_inhom_arrays()

    if case == "evl_list_upd_array":
        kwargs = dict(ini_array=ini, upd_array=upd, evl_array=[evl, evl])
    elif case == "upd_list_evl_array":
        kwargs = dict(ini_array=ini, upd_array=[upd], evl_array=evl)
    elif case == "hom_hidden":
        kwargs = dict(ini_array=ini, upd_array=upd[:1], evl_array=evl)
    elif case == "hom_evl":
        kwargs = dict(ini_array=ini, upd_array=upd, evl_array=np.array([0.0]))
    elif case == "hom_mediation":
        # M_curr = 2 but M_prev = 3, while ini and evl have M = 2.
        upd3 = np.zeros((2, 2, 3))
        upd3[:, 0, :] = 1.0
        kwargs = dict(ini_array=ini, upd_array=upd3, evl_array=evl)
    elif case == "inhom_empty":
        kwargs = dict(ini_array=ini, upd_array=[], evl_array=[])
    elif case == "inhom_length":
        kwargs = dict(ini_array=i_ini, upd_array=i_upd[:1], evl_array=i_evl)
    elif case == "inhom_evl0":
        i_evl[0] = np.array([0.0, 1.0, 0.0])
        kwargs = dict(ini_array=i_ini, upd_array=i_upd, evl_array=i_evl)
    elif case == "inhom_hidden":
        i_upd[1] = i_upd[1][:1]
        kwargs = dict(ini_array=i_ini, upd_array=i_upd, evl_array=i_evl)
    elif case == "inhom_prev":
        # upd[1] maps 3 previous states to 2 current, but evl[1] has 2 states.
        upd3 = np.zeros((2, 2, 3))
        upd3[:, 0, :] = 1.0
        i_upd[1] = upd3
        kwargs = dict(ini_array=i_ini, upd_array=i_upd, evl_array=i_evl)
    else:
        # upd[0] maps 2 previous states to 3 current, but evl[1] has 2 states.
        upd3 = np.zeros((2, 3, 2))
        upd3[:, 0, :] = 1.0
        i_upd[0] = upd3
        kwargs = dict(ini_array=i_ini, upd_array=i_upd, evl_array=i_evl)

    with pytest.raises(InvalidInputError, match=match):
        MVR_MatVecRepn(**kwargs)


def test_matvec_repn_accepts_plain_nested_lists():
    ini, upd, evl = valid_hom_arrays()
    repn = MVR_MatVecRepn(
        ini_array=ini.tolist(), upd_array=upd.tolist(), evl_array=evl.tolist()
    )
    assert repn.num_mediation_states == 2
    assert np.array_equal(repn.upd_array, upd)

    i_ini, i_upd, i_evl = valid_inhom_arrays()
    repn = MVR_MatVecRepn(
        ini_array=i_ini.tolist(),
        upd_array=[u.tolist() for u in i_upd],
        evl_array=[e.tolist() for e in i_evl],
    )
    assert repn.time_horizon == 2
    assert repn.num_mediation_states == [2, 2, 2]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("ini_array", [[1.0, 0.0], [0.0]], "ini_array must be a rectangular array"),
        ("ini_array", [["a", 0.0], [0.0, 1.0]], "ini_array must be a rectangular"),
        (
            "upd_array",
            [[[1.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [0.0]]],
            "upd_array must be a rectangular array",
        ),
        ("evl_array", [0.0, [1.0]], "evl_array must be a rectangular array"),
    ],
)
def test_matvec_repn_reports_ragged_nested_lists(field, value, match):
    # Ragged lists used to raise a NumPy ValueError or become object arrays
    # that failed with a misleading "must be a 3D array".
    ini, upd, evl = valid_hom_arrays()
    kwargs = dict(ini_array=ini, upd_array=upd, evl_array=evl)
    kwargs[field] = value

    with pytest.raises(InvalidInputError, match=match):
        MVR_MatVecRepn(**kwargs)


def test_matvec_repn_names_the_ragged_per_time_array():
    i_ini, i_upd, i_evl = valid_inhom_arrays()
    i_evl[1] = [0.0, [1.0]]

    with pytest.raises(
        InvalidInputError, match=r"evl_array\[1\] must be a rectangular"
    ):
        MVR_MatVecRepn(ini_array=i_ini, upd_array=i_upd, evl_array=i_evl)


def test_matvec_repn_check_errors_false_skips_validation():
    # Fractional, non-normalised arrays with mismatched shapes are stored as is.
    ini = np.array([[0.5, 0.5]])
    upd = np.full((1, 2, 2), 0.25)
    evl = np.array([0.5, 0.5, 0.5])

    repn = MVR_MatVecRepn(
        ini_array=ini, upd_array=upd, evl_array=evl, check_errors=False
    )

    assert np.array_equal(repn.ini_array, ini)
    assert repn.num_hidden_states == 1
    assert repn.num_mediation_states == 2
    assert not repn.evl_array.flags.writeable


def test_matvec_repn_accepts_inhom_arrays_and_caches_index_maps():
    ini, upd, evl = valid_inhom_arrays()

    repn = MVR_MatVecRepn(ini_array=ini, upd_array=upd, evl_array=evl)

    assert repn.time_horizon == 2
    assert repn.num_mediation_states == [2, 2, 2]
    assert repn.mediation_states == [[0, 1], [0, 1], [0, 1]]
    assert list(repn.ini_idx) == [0, 1]
    assert len(repn.next_idx) == 2
    assert all(not arr.flags.writeable for arr in repn.upd_array)
    assert all(not arr.flags.writeable for arr in repn.next_idx)


@pytest.mark.parametrize(
    "duplicate", [copy.deepcopy, lambda r: pickle.loads(pickle.dumps(r))]
)
@pytest.mark.parametrize("arrays", [valid_hom_arrays, valid_inhom_arrays])
def test_matvec_repn_copies_stay_read_only(duplicate, arrays):
    ini, upd, evl = arrays()
    repn = duplicate(MVR_MatVecRepn(ini_array=ini, upd_array=upd, evl_array=evl))

    for name in ("ini_array", "upd_array", "evl_array", "ini_idx", "next_idx"):
        value = getattr(repn, name)
        for arr in value if isinstance(value, list) else [value]:
            assert not arr.flags.writeable, name


# ---------------------------------------------------------------------
# MVRConstraint / mvr_constraint_fn
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "func,match",
    [
        (lambda hmm, data, extra: None, "more than 2 arguments"),
        (lambda: None, "takes none"),
    ],
    ids=["three", "none"],
)
def test_mvr_constraint_takes_one_or_two_arguments(func, match):
    # A function without arguments was accepted and then failed with a
    # TypeError when the constraint was called.
    with pytest.raises(ValueError, match=match):
        MVRConstraint(func=func)


def test_mvr_constraint_without_function():
    constraint = MVRConstraint()

    assert constraint.num_args is None
    assert constraint.name == "Unnamed constraint"

    with pytest.raises(
        InvalidInputError,
        match="In constraint Unnamed constraint, the actual constraint function is not defined",
    ):
        constraint(make_hmm())


def test_mvr_constraint_names():
    def visit_a(hidden_markov_model):
        return None

    assert MVRConstraint(func=visit_a).name == "visit_a"
    assert MVRConstraint(func=visit_a, name="custom").name == "custom"
    assert MVRConstraint(name="no_func").name == "no_func"
    assert mvr_constraint_fn()(visit_a).name == "visit_a"
    assert mvr_constraint_fn(name="custom")(visit_a).name == "custom"


def test_mvr_constraint_passes_data_to_two_argument_functions():
    hmm = make_hmm(hidden_states=ALPHABET)
    seen = []

    @mvr_constraint_fn(name="with_data")
    def with_data(hidden_markov_model, data):
        seen.append((hidden_markov_model, data))
        return HomMVR(**hom_kwargs())

    @mvr_constraint_fn()
    def without_data(hidden_markov_model):
        seen.append((hidden_markov_model,))
        return HomMVR(**hom_kwargs())

    assert with_data.num_args == 2
    assert without_data.num_args == 1

    assert isinstance(with_data(hmm, {"k": 1}), HomMVR)
    assert isinstance(with_data(hmm), HomMVR)
    assert isinstance(without_data(hmm, {"ignored": True}), HomMVR)

    assert seen == [(hmm, {"k": 1}), (hmm, None), (hmm,)]


def test_mvr_constraint_rejects_non_mvr_results():
    @mvr_constraint_fn(name="bad_two_arg")
    def bad(hidden_markov_model, data):
        return "not an MVR"

    with pytest.raises(
        InvalidInputError,
        match="In constraint bad_two_arg, the returned object is not an MVR object",
    ):
        bad(make_hmm())


# ---------------------------------------------------------------------
# MVROperator / mvr_operator_fn (the operator extension point)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("arity", [0, -1, 1.0, "1", True])
def test_operator_rejects_invalid_arity(arity):
    with pytest.raises(InvalidInputError, match="arity must be a positive integer"):
        mvr_operator_fn(arity=arity)(lambda mvrs: mvrs[0])


def test_operator_names_and_docs():
    def my_op(mvrs):
        """My docstring."""
        return mvrs[0]

    default = mvr_operator_fn()(my_op)
    named = mvr_operator_fn(name="renamed", arity=1)(my_op)

    assert isinstance(default, MVROperator)
    assert default.name == "my_op"
    assert default.__doc__ == "My docstring."
    assert default.arity is None
    assert named.name == "renamed"
    assert named.arity == 1

    unnamed = MVROperator()
    assert unnamed.name == "Unnamed operator"
    assert unnamed.func is None


def test_operator_without_function_raises():
    with pytest.raises(
        InvalidInputError,
        match="In operator Unnamed operator, the actual operator function is not defined",
    ):
        MVROperator()(HomMVR(**hom_kwargs()))


@pytest.mark.parametrize(
    "mvrs,match",
    [
        (None, "an MVR or list of MVRs must be provided"),
        ([], "empty list was passed in"),
        ("mvr", "input must be an MVR or list of MVRs"),
        ((contains_symbol_mvr("a"),), "input must be an MVR or list of MVRs"),
        ([contains_symbol_mvr("a"), "x"], "Object at index 1 is not an MVR object"),
    ],
)
def test_operator_rejects_invalid_mvr_inputs(mvrs, match):
    op = mvr_operator_fn()(lambda mvrs: mvrs[0])

    with pytest.raises(InvalidInputError, match=match):
        op(mvrs)


def test_operator_enforces_arity():
    op = mvr_operator_fn(arity=2)(lambda mvrs: mvrs[0])
    a = contains_symbol_mvr("a")

    with pytest.raises(
        InvalidInputError,
        match="operator arity is 2 but provided list has length 1",
    ):
        op(a)

    with pytest.raises(
        InvalidInputError,
        match="operator arity is 2 but provided list has length 3",
    ):
        op([a, a, a])

    assert op([a, a]) is a


def test_operator_rejects_mismatched_hidden_spaces():
    other = HomMVR(
        hidden_states=["a", "c"],
        mediation_states=[0],
        ini={"a": 0, "c": 0},
        upd={(0, "a"): 0, (0, "c"): 0},
        evl={0: True},
    )
    op = mvr_operator_fn()(lambda mvrs: mvrs[0])

    with pytest.raises(
        InvalidInputError,
        match=r"mvrs\[1\] must have the same hidden_states as mvrs\[0\]",
    ):
        op([contains_symbol_mvr("a"), other])

    with pytest.raises(InvalidInputError, match=r"mvrs\[1\] must have the same"):
        mvr_or([contains_symbol_mvr("a"), other])


def test_operator_accepts_reordered_hidden_states():
    reordered = HomMVR(
        hidden_states=["b", "a"],
        mediation_states=[0],
        ini={"a": 0, "b": 0},
        upd={(0, "a"): 0, (0, "b"): 0},
        evl={0: True},
    )

    result = mvr_or([contains_symbol_mvr("a"), reordered])

    assert result.hidden_states == list(ALPHABET)
    assert eval_mvr(result, ["b"]) is True


def test_operator_checks_the_hidden_markov_model():
    hmm = make_hmm(hidden_states=ALPHABET)
    reordered = HomMVR(
        hidden_states=["b", "a"],
        mediation_states=[0],
        ini={"a": 0, "b": 0},
        upd={(0, "a"): 0, (0, "b"): 0},
        evl={0: True},
    )

    with pytest.raises(
        InvalidInputError,
        match=r"The hidden states of the HMM and the \(first\) MVR do not exactly match",
    ):
        mvr_not(reordered, hidden_markov_model=hmm)

    with pytest.raises(InvalidInputError, match="do not exactly match"):
        mvr_not(contains_symbol_mvr("a"), hidden_markov_model=make_hmm(["a", "c"]))

    # The model is checked, not forwarded to the operator function.
    result = mvr_not(contains_symbol_mvr("a"), hidden_markov_model=hmm)
    assert eval_mvr(result, ["b"]) is True


def test_operator_rejects_non_mvr_output():
    op = mvr_operator_fn(name="broken")(lambda mvrs: "oops")

    with pytest.raises(
        InvalidInputError, match="The output of operator broken is not an MVR object"
    ):
        op(contains_symbol_mvr("a"))


def test_operator_forwards_arguments_and_prunes_new_results():
    calls = []

    @mvr_operator_fn(arity=1)
    def rebuild(mvrs, tag, *, flag=False):
        calls.append((tag, flag))
        mvr = mvrs[0]
        return HomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=list(mvr.mediation_states),
            ini=dict(mvr.ini),
            upd=dict(mvr.upd),
            evl=dict(mvr.evl),
        )

    orphan = orphan_state_mvr()
    result = rebuild(orphan, "t", flag=True)

    assert calls == [("t", True)]
    assert result is not orphan
    assert set(result.mediation_states) == {False, True}

    # Returning an input skips pruning and preserves identity.
    identity = mvr_operator_fn()(lambda mvrs: mvrs[-1])
    assert identity([contains_symbol_mvr("a"), orphan]) is orphan
    assert "orphan" in orphan.mediation_states


# ---------------------------------------------------------------------
# Built-in operators: type checks
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda m: mvr_timerange(m, [0, 1]), "mvr must be an instance of HomMVR"),
        (lambda m: mvr_not(m), "mvr must be an instance of HomMVR or InhomMVR"),
        (lambda m: mvr_not_yet(m), "mvr must be an instance of HomMVR or InhomMVR"),
        (lambda m: mvr_sattime(m), "mvr_sattime expects a HomMVR or InhomMVR"),
        (lambda m: mvr_count(m, "1"), "mvr_count expects a HomMVR or InhomMVR"),
        (
            lambda m: mvr_or([contains_symbol_mvr("a"), m]),
            r"mvrs\[1\] must be an instance of HomMVR or InhomMVR",
        ),
        (
            # It failed with a TypeError on None + int.
            lambda m: mvr_precedence([m, contains_symbol_mvr("a")], "<"),
            r"mvrs\[0\] must be an instance of HomMVR or InhomMVR",
        ),
    ],
)
def test_operators_reject_unknown_mvr_subclasses(call, match):
    with pytest.raises(InvalidInputError, match=match):
        call(OpaqueMVR())


@pytest.mark.filterwarnings("ignore:Use the AND operator")
def test_mvr_and_rejects_unknown_mvr_subclasses():
    with pytest.raises(InvalidInputError, match=r"mvrs\[0\] must be an instance"):
        mvr_and([OpaqueMVR(), contains_symbol_mvr("a")])


@pytest.mark.parametrize(
    "call,match",
    [
        (
            lambda h, i: mvr_setdiff([h, i]),
            "mvr_setdiff expects a list of exactly two HomMVRs",
        ),
        (
            lambda h, i: mvr_concatenate([i, h]),
            "mvr_concatenate expects a list of exactly two HomMVRs",
        ),
        (
            lambda h, i: mvr_concatenate_prefix([h, i]),
            "mvr_concatenate_prefix expects a list of exactly two HomMVRs",
        ),
        (lambda h, i: mvr_kfold_product(i, 2), "mvr_kfold_product expects a HomMVR"),
        (
            lambda h, i: mvr_kfold_product_prefix(i, 2),
            "mvr_kfold_product_prefix expects a HomMVR",
        ),
        (lambda h, i: mvr_kleene_closure(i), "mvr_kleene_closure expects a HomMVR"),
        (
            lambda h, i: mvr_kleene_closure_prefix(i),
            "mvr_kleene_closure_prefix expects a HomMVR",
        ),
        (lambda h, i: mvr_reverse(i), "mvr_reverse expects a HomMVR"),
    ],
)
def test_homogeneous_only_operators_reject_inhom_mvrs(call, match):
    hom = contains_symbol_mvr("a")
    inhom = trivial_inhom_mvr(time_horizon=2)

    with pytest.raises(InvalidInputError, match=match):
        call(hom, inhom)


@pytest.mark.parametrize("op", [mvr_kfold_product, mvr_kfold_product_prefix])
@pytest.mark.parametrize("k", [0, -2, 1.5, "2", None, True, False])
def test_kfold_products_reject_invalid_k(op, k):
    with pytest.raises(
        InvalidInputError, match="k must be an integer greater than or equal to 1"
    ):
        op(contains_symbol_mvr("a"), k)


@pytest.mark.parametrize(
    "condition,match",
    [
        (2, "condition must be a string"),
        ("[3,1]", "count range must be nonempty"),
        ("(2,2]", "count range must be nonempty"),
        ("<0", 'condition "<0" is degenerate'),
        (">=0", 'condition ">=0" is degenerate'),
        ("=2", "invalid count condition '=2'"),
        ("[1,3)", r"invalid count condition '\[1,3\)'"),
        ("-1", "invalid count condition '-1'"),
    ],
)
@pytest.mark.parametrize("inhom", [False, True])
def test_mvr_count_rejects_invalid_conditions(condition, match, inhom):
    mvr = trivial_inhom_mvr(time_horizon=2) if inhom else contains_symbol_mvr("a")

    with pytest.raises(InvalidInputError, match=match):
        mvr_count(mvr, condition)


@pytest.mark.parametrize(
    "time_range,match",
    [
        ((0, 1), "time range must be a list of two nonnegative integers"),
        ([-1, 2], "time range must be a list of two nonnegative integers"),
        ([0, 1, 2], "time range must be a list of two nonnegative integers"),
        ([0.0, 1], "time range must be a list of two nonnegative integers"),
        ([2, 1], "with start <= end"),
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_mvr_timerange_rejects_invalid_ranges(time_range, match, inplace):
    mvr = contains_symbol_mvr("a")

    with pytest.raises(InvalidInputError, match=match):
        mvr_timerange(mvr, time_range, inplace=inplace)

    assert mvr.time_range is None


def test_mvr_timerange_inplace_checks_the_inhom_horizon():
    inhom = trivial_inhom_mvr(time_horizon=2)

    with pytest.raises(InvalidInputError, match="longer than the time horizon"):
        mvr_timerange(inhom, [1, 4], inplace=True)

    assert inhom.time_range is None
    assert mvr_timerange(inhom, [1, 3], inplace=True) is inhom
    assert inhom.time_range == [1, 3]


# ---------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------


@pytest.mark.parametrize("transitions", [5, "ab", {"a": "b"}, None])
def test_current_transition_rejects_non_collections(transitions):
    with pytest.raises(
        InvalidInputError, match="transitions must be a list, tuple, set, or frozenset"
    ):
        mvr_current_transition(make_hmm(), transitions)


def test_regex_requires_greenery(monkeypatch):
    primitives = importlib.import_module("conin.constraints.mvr.primitives")
    monkeypatch.setattr(primitives, "greenery_available", False)

    with pytest.raises(ImportError, match="Missing import greenery"):
        mvr_regex(make_hmm(), "<a>")

    # The model is validated first.
    with pytest.raises(InvalidInputError, match="required argument"):
        mvr_regex(None, "<a>")
