import re
import warnings

from collections.abc import Callable
from itertools import combinations, product
from typing import Any, Literal

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import HomMVR, InhomMVR, _normalize_time_range
from conin.operators import mvr_operator_fn

MVR = HomMVR | InhomMVR
PrecedenceRelation = Literal["<", "<=", ">", ">="]


def _combined_time_horizon(mvrs: list[MVR]) -> int | None:
    """Return the minimum inhomogeneous horizon, or None when all are homogeneous."""
    horizons = [mvr.time_horizon for mvr in mvrs if isinstance(mvr, InhomMVR)]

    if len(horizons) == 0:
        return None

    min_horizon = min(horizons)

    if len(set(horizons)) > 1:
        warnings.warn(
            f"Combining inhomogeneous MVRs with unequal horizons; using minimum horizon {min_horizon}.",
            UserWarning,
        )

    return min_horizon


# ------------------------------------------------------------------
# Set TimeRange
# ------------------------------------------------------------------

# Every other operator drops ``_time_range``; apply this operator last when
# building a subsequence constraint.


@mvr_operator_fn(arity=1)
def mvr_timerange(
    mvrs: list[MVR],
    time_range: list[int] = None,
    inplace: bool = True,
) -> MVR:
    """Set an MVR's time range, mutating it by default."""
    mvr = mvrs[0]

    if not isinstance(mvr, (HomMVR, InhomMVR)):
        raise InvalidInputError("mvr must be an instance of HomMVR or InhomMVR.")

    if inplace:
        time_horizon = mvr.time_horizon if isinstance(mvr, InhomMVR) else None
        mvr._time_range = _normalize_time_range(time_range, time_horizon)
        return mvr

    if isinstance(mvr, HomMVR):
        result = HomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=list(mvr.mediation_states),
            ini=dict(mvr.ini),
            upd=dict(mvr.upd),
            evl=dict(mvr.evl),
            time_range=time_range,
            name=mvr.name,
        )
    else:
        result = InhomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=[
                list(mediation_states_t) for mediation_states_t in mvr.mediation_states
            ],
            ini=dict(mvr.ini),
            upd=[dict(upd_t) for upd_t in mvr.upd],
            evl=[dict(evl_t) for evl_t in mvr.evl],
            time_range=time_range,
            name=mvr.name,
        )

    result._prefix = mvr._prefix
    return result


# ------------------------------------------------------------------
# Boolean Operators: AND/OR/NOT
# ------------------------------------------------------------------


def _boolean_combine_mvrs(
    mvrs: list[MVR],
    bool_reducer: Callable[[list[bool]], bool],
) -> MVR:
    """Build the product MVR for a Boolean reduction."""
    if len(mvrs) == 0:
        raise InvalidInputError("mvrs must be a nonempty iterable of MVRs.")

    for i, mvr in enumerate(mvrs):
        if not isinstance(mvr, (HomMVR, InhomMVR)):
            raise InvalidInputError(
                f"mvrs[{i}] must be an instance of HomMVR or InhomMVR."
            )

    hidden_states = mvrs[0].hidden_states
    time_horizon = _combined_time_horizon(mvrs)
    num_mvrs = len(mvrs)

    # Homogeneous case: all inputs are HomMVR.
    if time_horizon is None:
        component_mediation_spaces = [list(mvr.mediation_states) for mvr in mvrs]

        mediation_states = list(product(*component_mediation_spaces))

        ini = {h: tuple(mvr.ini[h] for mvr in mvrs) for h in hidden_states}

        upd = {}

        for m_prev_tuple in mediation_states:
            for h_curr in hidden_states:
                m_curr_tuple = tuple(
                    mvrs[i].upd[(m_prev_tuple[i], h_curr)] for i in range(num_mvrs)
                )

                upd[(m_prev_tuple, h_curr)] = m_curr_tuple

        evl = {}

        for m_tuple in mediation_states:
            component_values = [mvrs[i].evl[m_tuple[i]] for i in range(num_mvrs)]

            evl[m_tuple] = bool(bool_reducer(component_values))

        return HomMVR(
            hidden_states=hidden_states,
            mediation_states=mediation_states,
            ini=ini,
            upd=upd,
            evl=evl,
        )

    # Inhomogeneous case: at least one input is InhomMVR.
    mediation_states = []

    for t in range(time_horizon + 1):
        component_mediation_spaces_t = [
            (
                list(mvr.mediation_states)
                if isinstance(mvr, HomMVR)
                else list(mvr.mediation_states[t])
            )
            for mvr in mvrs
        ]

        mediation_states_t = list(product(*component_mediation_spaces_t))
        mediation_states.append(mediation_states_t)

    ini = {h: tuple(mvr.ini[h] for mvr in mvrs) for h in hidden_states}

    upd = []

    for t in range(time_horizon):
        upd_t = {}

        for m_prev_tuple in mediation_states[t]:
            for h_curr in hidden_states:
                m_curr_tuple = tuple(
                    (
                        mvrs[i].upd[(m_prev_tuple[i], h_curr)]
                        if isinstance(mvrs[i], HomMVR)
                        else mvrs[i].upd[t][(m_prev_tuple[i], h_curr)]
                    )
                    for i in range(num_mvrs)
                )

                upd_t[(m_prev_tuple, h_curr)] = m_curr_tuple

        upd.append(upd_t)

    evl = []

    for t in range(time_horizon + 1):
        evl_t = {}

        for m_tuple in mediation_states[t]:
            component_values = [
                (
                    mvrs[i].evl[m_tuple[i]]
                    if isinstance(mvrs[i], HomMVR)
                    else mvrs[i].evl[t][m_tuple[i]]
                )
                for i in range(num_mvrs)
            ]

            evl_t[m_tuple] = bool(bool_reducer(component_values))

        evl.append(evl_t)

    return InhomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


@mvr_operator_fn(arity=None)
def mvr_and(
    mvrs: list[MVR],
) -> MVR:
    """Construct the logical AND of a nonempty list of MVRs."""
    warnings.warn(
        "Use the AND operator sparingly. It's generally more efficient to provide a list of MVRs to downstream algorithms.",
        UserWarning,
    )
    result = _boolean_combine_mvrs(
        mvrs,
        all,
    )

    # Respects prefix property: output has _prefix=True if any input does.
    result._prefix = any(mvr.prefix for mvr in mvrs)
    return result


@mvr_operator_fn(arity=None)
def mvr_or(
    mvrs: list[MVR],
) -> MVR:
    """Construct the logical OR of a nonempty list of MVRs."""
    # Deliberately does not propagate prefix, unlike mvr_and: union does not
    # preserve prefix-freeness. {a} and {aa} are each prefix-free, {a, aa} is not.
    return _boolean_combine_mvrs(
        mvrs,
        any,
    )


@mvr_operator_fn(arity=1)
def mvr_not(
    mvrs: list[MVR],
) -> MVR:
    """Construct the logical complement of one MVR."""
    mvr = mvrs[0]
    if isinstance(mvr, HomMVR):
        return HomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=list(mvr.mediation_states),
            ini=dict(mvr.ini),
            upd=dict(mvr.upd),
            evl={m: not value for m, value in mvr.evl.items()},
        )

    if isinstance(mvr, InhomMVR):
        return InhomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=[
                list(mediation_states_t) for mediation_states_t in mvr.mediation_states
            ],
            ini=dict(mvr.ini),
            upd=[dict(upd_t) for upd_t in mvr.upd],
            evl=[{m: not value for m, value in evl_t.items()} for evl_t in mvr.evl],
        )

    raise InvalidInputError("mvr must be an instance of HomMVR or InhomMVR.")


# ------------------------------------------------------------------
# Not Yet/ Already Satisfied
# ------------------------------------------------------------------


@mvr_operator_fn(arity=1)
def mvr_not_yet(
    mvrs: list[MVR],
) -> MVR:
    """Accept while the input has never accepted through the current time."""
    mvr = mvrs[0]

    if isinstance(mvr, HomMVR):
        mediation_states = [
            (m, not_yet_flag)
            for m in mvr.mediation_states
            for not_yet_flag in [False, True]
        ]

        ini = {}

        for h in mvr.hidden_states:
            m0 = mvr.ini[h]
            not_yet_flag0 = not mvr.evl[m0]
            ini[h] = (m0, not_yet_flag0)

        upd = {}

        for m_prev in mvr.mediation_states:
            for not_yet_flag_prev in [False, True]:
                for h_curr in mvr.hidden_states:
                    m_curr = mvr.upd[(m_prev, h_curr)]
                    not_yet_flag_curr = not_yet_flag_prev and not mvr.evl[m_curr]

                    upd[((m_prev, not_yet_flag_prev), h_curr)] = (
                        m_curr,
                        not_yet_flag_curr,
                    )

        evl = {
            (m, not_yet_flag): not_yet_flag
            for m in mvr.mediation_states
            for not_yet_flag in [False, True]
        }

        return HomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=mediation_states,
            ini=ini,
            upd=upd,
            evl=evl,
        )

    if isinstance(mvr, InhomMVR):
        mediation_states = []

        for t in range(mvr.time_horizon + 1):
            mediation_states_t = [
                (m, not_yet_flag)
                for m in mvr.mediation_states[t]
                for not_yet_flag in [False, True]
            ]

            mediation_states.append(mediation_states_t)

        ini = {}

        for h in mvr.hidden_states:
            m0 = mvr.ini[h]
            not_yet_flag0 = not mvr.evl[0][m0]
            ini[h] = (m0, not_yet_flag0)

        upd = []

        for t in range(mvr.time_horizon):
            upd_t = {}

            for m_prev in mvr.mediation_states[t]:
                for not_yet_flag_prev in [False, True]:
                    for h_curr in mvr.hidden_states:
                        m_curr = mvr.upd[t][(m_prev, h_curr)]
                        not_yet_flag_curr = (
                            not_yet_flag_prev and not mvr.evl[t + 1][m_curr]
                        )

                        upd_t[((m_prev, not_yet_flag_prev), h_curr)] = (
                            m_curr,
                            not_yet_flag_curr,
                        )

            upd.append(upd_t)

        evl = []

        for t in range(mvr.time_horizon + 1):
            evl_t = {
                (m, not_yet_flag): not_yet_flag
                for m in mvr.mediation_states[t]
                for not_yet_flag in [False, True]
            }

            evl.append(evl_t)

        return InhomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=mediation_states,
            ini=ini,
            upd=upd,
            evl=evl,
        )

    raise InvalidInputError("mvr must be an instance of HomMVR or InhomMVR.")


@mvr_operator_fn(arity=1)
def mvr_already_satisfied(
    mvrs: list[MVR],
) -> MVR:
    """Accept once the input has accepted at any time through the present."""
    return mvr_not(
        mvr_not_yet(
            mvrs,
        ),
    )


# ------------------------------------------------------------------
# Sat_Time: Prefix-Free
# ------------------------------------------------------------------


@mvr_operator_fn(arity=1)
def mvr_sattime(
    mvrs: list[MVR],
) -> MVR:
    """Return the prefix-free first-acceptance language, preserving certified inputs."""
    mvr = mvrs[0]

    if not isinstance(mvr, (HomMVR, InhomMVR)):
        raise InvalidInputError("mvr_sattime expects a HomMVR or InhomMVR.")

    # sattime(L) == L when L is prefix-free, so the input is already the answer.
    if mvr.prefix:
        return mvr

    # Create a fresh absorbing fail state.
    if isinstance(mvr, HomMVR):
        existing_states = set(mvr.mediation_states)
    else:
        existing_states = {
            m for mediation_states_t in mvr.mediation_states for m in mediation_states_t
        }

    fail_state = ("__sattime_fail__",)

    if fail_state in existing_states:
        i = 0
        while True:
            fail_state = ("__sattime_fail__", i)
            if fail_state not in existing_states:
                break
            i += 1

    # Homogeneous case
    if isinstance(mvr, HomMVR):
        hidden_states = list(mvr.hidden_states)
        mediation_states = list(mvr.mediation_states) + [fail_state]

        ini = dict(mvr.ini)

        upd = {}

        for m_prev in mediation_states:
            for h_curr in hidden_states:
                if m_prev == fail_state:
                    m_curr = fail_state
                elif mvr.evl[m_prev]:
                    m_curr = fail_state
                else:
                    m_curr = mvr.upd[(m_prev, h_curr)]

                upd[(m_prev, h_curr)] = m_curr

        evl = {m: mvr.evl[m] for m in mvr.mediation_states}
        evl[fail_state] = False

        result = HomMVR(
            hidden_states=hidden_states,
            mediation_states=mediation_states,
            ini=ini,
            upd=upd,
            evl=evl,
        )
        result._prefix = True

        return result

    # Inhomogeneous case
    hidden_states = list(mvr.hidden_states)

    mediation_states = [
        list(mediation_states_t) + [fail_state]
        for mediation_states_t in mvr.mediation_states
    ]

    ini = dict(mvr.ini)

    upd = []

    for t in range(mvr.time_horizon):
        upd_t = {}

        for m_prev in mediation_states[t]:
            for h_curr in hidden_states:
                if m_prev == fail_state:
                    m_curr = fail_state
                elif mvr.evl[t][m_prev]:
                    m_curr = fail_state
                else:
                    m_curr = mvr.upd[t][(m_prev, h_curr)]

                upd_t[(m_prev, h_curr)] = m_curr

        upd.append(upd_t)

    evl = []

    for t in range(mvr.time_horizon + 1):
        evl_t = {m: mvr.evl[t][m] for m in mvr.mediation_states[t]}
        evl_t[fail_state] = False

        evl.append(evl_t)

    result = InhomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )

    result._prefix = True

    return result


# ------------------------------------------------------------------
# Regular Language Operators: HomMVR ONLY
# ------------------------------------------------------------------


def _powerset_frozensets(states: list[Any]) -> list[frozenset[Any]]:
    """Return the powerset as a list of frozensets."""
    return [
        frozenset(subset)
        for r in range(len(states) + 1)
        for subset in combinations(states, r)
    ]


@mvr_operator_fn(arity=2)
def mvr_setdiff(
    mvrs: list[HomMVR],
) -> HomMVR:
    """Recognize the set difference of two homogeneous MVR languages."""
    if not all(isinstance(mvr, HomMVR) for mvr in mvrs):
        raise InvalidInputError("mvr_setdiff expects a list of exactly two HomMVRs.")

    mvr1, mvr2 = mvrs

    hidden_states = mvrs[0].hidden_states

    mediation_states = list(
        product(
            mvr1.mediation_states,
            mvr2.mediation_states,
        )
    )

    ini = {
        h: (
            mvr1.ini[h],
            mvr2.ini[h],
        )
        for h in hidden_states
    }

    upd = {}

    for m1_prev in mvr1.mediation_states:
        for m2_prev in mvr2.mediation_states:
            for h_curr in hidden_states:
                m1_curr = mvr1.upd[(m1_prev, h_curr)]
                m2_curr = mvr2.upd[(m2_prev, h_curr)]

                upd[((m1_prev, m2_prev), h_curr)] = (
                    m1_curr,
                    m2_curr,
                )

    evl = {
        (m1, m2): mvr1.evl[m1] and not mvr2.evl[m2]
        for m1 in mvr1.mediation_states
        for m2 in mvr2.mediation_states
    }

    result = HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )

    # Respects prefix property: output has _prefix=True if mvr1 does.
    result._prefix = mvr1.prefix
    return result


@mvr_operator_fn(arity=2)
def mvr_concatenate(
    mvrs: list[HomMVR],
) -> HomMVR:
    """Recognize the nonempty language ``L(mvr1) L(mvr2)``."""
    if not all(isinstance(mvr, HomMVR) for mvr in mvrs):
        raise InvalidInputError(
            "mvr_concatenate expects a list of exactly two HomMVRs."
        )

    mvr1, mvr2 = mvrs

    hidden_states = mvrs[0].hidden_states

    mvr2_subsets = _powerset_frozensets(list(mvr2.mediation_states))

    # Track all possible mediation states based on split points in M1.
    mediation_states = list(
        product(
            mvr1.mediation_states,
            mvr2_subsets,
        )
    )

    ini = {
        h: (
            mvr1.ini[h],
            frozenset(),
        )
        for h in hidden_states
    }

    upd = {}

    for m1_prev in mvr1.mediation_states:
        for active_m2_states_prev in mvr2_subsets:
            for h_curr in hidden_states:
                m1_curr = mvr1.upd[(m1_prev, h_curr)]

                active_m2_states_curr = {
                    mvr2.upd[(m2_prev, h_curr)] for m2_prev in active_m2_states_prev
                }

                if mvr1.evl[m1_prev]:
                    active_m2_states_curr.add(mvr2.ini[h_curr])

                active_m2_states_curr = frozenset(active_m2_states_curr)

                upd[((m1_prev, active_m2_states_prev), h_curr)] = (
                    m1_curr,
                    active_m2_states_curr,
                )

    evl = {
        (m1, active_m2_states): any(mvr2.evl[m2] for m2 in active_m2_states)
        for m1 in mvr1.mediation_states
        for active_m2_states in mvr2_subsets
    }

    result = HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )

    # Respects prefix property: output has _prefix=True if both inputs do.
    result._prefix = mvr1.prefix and mvr2.prefix
    return result


@mvr_operator_fn(arity=2)
def mvr_concatenate_prefix(
    mvrs: list[HomMVR],
) -> HomMVR:
    """Concatenate the prefix-free forms without a powerset construction."""
    if not all(isinstance(mvr, HomMVR) for mvr in mvrs):
        raise InvalidInputError(
            "mvr_concatenate_prefix expects a list of exactly two HomMVRs."
        )

    mvr1, mvr2 = mvrs

    mvr1 = mvr_sattime(mvr1)
    mvr2 = mvr_sattime(mvr2)

    hidden_states = mvrs[0].hidden_states

    mediation_states = [("first", m1) for m1 in mvr1.mediation_states] + [
        ("second", m2) for m2 in mvr2.mediation_states
    ]

    ini = {h: ("first", mvr1.ini[h]) for h in hidden_states}

    upd = {}

    for m1_prev in mvr1.mediation_states:
        for h_curr in hidden_states:
            if mvr1.evl[m1_prev]:
                m_curr = ("second", mvr2.ini[h_curr])
            else:
                m_curr = ("first", mvr1.upd[(m1_prev, h_curr)])

            upd[(("first", m1_prev), h_curr)] = m_curr

    for m2_prev in mvr2.mediation_states:
        for h_curr in hidden_states:
            upd[(("second", m2_prev), h_curr)] = (
                "second",
                mvr2.upd[(m2_prev, h_curr)],
            )

    evl = {("first", m1): False for m1 in mvr1.mediation_states}

    evl.update({("second", m2): mvr2.evl[m2] for m2 in mvr2.mediation_states})

    result = HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )
    result._prefix = True

    return result


@mvr_operator_fn(arity=1)
def mvr_kfold_product(
    mvrs: list[HomMVR],
    k: int,
) -> HomMVR:
    """Recognize ``L(mvr)^k`` for ``k >= 1``."""
    mvr = mvrs[0]

    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kfold_product expects a HomMVR.")

    if not isinstance(k, int) or k < 1:
        raise InvalidInputError("k must be an integer greater than or equal to 1.")

    if k == 1:
        result = HomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=list(mvr.mediation_states),
            ini=dict(mvr.ini),
            upd=dict(mvr.upd),
            evl=dict(mvr.evl),
        )

        # Respects prefix property: output has _prefix=True if input does.
        result._prefix = mvr.prefix
        return result

    return mvr_concatenate(
        [
            mvr_kfold_product(
                mvr,
                k - 1,
            ),
            mvr,
        ],
    )


@mvr_operator_fn(arity=1)
def mvr_kfold_product_prefix(
    mvrs: list[HomMVR],
    k: int,
) -> HomMVR:
    """Recognize ``sattime(L(mvr))^k`` without a powerset construction."""
    mvr = mvrs[0]

    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kfold_product_prefix expects a HomMVR.")

    if not isinstance(k, int) or k < 1:
        raise InvalidInputError("k must be an integer greater than or equal to 1.")

    if k == 1:
        return mvr_sattime(
            mvr,
        )

    return mvr_concatenate_prefix(
        [
            mvr_kfold_product_prefix(
                mvr,
                k - 1,
            ),
            mvr,
        ],
    )


@mvr_operator_fn(arity=1)
def mvr_kleene_closure(
    mvrs: list[HomMVR],
) -> HomMVR:
    """Construct the nonempty Kleene closure of a homogeneous MVR."""
    mvr = mvrs[0]

    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kleene_closure expects a HomMVR.")

    hidden_states = mvrs[0].hidden_states

    # Track all possible mediation states based on split points in the past.
    mediation_states = _powerset_frozensets(list(mvr.mediation_states))

    ini = {h: frozenset({mvr.ini[h]}) for h in hidden_states}

    upd = {}

    for active_states_prev in mediation_states:
        prefix_already_decomposable = any(mvr.evl[m] for m in active_states_prev)

        for h_curr in hidden_states:
            active_states_curr = {
                mvr.upd[(m_prev, h_curr)] for m_prev in active_states_prev
            }

            if prefix_already_decomposable:
                active_states_curr.add(mvr.ini[h_curr])

            active_states_curr = frozenset(active_states_curr)

            upd[(active_states_prev, h_curr)] = active_states_curr

    evl = {
        active_states: any(mvr.evl[m] for m in active_states)
        for active_states in mediation_states
    }

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


@mvr_operator_fn(arity=1)
def mvr_kleene_closure_prefix(
    mvrs: list[HomMVR],
) -> HomMVR:
    """Construct ``sattime(L(mvr))^+`` without a powerset construction."""
    mvr = mvrs[0]

    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kleene_closure_prefix expects a HomMVR.")

    mvr = mvr_sattime(
        mvr,
    )

    hidden_states = mvr.hidden_states
    mediation_states = mvr.mediation_states

    ini = {h: mvr.ini[h] for h in hidden_states}

    upd = {}

    for m_prev in mediation_states:
        for h_curr in hidden_states:
            if mvr.evl[m_prev]:
                m_curr = mvr.ini[h_curr]
            else:
                m_curr = mvr.upd[(m_prev, h_curr)]

            upd[(m_prev, h_curr)] = m_curr

    evl = dict(mvr.evl)

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


@mvr_operator_fn(arity=1)
def mvr_reverse(
    mvrs: list[HomMVR],
) -> HomMVR:
    """Recognize the nonempty reversal of ``L(mvr)``."""
    mvr = mvrs[0]

    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_reverse expects a HomMVR.")

    hidden_states = mvr.hidden_states
    original_mediation_states = mvr.mediation_states

    subsets = _powerset_frozensets(original_mediation_states)

    mediation_states = list(
        product(
            subsets,
            [False, True],
        )
    )

    accepting_states = frozenset(m for m in original_mediation_states if mvr.evl[m])

    def reverse_preimage(
        target_states: frozenset[Any],
        h: Any,
    ) -> frozenset[Any]:
        return frozenset(
            m_prev
            for m_prev in original_mediation_states
            if mvr.upd[(m_prev, h)] in target_states
        )

    ini = {}

    for h in hidden_states:
        possible_previous_states = reverse_preimage(
            accepting_states,
            h,
        )

        accepts_now = mvr.evl[mvr.ini[h]]

        ini[h] = (
            possible_previous_states,
            accepts_now,
        )

    upd = {}

    for possible_previous_states_prev in subsets:
        for accepts_prev in [False, True]:
            for h_curr in hidden_states:
                possible_previous_states_curr = reverse_preimage(
                    possible_previous_states_prev,
                    h_curr,
                )

                accepts_curr = mvr.ini[h_curr] in possible_previous_states_prev

                upd[
                    (
                        (
                            possible_previous_states_prev,
                            accepts_prev,
                        ),
                        h_curr,
                    )
                ] = (
                    possible_previous_states_curr,
                    accepts_curr,
                )

    evl = {
        (possible_previous_states, accepts_now): accepts_now
        for possible_previous_states in subsets
        for accepts_now in [False, True]
    }

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


# ------------------------------------------------------------------
# Precedence
# ------------------------------------------------------------------
@mvr_operator_fn(arity=2)
def mvr_precedence(
    mvrs: list[MVR],
    relation: PrecedenceRelation,
) -> MVR:
    """Compare first satisfaction times; an absent time behaves as infinity."""
    if relation not in ["<", "<=", ">", ">="]:
        raise InvalidInputError('relation must be one of "<", "<=", ">", ">=".')

    mvr1, mvr2 = mvrs

    hidden_states = mvrs[0].hidden_states
    bool_states = [False, True]

    def update_ok(
        seen1_prev: bool,
        seen2_prev: bool,
        event1_curr: bool,
        event2_curr: bool,
        ok_prev: bool,
    ) -> bool:
        """Update whether the precedence relation has been established."""
        if ok_prev:
            return True

        if relation == "<":
            return (seen1_prev and not seen2_prev) or (
                event1_curr and not seen2_prev and not event2_curr
            )

        if relation == "<=":
            return (not seen2_prev) and (seen1_prev or event1_curr)

        if relation == ">":
            return (seen2_prev and not seen1_prev) or (
                event2_curr and not seen1_prev and not event1_curr
            )

        # relation == ">="
        return (not seen1_prev) and (seen2_prev or event2_curr)

    homogeneous = isinstance(mvr1, HomMVR) and isinstance(mvr2, HomMVR)
    time_horizon = None if homogeneous else _combined_time_horizon(mvrs)
    num_slices = 1 if homogeneous else time_horizon + 1

    def states_at(mvr, t):
        return (
            mvr.mediation_states
            if isinstance(mvr, HomMVR)
            else mvr.mediation_states[t]
        )

    def evl_at(mvr, t, state):
        return mvr.evl[state] if isinstance(mvr, HomMVR) else mvr.evl[t][state]

    def upd_at(mvr, t, state, hidden):
        return (
            mvr.upd[(state, hidden)]
            if isinstance(mvr, HomMVR)
            else mvr.upd[t][(state, hidden)]
        )

    states_by_time = [
        list(
            product(
                states_at(mvr1, t),
                states_at(mvr2, t),
                bool_states,  # seen1
                bool_states,  # seen2
                bool_states,  # ok
            )
        )
        for t in range(num_slices)
    ]

    ini = {}

    for h in hidden_states:
        m1 = mvr1.ini[h]
        m2 = mvr2.ini[h]
        event1 = evl_at(mvr1, 0, m1)
        event2 = evl_at(mvr2, 0, m2)

        ini[h] = (
            m1,
            m2,
            event1,
            event2,
            update_ok(False, False, event1, event2, False),
        )

    updates_by_time = []
    num_updates = 1 if homogeneous else time_horizon

    for t in range(num_updates):
        upd_t = {}

        for state_prev in states_by_time[t]:
            m1_prev, m2_prev, seen1_prev, seen2_prev, ok_prev = state_prev

            for h_curr in hidden_states:
                m1_curr = upd_at(mvr1, t, m1_prev, h_curr)
                m2_curr = upd_at(mvr2, t, m2_prev, h_curr)
                event1_curr = (not seen1_prev) and evl_at(mvr1, t + 1, m1_curr)
                event2_curr = (not seen2_prev) and evl_at(mvr2, t + 1, m2_curr)
                seen1_curr = seen1_prev or event1_curr
                seen2_curr = seen2_prev or event2_curr

                upd_t[(state_prev, h_curr)] = (
                    m1_curr,
                    m2_curr,
                    seen1_curr,
                    seen2_curr,
                    update_ok(
                        seen1_prev,
                        seen2_prev,
                        event1_curr,
                        event2_curr,
                        ok_prev,
                    ),
                )

        updates_by_time.append(upd_t)

    evaluations_by_time = [
        {state: state[4] for state in states_t} for states_t in states_by_time
    ]

    if homogeneous:
        return HomMVR(
            hidden_states=hidden_states,
            mediation_states=states_by_time[0],
            ini=ini,
            upd=updates_by_time[0],
            evl=evaluations_by_time[0],
        )

    return InhomMVR(
        hidden_states=hidden_states,
        mediation_states=states_by_time,
        ini=ini,
        upd=updates_by_time,
        evl=evaluations_by_time,
    )


# ------------------------------------------------------------------
# Counts
# ------------------------------------------------------------------


@mvr_operator_fn(arity=1)
def mvr_count(
    mvrs: list[MVR],
    condition: str,
) -> MVR:
    """Apply ``condition`` to the count of greedy, non-overlapping acceptances."""
    mvr = mvrs[0]

    if not isinstance(mvr, (HomMVR, InhomMVR)):
        raise InvalidInputError("mvr_count expects a HomMVR or InhomMVR.")

    if not isinstance(condition, str):
        raise InvalidInputError("condition must be a string.")

    condition = condition.strip()

    def count_range_mvr(
        lower: int,
        upper: int,
    ) -> MVR:
        """
        Constructs the count-range MVR for counts in range [lower, upper].
        """
        if lower < 0 or upper < 0:
            raise InvalidInputError("count bounds must be nonnegative.")

        if upper < lower:
            raise InvalidInputError("count range must be nonempty.")

        fail_state = ("__count_fail__",)
        count_states = list(range(upper + 1))

        # Homogeneous case
        if isinstance(mvr, HomMVR):
            hidden_states = list(mvr.hidden_states)

            mediation_states = [
                (count, m) for count in count_states for m in mvr.mediation_states
            ] + [fail_state]

            ini = {}

            for h in hidden_states:
                m0 = mvr.ini[h]
                count0 = 1 if mvr.evl[m0] else 0

                if count0 > upper:
                    ini[h] = fail_state
                else:
                    ini[h] = (count0, m0)

            upd = {}

            for state_prev in mediation_states:
                for h_curr in hidden_states:
                    if state_prev == fail_state:
                        upd[(state_prev, h_curr)] = fail_state
                        continue

                    count_prev, m_prev = state_prev

                    # If the previous state was accepting, restart on the current symbol.
                    if mvr.evl[m_prev]:
                        m_curr = mvr.ini[h_curr]
                    else:
                        m_curr = mvr.upd[(m_prev, h_curr)]

                    count_curr = count_prev + 1 if mvr.evl[m_curr] else count_prev

                    if count_curr > upper:
                        upd[(state_prev, h_curr)] = fail_state
                    else:
                        upd[(state_prev, h_curr)] = (count_curr, m_curr)

            evl = {}

            for state in mediation_states:
                if state == fail_state:
                    evl[state] = False
                else:
                    count, _ = state
                    evl[state] = lower <= count <= upper

            return HomMVR(
                hidden_states=hidden_states,
                mediation_states=mediation_states,
                ini=ini,
                upd=upd,
                evl=evl,
            )

        # Inhomogeneous case.
        hidden_states = list(mvr.hidden_states)
        time_horizon = mvr.time_horizon

        mediation_states = []

        for t in range(time_horizon + 1):
            mediation_states_t = [
                (count, age, m)
                for count in count_states
                for age in range(t + 1)
                for m in mvr.mediation_states[age]
            ] + [fail_state]

            mediation_states.append(mediation_states_t)

        ini = {}

        for h in hidden_states:
            m0 = mvr.ini[h]
            count0 = 1 if mvr.evl[0][m0] else 0

            if count0 > upper:
                ini[h] = fail_state
            else:
                ini[h] = (count0, 0, m0)

        upd = []

        for t in range(time_horizon):
            upd_t = {}

            for state_prev in mediation_states[t]:
                for h_curr in hidden_states:
                    if state_prev == fail_state:
                        upd_t[(state_prev, h_curr)] = fail_state
                        continue

                    count_prev, age_prev, m_prev = state_prev

                    # If the previous local state was accepting, restart locally
                    # at age 0 on the current symbol.
                    if mvr.evl[age_prev][m_prev]:
                        age_curr = 0
                        m_curr = mvr.ini[h_curr]
                    else:
                        age_curr = age_prev + 1
                        m_curr = mvr.upd[age_prev][(m_prev, h_curr)]

                    count_curr = (
                        count_prev + 1 if mvr.evl[age_curr][m_curr] else count_prev
                    )

                    if count_curr > upper:
                        upd_t[(state_prev, h_curr)] = fail_state
                    else:
                        upd_t[(state_prev, h_curr)] = (
                            count_curr,
                            age_curr,
                            m_curr,
                        )

            upd.append(upd_t)

        evl = []

        for t in range(time_horizon + 1):
            evl_t = {}

            for state in mediation_states[t]:
                if state == fail_state:
                    evl_t[state] = False
                else:
                    count, _, _ = state
                    evl_t[state] = lower <= count <= upper

            evl.append(evl_t)

        return InhomMVR(
            hidden_states=hidden_states,
            mediation_states=mediation_states,
            ini=ini,
            upd=upd,
            evl=evl,
        )

    # Pattern 1: "k"
    exact_match = re.fullmatch(r"\d+", condition)

    if exact_match is not None:
        k = int(condition)
        return count_range_mvr(k, k)

    # Pattern 2: "[l,u]" or "(l,u]"
    range_match = re.fullmatch(
        r"([\[\(])\s*(\d+)\s*,\s*(\d+)\s*\]",
        condition,
    )

    if range_match is not None:
        left_bracket = range_match.group(1)
        left = int(range_match.group(2))
        upper = int(range_match.group(3))

        if left_bracket == "[":
            lower = left
        else:
            lower = left + 1

        return count_range_mvr(lower, upper)

    # Pattern 3: ">k", ">=k", "<k", "<=k"
    comparison_match = re.fullmatch(
        r"(>=|<=|>|<)\s*(\d+)",
        condition,
    )

    if comparison_match is not None:
        op = comparison_match.group(1)
        k = int(comparison_match.group(2))

        if op == "<=":
            return count_range_mvr(0, k)

        if op == "<":
            if k == 0:
                raise InvalidInputError(
                    'condition "<0" is degenerate because counts are nonnegative.'
                )

            return count_range_mvr(0, k - 1)

        if op == ">":
            return mvr_not(count_range_mvr(0, k))

        # op == ">="
        if k == 0:
            raise InvalidInputError(
                'condition ">=0" is degenerate because counts are nonnegative.'
            )

        return mvr_not(count_range_mvr(0, k - 1))

    raise InvalidInputError(
        f"invalid count condition {condition!r}; expected an exact count like '2', "
        "a range like '[1,3]' or '(1,3]', or an inequality like '<2', '<=2', '>2', '>=2'."
    )
