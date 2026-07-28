from collections.abc import Callable, Iterable
from itertools import combinations, product
from typing import Any

from conin.exceptions import InvalidInputError

# Adjust this import path as needed.
from conin.mvr import HomMVR, InhomMVR

MVR = HomMVR | InhomMVR


# ------------------------------------------------------------------
# Boolean Operators: AND/OR/NOT
# ------------------------------------------------------------------


def _common_hidden_states(mvrs: list[MVR]) -> list[Any]:
    """
    Checks that all MVRs have the same hidden state space.
    NOTE: the returned ordering is taken from mvrs[0].
    """
    hidden_states = list(mvrs[0].hidden_states)
    hidden_space = set(hidden_states)

    for i, mvr in enumerate(mvrs[1:], start=1):
        if set(mvr.hidden_states) != hidden_space:
            raise InvalidInputError(
                f"mvrs[{i}] must have the same hidden_states as mvrs[0]."
            )

    return hidden_states


def _combined_time_horizon(mvrs: list[MVR]) -> int | None:
    """
    Returns None if all MVRs are homogeneous.
    Otherwise returns the minimum time_horizon among the inhomogeneous MVRs.
    """
    horizons = [mvr.time_horizon for mvr in mvrs if isinstance(mvr, InhomMVR)]

    if len(horizons) == 0:
        return None

    min_horizon = min(horizons)

    print(f"Constructing inhomogeneous MVR using minimum time_horizon {min_horizon}")

    return min_horizon


def _boolean_combine_mvrs(
    mvrs: list[MVR],
    bool_reducer: Callable[[list[bool]], bool],
    *,
    initialize: bool = False,
) -> MVR:
    """
    IN:
        list of HomMVR and InhomMVR
        bool_reducer: generate function to aggregating a collection of booleans into a single value. ie. AND/OR/NOT
    OUT:
        single MVR, either HomMVR or InhomMVR depending on inputs

    Generic product construction for logical combinations of MVRs.
    Handles time-inhomogeneous MVRs by creating another time-inhomogeneous MVR up to the minimum of the individual time horizons.
    For mixes of inhomogeneous and homogeneous MVRs, homogeneous MVRs are treated as having infinite time horizon.
    """
    if len(mvrs) == 0:
        raise InvalidInputError("mvrs must be a nonempty iterable of MVRs.")

    for i, mvr in enumerate(mvrs):
        if not isinstance(mvr, (HomMVR, InhomMVR)):
            raise InvalidInputError(
                f"mvrs[{i}] must be an instance of HomMVR or InhomMVR."
            )

    hidden_states = _common_hidden_states(mvrs)
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
            initialize=initialize,
        )

    # Inhomogeneous case: at least one input is InhomMVR.
    mediation_states = []

    for t in range(time_horizon):
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

    for t in range(time_horizon - 1):
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

    for t in range(time_horizon):
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
        initialize=initialize,
    )


def mvr_and(
    mvrs: list[MVR],
    *,
    initialize: bool = False,
) -> MVR:
    """
    Constructs the logical AND of a nonempty iterable of MVRs.

    The resulting MVR evaluates true exactly when every input MVR evaluates true.
    """
    print(
        "Use the AND operator sparingly. It's more efficient to provide a list of MVRs to downstream algorithms"
    )
    return _boolean_combine_mvrs(
        mvrs,
        all,
        initialize=initialize,
    )


def mvr_or(
    mvrs: list[MVR],
    *,
    initialize: bool = False,
) -> MVR:
    """
    Constructs the logical OR of a nonempty iterable of MVRs.

    The resulting MVR evaluates true exactly when at least one input MVR
    evaluates true.
    """
    return _boolean_combine_mvrs(
        mvrs,
        any,
        initialize=initialize,
    )


def mvr_not(
    mvr: MVR,
    *,
    initialize: bool = False,
) -> MVR:
    """
    Constructs the logical NOT/complement of a single MVR. Creates a copy with eval reversed.
    """
    if isinstance(mvr, HomMVR):
        return HomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=list(mvr.mediation_states),
            ini=dict(mvr.ini),
            upd=dict(mvr.upd),
            evl={m: not value for m, value in mvr.evl.items()},
            initialize=initialize,
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
            initialize=initialize,
        )

    raise InvalidInputError("mvr must be an instance of HomMVR or InhomMVR.")


# ------------------------------------------------------------------
# Not Yet/ Already Satisfied
# ------------------------------------------------------------------


def mvr_not_yet(
    mvr: MVR,
    *,
    initialize: bool = False,
) -> MVR:
    """
    IN/OUT:
        single MVR

    Constructs the 'not yet satisfied' MVR.
    At time t, checks if the constraint has never been satisfied at times up to time t.
    """
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
            initialize=initialize,
        )

    if isinstance(mvr, InhomMVR):
        mediation_states = []

        for t in range(mvr.time_horizon):
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

        for t in range(mvr.time_horizon - 1):
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

        for t in range(mvr.time_horizon):
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
            initialize=initialize,
        )

    raise InvalidInputError("mvr must be an instance of HomMVR or InhomMVR.")


def mvr_already_satisfied(
    mvr: MVR,
    *,
    initialize: bool = False,
) -> MVR:
    """
    IN/OUT:
        single MVR

    Constructs the 'already satisfied' MVR.
    At time t, checks if the constraint has already satisfied at some time up to t.
    """
    return mvr_not(
        mvr_not_yet(
            mvr,
            initialize=False,
        ),
        initialize=initialize,
    )


# ------------------------------------------------------------------
# Sat_Time: Prefix-Free
# ------------------------------------------------------------------


def mvr_sattime(
    mvr: MVR,
    *,
    initialize: bool = False,
) -> MVR:
    """
    IN/OUT:
        single MVR

    Constructs the satisfaction-time MVR. Essentially create a prefix-free version of the constraint.
    As soon as we hit an accepint state (any m where evl(m) = True) we transtiion to an absorbing fail state.
    """
    if not isinstance(mvr, (HomMVR, InhomMVR)):
        raise InvalidInputError("mvr_sattime expects a HomMVR or InhomMVR.")

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

        return HomMVR(
            hidden_states=hidden_states,
            mediation_states=mediation_states,
            ini=ini,
            upd=upd,
            evl=evl,
            initialize=initialize,
        )

    # Inhomogeneous case
    hidden_states = list(mvr.hidden_states)

    mediation_states = [
        list(mediation_states_t) + [fail_state]
        for mediation_states_t in mvr.mediation_states
    ]

    ini = dict(mvr.ini)

    upd = []

    for t in range(mvr.time_horizon - 1):
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

    for t in range(mvr.time_horizon):
        evl_t = {m: mvr.evl[t][m] for m in mvr.mediation_states[t]}
        evl_t[fail_state] = False

        evl.append(evl_t)

    return InhomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
        initialize=initialize,
    )


# ------------------------------------------------------------------
# Regular Language Operators: HomMVR ONLY
# ------------------------------------------------------------------


def _powerset_frozensets(states: list[Any]) -> list[frozenset[Any]]:
    """
    Returns the powerset of states as a list of frozensets.
    """
    return [
        frozenset(subset)
        for r in range(len(states) + 1)
        for subset in combinations(states, r)
    ]


def mvr_setdiff(
    mvrs: list[HomMVR],
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs the set difference of two homogeneous MVRs.

    mvr_setdiff([mvr1, mvr2]) recognizes
        L(mvr1) \\ L(mvr2)
    """
    if len(mvrs) != 2 or not all(isinstance(mvr, HomMVR) for mvr in mvrs):
        raise InvalidInputError("mvr_setdiff expects a list of exactly two HomMVRs.")

    mvr1, mvr2 = mvrs

    if set(mvr1.hidden_states) != set(mvr2.hidden_states):
        raise InvalidInputError("mvr_setdiff inputs must have the same hidden_states.")

    hidden_states = list(mvr1.hidden_states)

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

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
        initialize=initialize,
    )


def mvr_concatenate(
    mvrs: list[HomMVR],
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs the (non-empty) concatenation of two homogeneous MVRs.

    mvr_concatenate([mvr1, mvr2]) recognizes

        L(mvr1) L(mvr2)

    We assume neither mvr accepts the empty string.
    """
    if len(mvrs) != 2 or not all(isinstance(mvr, HomMVR) for mvr in mvrs):
        raise InvalidInputError(
            "mvr_concatenate expects a list of exactly two HomMVRs."
        )

    mvr1, mvr2 = mvrs

    if set(mvr1.hidden_states) != set(mvr2.hidden_states):
        raise InvalidInputError(
            "mvr_concatenate inputs must have the same hidden_states."
        )

    hidden_states = list(mvr1.hidden_states)

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

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
        initialize=initialize,
    )


def mvr_concatenate_prefix(
    mvrs: list[HomMVR],
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs a prefix-free concatenation of two homogeneous MVRs.
    First applies mvr_sattime to generate prefix-free versions, then concatenates the without using a powerset construction.

    mvr_concatenate_prefix([mvr1, mvr2]) recognizes approximately

        sattime(L(mvr1)) sattime(L(mvr2))
    """
    if (
        not isinstance(mvrs, list)
        or len(mvrs) != 2
        or not all(isinstance(mvr, HomMVR) for mvr in mvrs)
    ):
        raise InvalidInputError(
            "mvr_concatenate_prefix expects a list of exactly two HomMVRs."
        )

    mvr1, mvr2 = mvrs

    if set(mvr1.hidden_states) != set(mvr2.hidden_states):
        raise InvalidInputError(
            "mvr_concatenate_prefix inputs must have the same hidden_states."
        )

    mvr1 = mvr_sattime(mvr1, initialize=False)
    mvr2 = mvr_sattime(mvr2, initialize=False)

    hidden_states = list(mvr1.hidden_states)

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

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
        initialize=initialize,
    )


def mvr_kfold_product(
    mvr: HomMVR,
    k: int,
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs the k-fold concatenation/product of a homogeneous MVR with itself.

    The resulting MVR recognizes

        L(mvr)^k

    for k >= 1.

    This is implemented recursively using mvr_concatenate.
    """
    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kfold_product expects a HomMVR.")

    if not isinstance(k, int) or k < 1:
        raise InvalidInputError("k must be an integer greater than or equal to 1.")

    if k == 1:
        return HomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=list(mvr.mediation_states),
            ini=dict(mvr.ini),
            upd=dict(mvr.upd),
            evl=dict(mvr.evl),
            initialize=initialize,
        )

    return mvr_concatenate(
        [
            mvr_kfold_product(
                mvr,
                k - 1,
                initialize=False,
            ),
            mvr,
        ],
        initialize=initialize,
    )


def mvr_kfold_product_prefix(
    mvr: HomMVR,
    k: int,
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs the k-fold prefix concatenation/product of a homogeneous MVR.

    Calls mvr_concatenate_prefix and avoids the powerset construction.

    For k >= 1, this recognizes approximately

        sattime(L(mvr))^k
    """
    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kfold_product_prefix expects a HomMVR.")

    if not isinstance(k, int) or k < 1:
        raise InvalidInputError("k must be an integer greater than or equal to 1.")

    if k == 1:
        return mvr_sattime(
            mvr,
            initialize=initialize,
        )

    return mvr_concatenate_prefix(
        [
            mvr_kfold_product_prefix(
                mvr,
                k - 1,
                initialize=False,
            ),
            mvr,
        ],
        initialize=initialize,
    )


def mvr_kleene_closure(
    mvr: HomMVR,
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs the Kleene closure of a homogeneous MVR, excluding the empty string
    """
    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kleene_closure expects a HomMVR.")

    hidden_states = list(mvr.hidden_states)

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
        initialize=initialize,
    )


def mvr_kleene_closure_prefix(
    mvr: HomMVR,
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs a prefix-free (nonempty) Kleene closure of a homogeneous MVR.

    First applies mvr_sattime to the input MVR.
    As resulting MVR is prefix-free, the Kleene construction avoid the powerset construction, only tracking the active segment.

    This recognizes approximately

        sattime(L(mvr))^+
    """
    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_kleene_closure_prefix expects a HomMVR.")

    mvr = mvr_sattime(
        mvr,
        initialize=False,
    )

    hidden_states = list(mvr.hidden_states)
    mediation_states = list(mvr.mediation_states)

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
        initialize=initialize,
    )


def mvr_reverse(
    mvr: HomMVR,
    *,
    initialize: bool = False,
) -> HomMVR:
    """
    Constructs the non-emtpy reversal of a homogeneous MVR.

    If the input MVR recognizes L, the resulting MVR recognizes

        reverse(L)
    """
    if not isinstance(mvr, HomMVR):
        raise InvalidInputError("mvr_reverse expects a HomMVR.")

    hidden_states = list(mvr.hidden_states)
    original_mediation_states = list(mvr.mediation_states)

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
        initialize=initialize,
    )
