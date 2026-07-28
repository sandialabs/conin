from collections.abc import Callable, Iterable
from itertools import product
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
    horizons = [
        mvr.time_horizon
        for mvr in mvrs
        if isinstance(mvr, InhomMVR)
    ]

    if len(horizons) == 0:
        return None

    min_horizon = min(horizons)

    print(
        f"Constructing inhomogeneous MVR using minimum time_horizon {min_horizon}"
    )

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

    # ------------------------------------------------------------------
    # Homogeneous case: all inputs are HomMVR.
    # ------------------------------------------------------------------
    if time_horizon is None:
        component_mediation_spaces = [
            list(mvr.mediation_states)
            for mvr in mvrs
        ]

        mediation_states = list(product(*component_mediation_spaces))

        ini = {
            h: tuple(
                mvr.ini[h]
                for mvr in mvrs
            )
            for h in hidden_states
        }

        upd = {}

        for m_prev_tuple in mediation_states:
            for h_curr in hidden_states:
                m_curr_tuple = tuple(
                    mvrs[i].upd[(m_prev_tuple[i], h_curr)]
                    for i in range(num_mvrs)
                )

                upd[(m_prev_tuple, h_curr)] = m_curr_tuple

        evl = {}

        for m_tuple in mediation_states:
            component_values = [
                mvrs[i].evl[m_tuple[i]]
                for i in range(num_mvrs)
            ]

            evl[m_tuple] = bool(bool_reducer(component_values))

        return HomMVR(
            hidden_states=hidden_states,
            mediation_states=mediation_states,
            ini=ini,
            upd=upd,
            evl=evl,
            initialize=initialize,
        )

    # ------------------------------------------------------------------
    # Inhomogeneous case: at least one input is InhomMVR.
    # ------------------------------------------------------------------
    mediation_states = []

    for t in range(time_horizon):
        component_mediation_spaces_t = [
            list(mvr.mediation_states)
            if isinstance(mvr, HomMVR)
            else list(mvr.mediation_states[t])
            for mvr in mvrs
        ]

        mediation_states_t = list(product(*component_mediation_spaces_t))
        mediation_states.append(mediation_states_t)

    ini = {
        h: tuple(
            mvr.ini[h]
            for mvr in mvrs
        )
        for h in hidden_states
    }

    upd = []

    for t in range(time_horizon - 1):
        upd_t = {}

        for m_prev_tuple in mediation_states[t]:
            for h_curr in hidden_states:
                m_curr_tuple = tuple(
                    mvrs[i].upd[(m_prev_tuple[i], h_curr)]
                    if isinstance(mvrs[i], HomMVR)
                    else mvrs[i].upd[t][(m_prev_tuple[i], h_curr)]
                    for i in range(num_mvrs)
                )

                upd_t[(m_prev_tuple, h_curr)] = m_curr_tuple

        upd.append(upd_t)

    evl = []

    for t in range(time_horizon):
        evl_t = {}

        for m_tuple in mediation_states[t]:
            component_values = [
                mvrs[i].evl[m_tuple[i]]
                if isinstance(mvrs[i], HomMVR)
                else mvrs[i].evl[t][m_tuple[i]]
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
    print("Use the AND operator sparingly. It's more efficient to provide a list of MVRs to downstream algorithms")
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
            evl={
                m: not value
                for m, value in mvr.evl.items()
            },
            initialize=initialize,
        )

    if isinstance(mvr, InhomMVR):
        return InhomMVR(
            hidden_states=list(mvr.hidden_states),
            mediation_states=[
                list(mediation_states_t)
                for mediation_states_t in mvr.mediation_states
            ],
            ini=dict(mvr.ini),
            upd=[
                dict(upd_t)
                for upd_t in mvr.upd
            ],
            evl=[
                {
                    m: not value
                    for m, value in evl_t.items()
                }
                for evl_t in mvr.evl
            ],
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
                    not_yet_flag_curr = (
                        not_yet_flag_prev
                        and not mvr.evl[m_curr]
                    )

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
                            not_yet_flag_prev
                            and not mvr.evl[t + 1][m_curr]
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