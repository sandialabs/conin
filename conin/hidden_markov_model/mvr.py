from conin.exceptions import InvalidInputError
from typing import Any
from itertools import product


class BaseMVR:
    """
    Base class for a mediation variable representation (MVR).

    An MVR consists of the following data:
    1. X, the hidden space of the HMM
    2. M_t, the state space of the MVR at time t
    3. ini: X -> M. initialization
    4. upd_t: M x X -> M. update at time t.
    5. evl_t: M -> 0/1. eval at time t.

    In the case where M and the above maps are constant over time, an MVR is a DFA.
    """

    hidden_states: Any
    mediation_states: Any
    ini: Any
    upd: Any
    evl: Any


class HomMVR(BaseMVR):
    """
    Class for time-homogeneous MVRs, i.e. DFAs.
    """

    def __init__(
        self,
        *,
        hidden_states: list[str],
        mediation_states: list[str],
        ini: dict[str, str],
        upd: dict[tuple[str, str], str],
        evl: dict[str, bool],
    ):
        # Validation
        h_space = set(hidden_states)
        m_space = set(mediation_states)
        mh_space = set(product(m_space, h_space))

        # ini
        if h_space != set(ini.keys()):
            raise InvalidInputError("domain(keys) of ini must match hidden_states")

        if not set(ini.values()) <= m_space:
            raise InvalidInputError(
                "range(values) of ini must be contained in mediation_states"
            )

        # upd
        if mh_space != set(upd.keys()):
            raise InvalidInputError(
                "domain(keys) of upd must match mediation_states x hidden_states"
            )

        if not set(upd.values()) <= m_space:
            raise InvalidInputError(
                "range(values) of upd must be contained in mediation_states"
            )

        # evl
        if m_space != set(evl.keys()):
            raise InvalidInputError("domain(keys) of evl must match mediation_states")

        if not all(isinstance(v, bool) for v in evl.values()):
            raise InvalidInputError("range(values) of evl must be boolean")

        self.hidden_states = hidden_states
        self.mediation_states = mediation_states
        self.ini = ini
        self.upd = upd
        self.evl = evl


class InhomMVR(BaseMVR):
    """
    Class for time-inhomogeneous MVRs.

    mediation_states, upd, and evl are now lists of their time-homogeneous counterparts.
    For example, mediation_states is a list of lists, each sublist the mediation space at that time.
    """

    def __init__(
        self,
        *,
        hidden_states: list[str],
        mediation_states: list[list[str]],
        ini: dict[str, str],
        upd: list[dict[tuple[str, str], str]],
        evl: list[dict[str, bool]],
    ):
        # Validation
        h_space = set(hidden_states)
        time_horizon = len(mediation_states)

        if time_horizon == 0:
            raise InvalidInputError("mediation_states must be nonempty")

        if time_horizon != len(evl):
            raise InvalidInputError("evl and mediation_states must be the same length")

        if len(upd) < time_horizon - 1:
            raise InvalidInputError(
                f"upd length {len(upd)} must be one less than mediation_states length {time_horizon}"
            )

        for t, m_space in enumerate(mediation_states):
            m_space = set(m_space)

            # ini
            if t == 0:
                if h_space != set(ini.keys()):
                    raise InvalidInputError(
                        "domain(keys) of ini must match hidden_states"
                    )

                if not set(ini.values()) <= m_space:
                    raise InvalidInputError(
                        "range(values) of ini must be contained in mediation_states at time 0"
                    )

            # upd
            if t > 0:
                mh_space = set(product(m_space_prev, h_space))
                upd_t_minus_1 = upd[t - 1]

                if mh_space != set(upd_t_minus_1.keys()):
                    raise InvalidInputError(
                        f"domain(keys) of upd at time {t - 1} must match mediation_states x hidden_states"
                    )

                if not set(upd_t_minus_1.values()) <= m_space:
                    raise InvalidInputError(
                        f"range(values) of upd at time {t - 1} must be contained in mediation_states"
                    )

            # evl
            evl_t = evl[t]

            if m_space != set(evl_t.keys()):
                raise InvalidInputError(
                    f"domain(keys) of evl at time {t} must match mediation_states"
                )

            if not all(isinstance(v, bool) for v in evl_t.values()):
                raise InvalidInputError(
                    f"range(values) of evl at time {t} must be boolean"
                )

            m_space_prev = m_space

        self.hidden_states = hidden_states
        self.mediation_states = mediation_states
        self.ini = ini
        self.upd = upd
        self.evl = evl
        self.time_horizon = time_horizon
