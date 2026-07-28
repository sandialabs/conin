import numpy as np

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

    @property
    def repn(self):
        return self.initialize()

    @repn.setter
    def repn(self, mvr_repn):
        self._repn = mvr_repn

    def initialize(self, avoid_reinitialization=True):
        """
        Converts the MVR into an integer-indexed NumPy array representation.
        """
        if avoid_reinitialization and getattr(self, "_repn", None) is not None:
            return self._repn

        init_array, update_array, eval_array = self._build_array_repn()

        self._repn = MVR_MatVecRepn(
            init_array=init_array,
            update_array=update_array,
            eval_array=eval_array,
        )

        return self._repn

    def _build_array_repn(self):
        raise NotImplementedError(
            "_build_array_repn must be implemented by subclasses."
        )


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
        initialize: bool = False,

    ):
        # Validation
        h_space = set(hidden_states)
        m_space = set(mediation_states)
        mh_space = set(product(m_space, h_space))

        #duplicates
        if len(hidden_states) != len(set(hidden_states)):
            raise InvalidInputError("hidden_states must not contain duplicates")
        
        if len(mediation_states) != len(set(mediation_states)):
            raise InvalidInputError("mediation_states must not contain duplicates")
        
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
        
        self._repn = None

        if initialize:
            self.initialize(True)

    def _build_array_repn(self):
        """
        Builds integer-indexed NumPy arrays for a homogeneous MVR.

        init_array[h, m]
        update_array[h_curr, m_curr, m_prev]
        eval_array[m]
        """
        hidden_to_internal = {
            h: i for i, h in enumerate(self.hidden_states)
        }
        mediation_to_internal = {
            m: i for i, m in enumerate(self.mediation_states)
        }

        H = len(self.hidden_states)
        M = len(self.mediation_states)

        init_array = np.zeros((H, M), dtype=float)
        update_array = np.zeros((H, M, M), dtype=float)
        eval_array = np.zeros((M,), dtype=float)

        # init_array[h, m]
        for h in self.hidden_states:
            h_idx = hidden_to_internal[h]
            m = self.ini[h]
            m_idx = mediation_to_internal[m]
            init_array[h_idx, m_idx] = 1.0

        # update_array[h_curr, m_curr, m_prev]
        for h_curr in self.hidden_states:
            h_idx = hidden_to_internal[h_curr]

            for m_prev in self.mediation_states:
                m_prev_idx = mediation_to_internal[m_prev]

                m_curr = self.upd[(m_prev, h_curr)]
                m_curr_idx = mediation_to_internal[m_curr]

                update_array[h_idx, m_curr_idx, m_prev_idx] = 1.0

        # eval_array[m]
        for m in self.mediation_states:
            m_idx = mediation_to_internal[m]
            eval_array[m_idx] = float(self.evl[m])

        return init_array, update_array, eval_array

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
        initialize: bool = False,

    ):
        # Validation
        h_space = set(hidden_states)
        time_horizon = len(mediation_states)

        #duplicates in hidden space
        if len(hidden_states) != len(set(hidden_states)):
            raise InvalidInputError("hidden_states must not contain duplicates")

        if time_horizon == 0:
            raise InvalidInputError("mediation_states must be nonempty")

        if time_horizon != len(evl):
            raise InvalidInputError("evl and mediation_states must be the same length")

        if len(upd) != time_horizon - 1:
            raise InvalidInputError(
                f"upd length {len(upd)} must be one less than mediation_states length {time_horizon}"
            )

        for t, m_space in enumerate(mediation_states):
            #duplicates in mediation space
            if len(m_space) != len(set(m_space)):
                raise InvalidInputError(
                    f"mediation_states at time {t} must not contain duplicates"
                )
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
        
        self._repn = None

        if initialize:
            self.initialize(True)

    def _build_array_repn(self):
        """
        Builds integer-indexed NumPy arrays for an inhomogeneous MVR.

        Returns
        -------
        init_array : np.ndarray
            Shape (H, M_0)

        update_array : list[np.ndarray]
            update_array[t] has shape (H, M_{t+1}, M_t)

            update_array[t][h_curr, m_curr, m_prev] = 1 iff

                upd[t][(m_prev, h_curr)] == m_curr

        eval_array : list[np.ndarray]
            eval_array[t] has shape (M_t,)
        """
        hidden_to_internal = {
            h: i for i, h in enumerate(self.hidden_states)
        }

        mediation_to_internal = [
            {m: i for i, m in enumerate(m_states_t)}
            for m_states_t in self.mediation_states
        ]

        H = len(self.hidden_states)
        T = self.time_horizon

        # ------------------------------------------------------------
        # init_array[h, m_0]
        # ------------------------------------------------------------
        M0 = len(self.mediation_states[0])
        init_array = np.zeros((H, M0), dtype=float)

        for h in self.hidden_states:
            h_idx = hidden_to_internal[h]

            m0 = self.ini[h]
            m0_idx = mediation_to_internal[0][m0]

            init_array[h_idx, m0_idx] = 1.0

        # ------------------------------------------------------------
        # eval_array[t][m_t]
        # ------------------------------------------------------------
        eval_array = []

        for t in range(T):
            Mt = len(self.mediation_states[t])
            eval_t = np.zeros((Mt,), dtype=float)

            for m in self.mediation_states[t]:
                m_idx = mediation_to_internal[t][m]
                eval_t[m_idx] = float(self.evl[t][m])

            eval_array.append(eval_t)

        # ------------------------------------------------------------
        # update_array[t][h_curr, m_curr, m_prev]
        #
        # t indexes the transition from time t to time t + 1.
        # ------------------------------------------------------------
        update_array = []

        for t in range(T - 1):
            M_prev = len(self.mediation_states[t])
            M_curr = len(self.mediation_states[t + 1])

            update_t = np.zeros((H, M_curr, M_prev), dtype=float)

            for h_curr in self.hidden_states:
                h_idx = hidden_to_internal[h_curr]

                for m_prev in self.mediation_states[t]:
                    m_prev_idx = mediation_to_internal[t][m_prev]

                    m_curr = self.upd[t][(m_prev, h_curr)]
                    m_curr_idx = mediation_to_internal[t + 1][m_curr]

                    update_t[h_idx, m_curr_idx, m_prev_idx] = 1.0

            update_array.append(update_t)

        return init_array, update_array, eval_array

class MVR_MatVecRepn:
    """
    Integer-indexed NumPy array representation of an MVR, analgous to HMM_MatVecRepn
    Handles both homogenoues and inhomogeneous cases.

    Homogeneous case
    ----------------
    init_array:
        shape (H, M)

    update_array:
        shape (H, M, M)

    eval_array:
        shape (M,)

    Inhomogeneous case
    ------------------
    init_array:
        shape (H, M_0)

    update_array:
        list of arrays where update_array[t] has shape (H, M_{t+1}, M_t)

    eval_array:
        list of arrays where eval_array[t] has shape (M_t,)
    """

    def __init__(
        self,
        *,
        init_array,
        update_array,
        eval_array,
        check_errors=True,
    ):
        self.load_init_array(init_array, check_errors=check_errors)
        self.load_update_array(update_array, check_errors=check_errors)
        self.load_eval_array(eval_array, check_errors=check_errors)

        if check_errors:
            self.check_dimensions()

        self.load_dimensions()

    def load_init_array(self, init_array, check_errors=True):
        """
        Loads the MVR initialization array.

        Homogeneous:
            init_array[h, m] = 1 iff ini[h] = m

        Inhomogeneous:
            init_array[h, m_0] = 1 iff ini[h] = m_0
        """
        init_array = np.asarray(init_array)

        if check_errors:
            if init_array.ndim != 2:
                raise InvalidInputError("init_array must be a 2D array.")

            if not np.all(init_array >= 0):
                raise InvalidInputError("init_array entries must be nonnegative.")

            if not np.all(np.isclose(init_array, 0) | np.isclose(init_array, 1)):
                raise InvalidInputError("init_array entries must be binary.")

            row_sums = init_array.sum(axis=1)
            if not np.all(np.isclose(row_sums, 1)):
                raise InvalidInputError("init_array rows must sum to 1.")

        self.init_array = init_array

    def load_update_array(self, update_array, check_errors=True):
        """
        Loads the MVR update array.

        Homogeneous:
            update_array[h_curr, m_curr, m_prev]

        Inhomogeneous:
            update_array[t][h_curr, m_curr, m_prev]
        """
        if isinstance(update_array, list):
            update_array = [np.asarray(arr) for arr in update_array]
        else:
            update_array = np.asarray(update_array)

        if check_errors:
            update_arrays = (
                update_array if isinstance(update_array, list) else [update_array]
            )

            for t, arr in enumerate(update_arrays):
                if arr.ndim != 3:
                    raise InvalidInputError(
                        f"update_array at index {t} must be a 3D array."
                    )

                if not np.all(arr >= 0):
                    raise InvalidInputError(
                        f"update_array at index {t} entries must be nonnegative."
                    )

                if not np.all(np.isclose(arr, 0) | np.isclose(arr, 1)):
                    raise InvalidInputError(
                        f"update_array at index {t} entries must be binary."
                    )

                # For every h_curr and m_prev, exactly one m_curr is selected.
                sums = arr.sum(axis=1)
                if not np.all(np.isclose(sums, 1)):
                    raise InvalidInputError(
                        f"update_array at index {t} must sum to 1 over the current mediation axis."
                    )

        self.update_array = update_array

    def load_eval_array(self, eval_array, check_errors=True):
        """
        Loads the MVR evaluation array.

        Homogeneous:
            eval_array[m]

        Inhomogeneous:
            eval_array[t][m_t]
        """
        if isinstance(eval_array, list):
            eval_array = [np.asarray(arr) for arr in eval_array]
        else:
            eval_array = np.asarray(eval_array)

        if check_errors:
            eval_arrays = eval_array if isinstance(eval_array, list) else [eval_array]

            for t, arr in enumerate(eval_arrays):
                if arr.ndim != 1:
                    raise InvalidInputError(
                        f"eval_array at index {t} must be a 1D array."
                    )

                if not np.all(arr >= 0):
                    raise InvalidInputError(
                        f"eval_array at index {t} entries must be nonnegative."
                    )

                if not np.all(np.isclose(arr, 0) | np.isclose(arr, 1)):
                    raise InvalidInputError(
                        f"eval_array at index {t} entries must be binary."
                    )

        self.eval_array = eval_array

    def check_dimensions(self):
        """
        Checks that init_array, update_array, and eval_array have compatible dimensions.
        """
        if isinstance(self.eval_array, list) != isinstance(self.update_array, list):
            raise InvalidInputError(
                "eval_array and update_array must either both be lists or both be arrays."
            )

        H_init, M_init = self.init_array.shape

        # Homogeneous case
        if not isinstance(self.eval_array, list):
            H_update, M_curr, M_prev = self.update_array.shape
            M_eval = self.eval_array.shape[0]

            if H_update != H_init:
                raise InvalidInputError(
                    "update_array hidden dimension must match init_array hidden dimension."
                )

            if M_eval != M_init:
                raise InvalidInputError(
                    "eval_array mediation dimension must match init_array mediation dimension."
                )

            if M_curr != M_init or M_prev != M_init:
                raise InvalidInputError(
                    "homogeneous update_array mediation dimensions must both match init_array mediation dimension."
                )

            return

        # Inhomogeneous case
        if len(self.eval_array) == 0:
            raise InvalidInputError("eval_array list must be nonempty.")

        if len(self.update_array) != len(self.eval_array) - 1:
            raise InvalidInputError(
                "update_array list length must be one less than eval_array list length."
            )

        M_eval_0 = self.eval_array[0].shape[0]

        if M_eval_0 != M_init:
            raise InvalidInputError(
                "eval_array[0] mediation dimension must match init_array mediation dimension."
            )

        for t, update_t in enumerate(self.update_array):
            H_update_t, M_curr, M_prev = update_t.shape

            if H_update_t != H_init:
                raise InvalidInputError(
                    f"update_array[{t}] hidden dimension must match init_array hidden dimension."
                )

            M_eval_prev = self.eval_array[t].shape[0]
            M_eval_curr = self.eval_array[t + 1].shape[0]

            if M_prev != M_eval_prev:
                raise InvalidInputError(
                    f"update_array[{t}] previous mediation dimension must match eval_array[{t}]."
                )

            if M_curr != M_eval_curr:
                raise InvalidInputError(
                    f"update_array[{t}] current mediation dimension must match eval_array[{t + 1}]."
                )

    def load_dimensions(self):
        """
        Updates dimension fields and integer-indexed state spaces.
        """
        self.num_hidden_states = self.init_array.shape[0]
        self.hidden_states = range(self.num_hidden_states)

        # Homogeneous case
        if not isinstance(self.eval_array, list):
            self.num_mediation_states = self.init_array.shape[1]
            self.mediation_states = range(self.num_mediation_states)
            return

        # Inhomogeneous case
        self.time_horizon = len(self.eval_array)
        self.num_mediation_states = [
            arr.shape[0] for arr in self.eval_array
        ]
        self.mediation_states = [
            range(num_states) for num_states in self.num_mediation_states
        ]