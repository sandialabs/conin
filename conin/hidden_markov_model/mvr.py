from __future__ import annotations

import numpy as np

from conin.exceptions import InvalidInputError
from typing import Any
from itertools import product


class BaseMVR:
    """Base class for mediation variable representations (MVRs).

    An MVR augments a hidden Markov model with an auxiliary state process used to
    track feasibility conditions over hidden-state sequences.
    """

    hidden_states: Any
    mediation_states: Any
    ini: Any
    upd: Any
    evl: Any

    _repn: MVR_MatVecRepn  # numerical representation like HMM_MatVecRepn
    _prefix: bool  # prefix-free tag.

    def __init__(self):
        """Initialize the base MVR state."""
        self._repn = None
        self._prefix = False

    @property
    def prefix(self):
        """Return whether the MVR is marked as prefix-free.

        Returns
        -------
        bool
            Prefix-free tag for the representation.
        """
        return self._prefix

    @property
    def repn(self):
        """Return the numeric array representation of the MVR.

        Returns
        -------
        MVR_MatVecRepn
            Integer-indexed array representation of the MVR.
        """
        if self._repn is None:
            self.initialize()
        return self._repn

    def initialize(self):
        """Convert the MVR into an integer-indexed NumPy array representation.

        Returns
        -------
        MVR_MatVecRepn
            Integer-indexed array representation of the MVR.
        """
        if getattr(self, "_repn", None) is not None:
            return self._repn

        ini_array, upd_array, evl_array = self._build_array_repn()

        self._repn = MVR_MatVecRepn(
            ini_array=ini_array,
            upd_array=upd_array,
            evl_array=evl_array,
        )

        return self._repn

    def _build_array_repn(self):
        raise NotImplementedError(
            "_build_array_repn must be implemented by subclasses."
        )


class HomMVR(BaseMVR):
    """Time-homogeneous mediation variable representation."""

    def __init__(
        self,
        *,
        hidden_states: list[Any],
        mediation_states: list[Any],
        ini: dict[Any, Any],
        upd: dict[tuple[Any, Any], Any],
        evl: dict[Any, bool],
    ):
        # Validation
        h_space = set(hidden_states)
        m_space = set(mediation_states)
        mh_space = set(product(m_space, h_space))

        # duplicates
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

        super().__init__()
        self.hidden_states = list(hidden_states)
        self.mediation_states = list(mediation_states)
        self.ini = dict(ini)
        self.upd = dict(upd)
        self.evl = dict(evl)
        # Build repn
        self.initialize()

    def _build_array_repn(self):
        """Build integer-indexed NumPy arrays for a homogeneous MVR.

        Returns
        -------
        tuple
            Tuple ``(ini_array, upd_array, evl_array)`` describing the homogeneous
            MVR.
        """
        hidden_to_internal = {h: i for i, h in enumerate(self.hidden_states)}
        mediation_to_internal = {m: i for i, m in enumerate(self.mediation_states)}

        H = len(self.hidden_states)
        M = len(self.mediation_states)

        ini_array = np.zeros((H, M), dtype=float)
        upd_array = np.zeros((H, M, M), dtype=float)
        evl_array = np.zeros((M,), dtype=float)

        # ini_array[h, m]
        for h in self.hidden_states:
            h_idx = hidden_to_internal[h]
            m = self.ini[h]
            m_idx = mediation_to_internal[m]
            ini_array[h_idx, m_idx] = 1.0

        # upd_array[h_curr, m_curr, m_prev]
        for h_curr in self.hidden_states:
            h_idx = hidden_to_internal[h_curr]

            for m_prev in self.mediation_states:
                m_prev_idx = mediation_to_internal[m_prev]

                m_curr = self.upd[(m_prev, h_curr)]
                m_curr_idx = mediation_to_internal[m_curr]

                upd_array[h_idx, m_curr_idx, m_prev_idx] = 1.0

        # evl_array[m]
        for m in self.mediation_states:
            m_idx = mediation_to_internal[m]
            evl_array[m_idx] = float(self.evl[m])

        return ini_array, upd_array, evl_array


class InhomMVR(BaseMVR):
    """Time-inhomogeneous mediation variable representation.

    ``mediation_states``, ``upd``, and ``evl`` vary over time instead of being
    constant across the horizon.
    """

    def __init__(
        self,
        *,
        hidden_states: list[Any],
        mediation_states: list[list[Any]],
        ini: dict[Any, Any],
        upd: list[dict[tuple[Any, Any], Any]],
        evl: list[dict[Any, bool]],
    ):
        # Validation
        h_space = set(hidden_states)
        time_horizon = len(mediation_states)

        # duplicates in hidden space
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

        m_space_prev = None
        for t, m_space in enumerate(mediation_states):
            # duplicates in mediation space
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

        super().__init__()
        self.hidden_states = list(hidden_states)
        self.mediation_states = [
            list(mediation_states_t) for mediation_states_t in mediation_states
        ]
        self.ini = dict(ini)
        self.upd = [dict(upd_t) for upd_t in upd]
        self.evl = [dict(evl_t) for evl_t in evl]
        self.time_horizon = time_horizon

        # Build repn
        self.initialize()

    def _build_array_repn(self):
        """Build integer-indexed NumPy arrays for an inhomogeneous MVR.

        Returns
        -------
        ini_array : np.ndarray
            Array of shape ``(H, M_0)``.
        upd_array : list[np.ndarray]
            ``upd_array[t]`` has shape ``(H, M_{t+1}, M_t)``.
        evl_array : list[np.ndarray]
            ``evl_array[t]`` has shape ``(M_t,)``.
        """
        hidden_to_internal = {h: i for i, h in enumerate(self.hidden_states)}

        mediation_to_internal = [
            {m: i for i, m in enumerate(m_states_t)}
            for m_states_t in self.mediation_states
        ]

        H = len(self.hidden_states)
        T = self.time_horizon

        # ini_array[h, m_0]
        M0 = len(self.mediation_states[0])
        ini_array = np.zeros((H, M0), dtype=float)

        for h in self.hidden_states:
            h_idx = hidden_to_internal[h]

            m0 = self.ini[h]
            m0_idx = mediation_to_internal[0][m0]

            ini_array[h_idx, m0_idx] = 1.0

        # evl_array[t][m_t]
        evl_array = []

        for t in range(T):
            Mt = len(self.mediation_states[t])
            eval_t = np.zeros((Mt,), dtype=float)

            for m in self.mediation_states[t]:
                m_idx = mediation_to_internal[t][m]
                eval_t[m_idx] = float(self.evl[t][m])

            evl_array.append(eval_t)

        # upd_array[t][h_curr, m_curr, m_prev]. t indexes the transition from time t to time t + 1.
        upd_array = []

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

            upd_array.append(update_t)

        return ini_array, upd_array, evl_array


class MVR_MatVecRepn:
    """Integer-indexed NumPy array representation of an MVR.

    Supports both homogeneous and inhomogeneous mediation variable
    representations.
    """

    def __init__(
        self,
        *,
        ini_array,
        upd_array,
        evl_array,
        check_errors=True,
    ):
        self.load_ini_array(ini_array, check_errors=check_errors)
        self.load_upd_array(upd_array, check_errors=check_errors)
        self.load_evl_array(evl_array, check_errors=check_errors)

        if check_errors:
            self.check_dimensions()

        self.load_dimensions()

    def load_ini_array(self, ini_array, check_errors=True):
        """Load the MVR initialization array.

        Parameters
        ----------
        ini_array : array-like
            Initialization array for the MVR.
        check_errors : bool, optional
            If ``True``, validate dimensions and binary structure.

        Raises
        ------
        InvalidInputError
            If ``ini_array`` has invalid shape or entries.
        """
        ini_array = np.asarray(ini_array)

        if check_errors:
            if ini_array.ndim != 2:
                raise InvalidInputError("ini_array must be a 2D array.")

            if not np.all(ini_array >= 0):
                raise InvalidInputError("ini_array entries must be nonnegative.")

            if not np.all(np.isclose(ini_array, 0) | np.isclose(ini_array, 1)):
                raise InvalidInputError("ini_array entries must be binary.")

            row_sums = ini_array.sum(axis=1)
            if not np.all(np.isclose(row_sums, 1)):
                raise InvalidInputError("ini_array rows must sum to 1.")

        self.ini_array = ini_array

    def load_upd_array(self, upd_array, check_errors=True):
        """Load the MVR update array or arrays.

        Parameters
        ----------
        upd_array : array-like or list of array-like
            Update array for a homogeneous MVR or time-indexed update arrays for an
            inhomogeneous MVR.
        check_errors : bool, optional
            If ``True``, validate dimensions and binary structure.

        Raises
        ------
        InvalidInputError
            If any update array has invalid shape or entries.
        """
        if isinstance(upd_array, list):
            upd_array = [np.asarray(arr) for arr in upd_array]
        else:
            upd_array = np.asarray(upd_array)

        if check_errors:
            upd_arrays = upd_array if isinstance(upd_array, list) else [upd_array]

            for t, arr in enumerate(upd_arrays):
                if arr.ndim != 3:
                    raise InvalidInputError(
                        f"upd_array at index {t} must be a 3D array."
                    )

                if not np.all(arr >= 0):
                    raise InvalidInputError(
                        f"upd_array at index {t} entries must be nonnegative."
                    )

                if not np.all(np.isclose(arr, 0) | np.isclose(arr, 1)):
                    raise InvalidInputError(
                        f"upd_array at index {t} entries must be binary."
                    )

                # For every h_curr and m_prev, exactly one m_curr is selected.
                sums = arr.sum(axis=1)
                if not np.all(np.isclose(sums, 1)):
                    raise InvalidInputError(
                        f"upd_array at index {t} must sum to 1 over the current mediation axis."
                    )

        self.upd_array = upd_array

    def load_evl_array(self, evl_array, check_errors=True):
        """Load the MVR evaluation array or arrays.

        Parameters
        ----------
        evl_array : array-like or list of array-like
            Evaluation array for a homogeneous MVR or time-indexed evaluation arrays
            for an inhomogeneous MVR.
        check_errors : bool, optional
            If ``True``, validate dimensions and binary structure.

        Raises
        ------
        InvalidInputError
            If any evaluation array has invalid shape or entries.
        """
        if isinstance(evl_array, list):
            evl_array = [np.asarray(arr) for arr in evl_array]
        else:
            evl_array = np.asarray(evl_array)

        if check_errors:
            evl_arrays = evl_array if isinstance(evl_array, list) else [evl_array]

            for t, arr in enumerate(evl_arrays):
                if arr.ndim != 1:
                    raise InvalidInputError(
                        f"evl_array at index {t} must be a 1D array."
                    )

                if not np.all(arr >= 0):
                    raise InvalidInputError(
                        f"evl_array at index {t} entries must be nonnegative."
                    )

                if not np.all(np.isclose(arr, 0) | np.isclose(arr, 1)):
                    raise InvalidInputError(
                        f"evl_array at index {t} entries must be binary."
                    )

        self.evl_array = evl_array

    def check_dimensions(self):
        """Validate that the initialization, update, and evaluation arrays are compatible.

        Raises
        ------
        InvalidInputError
            If the supplied arrays are inconsistent with each other.
        """
        if isinstance(self.evl_array, list) != isinstance(self.upd_array, list):
            raise InvalidInputError(
                "evl_array and upd_array must either both be lists or both be arrays."
            )

        H_init, M_init = self.ini_array.shape

        # Homogeneous case
        if not isinstance(self.evl_array, list):
            H_update, M_curr, M_prev = self.upd_array.shape
            M_eval = self.evl_array.shape[0]

            if H_update != H_init:
                raise InvalidInputError(
                    "upd_array hidden dimension must match ini_array hidden dimension."
                )

            if M_eval != M_init:
                raise InvalidInputError(
                    "evl_array mediation dimension must match ini_array mediation dimension."
                )

            if M_curr != M_init or M_prev != M_init:
                raise InvalidInputError(
                    "homogeneous upd_array mediation dimensions must both match ini_array mediation dimension."
                )

            return

        # Inhomogeneous case
        if len(self.evl_array) == 0:
            raise InvalidInputError("evl_array list must be nonempty.")

        if len(self.upd_array) != len(self.evl_array) - 1:
            raise InvalidInputError(
                "upd_array list length must be one less than evl_array list length."
            )

        M_eval_0 = self.evl_array[0].shape[0]

        if M_eval_0 != M_init:
            raise InvalidInputError(
                "evl_array[0] mediation dimension must match ini_array mediation dimension."
            )

        for t, update_t in enumerate(self.upd_array):
            H_update_t, M_curr, M_prev = update_t.shape

            if H_update_t != H_init:
                raise InvalidInputError(
                    f"upd_array[{t}] hidden dimension must match ini_array hidden dimension."
                )

            M_eval_prev = self.evl_array[t].shape[0]
            M_eval_curr = self.evl_array[t + 1].shape[0]

            if M_prev != M_eval_prev:
                raise InvalidInputError(
                    f"upd_array[{t}] previous mediation dimension must match evl_array[{t}]."
                )

            if M_curr != M_eval_curr:
                raise InvalidInputError(
                    f"upd_array[{t}] current mediation dimension must match evl_array[{t + 1}]."
                )

    def load_dimensions(self):
        """Update cached dimension metadata for the numeric MVR representation."""
        self.num_hidden_states = self.ini_array.shape[0]
        self.hidden_states = range(self.num_hidden_states)

        # Homogeneous case
        if not isinstance(self.evl_array, list):
            self.num_mediation_states = self.ini_array.shape[1]
            self.mediation_states = range(self.num_mediation_states)
            return

        # Inhomogeneous case
        self.time_horizon = len(self.evl_array)
        self.num_mediation_states = [arr.shape[0] for arr in self.evl_array]
        self.mediation_states = [
            range(num_states) for num_states in self.num_mediation_states
        ]
