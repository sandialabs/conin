from __future__ import annotations

import math

import numpy as np
import torch

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import InhomMVR

NEG_INF = float("-inf")


# ======================================================================
# Model extraction and conversion
# ======================================================================


def _hmm_to_torch(hmm, *, log=False, dtype=torch.float32, device="cpu"):
    """
    Convert a hidden Markov model into torch tensors, optionally in log space.
    Zero probabilities map to ``-inf``, the max-plus absorbing element.
    """
    repn = hmm.repn

    if repn is None:
        raise InvalidInputError("hidden_markov_model.repn is missing.")

    def convert(array):
        # Take the log in the source precision. Casting first would send any
        # probability below the dtype's smallest subnormal to 0, and log(0) to
        # -inf, silently reclassifying an unlikely transition as impossible.
        array = np.asarray(array, dtype=np.float64)

        if log:
            with np.errstate(divide="ignore"):
                array = np.log(array)

        return torch.as_tensor(array, dtype=dtype, device=device)

    return (
        convert(repn.start_vec),
        convert(repn.transition_mat),
        convert(repn.emission_mat),
    )


def _observations_to_emit_weights(hmm, emission_mat, observed):
    """
    Convert external observation labels into per-time emission weights.
    The result follows whichever space ``emission_mat`` is given in.
    """
    obs_idx = []

    for o in observed:
        if o not in hmm.observed_to_internal:
            raise InvalidInputError(f"Unknown observed state: {o}")
        obs_idx.append(hmm.observed_to_internal[o])

    obs_idx = torch.as_tensor(
        obs_idx,
        dtype=torch.long,
        device=emission_mat.device,
    )

    return emission_mat[:, obs_idx].T


# ======================================================================
# MVR preprocessing
# ======================================================================


def _validate_time_range(mvr, T):
    """
    Check an MVR's time range against the observation length and return it.
    An MVR without a ``_time_range`` is enforced over the whole sequence.
    """
    if mvr._time_range is None:
        active_start, active_end = 0, T - 1
    else:
        active_start, active_end = mvr._time_range

    # Nonnegativity and ordering are enforced by the MVR constructors. Only the
    # fit against the observation length is checkable here.
    if active_end >= T:
        raise InvalidInputError(
            f"MVR time_range [{active_start}, {active_end}] exceeds "
            f"observation horizon T={T}."
        )

    # The constructor checks this only for an explicit time_range. For a
    # defaulted window it becomes "is the automaton long enough to span the
    # whole sequence", which construction had no T to answer.
    if isinstance(mvr, InhomMVR) and active_end - active_start > mvr.time_horizon:
        raise InvalidInputError(
            "InhomMVR time_horizon is too short for requested inference range. "
            f"Required local horizon {active_end - active_start}, "
            f"but mvr.time_horizon is {mvr.time_horizon}."
        )

    return active_start, active_end


def _build_mvr_factor_info(
    mvr,
    T,
    *,
    log=False,
    dtype=torch.float32,
    device="cpu",
):
    """
    Precompute the per-MVR tensors an inference recursion needs, by global time.
    Keying by global time applies the local-time offset once, here, so no
    consumer repeats it and homogeneous and inhomogeneous MVRs read alike.
    """
    repn = mvr.repn
    a, b = _validate_time_range(mvr, T)
    is_inhom = isinstance(mvr, InhomMVR)

    def to_idx(array):
        return torch.as_tensor(array, dtype=torch.long, device=device)

    def to_evl(evl_array):
        evl = torch.as_tensor(evl_array, dtype=dtype, device=device)
        return torch.log(evl) if log else evl

    # MVR_CHMM aligned the hidden ordering and MVR_MatVecRepn already reduced
    # the one-hot ini/upd arrays to index maps, so only a dtype/device
    # conversion is left. A homogeneous MVR reuses one tensor across its window.
    if is_inhom:
        evl_at = {t: to_evl(repn.evl_array[t - a]) for t in range(a, b + 1)}
        next_at = {t: to_idx(repn.next_idx[t - a - 1]) for t in range(a + 1, b + 1)}
        labels_at = {t: mvr.mediation_states[t - a] for t in range(a, b + 1)}
    else:
        evl, next_idx = to_evl(repn.evl_array), to_idx(repn.next_idx)
        evl_at = {t: evl for t in range(a, b + 1)}
        next_at = {t: next_idx for t in range(a + 1, b + 1)}
        labels_at = {t: mvr.mediation_states for t in range(a, b + 1)}

    return {
        "a": a,
        "b": b,
        "ini_idx": to_idx(repn.ini_idx),
        "next_idx_at": next_at,
        "evl_at": evl_at,
        "labels_at": labels_at,
    }


def _c_strides(dims):
    """
    Return the C-order strides of a shape tuple.
    The order must match ``np.unravel_index``, which decodes the flat indices
    these strides encode.
    """
    strides = []
    total = 1

    for dim in reversed(dims):
        strides.append(total)
        total *= dim

    strides.reverse()

    return strides


# ======================================================================
# Destination index construction
# ======================================================================


def _build_destination_index(
    K,
    mvr_infos,
    prev_active,
    curr_active,
    prev_dims,
    curr_dims,
    t,
    device,
):
    """
    Return the flat destination index of each predecessor mediation state.
    The MVRs are folded in one at a time as strided terms, so no product
    transition tensor is ever materialized.
    """
    curr_strides = _c_strides(curr_dims)
    prev_total = math.prod(prev_dims)

    prev_position = {mvr_i: pos for pos, mvr_i in enumerate(prev_active)}

    # MVRs active at t - 1 but not at t contribute no stride, so their
    # predecessor states collapse onto a shared destination and the scatter
    # maximizes over them. MVRs inactive at both times appear in neither space.
    dest = torch.zeros((K,) + prev_dims, dtype=torch.long, device=device)

    for pos, mvr_i in enumerate(curr_active):
        info = mvr_infos[mvr_i]
        stride = curr_strides[pos]

        if mvr_i in prev_position:
            # Active across the transition: follow the deterministic update.
            next_idx = info["next_idx_at"][t]

            view = [1] * (1 + len(prev_dims))
            view[0] = K
            view[1 + prev_position[mvr_i]] = next_idx.shape[1]

            dest = dest + next_idx.reshape(view) * stride
        else:
            # Enters the lattice at t; this can only be its window start.
            if t != info["a"]:
                raise RuntimeError(
                    "Internal active-set inconsistency: MVR appears in current "
                    "but not previous at a non-start time."
                )

            view = [1] * (1 + len(prev_dims))
            view[0] = K

            dest = dest + info["ini_idx"].reshape(view) * stride

    return dest.reshape(K, prev_total)


# ======================================================================
# Decoding
# ======================================================================


def _decode_dynamic_augmented_path(
    augmented_index_path,
    active_by_time,
    mvr_infos,
):
    """Decode augmented index tuples into per-time dictionaries."""
    decoded = []

    for t, aug_idx in enumerate(augmented_index_path):
        active = active_by_time[t]

        entry = {
            "hidden_index": int(aug_idx[0]),
            "mvr_indices": {},
            "mvr_states": {},
        }

        for axis_pos, mvr_i in enumerate(active, start=1):
            m_idx = int(aug_idx[axis_pos])
            info = mvr_infos[mvr_i]

            entry["mvr_indices"][mvr_i] = m_idx
            entry["mvr_states"][mvr_i] = info["labels_at"][t][m_idx]

        decoded.append(entry)

    return decoded


# ======================================================================
# Viterbi
# ======================================================================


def viterbi_torch_mvr_chmm(
    model,
    observed,
    *,
    dtype=torch.float32,
    device="cpu",
    normalize=True,
    return_augmented=True,
    return_score=False,
):
    """Log-space Viterbi for an MVR-augmented constrained HMM.

    The recursion runs in the max-plus semiring: products of probabilities
    become sums of log probabilities and the maximization is unchanged, so no
    ``logsumexp`` is required and the decoded path is identical to what an
    exact probability-space implementation would produce.

    Augmented axes are dynamic. Rather than padding every MVR out to the full
    horizon, the value tensor carries only the MVRs enforced at the current
    time::

        V[t].shape == (K,) + tuple(M_i(t) for i active at t)

    An MVR is active over ``[a_i, b_i]``, taken from its ``time_range`` and
    defaulting to ``[0, T - 1]``. It is initialized at ``a_i``, evaluated at
    ``b_i``, and absent otherwise.

    The MVR updates are deterministic, so each transition is a scatter rather
    than a dense mediation-space multiply. The MVRs are folded into a single
    destination index one at a time, so no product automaton is constructed and
    the working tensors scale with the mediation space rather than its square.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints.
    observed : list
        Observed sequence in external observed-state labels.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors. ``float32`` is ample in log space.
    device : str or torch.device, optional
        Torch device.
    normalize : bool, optional
        If ``True``, shift each slice so its maximum is ``0``. Log space does
        not require this for stability; it is retained for comparability and
        costs one subtraction per step.
    return_augmented : bool, optional
        If ``True``, also return the decoded augmented path.
    return_score : bool, optional
        If ``True``, also return the log probability of the best path.

    Returns
    -------
    hidden_path : list
        Most likely feasible hidden-state path in external labels.
    augmented_path : list[dict], optional
        Returned when ``return_augmented`` is ``True``.
    best_logprob : float, optional
        Returned when ``return_score`` is ``True``.

    Raises
    ------
    InvalidInputError
        If the observed sequence is empty, contains an unknown label, or if the
        constraints admit no feasible path.
    """
    if len(observed) == 0:
        raise InvalidInputError("observed sequence must be nonempty.")

    hmm = getattr(model, "hidden_markov_model", None) or getattr(model, "hmm", None)

    if hmm is None:
        raise InvalidInputError("Missing hidden_markov_model from MVR_CHMM")

    constraints = list(getattr(model, "constraints", None) or [])
    T = len(observed)

    log_start_vec, log_transition_mat, log_emission_mat = _hmm_to_torch(
        hmm,
        log=True,
        dtype=dtype,
        device=device,
    )

    log_emit_weights = _observations_to_emit_weights(
        hmm,
        log_emission_mat,
        observed,
    )

    K = int(log_start_vec.shape[0])

    mvr_infos = [
        _build_mvr_factor_info(
            mvr,
            T,
            log=True,
            dtype=dtype,
            device=device,
        )
        for mvr in constraints
    ]

    active_by_time = [
        [i for i, info in enumerate(mvr_infos) if t in info["evl_at"]]
        for t in range(T)
    ]
    dims_by_time = [
        tuple(mvr_infos[i]["evl_at"][t].shape[0] for i in active_by_time[t])
        for t in range(T)
    ]
    shapes = [(K,) + dims_by_time[t] for t in range(T)]

    # log_transition_mat is indexed [h_prev, h_curr]; the recursion wants
    # h_curr on the leading axis.
    log_transition_t = log_transition_mat.transpose(0, 1).contiguous()

    backptr = []
    log_score = 0.0

    # ------------------------------------------------------------------
    # Initialization at t = 0.
    #
    # Every MVR active at t = 0 starts there, so the joint mediation state is a
    # function of the hidden state alone and V has exactly K finite entries.
    # ------------------------------------------------------------------
    active_0 = active_by_time[0]
    dims_0 = dims_by_time[0]
    strides_0 = _c_strides(dims_0)
    total_0 = math.prod(dims_0)

    flat_ini = torch.zeros(K, dtype=torch.long, device=device)

    for pos, mvr_i in enumerate(active_0):
        flat_ini = flat_ini + mvr_infos[mvr_i]["ini_idx"] * strides_0[pos]

    V_prev = torch.full(
        (K, total_0),
        NEG_INF,
        dtype=dtype,
        device=device,
    )
    V_prev[torch.arange(K, device=device), flat_ini] = (
        log_start_vec + log_emit_weights[0]
    )
    V_prev = V_prev.reshape(shapes[0])

    for pos, mvr_i in enumerate(active_0):
        info = mvr_infos[mvr_i]

        if info["b"] == 0:
            log_evl = info["evl_at"][0]
            view = [1] * len(shapes[0])
            view[1 + pos] = log_evl.shape[0]
            V_prev = V_prev + log_evl.reshape(view)

    scale = V_prev.max()

    if not torch.isfinite(scale):
        raise InvalidInputError("No feasible augmented path at time 0.")

    if normalize:
        log_score += scale.item()
        V_prev = V_prev - scale

    # ------------------------------------------------------------------
    # Forward pass.
    # ------------------------------------------------------------------
    for t in range(1, T):
        prev_active = active_by_time[t - 1]
        curr_active = active_by_time[t]

        prev_dims = dims_by_time[t - 1]
        curr_dims = dims_by_time[t]

        prev_total = math.prod(prev_dims)
        curr_total = math.prod(curr_dims)

        V_prev_flat = V_prev.reshape(K, prev_total)

        # Maximize over the predecessor hidden state first. This is valid
        # because the mediation destination depends on h_curr and m_prev but
        # never on h_prev, so the two maximizations commute. It removes a
        # factor of K from every tensor built below.
        #
        # candidate[h_curr, h_prev, m_prev]
        candidate = V_prev_flat.unsqueeze(0) + log_transition_t.unsqueeze(-1)

        # best[h_curr, m_prev], best_h_prev[h_curr, m_prev]
        best, best_h_prev = candidate.max(dim=1)
        best = best + log_emit_weights[t].unsqueeze(-1)

        dest = _build_destination_index(
            K,
            mvr_infos,
            prev_active,
            curr_active,
            prev_dims,
            curr_dims,
            t,
            device,
        )

        # Scatter-max the predecessor mediation states onto their successors.
        V_curr_flat = torch.full(
            (K, curr_total),
            NEG_INF,
            dtype=dtype,
            device=device,
        )
        V_curr_flat.scatter_reduce_(1, dest, best, reduce="amax", include_self=True)

        # Recover the winning predecessor mediation state. The scattered value
        # is copied verbatim from one source entry, so exact equality is the
        # right test; ties are broken toward the smallest source index.
        sentinel = prev_total
        source = torch.arange(prev_total, device=device).unsqueeze(0).expand(K, -1)

        is_winner = torch.isfinite(best) & (best == V_curr_flat.gather(1, dest))

        winner_source = torch.where(
            is_winner,
            source,
            torch.full_like(source, sentinel),
        )

        m_prev_ptr = torch.full(
            (K, curr_total),
            sentinel,
            dtype=torch.long,
            device=device,
        )
        m_prev_ptr.scatter_reduce_(
            1,
            dest,
            winner_source,
            reduce="amin",
            include_self=True,
        )

        # Unreachable destinations hold the sentinel and are -inf valued, so
        # their back pointer is never followed; clamp only to keep the gather
        # in bounds.
        m_prev_ptr = m_prev_ptr.clamp(max=prev_total - 1)
        h_prev_ptr = best_h_prev.gather(1, m_prev_ptr)

        # Flat index into shapes[t - 1], whose C-order ravel is exactly
        # h_prev * prev_total + m_prev.
        backptr.append((h_prev_ptr * prev_total + m_prev_ptr).reshape(shapes[t]))

        V_curr = V_curr_flat.reshape(shapes[t])

        # Evaluate any MVR whose enforcement window closes here.
        for pos, mvr_i in enumerate(curr_active):
            info = mvr_infos[mvr_i]

            if t == info["b"]:
                log_evl = info["evl_at"][t]
                view = [1] * len(shapes[t])
                view[1 + pos] = log_evl.shape[0]
                V_curr = V_curr + log_evl.reshape(view)

        scale = V_curr.max()

        if not torch.isfinite(scale):
            raise InvalidInputError(f"No feasible augmented path at time {t}.")

        if normalize:
            log_score += scale.item()
            V_curr = V_curr - scale

        V_prev = V_curr

    # ------------------------------------------------------------------
    # Termination.
    #
    # With normalization the running total already holds the full log
    # probability and the residual maximum is 0; without it, the residual
    # maximum is the whole score.
    # ------------------------------------------------------------------
    final_scale = V_prev.max()

    if not torch.isfinite(final_scale):
        raise InvalidInputError("No feasible augmented path at final time.")

    log_score += final_scale.item()

    final_flat = int(torch.argmax(V_prev).item())
    final_idx = tuple(int(x) for x in np.unravel_index(final_flat, shapes[T - 1]))

    # ------------------------------------------------------------------
    # Backtracking.
    # ------------------------------------------------------------------
    augmented_index_path = [final_idx]
    curr_idx = final_idx

    for t in range(T - 1, 0, -1):
        prev_flat = int(backptr[t - 1][curr_idx].item())
        prev_idx = tuple(int(x) for x in np.unravel_index(prev_flat, shapes[t - 1]))

        augmented_index_path.append(prev_idx)
        curr_idx = prev_idx

    augmented_index_path.reverse()

    hidden_path = [
        hmm.hidden_to_external[aug_idx[0]] for aug_idx in augmented_index_path
    ]

    if return_augmented:
        augmented_path = _decode_dynamic_augmented_path(
            augmented_index_path,
            active_by_time,
            mvr_infos,
        )

        if return_score:
            return hidden_path, augmented_path, log_score

        return hidden_path, augmented_path

    if return_score:
        return hidden_path, log_score

    return hidden_path
