from __future__ import annotations

import math

import numpy as np
import torch

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import InhomMVR

NEG_INF = float("-inf")

# Running totals are held wider than the tensors they sum. Each step contributes
# a value near 0, but the total grows with the horizon -- a few thousand nats over
# a long run -- and in float32 every later addition would round at the ULP of that
# total. Measured at T=20000 that is the difference between 3.4 and 0.0006 nats.
ACCUM_DTYPE = torch.float64


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


def _resolve_horizon(observed, time_horizon):
    """
    Resolve the inference horizon and return it with ``(time, label)`` pairs.
    Need to handle both a list of observations or a dict of time:observation.
    A dense list is implicitly timed by position; a map is explicitly timed.
    """
    if isinstance(observed, dict):
        # For a dict {time:obs}, user-defined time_horizon is required
        if time_horizon is None:
            raise InvalidInputError(
                "time_horizon is required when observed is a {time: label} map."
            )
        items = list(observed.items())
    else:
        items = list(enumerate(observed))

        if time_horizon is None:
            if not items:
                raise InvalidInputError(
                    "observed sequence must be nonempty when time_horizon is "
                    "not given."
                )
            time_horizon = len(items)
        elif time_horizon < len(items):
            raise InvalidInputError(
                f"time_horizon={time_horizon} is shorter than the observed "
                f"sequence of length {len(items)}."
            )

    if not isinstance(time_horizon, (int, np.integer)) or time_horizon < 1:
        raise InvalidInputError(
            f"time_horizon must be a positive integer, got {time_horizon!r}."
        )

    return int(time_horizon), items


def _resolve_emit_weights(hmm, emission_mat, observed, time_horizon=None, *, log=False):
    """
    Resolve the horizon and build the dense ``(T, K)`` table of emission weights.
    Times carrying no observation get the identity of the working space, so they
    are scored by the transition alone rather than being dropped from the chain.
    Also returns the set of observed times, which decides where that happens.
    """
    T, items = _resolve_horizon(observed, time_horizon)
    K = int(emission_mat.shape[0])

    times, obs_idx = [], []

    for t, o in items:
        if not isinstance(t, (int, np.integer)):
            raise InvalidInputError(f"Observation time must be an integer: {t!r}")
        if not 0 <= t < T:
            raise InvalidInputError(
                f"Observation time {t} is outside the horizon [0, {T})."
            )
        if o not in hmm.observed_to_internal:
            raise InvalidInputError(f"Unknown observed state: {o}")

        times.append(int(t))
        obs_idx.append(hmm.observed_to_internal[o])

    # 0 and 1 are the additive and multiplicative identities respectively, so an
    # unobserved time contributes nothing in whichever space the caller is using.
    identity = 0.0 if log else 1.0
    weights = torch.full(
        (T, K),
        identity,
        dtype=emission_mat.dtype,
        device=emission_mat.device,
    )

    if times:
        rows = torch.as_tensor(times, dtype=torch.long, device=emission_mat.device)
        cols = torch.as_tensor(obs_idx, dtype=torch.long, device=emission_mat.device)
        weights[rows] = emission_mat[:, cols].T

    return T, weights, set(times)


def _validate_time_range(mvr, T):
    """
    Check an MVR's time range against the inference horizon and return it.
    An MVR without a ``_time_range`` is enforced over the whole horizon.
    """
    if mvr._time_range is None:
        active_start, active_end = 0, T - 1
    else:
        active_start, active_end = mvr._time_range

    # The window must fit inside the inference horizon.
    if active_end >= T:
        raise InvalidInputError(
            f"MVR time_range [{active_start}, {active_end}] exceeds "
            f"observation horizon T={T}."
        )

    # The MVR must be long enough to run across the window.
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
    Precompute the per-MVR tensors, indexed by global time.
    """
    repn = mvr.repn
    a, b = _validate_time_range(mvr, T)
    is_inhom = isinstance(mvr, InhomMVR)

    def to_idx(array):
        return torch.as_tensor(array, dtype=torch.long, device=device)

    def to_evl(evl_array):
        evl = torch.as_tensor(evl_array, dtype=dtype, device=device)
        return torch.log(evl) if log else evl

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
    Return the C-order strides of a shape tuple, matching ``np.unravel_index``.
    """
    strides = []
    total = 1

    for dim in reversed(dims):
        strides.append(total)
        total *= dim

    strides.reverse()

    return strides


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


def _apply_closing_evl(V, t, mvr_infos, curr_active, *, log, lead=1):
    """
    Fold in the acceptance factor of every MVR whose window closes at ``t``.
    ``lead`` is the number of axes ahead of the mediation axes in ``V``.
    """
    for pos, mvr_i in enumerate(curr_active):
        info = mvr_infos[mvr_i]

        if t != info["b"]:
            continue

        evl = info["evl_at"][t]
        view = [1] * V.dim()
        view[lead + pos] = evl.shape[0]

        V = V + evl.reshape(view) if log else V * evl.reshape(view)

    return V


def _maxplus_step(
    V_prev,
    t,
    *,
    K,
    mvr_infos,
    prev_active,
    curr_active,
    prev_dims,
    curr_dims,
    log_transition_t,
    log_emit_t,
    dtype,
    device,
):
    """
    Advance one time step in the max-plus semiring without forming a product MVR.
    Returns the new values and a flat backpointer into the predecessor space.
    """
    prev_total = math.prod(prev_dims)
    curr_total = math.prod(curr_dims)

    V_prev_flat = V_prev.reshape(K, prev_total)

    # Maximize over the predecessor hidden state first. The scatter destination
    # depends on h_curr and m_prev but never on h_prev, so the two maximizations
    # commute and this keeps a factor of K out of every tensor built below.
    candidate = V_prev_flat.unsqueeze(0) + log_transition_t.unsqueeze(-1)

    # best[h_curr, m_prev], best_h_prev[h_curr, m_prev]
    best, best_h_prev = candidate.max(dim=1)
    best = best + log_emit_t.unsqueeze(-1)

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
    V_curr_flat = torch.full((K, curr_total), NEG_INF, dtype=dtype, device=device)
    V_curr_flat.scatter_reduce_(1, dest, best, reduce="amax", include_self=True)

    # Recover the best predecessor mediation state, breaking ties toward
    # the smallest source index.
    sentinel = prev_total
    source = torch.arange(prev_total, device=device).unsqueeze(0).expand(K, -1)

    is_winner = torch.isfinite(best) & (best == V_curr_flat.gather(1, dest))

    # `source` is an expanded view; a scalar `other` keeps it that way.
    winner_source = torch.where(is_winner, source, sentinel)

    m_prev_ptr = torch.full(
        (K, curr_total),
        sentinel,
        dtype=torch.long,
        device=device,
    )
    m_prev_ptr.scatter_reduce_(1, dest, winner_source, reduce="amin", include_self=True)

    # sentinel marks unreached destinations; clamp it back into range.
    m_prev_ptr = m_prev_ptr.clamp(max=prev_total - 1)
    h_prev_ptr = best_h_prev.gather(1, m_prev_ptr)

    backptr = (h_prev_ptr * prev_total + m_prev_ptr).reshape((K,) + curr_dims)
    V_curr = V_curr_flat.reshape((K,) + curr_dims)

    # Evaluate any MVR whose enforcement window closes here.
    V_curr = _apply_closing_evl(V_curr, t, mvr_infos, curr_active, log=True)

    return V_curr, backptr


# ======================================================================
# Decoding
# ======================================================================


def _decode_dynamic_augmented_path(
    augmented_index_path,
    active_by_time,
    mvr_infos,
    times=None,
):
    """Decode augmented index tuples into per-time dictionaries."""
    if times is None:
        times = range(len(augmented_index_path))

    decoded = []

    for t, aug_idx in zip(times, augmented_index_path):
        active = active_by_time[t]

        entry = {
            "time": t,
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
