from __future__ import annotations

import math

import numpy as np
import torch

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import InhomMVR

NEG_INF = float("-inf")

# Running totals are held wider than the tensors they sum; see CLAUDE.md.
ACCUM_DTYPE = torch.float64


def _hmm_to_torch(hmm, *, log=False, dtype=torch.float32, device="cpu"):
    """Convert a hidden Markov model into torch tensors, optionally in log space."""
    repn = hmm.repn

    if repn is None:
        raise InvalidInputError("hidden_markov_model.repn is missing.")

    def convert(array):
        # Log in the source precision: casting first would flush subnormals to -inf.
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
    """Resolve the inference horizon and return it with ``(time, label)`` pairs."""
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
    """Build the dense ``(T, K)`` emission-weight table and ``{time: observed index}``."""
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

    # The identity of the working space, so an unobserved time contributes nothing.
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

    return T, weights, dict(zip(times, obs_idx))


def _validate_time_range(mvr, T):
    """Check an MVR's time range against the inference horizon and return it."""
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


def _build_augmented_axes(K, T, mvr_infos):
    """Resolve which MVRs are enforced at each time and the shape that implies."""
    active_by_time = [
        [i for i, info in enumerate(mvr_infos) if t in info["evl_at"]] for t in range(T)
    ]
    dims_by_time = [
        tuple(mvr_infos[i]["evl_at"][t].shape[0] for i in active_by_time[t])
        for t in range(T)
    ]
    shapes = [(K,) + dims_by_time[t] for t in range(T)]

    return active_by_time, dims_by_time, shapes


def _build_static_context(constraints, K, T, *, log, dtype, device):
    """Build the parts of an inference context that do not depend on HMM parameters."""
    mvr_infos = [
        _build_mvr_factor_info(mvr, T, log=log, dtype=dtype, device=device)
        for mvr in constraints
    ]

    active_by_time, dims_by_time, shapes = _build_augmented_axes(K, T, mvr_infos)

    return {
        "T": T,
        "mvr_infos": mvr_infos,
        "active_by_time": active_by_time,
        "dims_by_time": dims_by_time,
        "shapes": shapes,
        "dest_cache": {},
    }


def _model_parts(model):
    """Pull the hidden Markov model and the MVR constraints out of an MVR_CHMM."""
    hmm = getattr(model, "hidden_markov_model", None) or getattr(model, "hmm", None)

    if hmm is None:
        raise InvalidInputError("Missing hidden_markov_model from MVR_CHMM")

    return hmm, list(getattr(model, "constraints", None) or [])


def _build_sumprod_ctx(
    model,
    observed,
    *,
    time_horizon=None,
    dtype=torch.float32,
    device="cpu",
    static=None,
):
    """Assemble the context the log-space sum-product steps read."""
    hmm, constraints = _model_parts(model)

    # Both forms kept: _sum_step contracts against the probability-space matrix,
    # while _maxplus_step and the transition posteriors want the log.
    _, transition_mat, _ = _hmm_to_torch(hmm, log=False, dtype=dtype, device=device)
    log_start_vec, log_transition_mat, log_emission_mat = _hmm_to_torch(
        hmm, log=True, dtype=dtype, device=device
    )

    T, log_emit_weights, observed_index = _resolve_emit_weights(
        hmm, log_emission_mat, observed, time_horizon, log=True
    )

    K = int(log_start_vec.shape[0])

    if static is None:
        static = _build_static_context(
            constraints, K, T, log=True, dtype=dtype, device=device
        )
    elif static["T"] != T:
        raise InvalidInputError(
            f"Cached static context was built for horizon {static['T']}, "
            f"but this call resolves to {T}."
        )

    return {
        "hmm": hmm,
        "constraints": constraints,
        "K": K,
        "T": T,
        "log_start_vec": log_start_vec,
        "transition_mat": transition_mat,
        "log_transition_mat": log_transition_mat,
        # log_transition_mat is indexed [h_prev, h_curr]
        "log_transition_t": log_transition_mat.transpose(0, 1).contiguous(),
        "log_emit_weights": log_emit_weights,
        "observed_index": observed_index,
        "mvr_infos": static["mvr_infos"],
        "active_by_time": static["active_by_time"],
        "dims_by_time": static["dims_by_time"],
        "shapes": static["shapes"],
        "dest_cache": static["dest_cache"],
        "dtype": dtype,
        "device": device,
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
    """Return the flat destination index of each predecessor mediation state."""
    curr_strides = _c_strides(curr_dims)
    prev_total = math.prod(prev_dims)

    prev_position = {mvr_i: pos for pos, mvr_i in enumerate(prev_active)}

    # An MVR active at t-1 but not t contributes no stride, so its states collapse
    # onto a shared destination and the scatter reduces over them.
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


def _flat_ini_index(K, mvr_infos, active, dims, device):
    """
    Flat mediation index every MVR active at ``t = 0`` starts in, per hidden state.
    """
    strides = _c_strides(dims)
    flat_ini = torch.zeros(K, dtype=torch.long, device=device)

    for pos, mvr_i in enumerate(active):
        flat_ini = flat_ini + mvr_infos[mvr_i]["ini_idx"] * strides[pos]

    return flat_ini


def _apply_closing_evl(V, t, mvr_infos, curr_active, *, log, lead=1):
    """Fold in the acceptance factor of every MVR whose window closes at ``t``."""
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
    """Advance one max-plus step, returning the values and a flat backpointer."""
    prev_total = math.prod(prev_dims)
    curr_total = math.prod(curr_dims)

    V_prev_flat = V_prev.reshape(K, prev_total)

    # Maximize over h_prev first: the destination never depends on it, so the two
    # maximizations commute and this keeps a factor of K out of everything below.
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
# Sum-product
# ======================================================================
#
# In log space, which is not the usual choice here -- see the semiring notes in
# CLAUDE.md before changing it.


def _dest_at(ctx, t):
    """Destination index for the ``t-1 -> t`` transition, cached across callers."""
    cache = ctx["dest_cache"]

    if t not in cache:
        cache[t] = _build_destination_index(
            ctx["K"],
            ctx["mvr_infos"],
            ctx["active_by_time"][t - 1],
            ctx["active_by_time"][t],
            ctx["dims_by_time"][t - 1],
            ctx["dims_by_time"][t],
            t,
            ctx["device"],
        )

    return cache[t]


def _safe_shift(values, dim, keepdim):
    """Largest entry along ``dim``, with an all ``-inf`` slice reported as ``0``."""
    shift = values.amax(dim=dim, keepdim=keepdim)

    return torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))


def _initial_sumprod_message(ctx):
    """Batched ``(1, K, M_0)`` log-weights at ``t = 0``, with every factor from that time."""
    K = ctx["K"]
    device = ctx["device"]

    dims_0 = ctx["dims_by_time"][0]
    total_0 = math.prod(dims_0)

    flat_ini = _flat_ini_index(
        K, ctx["mvr_infos"], ctx["active_by_time"][0], dims_0, device
    )

    log_P = torch.full((1, K, total_0), NEG_INF, dtype=ctx["dtype"], device=device)
    log_P[0, torch.arange(K, device=device), flat_ini] = (
        ctx["log_start_vec"] + ctx["log_emit_weights"][0]
    )

    return _apply_closing_evl(
        log_P.reshape((1, K) + dims_0),
        0,
        ctx["mvr_infos"],
        ctx["active_by_time"][0],
        log=True,
        lead=2,
    ).reshape(1, K, total_0)


def _sum_step(ctx, log_P, t):
    """Advance a batch of augmented log-weights one step, summing over predecessors."""
    K = ctx["K"]
    B = log_P.shape[0]

    prev_dims = ctx["dims_by_time"][t - 1]
    curr_dims = ctx["dims_by_time"][t]
    prev_total = math.prod(prev_dims)
    curr_total = math.prod(curr_dims)

    # Shifting first makes this an ordinary contraction rather than a logsumexp, so
    # the einsum keeps its (B, K, M) shape. In place below: these are the bottleneck.
    shift = _safe_shift(log_P, dim=1, keepdim=True)
    log_contrib = torch.einsum(
        "bhm,hg->bgm", (log_P - shift).exp(), ctx["transition_mat"]
    )
    log_contrib.log_().add_(shift).add_(ctx["log_emit_weights"][t].reshape(1, K, 1))

    dest = _dest_at(ctx, t).unsqueeze(0).expand(B, K, prev_total)
    flat = (B, K, curr_total)

    # Log-domain scatter-add: take the max reaching each destination, add the
    # shifted contributions there, and put the shift back.
    peak = torch.full(flat, NEG_INF, dtype=ctx["dtype"], device=ctx["device"])
    peak.scatter_reduce_(2, dest, log_contrib, reduce="amax", include_self=True)
    peak.masked_fill_(peak == NEG_INF, 0.0)

    out = torch.zeros(flat, dtype=ctx["dtype"], device=ctx["device"])
    out.scatter_add_(2, dest, log_contrib.sub_(peak.gather(2, dest)).exp_())
    out.log_().add_(peak)

    return _apply_closing_evl(
        out.reshape((B, K) + curr_dims),
        t,
        ctx["mvr_infos"],
        ctx["active_by_time"][t],
        log=True,
        lead=2,
    ).reshape(flat)


def _sum_step_backward(ctx, log_beta, t):
    """Pull a message from ``t`` to ``t - 1``, also returning it ``evl``-applied at ``t``."""
    K = ctx["K"]
    curr_dims = ctx["dims_by_time"][t]

    log_beta_tilde = _apply_closing_evl(
        log_beta.reshape((K,) + curr_dims),
        t,
        ctx["mvr_infos"],
        ctx["active_by_time"][t],
        log=True,
        lead=1,
    ).reshape(K, -1)

    # upd is deterministic, so pulling backward is a gather -- the forward scatter's transpose.
    followed = log_beta_tilde.gather(1, _dest_at(ctx, t))
    term = followed + ctx["log_emit_weights"][t].reshape(K, 1)

    shift = _safe_shift(term, dim=0, keepdim=True)
    log_beta_prev = torch.einsum(
        "hg,gm->hm",
        ctx["transition_mat"],
        (term - shift).exp(),
    )
    log_beta_prev = log_beta_prev.log() + shift

    return log_beta_prev, log_beta_tilde


def _forward_messages(ctx):
    """Log-space forward recursion: every message, plus the accumulated shift."""
    log_P = _initial_sumprod_message(ctx)

    log_alpha = []
    log_norm = torch.zeros((), dtype=ACCUM_DTYPE, device=ctx["device"])

    for t in range(ctx["T"]):
        if t:
            log_P = _sum_step(ctx, log_P, t)

        shift = log_P.max()

        # Must run before the subtraction: -inf - -inf is NaN.
        if not torch.isfinite(shift):
            raise InvalidInputError(f"No feasible augmented path at time {t}.")

        log_norm = log_norm + shift
        log_P = log_P - shift

        log_alpha.append(log_P[0])

    return log_alpha, log_norm


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
