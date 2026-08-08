"""
Marginal-MAP inference over the HMM x MVR product for a subset of times. Finds
the maximizing sequence X_times of P(X_times | Y), where times is a user-defined
list of times. All other time-steps are marginalized out.
NOTE: Marginalized times can still have emissions.

Runs entirely in log space, including the sum-product eliminations. Read the
semiring notes in CLAUDE.md before changing that: a gap operator holds the
probability of satisfying a constraint across the gap, which decays
exponentially in its length and silently underflows in probability space.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from conin.exceptions import InvalidInputError

from .mvr_common import (
    ACCUM_DTYPE,
    NEG_INF,
    _apply_closing_evl,
    _build_destination_index,
    _build_mvr_factor_info,
    _c_strides,
    _decode_dynamic_augmented_path,
    _hmm_to_torch,
    _maxplus_step,
    _resolve_emit_weights,
)

# ======================================================================
# Helpers
# ======================================================================


def _dest_at(ctx, t):
    """Destination index for the ``t-1 -> t`` transition, cached across segments."""
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
    """
    Largest entry along ``dim``, with an all ``-inf`` slice reported as ``0``.
    Subtracting a ``-inf`` shift would turn that slice into ``NaN``.
    """
    shift = values.amax(dim=dim, keepdim=keepdim)

    return torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))


def _sum_step(ctx, log_P, t):
    """Advance a batch of augmented log-weights one step, summing over predecessors."""
    K = ctx["K"]
    B = log_P.shape[0]

    prev_dims = ctx["dims_by_time"][t - 1]
    curr_dims = ctx["dims_by_time"][t]
    prev_total = math.prod(prev_dims)
    curr_total = math.prod(curr_dims)

    # Contract the predecessor hidden state. Shifting by the per-(batch, mediation)
    # maximum first makes this an ordinary contraction rather than a logsumexp, so
    # the einsum keeps its (B, K, M) shape instead of growing a second hidden axis.
    # These tensors are the memory bottleneck of the whole algorithm, so the
    # arithmetic below is done in place wherever the operand is already a
    # temporary of ours.
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


def _forward_prefix(ctx, first_query):
    """Sum forward from ``t = 0``, giving the weight of every augmented state there."""
    K = ctx["K"]
    device = ctx["device"]

    dims_0 = ctx["dims_by_time"][0]
    total_0 = math.prod(dims_0)
    strides_0 = _c_strides(dims_0)

    flat_ini = torch.zeros(K, dtype=torch.long, device=device)

    for pos, mvr_i in enumerate(ctx["active_by_time"][0]):
        flat_ini = flat_ini + ctx["mvr_infos"][mvr_i]["ini_idx"] * strides_0[pos]

    log_P = torch.full((1, K, total_0), NEG_INF, dtype=ctx["dtype"], device=device)
    log_P[0, torch.arange(K, device=device), flat_ini] = (
        ctx["log_start_vec"] + ctx["log_emit_weights"][0]
    )

    log_P = _apply_closing_evl(
        log_P.reshape((1, K) + dims_0),
        0,
        ctx["mvr_infos"],
        ctx["active_by_time"][0],
        log=True,
        lead=2,
    ).reshape(1, K, total_0)

    for t in range(1, first_query + 1):
        log_P = _sum_step(ctx, log_P, t)

    return log_P[0]


def _is_allocation_failure(exc):
    """
    Whether an exception is torch failing to allocate rather than a real bug.
    Without this an ordinary ``RuntimeError`` would be relabelled as a memory
    problem and send the reader after the wrong thing.
    """
    if isinstance(exc, MemoryError):
        return True

    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True

    text = str(exc).lower()

    return any(
        marker in text
        for marker in ("out of memory", "can't allocate", "cannot allocate", "alloc")
    )


def _gap_is_bare(ctx, s, s_next):
    """Whether ``(s, s_next]`` carries no mediation axis and no interior observation."""
    if any(ctx["active_by_time"][t] for t in range(s, s_next + 1)):
        return False

    return not any(t in ctx["observed_times"] for t in range(s + 1, s_next))


def _gap_operator(ctx, s, s_next):
    """
    Sum out the interior of ``(s, s_next]`` into a transfer operator between its ends.
    Propagating a batched identity is what makes the result a matrix rather than
    a vector: the source is a maximized variable and so cannot be contracted.
    """
    K = ctx["K"]
    src_total = K * math.prod(ctx["dims_by_time"][s])
    dst_total = K * math.prod(ctx["dims_by_time"][s_next])

    if _gap_is_bare(ctx, s, s_next):
        # No mediation axis and nothing observed in between, so the whole
        # interior collapses to a power of the transition matrix. That matrix is
        # stochastic, so its powers stay stochastic and cannot underflow.
        operator = torch.linalg.matrix_power(ctx["transition_mat"], s_next - s)

        return operator.log() + ctx["log_emit_weights"][s_next].reshape(1, K)

    try:
        log_P = torch.full(
            (src_total, src_total),
            NEG_INF,
            dtype=ctx["dtype"],
            device=ctx["device"],
        )
        log_P.fill_diagonal_(0.0)
        log_P = log_P.reshape(src_total, K, -1)

        for t in range(s + 1, s_next + 1):
            log_P = _sum_step(ctx, log_P, t)
    except (RuntimeError, MemoryError) as exc:
        if not _is_allocation_failure(exc):
            raise

        # Torch reports the byte count but not the cause, which is always an MVR
        # window covering times that are being summed out.
        spanning = sorted(
            set(ctx["active_by_time"][s]) & set(ctx["active_by_time"][s_next])
        )
        raise InvalidInputError(
            f"Summing out the interior of ({s}, {s_next}] needs a "
            f"{src_total} x {dst_total} transfer operator "
            f"({src_total * dst_total} entries). MVRs {spanning} span this gap. "
            f"Align their time_range to a query time so the gap carries no "
            f"mediation axis."
        ) from exc

    return log_P.reshape(src_total, dst_total)


def _backward_suffix(ctx, last_query):
    """Sum backward from the end of the horizon down to the last query time."""
    K = ctx["K"]
    T = ctx["T"]

    # log 1 -- the empty future contributes nothing.
    log_beta = torch.zeros(
        (K, math.prod(ctx["dims_by_time"][T - 1])),
        dtype=ctx["dtype"],
        device=ctx["device"],
    )

    for t in range(T - 1, last_query, -1):
        curr_dims = ctx["dims_by_time"][t]

        log_beta = _apply_closing_evl(
            log_beta.reshape((K,) + curr_dims),
            t,
            ctx["mvr_infos"],
            ctx["active_by_time"][t],
            log=True,
            lead=1,
        ).reshape(K, -1)

        # The mediation update is deterministic, so pulling a message backward
        # through it is a gather -- the transpose of the forward scatter.
        followed = log_beta.gather(1, _dest_at(ctx, t))
        term = followed + ctx["log_emit_weights"][t].reshape(K, 1)

        shift = _safe_shift(term, dim=0, keepdim=True)
        log_beta = torch.einsum(
            "hg,gm->hm",
            ctx["transition_mat"],
            (term - shift).exp(),
        )
        log_beta = log_beta.log() + shift

    return log_beta


# ======================================================================
# Query times
# ======================================================================


def _resolve_query_times(query_times, T):
    """Normalize the query set to a sorted list of distinct times inside the horizon."""
    if query_times is None:
        return list(range(T))

    times = list(query_times)

    for t in times:
        if not isinstance(t, (int, np.integer)):
            raise InvalidInputError(f"Query time must be an integer: {t!r}")
        if not 0 <= t < T:
            raise InvalidInputError(f"Query time {t} is outside the horizon [0, {T}).")

    times = sorted({int(t) for t in times})

    if not times:
        raise InvalidInputError("query_times must name at least one time.")

    return times


# ======================================================================
# Marginal MAP
# ======================================================================


def marginal_map_torch_mvr_chmm(
    model,
    observed,
    *,
    time_horizon=None,
    query_times=None,
    dtype=torch.float32,
    device="cpu",
    return_augmented=True,
    return_score=False,
):
    """Marginal-MAP decoding of an MVR-augmented constrained HMM at chosen times.

    Returns the assignment to the hidden states at ``query_times`` that
    maximizes their joint posterior, with the hidden states at all other times
    summed out::

        argmax over x_S of  sum over x_notS of  P(x, y) * 1[constraints hold]

    **This maximizes a marginal rather than marginalizing a MAP.** The states
    reported here are in general not the states the joint MAP path from Viterbi
    takes at those times. Use ``viterbi_torch_mvr_chmm`` when the whole path is
    wanted.

    Marginalized times still consume a transition, still emit if they carry an
    observation, and are still driven through every MVR active at that time.
    What is dropped is only the requirement to commit to a single value there.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints.
    observed : list or dict
        Observed sequence in external observed-state labels, either a dense
        list timed by position or a sparse ``{time: label}`` map.
    time_horizon : int, optional
        Number of time steps to infer over. Defaults to ``len(observed)`` for a
        dense list and is required for a map.
    query_times : sequence of int, optional
        Times to maximize over. Defaults to every time, which reproduces
        Viterbi exactly and warns, since the dedicated implementation is
        cheaper.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors.
    device : str or torch.device, optional
        Torch device.
    return_augmented : bool, optional
        If ``True``, also return the decoded augmented path.
    return_score : bool, optional
        If ``True``, also return the log of the maximized marginal.

    Returns
    -------
    hidden_path : list
        Hidden states at the query times, in external labels, in time order.
    augmented_path : list[dict], optional
        Returned when ``return_augmented`` is ``True``. Each entry carries the
        global ``time`` it refers to.
    best_logprob : float, optional
        Returned when ``return_score`` is ``True``.
    """
    hmm = getattr(model, "hidden_markov_model", None) or getattr(model, "hmm", None)

    if hmm is None:
        raise InvalidInputError("Missing hidden_markov_model from MVR_CHMM")

    constraints = list(getattr(model, "constraints", None) or [])

    # Everything runs in log space, including the sum-product gap eliminations.
    # Probability space is not viable for those: a gap operator holds the
    # probability of satisfying a constraint across the gap, which decays
    # exponentially in its length and silently flushes to zero.
    _, transition_mat, _ = _hmm_to_torch(hmm, log=False, dtype=dtype, device=device)
    log_start_vec, log_transition_mat, log_emission_mat = _hmm_to_torch(
        hmm, log=True, dtype=dtype, device=device
    )

    T, log_emit_weights, observed_times = _resolve_emit_weights(
        hmm, log_emission_mat, observed, time_horizon, log=True
    )

    K = int(log_start_vec.shape[0])
    times = _resolve_query_times(query_times, T)

    if len(times) == T:
        warnings.warn(
            "query_times covers the whole horizon, so this reduces to Viterbi. "
            "viterbi_torch_mvr_chmm computes the same result more cheaply.",
            UserWarning,
            stacklevel=2,
        )

    mvr_infos = [
        _build_mvr_factor_info(mvr, T, log=True, dtype=dtype, device=device)
        for mvr in constraints
    ]

    active_by_time = [
        [i for i, info in enumerate(mvr_infos) if t in info["evl_at"]] for t in range(T)
    ]
    dims_by_time = [
        tuple(mvr_infos[i]["evl_at"][t].shape[0] for i in active_by_time[t])
        for t in range(T)
    ]
    shapes = [(K,) + dims_by_time[t] for t in range(T)]

    ctx = {
        "K": K,
        "T": T,
        "log_start_vec": log_start_vec,
        "transition_mat": transition_mat,
        "log_emit_weights": log_emit_weights,
        "observed_times": observed_times,
        "mvr_infos": mvr_infos,
        "active_by_time": active_by_time,
        "dims_by_time": dims_by_time,
        "dtype": dtype,
        "device": device,
        "dest_cache": {},
    }

    # log_transition_mat is indexed [h_prev, h_curr]
    log_transition_t = log_transition_mat.transpose(0, 1).contiguous()

    def check_feasible(V, where):
        if not torch.isfinite(V.max()):
            raise InvalidInputError(f"No feasible augmented path {where}.")

    # ------------------------------------------------------------------
    # Everything before the first query time is summed out.
    # ------------------------------------------------------------------
    V = _forward_prefix(ctx, times[0]).reshape(shapes[times[0]])
    check_feasible(V, f"at time {times[0]}")

    # Held wider than the tensors it sums, and read once at the end. With one
    # query time per step this accumulates T times, so float32 here would cost
    # nats at long horizons.
    log_score = torch.zeros((), dtype=ACCUM_DTYPE, device=device)

    # ------------------------------------------------------------------
    # Max-plus recursion over the coarse chain of query times.
    # ------------------------------------------------------------------
    backptr = []

    for prev_time, curr_time in zip(times, times[1:]):
        if curr_time == prev_time + 1:
            # Adjacent: no interior to sum out, so take the ordinary scatter
            # step and never materialize an operator.
            V, step_backptr = _maxplus_step(
                V,
                curr_time,
                K=K,
                mvr_infos=mvr_infos,
                prev_active=active_by_time[prev_time],
                curr_active=active_by_time[curr_time],
                prev_dims=dims_by_time[prev_time],
                curr_dims=dims_by_time[curr_time],
                log_transition_t=log_transition_t,
                log_emit_t=log_emit_weights[curr_time],
                dtype=dtype,
                device=device,
            )
        else:
            # Fold V into the operator in place: it is the largest tensor here,
            # so an extra copy of it is worth avoiding.
            candidate = _gap_operator(ctx, prev_time, curr_time)
            candidate += V.reshape(-1, 1)

            best, step_backptr = candidate.max(dim=0)

            V = best.reshape(shapes[curr_time])
            step_backptr = step_backptr.reshape(shapes[curr_time])

        backptr.append(step_backptr)

        scale = V.max()
        check_feasible(V, f"at time {curr_time}")

        log_score = log_score + scale
        V = V - scale

    # ------------------------------------------------------------------
    # Everything after the last query time is summed out.
    # ------------------------------------------------------------------
    V = V + _backward_suffix(ctx, times[-1]).reshape(shapes[times[-1]])
    check_feasible(V, "at the final query time")

    log_score = float(log_score + V.max())

    # ------------------------------------------------------------------
    # Backtracking over the coarse chain.
    # ------------------------------------------------------------------
    final_flat = int(torch.argmax(V).item())
    curr_idx = tuple(int(x) for x in np.unravel_index(final_flat, shapes[times[-1]]))

    augmented_index_path = [curr_idx]

    for position in range(len(times) - 1, 0, -1):
        prev_flat = int(backptr[position - 1][curr_idx].item())
        curr_idx = tuple(
            int(x) for x in np.unravel_index(prev_flat, shapes[times[position - 1]])
        )
        augmented_index_path.append(curr_idx)

    augmented_index_path.reverse()

    hidden_path = [
        hmm.hidden_to_external[aug_idx[0]] for aug_idx in augmented_index_path
    ]

    if return_augmented:
        augmented_path = _decode_dynamic_augmented_path(
            augmented_index_path,
            active_by_time,
            mvr_infos,
            times=times,
        )

        if return_score:
            return hidden_path, augmented_path, log_score

        return hidden_path, augmented_path

    if return_score:
        return hidden_path, log_score

    return hidden_path
