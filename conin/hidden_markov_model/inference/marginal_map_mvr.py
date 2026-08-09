"""
Marginal-MAP inference over the HMM x MVR product for a subset of times. The
product chain is itself an HMM, and this is plain marginal MAP over it: the
augmented state (hidden x mediation) is maximized at a user-defined list of
times and every other time-step is marginalized out.
NOTE: Marginalized times can still have emissions.

Runs entirely in log space, including the sum-product eliminations; read the
semiring notes in CLAUDE.md before changing that.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from conin.exceptions import InvalidInputError

from ..mvr_common import (
    ACCUM_DTYPE,
    NEG_INF,
    _build_sumprod_ctx,
    _decode_dynamic_augmented_path,
    _initial_sumprod_message,
    _maxplus_step,
    _sum_step,
    _sum_step_backward,
)

# ======================================================================
# Helpers
# ======================================================================


def _forward_prefix(ctx, first_query):
    """Sum forward from ``t = 0``, giving the weight of every augmented state there."""
    log_P = _initial_sumprod_message(ctx)

    for t in range(1, first_query + 1):
        log_P = _sum_step(ctx, log_P, t)

    return log_P[0]


def _is_allocation_failure(exc):
    """Whether an exception is torch failing to allocate rather than a real bug."""
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

    return not any(t in ctx["observed_index"] for t in range(s + 1, s_next))


def _gap_operator(ctx, s, s_next):
    """Sum out the interior of ``(s, s_next]`` into a transfer operator between its ends."""
    K = ctx["K"]
    src_total = K * math.prod(ctx["dims_by_time"][s])
    dst_total = K * math.prod(ctx["dims_by_time"][s_next])

    if _gap_is_bare(ctx, s, s_next):
        # A stochastic matrix stays stochastic under powering, so this cannot underflow.
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
        log_beta, _ = _sum_step_backward(ctx, log_beta, t)

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

    The HMM x MVR product is itself an HMM, and this is plain marginal MAP over
    it: the augmented state ``z = (x, m)`` is maximized at ``query_times`` and
    summed out everywhere else, with each constraint enforced by conditioning on
    ``evl(m) == True`` at the end of its window. The mediation half is maximized
    too, so the returned hidden path is the projection of an augmented argmax.
    Marginalized times still transition, still emit and still drive their MVRs;
    they simply do not commit to a value.

    Two answers this deliberately does not give. It is **not** Viterbi restricted
    to ``query_times`` -- maximizing a marginal is not marginalizing a maximum.
    And it is **not** ``argmax_h gamma[t]`` from ``forward_backward_mvr_chmm``,
    which sums the mediation out because a marginal is a sum.

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
    ctx = _build_sumprod_ctx(
        model,
        observed,
        time_horizon=time_horizon,
        dtype=dtype,
        device=device,
    )

    hmm = ctx["hmm"]
    K, T = ctx["K"], ctx["T"]
    mvr_infos = ctx["mvr_infos"]
    active_by_time = ctx["active_by_time"]
    dims_by_time = ctx["dims_by_time"]
    shapes = ctx["shapes"]
    log_emit_weights = ctx["log_emit_weights"]
    log_transition_t = ctx["log_transition_t"]

    times = _resolve_query_times(query_times, T)

    if len(times) == T:
        warnings.warn(
            "query_times covers the whole horizon, so this reduces to Viterbi. "
            "viterbi_torch_mvr_chmm computes the same result more cheaply.",
            UserWarning,
            stacklevel=2,
        )

    def check_feasible(V, where):
        if not torch.isfinite(V.max()):
            raise InvalidInputError(f"No feasible augmented path {where}.")

    # ------------------------------------------------------------------
    # Everything before the first query time is summed out.
    # ------------------------------------------------------------------
    V = _forward_prefix(ctx, times[0]).reshape(shapes[times[0]])
    check_feasible(V, f"at time {times[0]}")

    # Wider than the tensors it sums: with one query time per step this accumulates T times.
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
