"""
Satisfaction-time inference over the HMM x MVR product. Computes the distribution
over the time at which a designated target MVR **first** accepts, conditioned on the
observations and on every other constraint being satisfied.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from conin.exceptions import InvalidInputError

from ..mvr_common import (
    ACCUM_DTYPE,
    NEG_INF,
    _build_static_context,
    _build_sumprod_ctx,
    _initial_sumprod_message,
    _model_parts,
    _sum_step,
    _sum_step_backward,
)

# Swapping exactly these rebases a context onto a different constraint set.
_STATIC_KEYS = ("mvr_infos", "active_by_time", "dims_by_time", "shapes", "dest_cache")


# ======================================================================
# Helpers
# ======================================================================


def _resolve_target(constraints, target):
    """Resolve an index or an MVR name into an index into ``constraints``."""
    if not constraints:
        raise InvalidInputError(
            "Satisfaction time needs at least one constraint to use as the target."
        )

    if isinstance(target, str):
        matches = [i for i, mvr in enumerate(constraints) if mvr.name == target]

        if len(matches) != 1:
            known = sorted(m.name for m in constraints if m.name is not None)
            raise InvalidInputError(
                f"target={target!r} matches {len(matches)} constraints; the named "
                f"constraints are {known}."
            )

        return matches[0]

    # bool is an int subclass; target=True meaning index 1 is worse than an error.
    if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
        if not -len(constraints) <= target < len(constraints):
            raise InvalidInputError(
                f"target={target} is out of range for {len(constraints)} constraints."
            )

        return int(target) % len(constraints)

    raise InvalidInputError(
        f"target must be an integer index or a constraint name, got {target!r}."
    )


def _reduced_ctx(ctx, target):
    """Rebase a context onto every constraint but the target, for the backward pass."""
    others = [mvr for i, mvr in enumerate(ctx["constraints"]) if i != target]

    static = _build_static_context(
        others,
        ctx["K"],
        ctx["T"],
        log=True,
        dtype=ctx["dtype"],
        device=ctx["device"],
    )

    return {**ctx, "constraints": others, **{k: static[k] for k in _STATIC_KEYS}}


def _detach_target(ctx, target):
    """Detach the target from ``_apply_closing_evl``; returns its window."""
    infos = list(ctx["mvr_infos"])
    info = infos[target]

    infos[target] = {**info, "b": None}
    ctx["mvr_infos"] = infos

    return info["a"], info["b"]

def _split_on_target(ctx, log_P, t, target):
    """Split a forward message into ``(accepting at t, not accepting at t)``."""
    dims = ctx["dims_by_time"][t]
    pos = ctx["active_by_time"][t].index(target)
    accept = ctx["mvr_infos"][target]["evl_at"][t]

    # evl is binary, so its log mask is 0 / -inf and the complement swaps the two.
    reject = torch.where(
        torch.isfinite(accept),
        torch.full_like(accept, NEG_INF),
        torch.zeros_like(accept),
    )

    view = [1, 1] + [1] * len(dims)
    view[2 + pos] = accept.shape[0]

    shaped = log_P.reshape((log_P.shape[0], ctx["K"]) + dims)

    return (
        (shaped + accept.reshape(view)).reshape(log_P.shape),
        (shaped + reject.reshape(view)).reshape(log_P.shape),
    )


def _hit_messages(ctx, target, a, b):
    """Not-yet-accepted forward messages, as ``{t: (accepting branch, shift)}``."""
    log_P = _initial_sumprod_message(ctx)
    log_norm = torch.zeros((), dtype=ACCUM_DTYPE, device=ctx["device"])

    hits = {}

    for t in range(b + 1):
        if t:
            log_P = _sum_step(ctx, log_P, t)

        if t >= a:
            hit, log_P = _split_on_target(ctx, log_P, t, target)
            hits[t] = (hit[0], log_norm.clone())

        shift = log_P.max()

        # Not an error: every feasible path has accepted, so no later time has mass.
        if not torch.isfinite(shift):
            break

        log_norm = log_norm + shift
        log_P = log_P - shift

    return hits


def _satisfaction_weights(ctx, bctx, target, a, b, hits):
    """Log weight of the target first accepting at each time in ``[a, b]``."""
    K, T = ctx["K"], ctx["T"]

    log_w = torch.full((b - a + 1,), NEG_INF, dtype=ACCUM_DTYPE, device=ctx["device"])

    # log 1 -- the empty future contributes nothing.
    log_beta = torch.zeros(
        (K, math.prod(bctx["dims_by_time"][T - 1])),
        dtype=ctx["dtype"],
        device=ctx["device"],
    )
    log_norm = torch.zeros((), dtype=ACCUM_DTYPE, device=ctx["device"])

    for t in range(T - 1, a - 1, -1):
        if t in hits:
            hit, fwd_norm = hits[t]

            # beta has no target axis; put one back where the forward lattice keeps it.
            pos = ctx["active_by_time"][t].index(target)
            beta = log_beta.reshape((K,) + bctx["dims_by_time"][t]).unsqueeze(1 + pos)

            joint = hit.reshape((K,) + ctx["dims_by_time"][t]) + beta

            # Both shifts are needed: weights at different times must stay comparable.
            log_w[t - a] = (
                torch.logsumexp(joint.reshape(-1), dim=0).to(ACCUM_DTYPE)
                + fwd_norm
                + log_norm
            )

        if t == a:
            break

        log_beta, _ = _sum_step_backward(bctx, log_beta, t)

        # Not an error: a state can simply have no feasible future.
        shift = log_beta.max()

        if torch.isfinite(shift):
            log_norm = log_norm + shift
            log_beta = log_beta - shift

    return log_w


# ======================================================================
# Satisfaction Time
# ======================================================================


def sat_time_torch_mvr_chmm(
    model,
    observed,
    *,
    target,
    time_horizon=None,
    dtype=torch.float64,
    device="cpu",
    return_log_weights=False,
):
    """First satisfaction time distribution for a target MVR.

    Over the target's window ``[a, b]`` -- its ``time_range``, defaulting to the
    whole horizon -- returns the distribution of the earliest time its ``evl``
    holds, normalized over that window::

        w[t] = P(observed, other constraints satisfied, target first accepts at t)

    Every constraint other than the target is enforced as usual. Note that
    ``time_range`` initializes the automaton at ``a`` rather than slicing the
    full-horizon answer: ``[5, T-1]`` starts fresh at ``t = 5`` and never sees
    ``hidden[0..4]``.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints.
    observed : list or dict
        Observed sequence in external observed-state labels, either a dense list
        timed by position or a sparse ``{time: label}`` map. A time with no
        observation still consumes a transition and still drives every MVR active
        there.
    target : int or str
        Index into ``model.constraints`` (negative allowed) or an MVR ``name``. A
        name matching no constraint, or more than one, raises.
    time_horizon : int, optional
        Number of time steps. Defaults to ``len(observed)`` for a dense list and is
        required for a map.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors. Defaults to ``float64``, as in
        ``baum_welch_mvr``.
    device : str or torch.device, optional
        Torch device.
    return_log_weights : bool, optional
        If ``True``, also return the unnormalized log weights.

    Returns
    -------
    times : list[int]
        The target's window, ``[a, a + 1, ..., b]``.
    probs : torch.Tensor
        ``(b - a + 1,)`` distribution over ``times``, in ``float64``.
    log_weights : torch.Tensor, optional
        unnormalized log weights. probs = normalized weights.
    """
    _, constraints = _model_parts(model)
    target = _resolve_target(constraints, target)

    ctx = _build_sumprod_ctx(
        model,
        observed,
        time_horizon=time_horizon,
        dtype=dtype,
        device=device,
    )

    bctx = _reduced_ctx(ctx, target)
    a, b = _detach_target(ctx, target)

    hits = _hit_messages(ctx, target, a, b)
    log_w = _satisfaction_weights(ctx, bctx, target, a, b, hits)

    total = torch.logsumexp(log_w, dim=0)

    if not torch.isfinite(total):
        raise InvalidInputError(
            f"No feasible path has target constraint {target} satisfied within its "
            f"window [{a}, {b}]. Either the target is never accepted or the "
            f"remaining constraints are infeasible."
        )

    times = list(range(a, b + 1))
    probs = (log_w - total).exp()

    if return_log_weights:
        return times, probs, log_w

    return times, probs
