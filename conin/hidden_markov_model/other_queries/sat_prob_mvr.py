"""
Satisfaction-probability inference over the HMM x MVR product. Computes the probability
that a designated target MVR is satisfied, conditioned on the observations and on every
other constraint being satisfied.
"""

from __future__ import annotations

import torch

from conin.exceptions import InvalidInputError

from ..mvr_common import (
    ACCUM_DTYPE,
    _build_sumprod_ctx,
    _initial_sumprod_message,
    _model_parts,
    _sum_step,
)
from .sat_time_mvr import _detach_target, _resolve_target, _split_on_target


# ======================================================================
# Helpers
# ======================================================================


def _branch_weights(ctx, target, b):
    """Log weight of the target holding at ``b`` and of it failing there, as a ``(2,)``."""
    log_P = _initial_sumprod_message(ctx)
    log_norm = torch.zeros((), dtype=ACCUM_DTYPE, device=ctx["device"])

    for t in range(ctx["T"]):
        if t:
            log_P = _sum_step(ctx, log_P, t)

        # Split into the batch axis rather than close the window; the suffix weighs
        # the two branches apart.
        if t == b:
            log_P = torch.cat(_split_on_target(ctx, log_P, t, target), dim=0)

        shift = log_P.max()

        # The branches partition the lattice, so both dying means the rest is infeasible.
        if not torch.isfinite(shift):
            raise InvalidInputError(f"No feasible augmented path at time {t}.")

        log_norm = log_norm + shift
        log_P = log_P - shift

    return torch.logsumexp(log_P.reshape(2, -1), dim=1).to(ACCUM_DTYPE) + log_norm


# ======================================================================
# Satisfaction Probability
# ======================================================================


def sat_prob_torch_mvr_chmm(
    model,
    observed,
    *,
    target,
    time_horizon=None,
    dtype=torch.float64,
    device="cpu",
    return_log_weights=False,
):
    """Probability that a target MVR is satisfied.

    Returns, normalized between the two branches::

        w[True]  = P(observed, other constraints satisfied, target's evl holds at b)
        w[False] = P(observed, other constraints satisfied, target's evl fails at b)

    where ``b`` ends the target's ``time_range`` and defaults to the last time
    step. Every constraint other than the target is enforced as usual.

    That is **not** "satisfied at some time in ``[a, b]``". For that, pass
    ``mvr_already_satisfied(target)``, whose accept state is absorbing.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints.
    observed : list or dict
        Observed sequence in external observed-state labels, either a dense list
        timed by position or a sparse ``{time: label}`` map.
    target : int or str
        Index into ``model.constraints`` (negative allowed) or an MVR ``name``. A
        name matching no constraint, or more than one, raises.
    time_horizon : int, optional
        Number of time steps. Defaults to ``len(observed)`` for a dense list and is
        required for a map.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors. Defaults to ``float64``, as in
        ``sat_time_mvr``.
    device : str or torch.device, optional
        Torch device.
    return_log_weights : bool, optional
        If ``True``, also return the unnormalized log weights.

    Returns
    -------
    prob : float
        Probability that the target is satisfied.
    log_weights : torch.Tensor, optional
        ``(2,)`` unnormalized log weights ``[satisfied, violated]``, in ``float64``.
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

    _, b = _detach_target(ctx, target)

    log_w = _branch_weights(ctx, target, b)
    prob = float((log_w[0] - torch.logsumexp(log_w, dim=0)).exp())

    if return_log_weights:
        return prob, log_w

    return prob
