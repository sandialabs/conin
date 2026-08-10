"""
Constrained stopped sampling. Draws a stopping time from the first-satisfaction-time
distribution of a designated target MVR, then draws the hidden-state prefix running
up to that time.
"""

from __future__ import annotations

import copy

import torch

from conin.exceptions import InvalidInputError

from ..mvr_common import NEG_INF, _model_parts
from ..mvr_operators import mvr_sattime
from ..other_queries.sat_time_mvr import _resolve_target, sat_time_torch_mvr_chmm
from .ffbs_mvr import _draw, _positive_int, ffbs_torch_mvr_chmm


# ======================================================================
# Stopped sampling
# ======================================================================


def stopped_sampling_torch_mvr_chmm(
    model,
    observed,
    *,
    target,
    num_samples=1,
    min_length=None,
    time_horizon=None,
    dtype=torch.float64,
    device="cpu",
    generator=None,
    return_times=False,
):
    """Sample hidden-state prefixes that stop when a target MVR is first satisfied.

    Draws the satisfaction time ``tau`` of the ``target`` from its posterior distribution
    using ``sat_time_mvr``, then samples a path from ``P(x[0..tau] | observed, all
    constraints, target first accepts at tau)``. The conditioning uses the whole
    observation set and every constraint over the whole horizon, so an observation
    later than ``tau`` still informs the sampled path.

    ``min_length`` bounds the length of the sampled path(s) from below.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints. It is
        not modified.
    observed : list or dict
        Observed sequence in external observed-state labels, either a dense list
        timed by position or a sparse ``{time: label}`` map.
    target : int or str
        Index into ``model.constraints`` (negative allowed) or an MVR ``name``.
    num_samples : int, optional
        Number of prefixes to draw.
    min_length : int, optional
        Shortest prefix to allow. Stopping times below it are dropped and the rest
        renormalized; raises if that leaves no mass.
    time_horizon : int, optional
        Number of time steps. Defaults to ``len(observed)`` for a dense list and is
        required for a map.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors.
    device : str or torch.device, optional
        Torch device.
    generator : torch.Generator, optional
        Generator for the draws, for reproducibility.
    return_times : bool, optional
        If ``True``, also return the sampled stopping times.

    Returns
    -------
    paths : list[list]
        ``num_samples`` hidden-state paths in external labels, of varying length.
        ``len(paths[n]) - 1`` is the stopping time of sample ``n``.
    times : torch.Tensor, optional
        ``(num_samples,)`` sampled stopping times, returned when ``return_times``
        is ``True``.
    """
    _positive_int("num_samples", num_samples)

    if min_length is not None:
        _positive_int("min_length", min_length)

    _, constraints = _model_parts(model)
    index = _resolve_target(constraints, target)

    # A no-op for the satisfaction-time pass, load-bearing for the path pass.
    prefix_free = constraints[index]

    if not prefix_free.prefix:
        # mvr_sattime sets the tag but, like every operator, drops the window.
        prefix_free = mvr_sattime(constraints[index])
        prefix_free._time_range = constraints[index]._time_range

    constraints = list(constraints)
    constraints[index] = prefix_free

    # CHMM stores the constraint list by reference, so hand it a fresh one.
    model = copy.copy(model)
    model.constraints = constraints

    times, _, log_weights = sat_time_torch_mvr_chmm(
        model,
        observed,
        target=index,
        time_horizon=time_horizon,
        dtype=dtype,
        device=device,
        return_log_weights=True,
    )

    if min_length is not None:
        # A prefix ending at tau has length tau + 1. Masking log weights rather
        # than probs matters: a far tail is already exactly 0 once normalized.
        keep = torch.as_tensor(
            [t >= min_length - 1 for t in times], device=log_weights.device
        )
        log_weights = log_weights.masked_fill(~keep, NEG_INF)

        if not torch.isfinite(log_weights.max()):
            raise InvalidInputError(
                f"min_length={min_length} leaves no feasible stopping time in the "
                f"target's window [{times[0]}, {times[-1]}]."
            )

    drawn = _draw(log_weights, num_samples, generator)
    sampled_times = torch.as_tensor(times, device=drawn.device)[drawn]

    paths = [None] * num_samples

    # Narrowing to [a, tau] puts the target's evl exactly at tau and drops it after.
    for tau in sorted(set(sampled_times.tolist())):
        rows = [n for n, t in enumerate(sampled_times.tolist()) if t == tau]

        narrowed = copy.copy(prefix_free)
        narrowed._time_range = [int(times[0]), int(tau)]

        sub_constraints = list(constraints)
        sub_constraints[index] = narrowed

        sub_model = copy.copy(model)
        sub_model.constraints = sub_constraints

        sampled = ffbs_torch_mvr_chmm(
            sub_model,
            observed,
            num_samples=len(rows),
            time_horizon=time_horizon,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        for row, path in zip(rows, sampled):
            paths[row] = path[: tau + 1]

    if return_times:
        return paths, sampled_times

    return paths
