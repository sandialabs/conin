"""
Forward-filtering backward-sampling over the augmented chain. Draws hidden-state
paths from the posterior given the observations and the fact that the path satisfies
every constraint.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from conin.exceptions import InvalidInputError

from ..mvr_common import (
    NEG_INF,
    _build_sumprod_ctx,
    _decode_dynamic_augmented_path,
    _dest_at,
    _forward_messages,
)


# ======================================================================
# Helpers
# ======================================================================


def _positive_int(name, value):
    """Reject anything that is not a positive integer, ``bool`` included."""
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise InvalidInputError(f"{name} must be a positive integer, got {value!r}.")


def _draw(log_weights, num_samples, generator):
    """Draw flat indices from log-weights, independently per row."""
    weights = (log_weights - log_weights.amax(dim=-1, keepdim=True)).exp()

    # Without replacement=True, multinomial returns zero-probability categories.
    return torch.multinomial(
        weights, num_samples, replacement=True, generator=generator
    )


def _backward_sample(ctx, log_alpha, start_flat, start_time, *, generator):
    """Sample augmented states back to ``0`` from the indices at ``start_time``."""
    K = ctx["K"]
    N = int(start_flat.shape[0])
    device = ctx["device"]

    hidden = torch.empty((N, start_time + 1), dtype=torch.long, device=device)
    mediation = torch.empty((N, start_time + 1), dtype=torch.long, device=device)

    total = math.prod(ctx["dims_by_time"][start_time])
    hidden[:, start_time] = torch.div(start_flat, total, rounding_mode="floor")
    mediation[:, start_time] = start_flat % total

    # The update must land on the successor: an equality test, not a scatter.
    for t in range(start_time - 1, -1, -1):
        prev_total = math.prod(ctx["dims_by_time"][t])

        h_next = hidden[:, t + 1]
        dest = _dest_at(ctx, t + 1)

        reachable = dest[h_next] == mediation[:, t + 1].unsqueeze(1)

        log_transition = ctx["log_transition_mat"][:, h_next].T

        log_w = log_alpha[t].unsqueeze(0) + log_transition.unsqueeze(2)
        log_w = log_w.masked_fill(~reachable.unsqueeze(1), NEG_INF)

        flat = _draw(log_w.reshape(N, K * prev_total), 1, generator).squeeze(1)

        hidden[:, t] = torch.div(flat, prev_total, rounding_mode="floor")
        mediation[:, t] = flat % prev_total

    return hidden, mediation


def _decode_samples(ctx, hidden, mediation):
    """Decode each sample's augmented indices into the per-time dictionary form."""
    hidden = hidden.cpu().numpy()
    mediation = mediation.cpu().numpy()

    # Unravel once per time over every sample; only the label lookup is per-sample.
    coords_at = [
        np.unravel_index(mediation[:, t], dims) if dims else ()
        for t, dims in enumerate(ctx["dims_by_time"])
    ]

    return [
        _decode_dynamic_augmented_path(
            [
                (hidden[n, t],) + tuple(axis[n] for axis in coords_at[t])
                for t in range(ctx["T"])
            ],
            ctx["active_by_time"],
            ctx["mvr_infos"],
        )
        for n in range(hidden.shape[0])
    ]


# ======================================================================
# FFBS
# ======================================================================


def ffbs_torch_mvr_chmm(
    model,
    observed,
    *,
    num_samples=1,
    time_horizon=None,
    dtype=torch.float64,
    device="cpu",
    generator=None,
    return_indices=False,
    return_augmented=False,
):
    """Sample hidden paths from an MVR-constrained hidden Markov model.

    Draws ``num_samples`` independent paths from ``P(x | observed, constraints
    satisfied)`` via FFBS.

    ``generator`` must live on ``device``. Unlike ``viterbi_torch_mvr_chmm``,
    ``return_augmented`` defaults to ``False``: decoding mediation labels is a
    per-sample Python loop and is rarely wanted for a whole batch.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints.
    observed : list or dict
        Observed sequence in external observed-state labels, either a dense list
        timed by position or a sparse ``{time: label}`` map. A time with no
        observation still consumes a transition and still drives every MVR active
        there.
    num_samples : int, optional
        Number of paths to draw.
    time_horizon : int, optional
        Number of time steps. Defaults to ``len(observed)`` for a dense list and is
        required for a map.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors. Defaults to ``float64``, as in
        ``baum_welch_mvr``.
    device : str or torch.device, optional
        Torch device.
    generator : torch.Generator, optional
        Generator for the draws, for reproducibility.
    return_indices : bool, optional
        If ``True``, also return the sampled hidden states as integer indices.
    return_augmented : bool, optional
        If ``True``, also return the decoded augmented path of each sample.

    Returns
    -------
    paths : list[list]
        ``num_samples`` hidden-state paths in external labels.
    indices : torch.Tensor, optional
        ``(num_samples, T)`` hidden-state indices, returned when
        ``return_indices`` is ``True``.
    augmented : list[list[dict]], optional
        Returned when ``return_augmented`` is ``True``.
    """
    _positive_int("num_samples", num_samples)

    ctx = _build_sumprod_ctx(
        model,
        observed,
        time_horizon=time_horizon,
        dtype=dtype,
        device=device,
    )

    log_alpha, _ = _forward_messages(ctx)

    # log_alpha[T-1] carries every factor: no window can end past T-1.
    start_flat = _draw(log_alpha[ctx["T"] - 1].reshape(-1), num_samples, generator)

    hidden, mediation = _backward_sample(
        ctx,
        log_alpha,
        start_flat,
        ctx["T"] - 1,
        generator=generator,
    )

    hidden_to_external = ctx["hmm"].hidden_to_external
    paths = [[hidden_to_external[h] for h in row] for row in hidden.tolist()]

    result = (paths,)

    if return_indices:
        result += (hidden,)

    if return_augmented:
        result += (_decode_samples(ctx, hidden, mediation),)

    return result[0] if len(result) == 1 else result
