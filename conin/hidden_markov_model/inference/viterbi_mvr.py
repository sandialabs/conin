from __future__ import annotations

import math

import numpy as np
import torch

from conin.exceptions import InvalidInputError

from .mvr_common import (
    ACCUM_DTYPE,
    NEG_INF,
    _apply_closing_evl,
    _build_mvr_factor_info,
    _c_strides,
    _decode_dynamic_augmented_path,
    _hmm_to_torch,
    _maxplus_step,
    _resolve_emit_weights,
)

# ======================================================================
# Viterbi
# ======================================================================


def viterbi_torch_mvr_chmm(
    model,
    observed,
    *,
    time_horizon=None,
    dtype=torch.float32,
    device="cpu",
    return_augmented=True,
    return_score=False,
):
    """Log-space Viterbi for an MVR-augmented constrained HMM.

    Augmented axes are dynamic. The value tensor carries only the MVRs enforced
    at the current time::

        V[t].shape == (K,) + tuple(M_i(t) for i active at t)

    An MVR is active over ``[a_i, b_i]``, taken from its ``time_range`` and
    defaulting to ``[0, T - 1]``. It is initialized at ``a_i``, evaluated at
    ``b_i``, and absent otherwise.

    The MVR updates are deterministic, so each transition is a scatter rather
    than a dense mediation-space multiply. The MVRs are folded into a single
    destination index one at a time, so no product MVR is constructed and
    the working tensors scale with the mediation space rather than its square.

    Each slice is shifted so its maximum is ``0``, with the shift accumulated
    into the score. This is not optional: log space rules out overflow, but the
    ULP grows with the running magnitude, so without the shift every addition
    rounds at the ULP of a value that keeps growing. Measured at ``T = 20000``
    the score drifts by 5.49 nats unshifted against 0.0006 shifted, and because
    the rounding differs per entry it can also reorder near-tied paths.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints.
    observed : list or dict
        Observed sequence in external observed-state labels, either a dense
        list timed by position or a sparse ``{time: label}`` map. A time with
        no observation is scored by its transition alone; it stays in the
        chain rather than being dropped from it.
    time_horizon : int, optional
        Number of time steps to infer over. Defaults to ``len(observed)`` for a
        dense list and is required for a map. May exceed the number of
        observations, in which case the unobserved tail is inferred from the
        transitions and the constraints. Note that an MVR with no
        ``time_range`` then defaults to ``[0, time_horizon - 1]``, so it is
        enforced over the extended horizon and not merely the observed span.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors. ``float32`` is ample in log space.
    device : str or torch.device, optional
        Torch device.
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
    """
    hmm = getattr(model, "hidden_markov_model", None) or getattr(model, "hmm", None)

    if hmm is None:
        raise InvalidInputError("Missing hidden_markov_model from MVR_CHMM")

    constraints = list(getattr(model, "constraints", None) or [])

    log_start_vec, log_transition_mat, log_emission_mat = _hmm_to_torch(
        hmm,
        log=True,
        dtype=dtype,
        device=device,
    )

    T, log_emit_weights, _ = _resolve_emit_weights(
        hmm,
        log_emission_mat,
        observed,
        time_horizon,
        log=True,
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
        [i for i, info in enumerate(mvr_infos) if t in info["evl_at"]] for t in range(T)
    ]
    dims_by_time = [
        tuple(mvr_infos[i]["evl_at"][t].shape[0] for i in active_by_time[t])
        for t in range(T)
    ]
    shapes = [(K,) + dims_by_time[t] for t in range(T)]

    # log_transition_mat is indexed [h_prev, h_curr]
    log_transition_t = log_transition_mat.transpose(0, 1).contiguous()

    backptr = []
    # Kept on device and read once at the end. The per-step host sync comes from
    # the feasibility check below, not from here.
    log_score = torch.zeros((), dtype=ACCUM_DTYPE, device=device)

    # ------------------------------------------------------------------
    # Initialization at t = 0.
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
    V_prev = _apply_closing_evl(
        V_prev.reshape(shapes[0]), 0, mvr_infos, active_0, log=True
    )

    scale = V_prev.max()

    if not torch.isfinite(scale):
        raise InvalidInputError("No feasible augmented path at time 0.")

    log_score = log_score + scale
    V_prev = V_prev - scale

    # ------------------------------------------------------------------
    # Forward pass.
    # ------------------------------------------------------------------
    for t in range(1, T):
        V_curr, step_backptr = _maxplus_step(
            V_prev,
            t,
            K=K,
            mvr_infos=mvr_infos,
            prev_active=active_by_time[t - 1],
            curr_active=active_by_time[t],
            prev_dims=dims_by_time[t - 1],
            curr_dims=dims_by_time[t],
            log_transition_t=log_transition_t,
            log_emit_t=log_emit_weights[t],
            dtype=dtype,
            device=device,
        )

        backptr.append(step_backptr)

        scale = V_curr.max()

        if not torch.isfinite(scale):
            raise InvalidInputError(f"No feasible augmented path at time {t}.")

        log_score = log_score + scale
        V_prev = V_curr - scale

    # ------------------------------------------------------------------
    # Termination.
    # ------------------------------------------------------------------
    final_scale = V_prev.max()

    if not torch.isfinite(final_scale):
        raise InvalidInputError("No feasible augmented path at final time.")

    log_score = float(log_score + final_scale)

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
