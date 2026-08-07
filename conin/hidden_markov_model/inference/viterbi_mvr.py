from __future__ import annotations

import numpy as np
import torch

from conin.exceptions import InvalidInputError


def _get_mvr_chmm_parts(model):
    """
    Extract hidden_markov_model and constraints from an MVR_CHMM-like object.

    This is defensive because the exact parent CHMM attribute names were not shown.
    """
    hmm = getattr(model, "hidden_markov_model", None)
    if hmm is None:
        hmm = getattr(model, "hmm", None)
    if hmm is None:
        hmm = getattr(model, "_hidden_markov_model", None)

    constraints = getattr(model, "constraints", None)
    if constraints is None:
        constraints = getattr(model, "_constraints", None)
    if constraints is None:
        constraints = []

    if hmm is None:
        raise InvalidInputError(
            "Could not find hidden_markov_model on the supplied MVR_CHMM object."
        )

    return hmm, list(constraints)


def _hmm_to_torch(hmm, *, dtype=torch.float32, device="cpu"):
    """
    Convert a HiddenMarkovModel into torch tensors.

    Returns
    -------
    start_vec : torch.Tensor, shape (K,)
    transition_mat : torch.Tensor, shape (K, K)
    emission_mat : torch.Tensor, shape (K, O)
    """
    repn = hmm.repn
    if repn is None:
        raise InvalidInputError("hidden_markov_model.repn is missing.")

    start_vec = torch.as_tensor(repn.start_vec, dtype=dtype, device=device)
    transition_mat = torch.as_tensor(repn.transition_mat, dtype=dtype, device=device)
    emission_mat = torch.as_tensor(repn.emission_mat, dtype=dtype, device=device)

    return start_vec, transition_mat, emission_mat


def _observations_to_emit_weights(hmm, emission_mat, observed):
    """
    Convert external observation labels to emission weights.

    Returns
    -------
    emit_weights : torch.Tensor, shape (T, K)
        emit_weights[t, h] = P(observed[t] | hidden[t] = h)
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


def _is_inhom_mvr(mvr):
    """
    Return True if the MVR representation is time-inhomogeneous.
    """
    return isinstance(mvr.repn.evl_array, list)


def _mvr_active_range(mvr, T):
    """
    Return active global time range [a, b] for an MVR.

    If mvr._time_range is None, default to [0, T - 1].

    For an InhomMVR, require

        b - a <= mvr.time_horizon

    using the convention

        time_horizon = len(mediation_states) - 1.
    """
    if T <= 0:
        raise InvalidInputError("observed sequence must be nonempty.")

    time_range = getattr(mvr, "_time_range", None)

    if time_range is None:
        active_start = 0
        active_end = T - 1
    else:
        active_start, active_end = time_range

    if active_start < 0:
        raise InvalidInputError("MVR active_start must be nonnegative.")

    if active_end < active_start:
        raise InvalidInputError("MVR active_end must be >= active_start.")

    if active_end >= T:
        raise InvalidInputError(
            f"MVR time_range [{active_start}, {active_end}] exceeds "
            f"observation horizon T={T}."
        )

    if _is_inhom_mvr(mvr):
        required_local_horizon = active_end - active_start
        if required_local_horizon > mvr.time_horizon:
            raise InvalidInputError(
                "InhomMVR time_horizon is too short for requested inference range. "
                f"Required local horizon {required_local_horizon}, "
                f"but mvr.time_horizon is {mvr.time_horizon}."
            )

    return active_start, active_end


def _hidden_permutation_for_mvr(mvr, hmm_hidden_order):
    """
    Compute a permutation so that MVR arrays are reordered to match the HMM's
    internal hidden-state order.
    """
    mvr_hidden_order = list(mvr.hidden_states)

    if set(mvr_hidden_order) != set(hmm_hidden_order):
        raise InvalidInputError("MVR hidden states do not match HMM hidden states.")

    return [mvr_hidden_order.index(h) for h in hmm_hidden_order]


def _build_mvr_factor_info(
    mvr,
    T,
    hmm_hidden_order,
    *,
    dtype=torch.float32,
    device="cpu",
):
    """
    Build torch factors for one MVR without padding mediation dimensions.

    Returns a dictionary containing:
      - active interval [a, b]
      - homogeneous/inhomogeneous flag
      - ini tensor
      - upd tensor or list of tensors
      - evl tensor or list of tensors

    Hidden axes are reordered to match HMM hidden-state order.
    """
    repn = mvr.repn
    a, b = _mvr_active_range(mvr, T)
    hidden_perm = _hidden_permutation_for_mvr(mvr, hmm_hidden_order)
    is_inhom = _is_inhom_mvr(mvr)

    ini_np = np.asarray(repn.ini_array)[hidden_perm, :]
    ini = torch.as_tensor(ini_np, dtype=dtype, device=device)

    if is_inhom:
        upd = [
            torch.as_tensor(
                np.asarray(upd_t)[hidden_perm, :, :],
                dtype=dtype,
                device=device,
            )
            for upd_t in repn.upd_array
        ]
        evl = [
            torch.as_tensor(
                np.asarray(evl_t),
                dtype=dtype,
                device=device,
            )
            for evl_t in repn.evl_array
        ]
    else:
        upd = torch.as_tensor(
            np.asarray(repn.upd_array)[hidden_perm, :, :],
            dtype=dtype,
            device=device,
        )
        evl = torch.as_tensor(
            np.asarray(repn.evl_array),
            dtype=dtype,
            device=device,
        )

    return {
        "mvr": mvr,
        "a": a,
        "b": b,
        "is_inhom": is_inhom,
        "ini": ini,
        "upd": upd,
        "evl": evl,
    }


def _active_mvr_indices(mvr_infos, t):
    """
    Return indices of MVRs active at global time t.
    """
    return [i for i, info in enumerate(mvr_infos) if info["a"] <= t <= info["b"]]


def _mvr_dim_at_time(info, t):
    """
    Return the mediation dimension of an active MVR at global time t.
    """
    local_t = t - info["a"]

    if info["is_inhom"]:
        return int(info["evl"][local_t].shape[0])

    return int(info["evl"].shape[0])


def _shape_for_time(K, mvr_infos, active_indices, t):
    """
    Return augmented tensor shape at time t.

    Shape convention:
        (K, mediation dims for active MVRs in active_indices order)
    """
    return tuple([K] + [_mvr_dim_at_time(mvr_infos[i], t) for i in active_indices])


def _mvr_init_factor(info):
    """
    Return initialization factor for an MVR.

    Shape:
        (K, M_0)
    """
    return info["ini"]


def _mvr_update_factor(info, t):
    """
    Return update factor for an MVR active at both t - 1 and t.

    Shape:
        (K, M_curr, M_prev)

    Here t is the global current time, i.e. the transition is t - 1 -> t.
    """
    if info["is_inhom"]:
        local_update_index = t - info["a"] - 1
        return info["upd"][local_update_index]

    return info["upd"]


def _mvr_eval_factor(info, t):
    """
    Return evaluation factor for an MVR at global time t.

    Shape:
        (M_t,)
    """
    if info["is_inhom"]:
        local_t = t - info["a"]
        return info["evl"][local_t]

    return info["evl"]


def _decode_dynamic_augmented_path(
    augmented_index_path,
    active_by_time,
    mvr_infos,
):
    """
    Decode dynamic augmented index tuples into dictionaries.

    Each augmented index tuple has shape:

        (hidden_index, active_mvr_0_state_index, active_mvr_1_state_index, ...)

    where active MVRs vary by time.
    """
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
            mvr = info["mvr"]
            local_t = t - info["a"]

            entry["mvr_indices"][mvr_i] = m_idx

            if info["is_inhom"]:
                entry["mvr_states"][mvr_i] = mvr.mediation_states[local_t][m_idx]
            else:
                entry["mvr_states"][mvr_i] = mvr.mediation_states[m_idx]

        decoded.append(entry)

    return decoded


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
    """
    Einsum-based Viterbi for an MVR-augmented constrained HMM using dynamic
    MVR axes.

    Unlike the older dna_algorithms.py implementation, this version does not
    assume a fixed augmented tensor shape over time. Instead,

        V[t].shape == (K, d_i(t) for i active at t)

    where an MVR is active only over its time_range. If time_range is None,
    the MVR is active over [0, T - 1].

    Parameters
    ----------
    model : MVR_CHMM-like
        Object containing a hidden_markov_model and a list of MVR constraints.
    observed : list
        Observed sequence using external observed-state labels.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors.
    device : str or torch.device, optional
        Torch device.
    normalize : bool, optional
        If True, divide each Viterbi slice by its maximum value.
    return_augmented : bool, optional
        If True, return decoded dynamic augmented path information.
    return_score : bool, optional
        If True, return best-path log score estimate.

    Returns
    -------
    hidden_path : list
        Best hidden-state path in external labels.

    Optionally also returns:
      - augmented_path : list[dict]
      - best_logprob : float
    """
    if len(observed) == 0:
        raise InvalidInputError("observed sequence must be nonempty.")

    hmm, constraints = _get_mvr_chmm_parts(model)
    T = len(observed)

    hmm_hidden_order = list(hmm.hidden_to_external)

    start_vec, transition_mat, emission_mat = _hmm_to_torch(
        hmm,
        dtype=dtype,
        device=device,
    )

    emit_weights = _observations_to_emit_weights(
        hmm,
        emission_mat,
        observed,
    )

    K = int(start_vec.shape[0])
    C = len(constraints)

    mvr_infos = [
        _build_mvr_factor_info(
            mvr,
            T,
            hmm_hidden_order,
            dtype=dtype,
            device=device,
        )
        for mvr in constraints
    ]

    active_by_time = [_active_mvr_indices(mvr_infos, t) for t in range(T)]

    shapes = [_shape_for_time(K, mvr_infos, active_by_time[t], t) for t in range(T)]

    # Each time step uses at most:
    #   current hidden + current active MVR axes
    #   previous hidden + previous active MVR axes
    #
    # PyTorch integer einsum labels must be less than 52.
    max_active = max(len(active) for active in active_by_time) if T > 0 else 0
    if 2 * (1 + max_active) >= 52:
        raise InvalidInputError(
            "Too many simultaneously active MVR constraints for this einsum "
            "implementation. PyTorch integer einsum subscripts must be < 52."
        )

    backptr = []
    log_score = 0.0

    # ------------------------------------------------------------------
    # Initialization at t = 0.
    # ------------------------------------------------------------------
    active0 = active_by_time[0]
    curr_shape = shapes[0]

    curr_indices = list(range(1 + len(active0)))
    curr_h_label = curr_indices[0]

    curr_m_label = {mvr_i: curr_indices[pos + 1] for pos, mvr_i in enumerate(active0)}

    einsum_args = [
        start_vec,
        [curr_h_label],
        emit_weights[0],
        [curr_h_label],
    ]

    for mvr_i in active0:
        info = mvr_infos[mvr_i]
        m_label = curr_m_label[mvr_i]

        # Since the MVR is active at t=0, it must have a == 0.
        einsum_args.extend(
            [
                _mvr_init_factor(info),
                [curr_h_label, m_label],
            ]
        )

        # If the MVR evaluates at t=0, apply evl immediately.
        if info["b"] == 0:
            einsum_args.extend(
                [
                    _mvr_eval_factor(info, 0),
                    [m_label],
                ]
            )

    einsum_args.append(curr_indices)

    V_prev = torch.einsum(*einsum_args)

    if tuple(V_prev.shape) != curr_shape:
        raise RuntimeError(
            f"Internal shape mismatch at t=0. "
            f"Expected {curr_shape}, got {tuple(V_prev.shape)}."
        )

    scale = V_prev.max()

    if scale.item() <= 0:
        raise InvalidInputError("No feasible augmented path at time 0.")

    if normalize:
        log_score += torch.log(scale).item()
        V_prev = V_prev / scale

    # ------------------------------------------------------------------
    # Forward Viterbi pass.
    # ------------------------------------------------------------------
    for t in range(1, T):
        prev_active = active_by_time[t - 1]
        curr_active = active_by_time[t]

        prev_shape = shapes[t - 1]
        curr_shape = shapes[t]

        # Current labels:
        #   current hidden plus current active MVR axes.
        curr_indices = list(range(1 + len(curr_active)))
        curr_h_label = curr_indices[0]

        # Previous labels:
        #   previous hidden plus previous active MVR axes.
        offset = len(curr_indices)
        prev_indices = list(range(offset, offset + 1 + len(prev_active)))
        prev_h_label = prev_indices[0]

        curr_m_label = {
            mvr_i: curr_indices[pos + 1] for pos, mvr_i in enumerate(curr_active)
        }

        prev_m_label = {
            mvr_i: prev_indices[pos + 1] for pos, mvr_i in enumerate(prev_active)
        }

        einsum_args = [
            V_prev,
            prev_indices,
            transition_mat,
            [prev_h_label, curr_h_label],
            emit_weights[t],
            [curr_h_label],
        ]

        # Add MVR factors.
        #
        # Cases:
        #   1. MVR active at current but not previous: it starts at t.
        #      Use ini[h_curr, m_curr].
        #
        #   2. MVR active at both previous and current: use upd.
        #
        #   3. MVR active at previous but not current: it ended at t - 1.
        #      No factor is needed. Its previous axis remains in prev_indices
        #      and will be maximized out by the Viterbi max.
        #
        #   4. MVR inactive at both: no axis and no factor.
        for mvr_i, info in enumerate(mvr_infos):
            in_prev = mvr_i in prev_m_label
            in_curr = mvr_i in curr_m_label

            if in_curr and not in_prev:
                # MVR starts at current time t.
                # This can happen only when t == a_i.
                if t != info["a"]:
                    raise RuntimeError(
                        "Internal active-set inconsistency: MVR appears in current "
                        "but not previous at a non-start time."
                    )

                einsum_args.extend(
                    [
                        _mvr_init_factor(info),
                        [curr_h_label, curr_m_label[mvr_i]],
                    ]
                )

            elif in_curr and in_prev:
                # MVR is active across transition t - 1 -> t.
                einsum_args.extend(
                    [
                        _mvr_update_factor(info, t),
                        [
                            curr_h_label,
                            curr_m_label[mvr_i],
                            prev_m_label[mvr_i],
                        ],
                    ]
                )

            elif in_prev and not in_curr:
                # MVR ended at t - 1.
                # Evaluation was already applied at t - 1 if needed.
                # No transition factor is needed; prev mediation state is
                # simply part of the predecessor state maximized over.
                pass

            else:
                # inactive in both
                pass

        # Apply evaluation factors for MVRs that end at current time.
        for mvr_i in curr_active:
            info = mvr_infos[mvr_i]

            if t == info["b"]:
                einsum_args.extend(
                    [
                        _mvr_eval_factor(info, t),
                        [curr_m_label[mvr_i]],
                    ]
                )

        # Output contains current augmented state first, then previous
        # augmented state. The previous part is flattened and maximized over.
        output_indices = curr_indices + prev_indices
        einsum_args.append(output_indices)

        candidate = torch.einsum(*einsum_args)

        expected_candidate_shape = curr_shape + prev_shape
        if tuple(candidate.shape) != expected_candidate_shape:
            raise RuntimeError(
                f"Internal candidate shape mismatch at t={t}. "
                f"Expected {expected_candidate_shape}, got {tuple(candidate.shape)}."
            )

        candidate = candidate.reshape(curr_shape + (-1,))

        V_curr, max_prev_flat = torch.max(candidate, dim=-1)
        backptr.append(max_prev_flat)

        scale = V_curr.max()

        if scale.item() <= 0:
            raise InvalidInputError(f"No feasible augmented path at time {t}.")

        if normalize:
            log_score += torch.log(scale).item()
            V_curr = V_curr / scale

        V_prev = V_curr

    # ------------------------------------------------------------------
    # Termination.
    # ------------------------------------------------------------------
    final_scale = V_prev.max()

    if final_scale.item() <= 0:
        raise InvalidInputError("No feasible augmented path at final time.")

    final_flat = int(torch.argmax(V_prev).detach().cpu().item())
    final_idx = tuple(int(x) for x in np.unravel_index(final_flat, shapes[T - 1]))

    if not normalize:
        log_score = torch.log(final_scale).item()

    # ------------------------------------------------------------------
    # Backtracking.
    # ------------------------------------------------------------------
    augmented_index_path = [final_idx]
    curr_idx = final_idx

    for t in range(T - 1, 0, -1):
        prev_flat = int(backptr[t - 1][curr_idx].detach().cpu().item())

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
